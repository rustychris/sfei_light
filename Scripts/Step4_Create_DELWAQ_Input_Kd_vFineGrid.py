# -*- coding: utf-8 -*-
"""
This script is modified from Taylor Winchell's script on google drive
https://drive.google.com/drive/u/0/folders/1-SHOKYV0BvsDVIHDn3Ji1OZtWfKR2PXT 

Instead of generating DWAQ input directly, this script generates a netcdf file that
will be read by the stompy waq_scenario script to create spatially and temporally 
varying light field. 

@author: zhenlinz
"""

# Read in netCDF input file (spec'ed from version) from "...\TWDR_Method_fy21\Data_Kd_Shifted"
# Read in polygons associated with each Kd time-series
# Read in DELWAQ bay-model grid
# Extrapolate time-series of Kd to polygons
# Smooth polygons onto grid using N-iterations of nearest-neighboor averaging at each time step
# Write output netCDF file for use as DELWAQ input, or for aggregating to the agg grid
# Some bugs in xarray writing netcdf files to Google Drive, so write to local space
# then copy to "...\TWDR_Method_fy21\Data_DELWAQ_InputFiles"

import matplotlib.pyplot as plt
from stompy.model.delft import dfm_grid
import pandas as pd
from stompy.spatial import wkb2shp
from shapely import geometry
import numpy as np
import pandas as pd
import os
import xarray as xr 
from scipy import sparse
from scipy.sparse import linalg
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta


start = np.datetime64('2017-08-01 00:00:00')
end = np.datetime64('2018-10-01 00:00:00')

version = 'Kd_PropShift' # Kd Kd_PropShift or Kd_DiffShift

SmoothIterations = 50 # nearest-neighbor averagin iterations to smooth polygon time-series

dir_input = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Kd_Shifted'
file_input = os.path.join(dir_input,version+'_LongTermHourly.nc')

dir_output = r'C:\hello' # errors writing to Google Drive for some reason, xarray sucks...
file_output = os.path.join(dir_output,version+'_forDELWAQ_' + pd.to_datetime(start).strftime('%Y%m%d')+'_to_'+pd.to_datetime(end).strftime('%Y%m%d')+'.nc')

#dir_output = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_DELWAQ_InputFiles'
#file_output = os.path.join(dir_output,version+'_forDELWAQ_' + pd.to_datetime(start).strftime('%Y%m%d')+'_to_'+pd.to_datetime(end).strftime('%Y%m%d')+'.nc')

dir_grid = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Grid'
file_grid = os.path.join(dir_grid,'wy2013c_waqgeom.nc')


# %% Load input data

# load xarray dataset
ds = xr.open_dataset(file_input) 

# Load Del-waq grid 
g=dfm_grid.DFMGrid(file_grid)
cc=g.cells_centroid() # the xy locations we are aiming for


# %%
    
# Load qgis polygons to determine where each station's polygon lies withing the delwaq grid 
extrap_polygons=wkb2shp.shp2geom(os.path.join(dir_grid,'light_field_polygons.shp'))

# Compute masks of where each polygon from extrap_polygons intersects with the Delwaq grid 
# precompute masks for when we apply this many times
# probably takes 20sec
masks=[ g.select_cells_intersecting(p)
        for p in extrap_polygons['geom'] ]

# Setup the pieces required for neighbor smoothing of cells 
N=g.Ncells() # number of cells 
D=sparse.dok_matrix((N,N),np.float64) # .dok creates a normal matrix, but takes up less memory. 

# This for loop sets up the D matrix. Which essentially creates one grid for each cell. 
# The grid weights the neighbors of that cell accordingly. It results in a sparse matrix of ~50000x50000
# This should only need to be run once. 
f=1.
for c in range(g.Ncells()):
    nbrs=np.array( g.cell_to_cells(c) )
    nbrs=nbrs[nbrs>=0]
    D[c,c]=1-f
    for nbr in nbrs:
        D[c,nbr] = f/float(len(nbrs))

Dcsr=D.tocsr() # puts in csr format for easier calculations. compressed sparse row 

#%%# These are the steps which are done per output time step

days = ds.time.values# Days of data 
#days = pd.to_datetime(ds.time.values).to_numpy() # Days of data 

index1 = np.where(pd.to_datetime(ds.time.values).to_numpy() == start)[0] # start time
index2 = np.where(pd.to_datetime(ds.time.values).to_numpy() == end)[0] # end time 

counter = 1
#for day in days[2866:len(days)]:
for day in days[index1[0]:index2[0]+1]: 
    
    f_polyfill=np.zeros(g.Ncells(),'f8') # zeros array with the length of the number of cells in the Delwaq grid 
    
    # This for loop matches the station values with the correct grid cell 
    for mask,extrap_polygon in zip(masks,extrap_polygons):
        polygon_name = extrap_polygon[0]
        #print(polygon_name)
        f_polyfill[mask] = ds.light_ext_coef.loc[dict(time = day, station = polygon_name)].values # set the values of the grid array
    
    ## Smoothing Section 
    f_smooth=f_polyfill.copy() # idempotent. 
    
    # This for loop does the smoothing. 
    for it in range(SmoothIterations):
        f_smooth=Dcsr.dot(f_smooth)
    
    if counter == 1: 
        f_smooth_agg = f_smooth.copy() # f smooth aggregated 
    else:
        newrow = f_smooth.copy() 
        f_smooth_agg = np.vstack([f_smooth_agg, newrow])
    
    counter = counter + 1
    print(counter)    

# This fails... maybe a stompy change
# Define results plotting function 
#def plot_result(num,title,values):
#    plt.figure(num).clf()
#    ccoll=g.plot_cells(values=values,cmap='CMRmap_r') #'copper_r' 
#
#    scat = plt.scatter(ds.utm_E.values, ds.utm_N.values, cmap = 'CMRmap_r',  alpha = 0, s = 40, zorder=2)    
##    scat.set_edgecolor('k')
#    scat.set_clim(ccoll.get_clim())
#    
#    plt.gca().set_title(title)
#    plt.gca().set_facecolor('Gainsboro')
#    plt.axis('equal')
    
#plot_result(1,'Polygon fill smooth',f_polyfill) # plot un-smoothing result
#plot_result(2,'Polygon fill smooth',newrow) # plot un-smoothing result


#%% Generate a netcdf file for DELWAQ input
    
cells = np.linspace(0,N-1,N).astype(int)
dataout = xr.DataArray(data=f_smooth_agg, coords=[days[index1[0]:index2[0]+1],cells], 
                       dims=['time', 'nFlowElem'],attrs={'unit':'m-1',
                            'variable name':'light_ext_coef',
                            'source':'It is a long story'})
    
dataout.to_netcdf(file_output)
