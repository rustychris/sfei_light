# -*- coding: utf-8 -*-
"""
This script is derived from the steps in RHStep4_Create_DELWAQ_Input_kd_vFineGrid,
but streamlines the calculation to yield predicted Kd at a finite number of stations
for specific times.

To that end, spatial weights are calculated once, and long time series of predictions
at the specified locations are possible (whereas RHStep4... is optimized to predict
at all points, for a short period of time)
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
from stompy import utils
from shapely import geometry
import numpy as np
import pickle
import pandas as pd
import os
import xarray as xr 
from scipy import sparse
from scipy.sparse import linalg
from stompy.io.local import usgs_sfbay
from stompy.spatial import proj_utils

from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta


version = 'Kd_PropShift' # Kd Kd_PropShift or Kd_DiffShift
SmoothIterations = 50 # nearest-neighbor averagin iterations to smooth polygon time-series

dir_input = '../Data_Kd_Shifted_RH'
file_input = os.path.join(dir_input,version+'_LongTermHourly.nc')

dir_output = '../Figures_ErrorPostExtrap' 
#file_output = os.path.join(dir_output,version+'_forDELWAQ_' + pd.to_datetime(start).strftime('%Y%m%d')+'_to_'+pd.to_datetime(end).strftime('%Y%m%d')+'.nc')

dir_grid = '../Grid'
file_grid = os.path.join(dir_grid,'wy2013c_waqgeom.nc')


# %% Load input data

# load xarray dataset
ds = xr.open_dataset(file_input) 
# time comes in as a string...
ds['time']=('time',),ds.time.values.astype(np.datetime64)

# Load Del-waq grid 
g=dfm_grid.DFMGrid(file_grid,cleanup=True)
cc=g.cells_centroid() # the xy locations we are aiming for


# %%
    
# Load qgis polygons to determine where each station's polygon lies withing the delwaq grid 
extrap_polygons=wkb2shp.shp2geom(os.path.join(dir_grid,'light_field_polygons.shp'))

# Compute masks of where each polygon from extrap_polygons intersects with the Delwaq grid 
# precompute masks for when we apply this many times
# probably takes 20sec
# select by center so that each cell hits exactly one polygon.
masks=[ g.select_cells_intersecting(p, by_center='centroid')
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

#%%


def timeseries_at_xy(sample_xy):
    c=g.select_cells_nearest(sample_xy)
    
    # 16 masks in shapefile order., 16 stations, match by names.
    
    # I want a vector of weights that lines up with ds.station
    
    weights=np.zeros(ds.dims['station'])
    
    for i,station in enumerate(ds.station.values):
        mask_i=np.nonzero(extrap_polygons['sta_zones']==station)[0][0]
        f_polyfill=np.zeros(g.Ncells(),'f8') 
        f_polyfill[masks[mask_i]]=1.0
    
        ## Smoothing Section 
        f_smooth=f_polyfill.copy() # idempotent. 
        
        # This for loop does the smoothing. 
        for it in range(SmoothIterations):
            f_smooth=Dcsr.dot(f_smooth)
    
        if 0:    
            plt.figure(100+i).clf()
            ccoll=g.plot_cells(values=f_smooth,clim=[0,0.01])
            plt.axis('equal')
        weights[i]=f_smooth[c]
    
    assert np.allclose(weights.sum(),1.0)    
    
    pred_timeseries=(ds.light_ext_coef.values * weights[None,:]).sum(axis=1)
    result_ds=xr.Dataset()
    result_ds['Kd']=('time',),pred_timeseries
    result_ds['time']=ds.time
    return result_ds

sample_xy=[558312., 4171395.]

result=timeseries_at_xy(sample_xy)

#%% 
# This can then be compared agains a cruise, mooring, etc.

# Load USGS cruise data:    
dir_cruise = '../Data_Cruise'
file_cruisessc = os.path.join(dir_cruise,'Curated_SiteSpec_SSCandKD_2_to_36.p')
cruise = pickle.load(open(file_cruisessc,'rb'))

#%%
# cruise.sites, array of station indices.
# cruise[2].ts_pst
# cruise[2].ssc_mgL
#   ..Kd
# chla_ugl.

if not os.path.exists(dir_output):
    os.makedirs(dir_output)
    
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')

for site in cruise.sites:
    print(site)
    ll=usgs_sfbay.station_number_to_lonlat(site)
    xy=ll2utm(ll)
    prd=timeseries_at_xy(xy)
    obs=xr.Dataset()
    obs['time']=('time',),cruise[site].ts_pst
    obs['Kd']=('time',),cruise[site].Kd
    
    t_min=max(prd.time.values[0],
              obs.time.values[0])
    t_max=min(prd.time.values[-1],
              obs.time.values[-1])
    
    prd=prd.isel(time=(prd.time.values>=t_min) & (prd.time.values<=t_max))
    obs=obs.isel(time=(obs.time.values>=t_min) & (obs.time.values<=t_max))
    
    if len(prd.time)==0 or len(obs.time)==0:
        print(f"Site {site}: not enough overlap")
        continue
    
    for data in [prd, obs]:
        data['z_photo']=4./data['Kd']
        data['dn']=('time',),utils.to_dnum(data['time'].values)
        
    obs['pred']=('time',),np.interp(obs['dn'].values,
                                    prd['dn'].values,prd['z_photo'].values,
                                    left=np.nan,right=np.nan)
    # Metrics against the instantaneous values:
    pval=obs['pred'].values
    oval=obs['z_photo'].values
    
    valid=np.isfinite(pval * oval)
    
    instant_metrics=[f"USGS Station {site}\n",
                     f"Observed mean: {oval[valid].mean():.2f}m",
                     f"Predicted mean: {pval[valid].mean():.2f}m",
                     f"RMSE: {np.sqrt(np.nanmean( (oval-pval)**2)):.2f}m",
                     f"Obs. std: {np.std(oval[valid]):.2f}m",
                     f"Pred. std: {np.std(pval[valid]):.2f}m",
                     f"R$^2$: {np.corrcoef(oval[valid],pval[valid])[0,1]**2:.3f}"]
    txt="\n".join(instant_metrics)
    print(txt)
    
    fig=plt.figure(site)
    fig.clf()
    fig.set_size_inches((7,4),forward=True)
    fig,ax=plt.subplots(num=site)
    ax.plot(prd.time,prd.z_photo,label='Pred.')
    ax.plot(obs.time,obs.z_photo,lw=0,marker='.',label='Obs.')
    ax.axis(xmin=t_min,xmax=t_max)
    ax.set_ylabel('4/K$_D$ (m)')
    ax.text(1.03,0.98,txt,transform=ax.transAxes,va='top')
    fig.subplots_adjust(right=0.7)
    fig.savefig(os.path.join(dir_output,f"timeseries_and_metrics-USGS{site}.png"),
                dpi=150)

    
