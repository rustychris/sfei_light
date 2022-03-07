# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:55:32 2019
Take the interpolated full-resolution light field and perform a spatial aggregation with the aggregated grid
the module dpp can be found on hpc:/hpcvol1/zhenlin/Ztoolbox
@author: zhenlinz
"""

import dwaq.PostProcessing as dpp
import geopandas as gpd
import numpy as np
import logging
import xarray as xr
import os

input_name = 'Kd_PropShift_forDELWAQ_20171001_to_20181001.nc'

#dir_input = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_DELWAQ_InputFiles'
dir_input = r'C:\hello'
file_input = os.path.join(dir_input,input_name)

dir_output = r'C:\hello'
file_output = os.path.join(dir_output,input_name[:-3]+'_AggGrid_141.nc')

dir_grid = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Grid'
file_finegrid = os.path.join(dir_grid,'wy2013c_waqgeom.nc')
file_agggrid = os.path.join(dir_grid,'flowgeom141.nc')

gridf = dpp.dwaqGrid(file_finegrid)
grida = dpp.dwaqGrid(file_agggrid)

gpdf = gridf.toGeopandasPoly()
gpda = grida.toGeopandasPoly()

mapping = gpd.sjoin(gpdf, gpda, how="left", op='within')

# The following should not happen but just a quick check
#if np.any(np.isnan(mapping.index_right.values)):
#    contiguous=False
#    indnan = np.where(np.isnan(mapping.index_right.values))[0]
#    logging.warning("unclassfied cells identified: correction needed at {}".format(indnan))
    
# %% Some really messed up workspace clearing to handle xarray bugs
    
for name in dir():
    if not (name.startswith('file_output')) | (name.startswith('_')) | (name.startswith('gpd')) | (name.startswith('file')) | (name.startswith('mapping')):
        del globals()[name]
        
import xarray as xr
import numpy as np
import os

#%% Is this a fast way to aggregate?
light = xr.open_dataset(file_input)
light_full = light.__xarray_dataarray_variable__.values
light_avg = []
for i in np.arange(len(gpda)): # for each aggregated polygon
    poly_cells = np.nonzero(mapping.index_right==i)[0] # the included high resolution grid cells
    light_i = light_full[:,poly_cells].mean(axis=1)
    light_avg.append(light_i)
   
    
time = light.time.values
nFlowElem = gpda.index.values
light_avg = np.asarray(light_avg).T
light_new = xr.DataArray(light_avg,coords=[time,nFlowElem],dims=['time','nFlowElem'])
light_new = light_new.to_dataset(name='kd')

light_new.to_netcdf(file_output)

