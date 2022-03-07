# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:58:17 2021

@author: derekr
"""

import geopandas as gpd
import numpy as np
import xarray as xr
import os


OutputTimeStep = np.timedelta64(1,'h')

dir_lightfield = r'C:\hello'

file_output = os.path.join(dir_lightfield,'Kd_PropShift_forDELWAQ_20120801_to_20181001_AggGrid_141.nc')

files = ['Kd_PropShift_forDELWAQ_20120801_to_20131001_AggGrid_141.nc',
         'Kd_PropShift_forDELWAQ_20131001_to_20141001_AggGrid_141.nc',
         'Kd_PropShift_forDELWAQ_20141001_to_20151001_AggGrid_141.nc',
         'Kd_PropShift_forDELWAQ_20151001_to_20161001_AggGrid_141.nc',
         'Kd_PropShift_forDELWAQ_20161001_to_20171001_AggGrid_141.nc',
         'Kd_PropShift_forDELWAQ_20171001_to_20181001_AggGrid_141.nc']

# %% Read and org data from multiple files

wydict = {}

s = np.datetime64('2222-01-01') # initialize start date as late date
e = np.datetime64('1000-01-01') # initialize end date as early date

for n,file in enumerate(files):
    
    file_path = os.path.join(dir_lightfield,file)
    data = xr.open_dataset(file_path)
    data.close()
    
    wydict[n] = {}
    wydict[n]['ts'] = data.time.values.astype('datetime64')
    wydict[n]['nFlowElem'] = data.nFlowElem.values
    wydict[n]['Kd'] = data.kd.values
    
    if wydict[n]['ts'][0]<s:
        s = wydict[n]['ts'][0]
        
    if wydict[n]['ts'][-1]>e:
        e = wydict[n]['ts'][-1]

# %% Build the output dictionary
        
aggdict = {}

aggdict['ts'] = np.arange(s,e+OutputTimeStep,OutputTimeStep)
aggdict['nFlowElem'] = wydict[0]['nFlowElem'] # assuming nFlowElem is the same for all of these

aggdict['kd'] = np.ones((len(aggdict['ts']),len(aggdict['nFlowElem'])))

for n in wydict.keys():
    
    _,iA,iB = np.intersect1d(aggdict['ts'],wydict[n]['ts'],return_indices=True)
    
    aggdict['kd'][iA,:] = wydict[n]['Kd'][iB,:]
    
# %% Dump back into netcdf and write output
    
output = xr.DataArray(aggdict['kd'],coords=[aggdict['ts'],aggdict['nFlowElem']],dims=['time','nFlowElem'])
output = output.to_dataset(name='kd')

output.to_netcdf(file_output)