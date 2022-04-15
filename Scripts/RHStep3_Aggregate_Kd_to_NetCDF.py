# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:09:02 2018

@author: derekr
"""

# This script builds netCDF files of Kd time-series that will 
# will be converted to Delwaq input in subsequent steps. 
# Input Kd time-series are read from TWDR_Method_fy21\Data_Kd_Shifted.
# Site/polygon coordinates are read in from Station_Polygon_Coordinates.csv

# Since observed/modeled time-series do not exist for Golden Gate or Ocean,
# we estimates these as a function of nearby Alcatraz Island,
# see the Alcatraz_to_GoldenGate and Alcatraz_to_Ocean conversion factors. 

# Independent netCDF files for the orignal Kd, and both shifted versions (prop/ratio and difference shfited)
# are written to '...\TWDR_Method_fy21\Data_Kd_Shifted'

import pandas as pd
import xarray as xr
import numpy as np 
import os

Alcatraz_to_GoldenGate = 0.75 # estimate Golden Gate as f(Alcatraz)
Alcatraz_to_Ocean = 0.5 # estimate Ocean as f(Alcatraz)

dir_data = '../Data_Kd_Shifted_RH'

dir_coords = '..'
file_coords = os.path.join(dir_coords,'Station_Polygon_Coordinates.csv')

# %%

station_info = pd.read_csv(file_coords)

stations = list(station_info.Site)

for n in range(len(stations)):
    
    sta = stations[n]
    
    if (sta!='Golden_Gate_Bridge') & (sta!='Ocean'): # there is no input data for these sites
        data = pd.read_csv(os.path.join(dir_data,sta+'_Kd_Hourly_LongTerm.csv'))
    
        if n==0: # if we are just getting started, pre-allocate the output array
            
            ts_pst = data.ts_pst
            
            kd = np.ones((len(data.ts_pst),len(stations)))*np.nan
            kd_diffshift = np.ones((len(data.ts_pst),len(stations)))*np.nan
            kd_propshift = np.ones((len(data.ts_pst),len(stations)))*np.nan
        
        kd[:,n] = data['Kd'].to_numpy()
        kd_diffshift[:,n] = data['Kd_DiffShifted'].to_numpy()
        kd_propshift[:,n] = data['Kd_PropShifted'].to_numpy()

# Synthesize Golden Gate and Ocean as f(Alcatraz)        
nAlcatraz = np.array(stations) == 'Alcatraz_Island'        
nGoldenGate = np.array(stations) =='Golden_Gate_Bridge'
nOcean = np.array(stations) == 'Ocean'

kd[:,nGoldenGate] = kd[:,nAlcatraz] * Alcatraz_to_GoldenGate
kd_diffshift[:,nGoldenGate] = kd_diffshift[:,nAlcatraz] * Alcatraz_to_GoldenGate
kd_propshift[:,nGoldenGate] = kd_propshift[:,nAlcatraz] * Alcatraz_to_GoldenGate

kd[:,nOcean] = kd[:,nAlcatraz] * Alcatraz_to_Ocean
kd_diffshift[:,nOcean] = kd_diffshift[:,nAlcatraz] * Alcatraz_to_Ocean
kd_propshift[:,nOcean] = kd_propshift[:,nAlcatraz] * Alcatraz_to_Ocean


# %% Write output net cdf files

#Set coordinates 
coords = {'time': (['time'], ts_pst),
        'station': (['station'], stations), 
        'utm_E': (['utm_E'],station_info.utm_e.to_numpy()),
        'utm_N': (['utm_N'], station_info.utm_n.to_numpy()),
        'latitude': (['latitude'],station_info.lat.to_numpy()), 
        'longitude': (['longitude'],station_info.lon.to_numpy())}

#Create netcdf datasets for use in script to build gridded light field
Kd_LongTermHourly = xr.Dataset({'light_ext_coef': (['time','station'],kd)},coords=coords)
Kd_LongTermHourly.light_ext_coef.attrs['units'] = 'm-1'
Kd_LongTermHourly.to_netcdf(os.path.join(dir_data,'Kd_LongTermHourly.nc'))

Kd_DiffShift_LongTermHourly = xr.Dataset({'light_ext_coef': (['time','station'],kd_diffshift)},coords=coords)
Kd_DiffShift_LongTermHourly.light_ext_coef.attrs['units'] = 'm-1'
Kd_DiffShift_LongTermHourly.to_netcdf(os.path.join(dir_data,'Kd_DiffShift_LongTermHourly.nc'))

Kd_PropShift_LongTermHourly = xr.Dataset({'light_ext_coef': (['time','station'],kd_propshift)},coords=coords)
Kd_PropShift_LongTermHourly.light_ext_coef.attrs['units'] = 'm-1'
Kd_PropShift_LongTermHourly.to_netcdf(os.path.join(dir_data,'Kd_PropShift_LongTermHourly.nc'))

