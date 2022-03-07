# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:28:09 2020

@author: derekr
"""
# This script builds single observational SSC time-series from the scattering
# of raw SSC and turbidity files available at each site/sub-folder shown
# in "...\TWDR_Method_fy21\Data_SSC_Raw."

# This script uses the meta data file DataInfo_Raw_LightFieldData.xlsx
# to handle different raw data file types. 

# Time-series are built around the Priority=1 reference sensor. 
# SSC data from all other sensors are normalized to the Priority1 station using
# a log-linear fit between the two sensors' overlapping data. If there are no overlapping
# data (San Mateo Bridge, only), the unconverted SSC data are used. If only turbidity data
# are available, the turbidity data are converted to SSC using a conversion calculated
# from a linear fit of whatever overlapping SSC and turbidity data have already been incorporated. 
# The converted turbidity data, now as SSC, are then normalized to the reference station, 
# as with all stations. Plots and output CSVs (with flags marking priority stations are 
# written to Data_SSC_Processed


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from dotmap import DotMap
import os
import pickle
from sklearn.linear_model import LinearRegression
from math import e

TimeStep = 15 # minutes, output time step

MinOverlap= 100 # minimum overlapping points for any linear regression

dir_parent = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_SSC_Raw'
file_datainfo = os.path.join(dir_parent,'DataInfo_Raw_LightFieldData.xlsx')

dir_output = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_SSC_Processed'

# Default turb to ssc conversion (applied if no overlapping turbidity and SSC data are available to
# back out a site-specific conversion from USGS)
# SSC = a * turb^b
turb2ssc_a = 4.35
turb2ssc_b = 0.834

# Notes
# First file (primary sensor) must have SSC data, later ones can have just turbidity and will determine SSC-turb relationships as needed from primary site

# %% Handle the data, site by site, and write output

info = pd.read_excel(file_datainfo,'data')
sites = list(np.unique(info.site))

for site in sites:
#for site in ['Dumbarton_Bridge']:
    
    data = DotMap() # Output data structure
        
    dir_site = os.path.join(dir_parent,site) # sub-folder with site-specific files
    
    # Site-specific meta data
    priority = info.loc[info.site==site,'file_priority'].to_numpy()
    sscfiles = info.loc[info.site==site,'ssc_file'].to_numpy(dtype='str')
    turbfiles = info.loc[info.site==site,'turbidity_file'].to_numpy()
    fileformat = info.loc[info.site==site,'file_format'].to_numpy()
    names = info.loc[info.site==site,'Name'].to_numpy()
    
    # Initialize output time-series limits at opposite extremes to be over-written as input files are read in
    # Limits will ultimately reflect the earliest and latest available data.      
    start = np.datetime64('2100-01-01')
    end = np.datetime64('1900-01-01')
    
    sraw = DotMap() # Site-specific data structure for handling; over-written for each site in loop
    
    for n in np.sort(priority): # Loop over site-specific files in order of spec'ed priority
        
        idx = priority==n # meta data file line index
        
        # File for this priority, and its format
        sscfile = str(sscfiles[idx][0])
        turbfile = str(turbfiles[idx][0])
        fmt = str(fileformat[idx][0])
        
        if '.csv' in sscfile: # if there is an SSC CSV, handle the raw data, if not, makes this dictionary value empty
            if fmt == 'usgs_v1': # if there is a file here, check the format
                rawssc = pd.read_csv(os.path.join(dir_site,sscfile),comment='#')
                rawssc['ts_pst'] = pd.to_datetime(rawssc[' Timestamp (UTC-08:00)'])
                sraw[n]['ssc']['ts_pst'] = rawssc['ts_pst'].to_numpy()
                sraw[n]['ssc']['ssc_mgL'] = rawssc[' Value'].to_numpy()
                
                # Reset start and end times if these new data expand the limits
                if sraw[n]['ssc']['ts_pst'][0]<start:
                    start = sraw[n]['ssc']['ts_pst'][0]
                if sraw[n]['ssc']['ts_pst'][-1]>end:
                    end = sraw[n]['ssc']['ts_pst'][-1]
            
            elif fmt == 'usgs_v2': # if there is a file here, check the format
                    rawssc = pd.read_csv(os.path.join(dir_site,sscfile),na_values='Eqp')
                    rawssc['ts_pst'] = pd.to_datetime(rawssc['ts_pst'])
                    sraw[n]['ssc']['ts_pst'] = rawssc['ts_pst'].to_numpy()
                    sraw[n]['ssc']['ssc_mgL'] = rawssc['SSC'].to_numpy()
                    
                    # Reset start and end times if these new data expand the limits
                    if sraw[n]['ssc']['ts_pst'][0]<start:
                        start = sraw[n]['ssc']['ts_pst'][0]
                    if sraw[n]['ssc']['ts_pst'][-1]>end:
                        end = sraw[n]['ssc']['ts_pst'][-1]
    
            else: # no SFEI style SSC data # if there is a file here, check the format - there aren't 
                    print('Unknown SSC file format')
        else: # if not a CSV, make it empty
            sraw[n]['ssc'] = {}
                
        if '.csv' in turbfile: # if there is a Turb CSV, handle the raw data, if not, makes this dictionary value empty
            if fmt == 'usgs_v1':
                rawturb = pd.read_csv(os.path.join(dir_site,turbfile),comment='#')
                rawturb['ts_pst'] = pd.to_datetime(rawturb[' Timestamp (UTC-08:00)'])  
                sraw[n]['turb']['ts_pst'] = rawturb['ts_pst'].to_numpy()
                sraw[n]['turb']['turb_fnu'] = rawturb[' Value'].to_numpy()
                
                # Reset start and end times if these new data expand the limits
                if sraw[n]['turb']['ts_pst'][0]<start:
                    start = sraw[n]['turb']['ts_pst'][0]
                if sraw[n]['turb']['ts_pst'][-1]>end:
                    end = sraw[n]['turb']['ts_pst'][-1]
                
            elif fmt == 'sfei':
                rawturb = pd.read_csv(os.path.join(dir_site,turbfile))
                rawturb['ts_pst'] = pd.to_datetime(rawturb['ts_pst'])
                sraw[n]['turb']['ts_pst'] = rawturb['ts_pst'].to_numpy()
                sraw[n]['turb']['turb_fnu'] = rawturb['Turb'].to_numpy()
                
                # Reset start and end times if these new data expand the limits
                if sraw[n]['turb']['ts_pst'][0]<start:
                    start = sraw[n]['turb']['ts_pst'][0]
                if sraw[n]['turb']['ts_pst'][-1]>end:
                    end = sraw[n]['turb']['ts_pst'][-1]
                    
            elif fmt == 'usgs_v2':
                rawturb = pd.read_csv(os.path.join(dir_site,turbfile),na_values='Eqp')
                rawturb['ts_pst'] = pd.to_datetime(rawturb['ts_pst'])
                sraw[n]['turb']['ts_pst'] = rawturb['ts_pst'].to_numpy()
                sraw[n]['turb']['turb_fnu'] = rawturb['Turb'].to_numpy()
                
                if sraw[n]['turb']['ts_pst'][0]<start:
                    start = sraw[n]['turb']['ts_pst'][0]
                if sraw[n]['turb']['ts_pst'][-1]>end:
                    end = sraw[n]['turb']['ts_pst'][-1]
                    
            else: # no fmt is messed
                    print('Unknown Turb file format')
                
    # Build out the output data structure based on the known start and end times
    data.ts_pst = np.arange(start,end+np.timedelta64(TimeStep,'m'),np.timedelta64(TimeStep,'m'))                               
    data.ssc_mgL = np.ones(len(data.ts_pst))*np.nan
    data.turb = np.ones(len(data.ts_pst))*np.nan
    data.flags = np.ones(len(data.ts_pst))*np.nan
    
    # Mesh on data from primary site (must have SSC!)
    n = 1
    _,iA,iB = np.intersect1d(data.ts_pst,sraw[n]['ssc']['ts_pst'],return_indices=True)
    data.ssc_mgL[iA] = sraw[n]['ssc']['ssc_mgL'][iB]
    data.flags[iA] = n
    if bool(sraw[n]['turb']):
        _,iA,iB = np.intersect1d(data.ts_pst,sraw[n]['turb']['ts_pst'],return_indices=True)
        data.turb[iA] = sraw[n]['turb']['turb_fnu'][iB]
    
    # if there are additional sensors, incorporate these data following the "priority" hierarchy
    if len(priority)>1: 
        for n in range(2,len(priority)+1): # loop over the additional sensors
            
            if bool(sraw[n]['ssc']): # is there SSC data for this site? 
                ssctemp = np.ones(len(data.ssc_mgL))*np.nan # If so, build a temporary time-series
                _,iA,iB = np.intersect1d(data.ts_pst,sraw[n]['ssc']['ts_pst'],return_indices=True)
                ssctemp[iA] = sraw[n]['ssc']['ssc_mgL'][iB]
            
            elif  bool(sraw[n]['turb']): # ok, then if no SSC is there turb data? There must be... or there'd be no row
                
                # we need a turb-SSC conversion for this site
                ssctemp = np.ones(len(data.ssc_mgL))*np.nan
                _,iA,iB = np.intersect1d(data.ts_pst,sraw[n]['turb']['ts_pst'],return_indices=True)
                
                # Try to make one using data incorporated from higher priority sensors
                iOverlap = (~np.isnan(data.ssc_mgL)) & (~np.isnan(data.turb))
                if np.sum(iOverlap)>MinOverlap: # If there's enough overlap for a conversion, fit a turb to ssc model, and make a temp SSC time-series
                    lm = LinearRegression().fit(data.turb[iOverlap].reshape(-1, 1),data.ssc_mgL[iOverlap].reshape(-1, 1))
                    ssctemp[iA] = sraw[n]['turb']['turb_fnu'][iB]*lm.coef_[0] + lm.intercept_[0] # get ssctemp from turbidity usng conversion 
                # if no overlapping data, use a default conversino!
                else: # use a default conversion!
                    ssctemp[iA] = turb2ssc_a * sraw[n]['turb']['turb_fnu'][iB]**turb2ssc_b
                    
    #        data.turb[iA] = sraw[n]['turb']['turb_fnu'][iB] # merge this in anyway, but after the fitting
            
            # Ok, now shift this SSC chunk (which may have been converted from turb) to match the "primary" site
            iCross = (~np.isnan(data.ssc_mgL)) & (~np.isnan(ssctemp))
            if np.sum(iCross)>MinOverlap:
                # A log-log fit
                x = np.log(ssctemp[iCross].reshape(-1,1))
                y = np.log(data.ssc_mgL[iCross].reshape(-1,1))
                x[x<=0] = 0.01
                y[y<=0] = 0.01
                
                lm = LinearRegression().fit(x,y)
                
                # Fill  the normalized SSC data into the general time-series 
                iFill = (np.isnan(data.ssc_mgL)) & (~np.isnan(ssctemp)) # wherever the main time-series is empty and this priority time-series has data
                data.ssc_mgL[iFill] = ssctemp[iFill]**lm.coef_[0] * e**lm.intercept_[0] # gotta normalize it using the log-log model
                data.flags[iFill] = n
                
            else: # But don't normalize it if you can't...
                print('Insufficient overlap to shift data from ' + site + ' ' +names[n-1])
                iFill = (np.isnan(data.ssc_mgL)) & (~np.isnan(ssctemp))
                data.ssc_mgL[iFill] = ssctemp[iFill]
                data.flags[iFill] = n
                
    
    # Make a plot of the data
    fig,axs = plt.subplots(figsize=(7,4))               
    for n in priority:
        iFlag = data.flags==n
        if n == 1:
            axs.scatter(data.ts_pst[iFlag],data.ssc_mgL[iFlag],label=names[priority==n][0],s=3)
        else:
            axs.scatter(data.ts_pst[iFlag],data.ssc_mgL[iFlag],label=names[priority==n][0]+' - Converted',s=3)
    axs.set_title(site)
    axs.set_ylabel('SSC (mg/L)')
    plt.legend(loc='upper left')
    fig.show()
    fig.savefig(os.path.join(dir_output,site+'_Processed_15min.png'))
    
#    # Write a CSV
    out = pd.DataFrame.from_dict(data)
    out.to_csv(os.path.join(dir_output,site+'Processed_15min.csv'),index=False)
    
    