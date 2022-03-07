# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:20:49 2020

@author: derekr
"""

# This script nudges filled Kd time-series ("...\TWDR_Method_fy21\Data_Constructed_SSC_and_Kd")
# to fit USGS Cruise observations of Kd.
# Corresponding cruise sites for nudging are spec'ed here: Match_Cruise_to_HFsite_for_Kd_Bending.xlsx
# The nudging/bending time-series is built as follows:
# 1 - Read in site-specific cruise Kd data and mesh onto an empty time-series matching Kd input at OutputTimeStep step
# 2 - If multiple cruise sites have data at the same time-step, collapse into average of sites. 
# 3 - Calculate both the ratio and the difference between the Kd hourly time-series and the cruise observations wherever there is a crusie observation
# 4 - Interpolate the ratio and diff time-series down to the OutputTime Step
# 5 - Smooth this interpolated time-series at SmoothShift_Window days. 
# 6 - Use the smoothed ratio and diff time-series to bend the original Kd input to better match Kd cruise observations. 
# 7 - Make plots of both approaches and write an output file with original Kd, Kd shifted with ratio method, and Kd shifted with difference method
# Output written to CSVs in "...\TWDR_Method_fy21\Data_Kd_Shifted"


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pickle
from dotmap import DotMap

OutputTimeStep = 1 # hours, must be at least 1 hour and no more than 24 hours

SmoothShift_Window = 30 # days, for smoothing offset/shift time-series
win = int(np.timedelta64(SmoothShift_Window,'D')/np.timedelta64(OutputTimeStep,'h'))

Example_WY = 2017 # an example water year for plotting a shorter window

# Analysis sites
#sites = ['Mallard_Island']
sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_01','Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']

# Input Kd data path 
input_ext = '_SSCandKd_Filled_2009_to_2018.csv'
dir_input = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Constructed_SSC_and_Kd'

dir_matchref = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21'
file_matchref = os.path.join(dir_matchref,'Match_Cruise_to_HFsite_for_Kd_Bending.xlsx')

dir_cruise = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Cruise'
file_cruisessc = os.path.join(dir_cruise,'Curated_SiteSpec_SSCandKD_2_to_36.p')

dir_output = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Kd_Shifted'

dir_figs = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Figures_KdOutput_vs_KdObserved'

# %%

#Load cruise data, but handling is site-specific
cruise = pickle.load(open(file_cruisessc,'rb'))

# Load cruise match table
matchref = pd.read_excel(file_matchref)


# %%

for site in sites:
    
    inp = pd.read_csv(os.path.join(dir_input,site+input_ext))
    
    data = DotMap()
    data.ts_pst = pd.to_datetime(inp['ts_pst']).to_numpy()
    data.kd = inp['kd'].to_numpy()
    data.flag = inp['flag'].to_numpy()
    
    iSite = matchref['Site'] == site
    if ',' in str(matchref['CruiseStations'][iSite].values[0]):
        matchsites = tuple(map(int,matchref['CruiseStations'][iSite].values[0].split(',')))
    else:
        matchsites = (matchref['CruiseStations'][iSite].values[0],)
    
    # start with an overshoot time-series; so we can mesh with both the model fit and with the prediction data
    cruise_ts = data.ts_pst
    cruise_kd_set = np.ones((len(cruise_ts),len(matchsites)))*np.nan

    j = 0
    for n in matchsites: # loop over cruise sites associated with this SSC site
        site_ts = pd.Series(cruise[n].ts_pst).dt.round(str(OutputTimeStep)+'h').to_numpy()
        _,iA,iB = np.intersect1d(cruise_ts,site_ts,return_indices=True)
        cruise_kd_set[iA,j] = cruise[n].Kd[iB] # array of cruise Kd
        j = j+1
        
    cruise_kd = np.nanmean(cruise_kd_set,axis=1) # collapse cruise_ssc data into site-averages when time steps overlap
    
    ovm_prop = cruise_kd / data.kd # modeled/measure proportion (at times when there is cruise data)
    ovm_diff = cruise_kd - data.kd # modeled - measured difference (at times when there is cruise data)
    
    iCruise = ~np.isnan(cruise_kd) # where there is cruise data
    ovm_prop_filled = np.interp(date2num(cruise_ts),date2num(cruise_ts[iCruise]),ovm_prop[iCruise])
    ovm_diff_filled = np.interp(date2num(cruise_ts),date2num(cruise_ts[iCruise]),ovm_diff[iCruise])
    
    ovm_prop_smooth = pd.Series(ovm_prop_filled).rolling(window=win,center=True,min_periods=1).mean().to_numpy()
    ovm_diff_smooth = pd.Series(ovm_diff_filled).rolling(window=win,center=True,min_periods=1).mean().to_numpy()
    
    data.kd_PropShifted = data.kd * ovm_prop_smooth
    data.kd_DiffShifted = data.kd + ovm_diff_smooth
    
    ### Output the data
    out = {}
    out['ts_pst'] = data.ts_pst
    out['Kd'] = np.round(data.kd,2)
    out['Kd_DiffShifted'] = np.round(data.kd_DiffShifted,2)
    out['Kd_PropShifted'] = np.round(data.kd_PropShifted,2)
    
    output = pd.DataFrame.from_dict(out)
    output.to_csv(os.path.join(dir_output,site+'_Kd_Hourly_LongTerm.csv'),index=False)
    
    ### Make some pretty plots ###
    fig,axs = plt.subplots(2,2,figsize=(18,10),sharey=True,sharex='row')
       
    iObs = data.flag==1
    kd_data = data.kd.copy()
    kd_data[~iObs] = np.nan
    kd_mod = data.kd.copy()
    kd_mod[iObs] = np.nan
    
    # Plot long term original Kd data
    axs[0,0].plot(data.ts_pst,kd_data,color='dodgerblue',label='Kd (from observed SSC)',zorder=1)
    axs[0,0].plot(data.ts_pst,kd_mod,color='orange',label='Kd (from modeled SSC)',zorder=2)
    axs[0,0].scatter(data.ts_pst,cruise_kd,facecolor='red',edgecolor='k',s=30,zorder=3,label='Cruise Kd')
    axs[0,0].set_xlim((data.ts_pst[0],data.ts_pst[-1]))
    axs[0,0].legend(loc='upper left')
    axs[0,0].set_title(site)
    axs[0,0].grid()
    axs[0,0].set_axisbelow(True)
    
    wyWindow = (np.datetime64(str(Example_WY-1)+'-10-01'),np.datetime64(str(Example_WY)+'-10-01'))
    iWY = (data.ts_pst>=wyWindow[0]) & (data.ts_pst<=wyWindow[-1]) 
    
    # Plot water year exampple original Kd data
    axs[1,0].plot(data.ts_pst[iWY],kd_data[iWY],color='dodgerblue',label='Kd (from observed SSC)',zorder=1)
    axs[1,0].plot(data.ts_pst[iWY],kd_mod[iWY],color='orange',label='Kd (from observed SSC)',zorder=2)
    axs[1,0].scatter(data.ts_pst,cruise_kd,facecolor='red',edgecolor='k',s=30,zorder=3,label='Cruise Kd')
    axs[1,0].set_xlim((wyWindow[0],wyWindow[1]))
    axs[1,0].set_title('Example Water Year ' + str(Example_WY))
    axs[1,0].grid()
    axs[1,0].set_axisbelow(True)
    
    # Plot long term shifted Kd data
    axs[0,1].plot(data.ts_pst,data.kd_DiffShifted,color='grey',label='Kd (diff shifted)',zorder=1)
    axs[0,1].plot(data.ts_pst,data.kd_PropShifted,color='mediumblue',label='Kd (prop shifted)',zorder=2)
    axs[0,1].scatter(data.ts_pst,cruise_kd,facecolor='red',edgecolor='k',s=30,zorder=3,label='Cruise Kd')
    axs[0,1].set_xlim((data.ts_pst[0],data.ts_pst[-1]))
    axs[0,1].legend(loc='upper left')
    axs[0,1].set_title(site+ ' - Shifted')
    axs[0,1].grid()
    axs[0,1].set_axisbelow(True)
    
    # Plot water year exampple shifted Kd data
    axs[1,1].plot(data.ts_pst[iWY],data.kd_DiffShifted[iWY],color='grey',label='Kd (diff shifted)',zorder=1)
    axs[1,1].plot(data.ts_pst[iWY],data.kd_PropShifted[iWY],color='mediumblue',label='Kd (prop shifted)',zorder=2)
    axs[1,1].scatter(data.ts_pst,cruise_kd,facecolor='red',edgecolor='k',s=30,zorder=3,label='Cruise Kd')
    axs[1,1].set_xlim((wyWindow[0],wyWindow[1]))
    axs[1,1].set_title('Example Water Year ' + str(Example_WY))
    axs[1,1].grid()
    axs[1,1].set_axisbelow(True)
    
    axs[0,0].set_ylim((0,10))
    
    plt.subplots_adjust(wspace=0.05)
    fig.show()
    fig.savefig(os.path.join(dir_figs,site+'_Kd_and_KdShifted_vs_Observations.png'))
