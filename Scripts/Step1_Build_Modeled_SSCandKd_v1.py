m# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:04:20 2020

@author: derekr
"""

# This script builds continous time-series at OutputTimeStep resolution
# over the window OutputStart to Output End.
# Output is observed data filled with modeled data as needed. 
# The continuity of this output is dependent on the combined continuity of
# the input and forcing data over that period. 

# This script reads SSC time-series from "...\TWDR_Method_fy21\Data_SSC_Processed"
# Builds models of these data based on overlap with forcing data set. 
# Uses model to fill SSC gaps over the output window where no observational are available. 

# A light attenuation (Kd) time-series is estimated from the filled SSC time-series
# based on a conversion developed from site-specific USGS cruise data
# Fit of near-surface (4 m and shallower) - matching to USGS sites spec'ed
# in the file Match_Cruise_to_HFsite_for_ssc2kd_and_gamtrends.xlsx ("...\TWDR_Method_fy21")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pickle
from dotmap import DotMap
from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, l, te
from sklearn.linear_model import LinearRegression
from math import e

# Analysis sites
# Note that Golden Gate Bridge does not have recent enough data (2000 forward) to overlap with the predictor data set (2000-2020)
sites = ['Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge',
         'Dumbarton_Bridge','San_Mateo_Bridge']
#sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
#         'Channel_Marker_01','Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
#         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']

OutputTimeStep = 1 # hours, must be at least 1 hour and no more than 24 hours

# Output date range; these dates (WYs 2010-2018) because forcing data are complete over this interval
OutputStart = np.datetime64('2009-10-01') # output start date
OutputEnd = np.datetime64('2018-10-01') # output end data

LowFreqWindow = 90 # days, for smoothing cruise ssc data to get a seasonal trend signal
lfwin = int(np.timedelta64(LowFreqWindow,'D')/np.timedelta64(OutputTimeStep,'h'))

# Input SSC data path
dir_sscdata = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_SSC_Processed'

# Path to file matching sites to USGS cruise sites 
# this info used for building long-term trend in forcing data and for applying SSC to Kd conversion
dir_matchref = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21'
file_matchref = os.path.join(dir_matchref,'Match_Cruise_to_HFsite_for_ssc2kd_and_gamtrends.xlsx')


# Paths of predictor/forcing variables. See readmes in paths for details
dir_wind = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Forcing\Wind'
file_wind = os.path.join(dir_wind,'ASOS-HWD_Wind_2000-2019.p')

dir_tidevel = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Forcing\TidalVelocity'
file_tidevel = os.path.join(dir_tidevel,'SMB_TidalVelocity.p')

dir_tides = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Forcing\TidalElevation'
file_tides = os.path.join(dir_tides,'NOAA_RedwoodCity_TideElev_2000_2019.p')

dir_inflows = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Forcing\Inflows'
file_delta = os.path.join(dir_inflows,'DayFlow_Q.p')
file_alameda = os.path.join(dir_inflows,'Alameda_Creek_Q.p')

dir_cruise = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Cruise'
file_cruisessc = os.path.join(dir_cruise,'Curated_SiteSpec_SSCandKD_2_to_36.p')

# Misc additional paths
dir_output = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Data_Constructed_SSC_and_Kd'
dir_gamfigs = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Figures_GAM_Components'
dir_ssc2kdfigs = r'D:\My Drive\1_Nutrient_Share\1_Projects_NUTRIENTS\07_FY21_NMS_Projects\FY2021_Mod_SedTransp_Light\3_ProjectWork_Analysis_Reporting\TWDR_Method_fy21\Figures_ssc2kd_fits'

# %% Load forcing data
# Average to the output time step, as needed
# Interpolate day flow data, as needed

# Load and handle wind data
wnd = pickle.load(open(file_wind,'rb')) # time series is complete
step = wnd.ts_pst[1] - wnd.ts_pst[0]
window = int(np.timedelta64(OutputTimeStep,'h')/step)
if window > 1: # smooth the data if the input time step is smaller than the output step
    wnd.spd = pd.Series(wnd.spd).rolling(window=window,min_periods=1).mean().to_numpy()


# Load and handle tidal velocity data
tdvel = pickle.load(open(file_tidevel,'rb')) # time series is complete
step = tdvel.ts_pst[1] - tdvel.ts_pst[0]
window = int(np.timedelta64(OutputTimeStep,'h')/step)
if window > 1: # smooth the data if the input time step is smaller than the output step
    tdvel.u_ms = pd.Series(tdvel.u_ms).rolling(window=window,min_periods=1).mean().to_numpy()

# Load and handle tidal water level data
wtrlvl = pickle.load(open(file_tides,'rb')) # time series is complete
step = wtrlvl.ts_pst[1] - wtrlvl.ts_pst[0]
window = int(np.timedelta64(OutputTimeStep,'h')/step)
if window > 1: # smooth the data if the input time step is smaller than the output step
    wtrlvl.wl = pd.Series(wtrlvl.wl).rolling(window=window,min_periods=1).mean().to_numpy()

# Load and handle Alameda Creek data
alamedaq = pickle.load(open(file_alameda,'rb')) # time series is complete
step = alamedaq.ts_pst[1] - alamedaq.ts_pst[0]
window = int(np.timedelta64(OutputTimeStep,'h')/step)
if window > 1: # smooth the data if the input time step is smaller than the output step
    alamedaq.q_cms = pd.Series(alamedaq.q_cms).rolling(window=window,min_periods=1).mean().to_numpy()

# Load and handle Delta Dayflow data - Note that this is daily data and needs different handling
deltaq = pickle.load(open(file_delta,'rb')) # time series is complete
deltaq.ts_pst = np.arange(deltaq.ts[0],deltaq.ts[-1]+np.timedelta64(OutputTimeStep,'h'),np.timedelta64(OutputTimeStep,'h'))
deltaq.q_cms = np.interp(date2num(deltaq.ts_pst),date2num(deltaq.ts),deltaq.q_cms)

#Load cruise data, but handling is site-specific
cruise = pickle.load(open(file_cruisessc,'rb'))
matchref = pd.read_excel(file_matchref)

# %% Build forcing data and model output time-series

model = DotMap() # model data structure
model.ts_pst = np.arange(OutputStart,OutputEnd+np.timedelta64(OutputTimeStep,'h'),np.timedelta64(OutputTimeStep,'h'))

##### Mesh in the  forcing time-series ####

# Wind
model.wnd = np.ones(len(model.ts_pst))*np.nan
_,iA,iB = np.intersect1d(model.ts_pst,wnd.ts_pst,return_indices=True)
model.wnd[iA] = wnd.spd[iB]

# tidal velocity
model.tdvel = np.ones(len(model.ts_pst))*np.nan
_,iA,iB = np.intersect1d(model.ts_pst,tdvel.ts_pst,return_indices=True)
model.tdvel[iA] = tdvel.u_ms[iB]

# Tidal water level
model.wl = np.ones(len(model.ts_pst))*np.nan
_,iA,iB = np.intersect1d(model.ts_pst,wtrlvl.ts_pst,return_indices=True)
model.wl[iA] = wtrlvl.wl[iB]

# Alameda Inflow (local inflow effects)
model.localq = np.ones(len(model.ts_pst))*np.nan
_,iA,iB = np.intersect1d(model.ts_pst,alamedaq.ts_pst,return_indices=True)
model.localq[iA] = alamedaq.q_cms[iB]

# Delta inflow (seasonal/controlled inflows)
model.deltaq = np.ones(len(model.ts_pst))*np.nan
_,iA,iB = np.intersect1d(model.ts_pst,deltaq.ts_pst,return_indices=True)
model.deltaq[iA] = deltaq.q_cms[iB]


# %% Loop over sites, build input data structure, build a model

data = DotMap()

for site in sites:
    
    ###### Build the corresponding cruise time-series for the predictions ##### 
    
    iSite = matchref['Site'] == site
    if ',' in str(matchref['CruiseStations'][iSite].values[0]):
        matchsites = tuple(map(int,matchref['CruiseStations'][iSite].values[0].split(',')))
    else:
        matchsites = (matchref['CruiseStations'][iSite].values[0],)
    
    # start with an overshoot time-series; so we can mesh with both the model fit and with the prediction data
    cruiseset_ts = np.arange(np.datetime64('1990-01-01'),OutputEnd+np.timedelta64(100,'D'),np.timedelta64(OutputTimeStep,'h'))
    cruiseset = np.ones((len(cruiseset_ts),len(matchsites)))*np.nan
    cruiseset_kd = np.ones((len(cruiseset_ts),len(matchsites)))*np.nan

    j = 0
    for n in matchsites: # loop over cruise sites associated with this SSC site
        ts_cruise = pd.Series(cruise[n].ts_pst).dt.round(str(OutputTimeStep)+'h').to_numpy()
        _,iA,iB = np.intersect1d(cruiseset_ts,ts_cruise,return_indices=True)
        cruiseset[iA,j] = cruise[n].ssc_mgL[iB] # array of cruise SSC data
        cruiseset_kd[iA,j] = cruise[n].Kd[iB] # array of cruise Kd data
        j = j+1
    cruise_ssc = np.nanmean(cruiseset,axis=1) # collapse cruise_ssc data into site-averages
    iCruise = ~np.isnan(cruise_ssc) # where there is cruise data
    cruise_filled = np.interp(date2num(cruiseset_ts),date2num(cruiseset_ts[iCruise]),cruise_ssc[iCruise]) # interpolate time-series of cruise data
    CruiseSet_SSC = pd.Series(cruise_filled).rolling(window=lfwin,center=True).mean().to_numpy() # smooth it in time - THIS IS THE LONG-TERM TREND forcing data
    
    ############################################################################
    
    ########### Handle the observed SSC data ###################################
    rawdf = pd.read_csv(os.path.join(dir_sscdata,site+'Processed_15min.csv'))
    rawdf['ts_pst'] = pd.to_datetime(rawdf['ts_pst'])
    
    # Smooth it as needed to the output time step
    raw = DotMap()
    raw.ts_pst = rawdf['ts_pst'].to_numpy()
    raw.ssc = rawdf['ssc_mgL'].to_numpy()
    step = raw.ts_pst[1] - raw.ts_pst[0]
    window = int(np.timedelta64(OutputTimeStep,'h')/step)
    if window > 1: # smooth the data if output step is greater than input time series step
        raw.ssc = pd.Series(raw.ssc).rolling(window=window,min_periods=1,center=True).mean().to_numpy()
        
    # Get the first and last dt values at the output step resolution  
    RoundTime = rawdf['ts_pst'].dt.round(str(OutputTimeStep)+'h')
    start = RoundTime.to_numpy()[0]
    end = RoundTime.to_numpy()[-1]
    
    # Build input SSC at output step resolution for modeling
    data[site].ts_pst = np.arange(start,end+np.timedelta64(OutputTimeStep,'h'),np.timedelta64(OutputTimeStep,'h'))
    _,iA,iB = np.intersect1d(data[site].ts_pst,raw.ts_pst,return_indices=True)
    data[site].ssc = np.ones(len(data[site].ts_pst))*np.nan
    data[site].ssc[iA] = raw.ssc[iB] 
    
    ############################################################################

    ##### Mesh in the  forcing time-series #####################################
    
    # Wind
    data[site].wnd = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,wnd.ts_pst,return_indices=True)
    data[site].wnd[iA] = wnd.spd[iB]
    
    # tidal velocity
    data[site].tdvel = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,tdvel.ts_pst,return_indices=True)
    data[site].tdvel[iA] = tdvel.u_ms[iB]
    
    # Tidal water level
    data[site].wl = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,wtrlvl.ts_pst,return_indices=True)
    data[site].wl[iA] = wtrlvl.wl[iB]
    
    # Alameda Inflow (local inflow effects)
    data[site].localq = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,alamedaq.ts_pst,return_indices=True)
    data[site].localq[iA] = alamedaq.q_cms[iB]
    
    # Delta inflow (seasonal/controlled inflows)
    data[site].deltaq = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,deltaq.ts_pst,return_indices=True)
    data[site].deltaq[iA] = deltaq.q_cms[iB]
    
    # Cruise signal (seasonal variability, smoothed discrete samples)
    data[site].cruise_ssc = np.ones(len(data[site].ts_pst))*np.nan
    _,iA,iB = np.intersect1d(data[site].ts_pst,cruiseset_ts,return_indices=True)
    data[site].cruise_ssc[iA] = CruiseSet_SSC[iB]
    
    ############################################################################
    
    ################ Fit a model ###############################################
    # 0 - wind, 1 - tidal vel, 2 - water level, 3 - alameda flow, 4 - delta flow, 5 - cruise ssc
    predictors = ['Wind Speed','Tidal Velocity (flood+)','Tidal Elevation','Alameda Flow','Delta Flow','Local Seasonal Cruise SSC']

    X = np.vstack( (data[site].wnd, data[site].tdvel, data[site].wl,
                              data[site].localq, data[site].deltaq, data[site].cruise_ssc )).T
    Y = data[site].ssc
    iXgood = ~np.isnan(X).any(axis=1)
    iYgood = (~np.isnan(Y))
    iGood =  iXgood & iYgood
    xGood = X[iGood,:]
    yGood = Y[iGood]
    
#    xTrain, xTest, yTrain, yTest = train_test_split(X,Y,train_size=0.8)
#    gam = LinearGAM().fit(xTrain,yTrain)
#    gam = LinearGAM(s(0,n_splines=10)+s(1,n_splines=10)+s(2,n_splines=10)+s(3,n_splines=10)+s(4,n_splines=10)+s(5,n_splines=10)).fit(Xtrain,Ytrain)
    gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5),n_splines=10).fit(xGood,yGood)
#    gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)).fit(X,Y)
    
    data[site].gam_ssc = np.ones(len(data[site].ts_pst))*np.nan
    data[site].gam_ssc[iXgood] = gam.predict(X[iXgood])
    
    # Plot GAM fits
    fig, axs = plt.subplots(1,len(predictors),figsize=(14,3.5));

    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        if i == 0:
            ax.set_ylim(-30,30)
        ax.set_title(predictors[i],fontsize=9)
    axs[0].set_ylabel('SSC (mg/L)')
    fig.text(0.04, 0.5,site, va='center', rotation='vertical',fontsize=12)
    fig.savefig(os.path.join(dir_gamfigs,site+'_gam_fits.png')) 
    
    ############################################################################
        
    ########### Generate model predictions for output time window ##############
    
    # Cruise signal (seasonal variability, smoothed discrete samples)
    model.cruise[site] = np.ones(len(model.ts_pst))*np.nan
    _,iA,iB = np.intersect1d(model.ts_pst,cruiseset_ts,return_indices=True)
    model.cruise[site][iA] = CruiseSet_SSC[iB]
    
    # Fulll set of forcing data, most variables the same for all sites, but the cruise forcing data varies by site
    model.X = np.vstack( (model.wnd, model.tdvel, model.wl,
                              model.localq, model.deltaq,model.cruise[site])).T
     
    # Get modeled SSC values                     
    model.gam_ssc[site] = gam.predict(model.X)
    model.gam_ssc[site][model.gam_ssc[site]<=0] = 0.1 # make negative SSC = 0.1 mg/L
    
    # Convert SSC to Kd using local regression of cruise SSC to cruise Kd
    iCruise = ~np.isnan(cruiseset) & (~np.isnan(cruiseset_kd))
    lm = LinearRegression().fit(np.log(cruiseset[iCruise].reshape(-1,1)),np.log(cruiseset_kd[iCruise].reshape(-1,1))) # 
    x = np.arange(0.01,np.nanmax(cruiseset),10)
    logx = np.log(x)
    yfitlog = lm.predict(logx.reshape(-1,1))
    yfit = e**yfitlog
    coef = e**lm.intercept_[0]
    exp = lm.coef_[0][0]
    
    model.gam_kd[site] = coef * model.gam_ssc[site]**exp
    
    # Make a figure of the site-specific SSC to Kd conversion
    fig,ax = plt.subplots(figsize=(6,3)) # 
    ax.scatter(cruiseset,cruiseset_kd,color='grey')
    ax.plot(x,yfit,color='k')
    ax.text(0.05,0.9,'Kd = '+str(round(coef,3))+' * SSS^('+str(round(exp,3))+')',transform=ax.transAxes)
    ax.set_title(site + ' - CruiseSites:' + str(matchsites))
    fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_to_Kd.png'))
    
    ############################################################################
    
    # Plot the output
    
    fig,axs = plt.subplots(1,2,figsize=(13,3.5))
    axs[0].plot(data[site].ts_pst,data[site].ssc,label='Observed (sensor)',color='k',zorder=1)
    axs[0].plot(data[site].ts_pst,data[site].gam_ssc,label='Modeled',color='darkred',zorder=2)
    for j in range(np.shape(cruiseset)[1]):
        axs[0].scatter(cruiseset_ts,cruiseset[:,j],label='Discrete - USGS '+str(matchsites[j]),zorder=3+j,s=3)
    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('SSC (mg/L)')
    
    axs[1].plot(model.ts_pst,model.gam_kd[site],label='Model Kd')
    for j in range(np.shape(cruiseset)[1]):
        axs[1].scatter(cruiseset_ts,cruiseset_kd[:,j],label='Discrete - USGS '+str(matchsites[j]),zorder=3+j,s=3)
    axs[1].set_xlim((model.ts_pst[0],model.ts_pst[-1]))
    axs[1].set_ylim((0,8))
    fig.show()
    
    ############### Create and write output #####################################
    output = {}
    output['ts_pst'] = model.ts_pst # output for 2010-2019
    output['ssc_mgL'] = model.gam_ssc[site].copy() # output is gam model
    output['kd'] = model.gam_kd[site].copy() # output is gam model
    output['flag'] = np.ones(len(output['ts_pst']))*2 # flag for: 'from model' 
    # over-write with real data, where available
    _,iA,iB = np.intersect1d(output['ts_pst'],data[site].ts_pst,return_indices=True)
    output['ssc_mgL'][iA] = data[site].ssc[iB].copy()
    output['kd'][iA] = coef * data[site].ssc[iB].copy()**exp
    output['flag'][iA] = 1 # flag for 'from observed high-freq SSC'
    # re-over-write with gam model where observations introduced nans
    iNaN = np.isnan(output['ssc_mgL']) # NaNs will be the same for ssc and kd since both built from observed ssc
    output['ssc_mgL'][iNaN] = model.gam_ssc[site][iNaN].copy()
    output['kd'][iNaN] = model.gam_kd[site][iNaN].copy()
    output['flag'][iNaN] = 2
    
    out = pd.DataFrame.from_dict(output)
#    out.to_csv(os.path.join(dir_output,site+'_SSCandKd_Filled_'+str(OutputStart)[:4]+'_to_'+str(OutputEnd)[:4]+'.csv'),index=False)
    
    ############################################################################