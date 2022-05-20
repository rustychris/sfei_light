# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:04:20 2020

@author: derekr
Adapted by rustyh
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

from stompy import utils
import seaborn as sns
from dotmap import DotMap
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

from pygam import LinearGAM, s, l, te
from sklearn.linear_model import LinearRegression
from math import e

# Analysis sites
# Note that Golden Gate Bridge does not have recent enough data (2000 forward) to overlap with the predictor data set (2000-2020)
#sites = ['Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge',
#         'Dumbarton_Bridge','San_Mateo_Bridge']

sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']

OutputTimeStep = 1 # hours, must be at least 1 hour and no more than 24 hours

# Output date range; these dates (WYs 2010-2018) because forcing data are complete over this interval
OutputStart = np.datetime64('2009-10-01') # output start date
OutputEnd = np.datetime64('2018-10-01') # output end data

LowFreqWindow = 90 # days, for smoothing cruise ssc data to get a seasonal trend signal
lfwin = int(np.timedelta64(LowFreqWindow,'D')/np.timedelta64(OutputTimeStep,'h'))

data_root=".."

# Input SSC data path
dir_sscdata = data_root+'/Data_SSC_Processed'

# Path to file matching sites to USGS cruise sites 
# this info used for building long-term trend in forcing data and for applying SSC to Kd conversion
dir_matchref = data_root
file_matchref = os.path.join(dir_matchref,'Match_Cruise_to_HFsite_for_ssc2kd_and_gamtrends.xlsx')


# Paths of predictor/forcing variables. See readmes in paths for details
dir_wind = data_root+'/Data_Forcing/Wind'
file_wind = os.path.join(dir_wind,'ASOS-HWD_Wind_2000-2019.p')

dir_tidevel = data_root+'/Data_Forcing/TidalVelocity'
file_tidevel = os.path.join(dir_tidevel,'SMB_TidalVelocity.p')

dir_tides = data_root+'/Data_Forcing/TidalElevation'
file_tides = os.path.join(dir_tides,'NOAA_RedwoodCity_TideElev_2000_2019.p')

dir_inflows = data_root+'/Data_Forcing/Inflows'
file_delta = os.path.join(dir_inflows,'DayFlow_Q.p')
file_alameda = os.path.join(dir_inflows,'Alameda_Creek_Q.p')

dir_cruise = data_root+'/Data_Cruise'
file_cruisessc = os.path.join(dir_cruise,'Curated_SiteSpec_SSCandKD_2_to_36.p')

# Misc additional paths
dir_output = data_root+'/Data_Constructed_SSC_and_Kd_RH'
dir_gamfigs = data_root+'/Figures_GAM_Components_RH'
dir_ssc2kdfigs = data_root+'/Figures_ssc2kd_fits_RH'

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


# %%
def cruise_ssc_at_site(site,times):
    """
    given name of a HF site and times (pst), return the aggregated ssc 
    observed by USGS stations nearby
    """
    iSite = matchref['Site'] == site
    matchsites = tuple(map(int,str(matchref['CruiseStations'][iSite].values[0]).split(',')))

    return cruise_ssc_at_stations(matchsites,times)

def cruise_ssc_at_stations(stations,times):
    """
    matchsites: sequence of USGS station numbers.
    times: times for which to return output.
    extract measured ssc at the given stations, average across stations, interpolate
    to hourly, spanning requested times, lowpass, and then narrow to the requested times.
    """
    # start with an overshoot time-series; so we can mesh with both the model fit and with the prediction data
    # cruiseset_ts = np.arange(np.datetime64('1990-01-01'),OutputEnd+np.timedelta64(100,'D'),np.timedelta64(OutputTimeStep,'h'))
    pad=np.timedelta64(100,'D')
    dt64=np.timedelta64(OutputTimeStep,'h')
    cruiseset_ts = np.arange(utils.floor_dt64(times.min(),dt64) - pad,
                             times.max() + pad,
                             dt64)
    cruiseset = np.ones((len(cruiseset_ts),len(stations)))*np.nan
    cruiseset_kd = np.ones((len(cruiseset_ts),len(stations)))*np.nan

    for j,n in enumerate(stations): # loop over cruise sites associated with this SSC site
        # times of the cruise data for a specific cruise station
        ts_cruise = pd.Series(cruise[n].ts_pst).dt.round(str(OutputTimeStep)+'h').to_numpy()
        # 
        _,iA,iB = np.intersect1d(cruiseset_ts,ts_cruise,return_indices=True)
        cruiseset[iA,j] = cruise[n].ssc_mgL[iB] # array of cruise SSC data
        #cruiseset_kd[iA,j] = cruise[n].Kd[iB] # array of cruise Kd data

    cruise_ssc = np.nanmean(cruiseset,axis=1) # collapse cruise_ssc data into site-averages
    iCruise = ~np.isnan(cruise_ssc) # where there is cruise data
    if not np.any(iCruise):
        import pdb
        pdb.set_trace()
    # interpolate time-series of cruise data
    cruise_filled = np.interp(date2num(cruiseset_ts),date2num(cruiseset_ts[iCruise]),cruise_ssc[iCruise])
    # smooth it in time - THIS IS THE LONG-TERM TREND forcing data
    CruiseSet_SSC = pd.Series(cruise_filled).rolling(window=lfwin,center=True).mean().to_numpy()

    return np.interp( date2num(times),
                      date2num(cruiseset_ts), CruiseSet_SSC)

#%%

# create an along/across channel coordinate system.
thalweg=np.array([[ 588380., 4147386.],[ 586723., 4147059.],
                  [ 584652., 4146406.], [ 582892., 4147168.], [ 580097., 4149563.],
                  [ 578440., 4150651.], [ 576784., 4152066.], [ 574610., 4153590.],
                  [ 573160., 4154897.], [ 570779., 4157292.], [ 566948., 4159469.],
                  [ 564049., 4161972.], [ 560425., 4166871.], [ 559080., 4171116.],
                  [ 558044., 4177104.], [ 555663., 4183200.], [ 553178., 4186574.],
                  [ 551936., 4191037.], [ 549554., 4195065.], [ 548312., 4198875.],
                  [ 549347., 4203229.], [ 552660., 4207801.], [ 556077., 4210632.],
                  [ 560529., 4212156.], [ 564463., 4213135.], [ 569019., 4213026.],
                  [ 571711., 4212373.], [ 572953., 4210632.], [ 574713., 4209978.],
                  [ 576887., 4211067.], [ 580615., 4212373.], [ 583410., 4213135.],
                  [ 587137., 4213135.], [ 590657., 4212700.], [ 595006., 4211502.],
                  [ 598422., 4212373.], [ 600493., 4214006.], [ 602253., 4213788.],
                  [ 604945., 4213026.]])
from shapely import geometry
chan_geom=geometry.LineString(thalweg)
def utm_to_aa(utm,chan_geom=chan_geom):
    """
    map utm coordinates to along-channel, across channel
    """
    pnts = [geometry.Point(xy) for xy in utm]
    alongs=[chan_geom.project(pnt) for pnt in pnts]
    on_axis=np.array([chan_geom.interpolate(along).coords[0]
                      for along in alongs])
    eps=10.0
    on_axis_eps=np.array([chan_geom.interpolate(along+eps).coords[0]
                         for along in alongs])
    tan=on_axis_eps - on_axis
    norm=tan[:,::-1]*np.r_[-1,1]
    norm = norm/ utils.mag(norm)[:,None]
    across=((utm-on_axis)*norm).sum(axis=1)
    return np.c_[alongs,across]
#%%    
import gam_common
g=gam_common.grid()

#%%
## 
# Compile all of the observed SSC into a global dataframe that can then
# be annotated with more local data.

rawdfs=[]

station_coords=pd.read_csv("../Station_Polygon_Coordinates.csv").set_index('Site')
stn_utm = station_coords[ ['utm_e','utm_n'] ].values
aa=utm_to_aa(stn_utm)
station_coords['along']=aa[:,0]
station_coords['across']=aa[:,1]


#%%

if 1: # location of stations relative to WAQ grid
    plt.figure(1).clf()
    #g.plot_edges(color='k',lw=0.7,alpha=0.4)
    g.contourf_node_values(g.nodes['depth'],np.linspace(-20,2,40),cmap='turbo',
                           extend='both')
    plt.plot( station_coords.utm_e, station_coords.utm_n, 'go')
    for idx,row in station_coords.iterrows():
        plt.text(row['utm_e'],row['utm_n'],f"{idx} {row['along']:.0f} x {row['across']:.0f}")

plt.axis('equal')

#%%

def get_global_ts(time_pst,source_time,source_value):
    result = np.full(len(time_pst),np.nan)
    _,iA,iB = np.intersect1d(time_pst,source_time,return_indices=True)
    result[iA] = source_value[iB]
    return result

def get_wind(site=None,utm=None,time_pst=None):
    return get_global_ts(time_pst=time_pst,
                         source_time=wnd.ts_pst,source_value=wnd.spd)

def get_tide_velocity(site=None,utm=None,time_pst=None):
    return get_global_ts(time_pst=time_pst,
                         source_time=tdvel.ts_pst,source_value=tdvel.u_ms)

def get_tide_elevation(site=None,utm=None,time_pst=None):
    return get_global_ts(time_pst=time_pst,
                         source_time=wtrlvl.ts_pst,
                         source_value=wtrlvl.wl)

def get_trib_flow(site=None,utm=None,time_pst=None):   
    return get_global_ts(time_pst=time_pst,
                         source_time=alamedaq.ts_pst,
                         source_value=alamedaq.q_cms)

def get_delta_flow(site=None,utm=None,time_pst=None):
    return get_global_ts(time_pst=time_pst,
                         source_time=deltaq.ts_pst,
                         source_value=deltaq.q_cms)

for site in sites:
    rawdf=pd.read_csv(os.path.join(dir_sscdata,site+'Processed_15min.csv'),
                      parse_dates=['ts_pst'])
    # downsample to the output time resolution
    rawdf=rawdf.resample('h',on='ts_pst').mean().reset_index()
    
    rawdf['site']=site
    utm=[station_coords.loc[site,'utm_e'],
         station_coords.loc[site,'utm_n']]
    rawdf['utm_e']=utm[0]
    rawdf['utm_n']=utm[1]
    rawdf['along']=station_coords.loc[site,'along']
    rawdf['across']=station_coords.loc[site,'across']
    
    rawdf['src']='hf_ssc'

    # per-site covariate handling:
    rawdf['usgs_lf']=cruise_ssc_at_site(site,rawdf.ts_pst)
    
    rawdf['wind'] = get_wind(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['tdvel'] = get_tide_velocity(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['wl'] = get_tide_elevation(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['storm'] = get_trib_flow(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['delta'] = get_delta_flow(site=site,utm=utm,time_pst=rawdf.ts_pst)    
        
    rawdfs.append(rawdf)

station_df=pd.concat(rawdfs)

# A bunch of those covariates we don't have long time series for...
# looks like wind, tdvel, wl, nd delta outflow all have about the 
# same number of entries. storm has about double those.

#%%

# Trim out missing SSC or covariate data
missing=( station_df['ssc_mgL'].isnull()
         | station_df['usgs_lf'].isnull()
         | station_df['wind'].isnull()
         | station_df['tdvel'].isnull()
         | station_df['storm'].isnull()
         | station_df['delta'].isnull() )

station_df_notnull=station_df[~missing]

# For starters all covariates handled at the station level, so already
# filled in.
# Write to global csv, ready for mgcv.

dest_dir="../DataFit00"
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
station_df_notnull.to_csv(os.path.join(dest_dir,"model-inputs.csv"))
    

## DR code below
# %% Loop over sites, build input data structure, build a model

for d in [dir_gamfigs, dir_ssc2kdfigs, dir_output]:
    if not os.path.exists(d):
        os.makedirs(d)

## 
data = DotMap()

for site in sites:
    
    ###### Build the corresponding cruise time-series for the predictions ##### 
    
    iSite = matchref['Site'] == site
    matchsites = tuple(map(int,str(matchref['CruiseStations'][iSite].values[0]).split(',')))
    
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
    
    # Full set of forcing data, most variables the same for all sites, but the cruise forcing data varies by site
    model.X = np.vstack( (model.wnd, model.tdvel, model.wl,
                              model.localq, model.deltaq,model.cruise[site])).T
     
    # Get modeled SSC values                     
    model.gam_ssc[site] = gam.predict(model.X)
    model.gam_ssc[site][model.gam_ssc[site]<=0] = 0.1 # make negative SSC = 0.1 mg/L
    
    # Convert SSC to Kd using local regression of cruise SSC to cruise Kd
    # 
    iCruise = ~np.isnan(cruiseset) & (~np.isnan(cruiseset_kd))
    # which of the match sites (columns) does each valid, ravelled
    # data point come from
    imatch = np.nonzero(iCruise)[1]

    if 0: # previous approach, using sklearn.
        lm = LinearRegression().fit(np.log(cruiseset[iCruise].reshape(-1,1)),np.log(cruiseset_kd[iCruise].reshape(-1,1))) # 
        x = np.linspace(0.01,np.nanmax(cruiseset),100)
        logx = np.log(x)
        yfitlog = lm.predict(logx.reshape(-1,1))
        yfit = e**yfitlog
        coef = e**lm.intercept_[0]
        exp = lm.coef_[0][0]
    
        model.gam_kd[site] = coef * model.gam_ssc[site]**exp
    
        # Make a figure of the site-specific SSC to Kd conversion
        fig,ax = plt.subplots(figsize=(6,3)) 
        ax.scatter(cruiseset,cruiseset_kd,color='grey')
        ax.plot(x,yfit,color='k')
        ax.text(0.05,0.9,'Kd = '+str(round(coef,3))+' * SSS^('+str(round(exp,3))+')',transform=ax.transAxes)
        ax.set_title(site + ' - CruiseSites:' + str(matchsites))
        fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_to_Kd.png'))    

    if 1: # statsmodels, report uncertainty, plot more info        
        # Fit with statsmodels to get some uncertainty info.
        df=pd.DataFrame(dict(ssc=cruiseset[iCruise].ravel(),
                             kd= cruiseset_kd[iCruise].ravel()))
        if 1: # limit the fit to when 4.5/Kd>0.5
            select=4.5/df['kd'] > 0.5
            print(f"{site}: limit fit to 4.5/Kd>0.5, retains {select.sum()} of {len(df)} samples")
            df=df[select]
        else:
            select=True
        mod=smf.ols('np.log(kd) ~ np.log(ssc)',df)
        fit=mod.fit() 
        
        pred=fit.get_prediction(pd.DataFrame(dict(ssc=model.gam_ssc[site])))    
        model.gam_kd[site] = np.exp(pred.predicted_mean)
    
        x = np.linspace(0.01,np.nanmax(cruiseset),100)
        pred=fit.get_prediction(pd.DataFrame(dict(ssc=x)))    
        fig,ax = plt.subplots(figsize=(6,3)) 
        ax.scatter(cruiseset[iCruise],cruiseset_kd[iCruise],3+4*select,imatch)
        df_pred=pred.summary_frame() # mean, mean_se, {obs,mean}_ci_{lower,upper}
        ax.plot(x,np.exp(df_pred['mean']),color='k')
        ax.fill_between(x,
                        np.exp(df_pred['obs_ci_lower']),
                        np.exp(df_pred['obs_ci_upper']),
                        color='tab:blue',alpha=0.1)
        ax.fill_between(x,
                        np.exp(df_pred['mean_ci_lower']),
                        np.exp(df_pred['mean_ci_upper']),
                        color='tab:blue',alpha=0.5)
        ax.plot(x,np.exp(df_pred['mean']),color='k')
        
        coef=np.exp(fit.params[0])
        exp=fit.params[1]
        cis=fit.conf_int().values # {intercept,exp} x {lower, upper}
        txts=[f'Kd = {coef:.3f} SSC$^{{{exp:.3f}}}$ */ {np.exp(np.std(fit.resid)):.3f}',
              f' coef $\in$ [{np.exp(cis[0,0]):.3f},{np.exp(cis[0,1]):.3f}]',
              f' exp $\in$ [{cis[1,0]:.3f},{cis[1,1,]:.3f}]']
    
        ax.text(0.02,0.95,"\n".join(txts),transform=ax.transAxes,va='top')
        ax.set_title(site + ' - CruiseSites:' + str(matchsites))
        ax.set_xlabel('SSC (mg/l)')
        ax.set_ylabel('K$_D$ (m$^{-1}$)')
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_to_Kd_errs.png'))    

    ############################################################################
    
    # Plot the output
    
    fig,axs = plt.subplots(1,2,figsize=(13,3.5))
    # RH: report RMSE here, too.
    rmse=np.sqrt(np.nanmean( (data[site].gam_ssc-data[site].ssc)**2 ))
    axs[0].plot(data[site].ts_pst,data[site].ssc,label='Observed (sensor)',color='k',zorder=1)
    axs[0].plot(data[site].ts_pst,data[site].gam_ssc,label=f'Modeled (RMSE {rmse:.1f}mg/l)',
                color='darkred',zorder=2)
    for j in range(np.shape(cruiseset)[1]):
        axs[0].scatter(cruiseset_ts,cruiseset[:,j],label='Discrete - USGS '+str(matchsites[j]),zorder=3+j,s=3)
    axs[0].legend(loc='upper left')
    axs[0].set_ylabel('SSC (mg/L)')
    
    axs[1].plot(model.ts_pst,model.gam_kd[site],label='Model Kd')
    for j in range(np.shape(cruiseset)[1]):
        axs[1].scatter(cruiseset_ts,cruiseset_kd[:,j],label='Discrete - USGS '+str(matchsites[j]),zorder=3+j,s=3)
    axs[1].set_xlim((model.ts_pst[0],model.ts_pst[-1]))
    axs[1].set_ylim((0,8))
    fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_Kd_and_rmse.png'))    
    
    ############### Create and write output #####################################
    output = {}
    output['ts_pst'] = model.ts_pst # output for 2010-2019
    output['ssc_mgL'] = model.gam_ssc[site].copy() # output is gam model
    output['kd'] = model.gam_kd[site].copy() # output is gam model
    output['kd_gam'] = model.gam_kd[site].copy() # keep kd from gam additionally to evaluate errors.
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
    out.to_csv(os.path.join(dir_output,site+'_SSCandKd_Filled_'+str(OutputStart)[:4]+'_to_'+str(OutputEnd)[:4]+'.csv'),index=False)
    
    ############################################################################
#%%

from stompy import filters

def plot_ssc_scatter(pred,obs,lowpass=None,num=100):
    """
    lowpass: lowpass cutoff in hours, or None for no filter.

    Parameters
    ----------
    pred : TYPE
        DESCRIPTION.
    obs : TYPE
        DESCRIPTION.
    lowpass : TYPE, optional
        DESCRIPTION. The default is None.
    num : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    if lowpass is not None:
        winsize=int(round(lowpass/OutputTimeStep))
        if winsize<=1:
            print("Lowpass cutoff {lowpass} leads to ineffective window size {winsize}")
        else:
            pred_lp=filters.lowpass_fir(pred,winsize)
            obs_lp =filters.lowpass_fir(obs,winsize)
            if 0: # visual check on filtering
                plt.figure(100+num).clf()
                plt.plot(pred,label='Predicted')
                plt.plot(pred_lp,label='Predicted, LP')
                plt.plot(obs,label='Observed')
                plt.plot(obs_lp,label='Observed, LP')
                plt.legend()
            # roughly grab independent samples
            pred=pred_lp[::winsize//2]
            obs=obs_lp[::winsize//2]
    valid=np.isfinite(pred+obs)
    pval=pred[valid]
    oval=obs[valid]
    
    fig=plt.figure(num)
    fig.set_size_inches([4.5,4])
    fig.clf()
    
    ax=fig.add_subplot(1,1,1)
    
    ax.plot( oval, pval, 'k.',ms=2,alpha=0.2)
    ax.axis('equal')
    ax.set_adjustable('box')
    
    ssc_max=np.percentile(oval,98)
    ax.axis(ymin=0,xmin=0,ymax=ssc_max,xmax=ssc_max)
    ax.plot([0,ssc_max],[0,ssc_max],'-',color='black',lw=2.5)
    ax.plot([0,ssc_max],[0,ssc_max],'-',color='yellow',lw=1.5)
    ax.set_xlabel('Observed (mg/l)')
    ax.set_ylabel('GAM (mg/l)')
    ax.text(0.05,0.99,site,transform=ax.transAxes,va='top')
    fig.subplots_adjust(top=0.95,right=0.95)
    
    if lowpass:
        # reducing by winsize//2 is too drastic. try
        # reducing to 8.
        breaks=np.percentile(oval,np.linspace(0,100,8))
    else:
        breaks=np.percentile(oval,np.linspace(0,100,50))
    breaks[-1]+=0.1
    breaks[0]-=0.1
    bins=np.searchsorted(breaks[:-1],oval)-1
    bin_centers=[]
    bin_lows=[]
    bin_highs=[]
    # last bin never looks quite right.
    for b,idxs in utils.enumerate_groups(bins):
        if b==bins.max(): break
        p_in_bin=pval[idxs]
        quarts=np.percentile(p_in_bin,[25,75])
        bin_centers.append(0.5*(breaks[b]+breaks[b+1]))
        bin_lows.append(quarts[0])
        bin_highs.append(quarts[1])
    ax.plot(bin_centers,bin_lows,'k-',lw=2.5)
    ax.plot(bin_centers,bin_lows,'r-')
    ax.plot(bin_centers,bin_highs,'k-',lw=2.5)
    ax.plot(bin_centers,bin_highs,'r-')
    return fig

for site in sites:
    pred=data[site].gam_ssc
    obs =data[site].ssc
    fig=plot_ssc_scatter(pred,obs,lowpass=60)
    fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_scatter_LP60h.png'))
    
    fig=plot_ssc_scatter(pred,obs)
    fig.savefig(os.path.join(dir_ssc2kdfigs,site+'_ssc_scatter.png'))
#%%
# Target diagram doesn't make sense because we're fitting the 
# model directly to the data. bias is zero.

recs=[]
all_obs=[]
all_obs_lp=[]
for site in sites:
    pred=data[site].gam_ssc
    obs =data[site].ssc
    bias = np.nanmean(pred-obs)
    ubrmse = np.sqrt( np.nanmean( (pred-bias-obs)**2))
    valid=np.isfinite(pred-obs)
    obs_std=np.std(obs[valid])
    mod_std=np.std(pred[valid])
    if mod_std<obs_std:
        ubrmse_sgn=-ubrmse
    else:
        ubrmse_sgn=ubrmse

    # And a 60h lp version:
    lowpass=60
    winsize=int(round(lowpass/OutputTimeStep))
    # samples aren't independent, but not a big deal for just calculating
    # error stats
    pred_lp=filters.lowpass_fir(pred,winsize)
    obs_lp =filters.lowpass_fir(obs,winsize)
    bias_lp=np.nanmean(pred_lp-obs_lp)
    ubrmse_lp = np.sqrt( np.nanmean( (pred_lp-bias_lp-obs_lp)**2))
    valid_lp=np.isfinite(pred_lp+obs_lp)
    obs_lp_std=np.std(obs_lp[valid_lp])
    mod_lp_std=np.std(pred_lp[valid_lp])
    if mod_lp_std<obs_lp_std:
        ubrmse_lp_sgn=-ubrmse_lp
    else:
        ubrmse_lp_sgn= ubrmse_lp
        
    recs.append(dict(site=site,bias=bias,ubrmse=ubrmse,
                     ubrmse_sgn=ubrmse_sgn,
                     bias_lp=bias_lp,
                     ubrmse_lp=ubrmse_lp,
                     ubrmse_lp_sgn=ubrmse_lp_sgn))
    all_obs.append(obs[valid])
    all_obs_lp.append(obs_lp[valid_lp])

global_std=np.std(np.concatenate(all_obs))    
global_std_lp=np.std(np.concatenate(all_obs_lp))
df_target=pd.DataFrame(recs)


plt.figure(300).clf()
fig,ax=plt.subplots(num=300)
ax.axhline(global_std,ls='--',zorder=-1,color='k',lw=1)
sns.barplot(data=df_target,x='site',y='ubrmse')
plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
fig.tight_layout()
fig.savefig(os.path.join(dir_ssc2kdfigs,'rmse.png'))


plt.figure(301).clf()
fig,ax=plt.subplots(num=301)
ax.axhline(global_std_lp,ls='--',zorder=-1,color='k',lw=1)
sns.barplot(data=df_target,x='site',y='ubrmse_lp')
plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
fig.tight_layout()
fig.savefig(os.path.join(dir_ssc2kdfigs,'rmse_lp.png'))
