# -*- coding: utf-8 -*-
"""
Test additional terms in GAM fits, possibly reaching into
R for better GAM support.

@author: rustyh, adapted from derekr
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
from stompy import filters
import seaborn as sns
from dotmap import DotMap
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

for d in [dir_gamfigs, dir_ssc2kdfigs, dir_output]:
    if not os.path.exists(d):
        os.makedirs(d)



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
    # This step might be reaching too far. Currently no control over extrapolating
    # very far in time.
    cruise_filled = np.interp(date2num(cruiseset_ts),date2num(cruiseset_ts[iCruise]),cruise_ssc[iCruise],
                              left=np.nan,right=np.nan)
    # smooth it in time - THIS IS THE LONG-TERM TREND forcing data
    CruiseSet_SSC = pd.Series(cruise_filled).rolling(window=lfwin,center=True).mean().to_numpy()

    # cruise data is already padded out 100 days, beyond which just return nan.
    return np.interp( date2num(times),
                      date2num(cruiseset_ts), CruiseSet_SSC, left=np.nan, right=np.nan)

#%%

from stompy.grid import unstructured_grid
g=unstructured_grid.UnstructuredGrid.read_dfm("../Grid/wy2013c_waqgeom.nc",cleanup=True)

#%%

import utm_to_aa

# Compile all of the observed SSC into a global dataframe that can then
# be annotated with more local data.

rawdfs=[]

station_coords=pd.read_csv("../Station_Polygon_Coordinates.csv").set_index('Site')
stn_utm = station_coords[ ['utm_e','utm_n'] ].values
aa=utm_to_aa.utm_to_aa(stn_utm)
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

station_dfs={} # site => DataFrame
for site in sites:
    rawdf=pd.read_csv(os.path.join(dir_sscdata,site+'Processed_15min.csv'),
                      parse_dates=['ts_pst'])
    # downsample to the output time resolution
    #rawdf=rawdf.resample('h',on='ts_pst').mean().reset_index()
    # Follow the original code! a straight resample like that introduces
    # a shift.
    # instead, smooth at the desired interval, then pull individual 
    # samples
    
    # verify we have evenly spaced data before filtering
    dt_s=np.diff(rawdf.ts_pst)/np.timedelta64(1,'s')
    assert np.all(dt_s==dt_s[0])
    dt_s=dt_s[0]
    assert dt_s==900 # could be dynamic, but need more info
    winsize=5 # odd avoids offsets 
    rawdf['ssc_mgL'] = filters.lowpass_fir(rawdf.ssc_mgL.values,winsize)
    # And *now* downsample. while mean() silently drops ts_pst b/c it doesn't
    # know how to average time, here it is explicit. 
    rawdf=rawdf.resample('h',on='ts_pst').first().drop('ts_pst',axis=1).reset_index()
        
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
    station_dfs[site] = rawdf
    
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
    raw.ssc_orig = rawdf['ssc_mgL'].to_numpy()
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
    # map short variable names to plottable pretty names
    var_labels={'wnd_speed':'Wind Speed',
                'u_tide':'Tidal Velocity (flood+)',
                'h_tide':'Tidal Elevation',
                'Qalameda':'Alameda Flow',
                'Qdelta':'Delta Flow',
                'localTrend':'Local Seasonal Cruise SSC'
                }
    
    predictors = ['wnd_speed','u_tide','h_tide','Qalameda','Qdelta','localTrend']

    # Stuff all this into a per-site dataframe
    df=data[site]
    #pd.DataFrame()
    #df['wnd_speed']=data[site].wnd
    #df['u_tide'] = data[site].tdvel

    
    X = np.vstack( (data[site].wnd, data[site].tdvel, data[site].wl,
                    data[site].localq, data[site].deltaq, data[site].cruise_ssc )).T
    Y = data[site].ssc
    iXgood = ~np.isnan(X).any(axis=1)
    iYgood = (~np.isnan(Y))
    iGood =  iXgood & iYgood
    xGood = X[iGood,:]
    yGood = Y[iGood]
    
    # Train on the first 80% of the data, test on last 20%
    # Don't use train_test_split, since it randomly chooses and 
    # with autocorrelation that's not a fair test.

    
        
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
        ax.set_title(var_labels[predictors[i]],fontsize=9)
    axs[0].set_ylabel('SSC (mg/L)')
    fig.text(0.04, 0.5,site, va='center', rotation='vertical',fontsize=12)
    # fig.savefig(os.path.join(dir_gamfigs,site+'_gam_fits.png')) 
    
    break

#%%

# Make sure the dataframe closely matches data[site] before ditching
# all look good.
# had to futz with filtering of ssc from 15 minute to an hour,
# and avoid too much extrapolation for cruise data.

site='Alcatraz_Island'

for i,(data_var,df_var) in enumerate( [
        ('cruise_ssc','usgs_lf'),
        #('ssc','ssc_mgL'),
        #('localq','storm'),
        #('tdvel','tdvel'),
        #('wl','wl'),
        #('wnd','wind')
        ]):    
    plt.figure(100+i).clf()
    plt.plot(station_dfs[site].ts_pst,
             station_dfs[site][df_var],
             label=f'{df_var} data frame')
    plt.plot(data[site].ts_pst,
             data[site][data_var],
             label=f'{data_var} dotmap')
    if data_var=='ssc':
        plt.plot(raw.ts_pst,raw.ssc_orig,label='raw 15 min')
        plt.plot(raw.ts_pst,raw.ssc,label='15 min with rolling')
    plt.legend(loc='upper right')
