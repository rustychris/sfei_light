# -*- coding: utf-8 -*-
"""
Test additional terms in GAM fits, possibly reaching into
R for better GAM support.

This script now just handles the preprocessing for each station, stuffing
observations into CSV files for subsequent fitting. 

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
import datetime
import pickle

from stompy import utils
from stompy import filters
from dotmap import DotMap
from stompy.spatial import interp_nn, proj_utils
from stompy.grid import unstructured_grid
import utm_to_aa

#%%

# Analysis sites
# Note that Golden Gate Bridge does not have recent enough data (2000 forward) to overlap with the predictor data set (2000-2020)
import gam_common
from gam_common import (sites,OutputTimeStep,OutputStart,OutputEnd)

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
# Now using local wind
#dir_wind = data_root+'/Data_Forcing/Wind'
#file_wind = os.path.join(dir_wind,'ASOS-HWD_Wind_2000-2019.p')

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
#wnd = pickle.load(open(file_wind,'rb')) # time series is complete
#step = wnd.ts_pst[1] - wnd.ts_pst[0]
#window = int(np.timedelta64(OutputTimeStep,'h')/step)
#if window > 1: # smooth the data if the input time step is smaller than the output step
#    wnd.spd = pd.Series(wnd.spd).rolling(window=window,min_periods=1).mean().to_numpy()

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
#model.wnd = np.ones(len(model.ts_pst))*np.nan
#_,iA,iB = np.intersect1d(model.ts_pst,wnd.ts_pst,return_indices=True)
#model.wnd[iA] = wnd.spd[iB]

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

g=unstructured_grid.UnstructuredGrid.read_dfm("../Grid/wy2013c_waqgeom.nc",cleanup=True)

#%%


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

# Local tide processing:
import xarray as xr
from stompy.grid import unstructured_grid
from stompy import harm_decomp

harmonics = xr.open_dataset("../Data_HydroHarmonics/harmonics-wy2013.nc")
grid_harmonics=unstructured_grid.UnstructuredGrid.read_ugrid(harmonics)

def get_local_tides(site, utm, time_pst):
    if site=='Mare_Island_Causeway':
        # default location pulls from intertidal area, and model is crap
        # up on Napa due to some bad bathy.
        print(f"Local tides special case for {site}")
        print("  Will pull tides from Carquinez Strait")
        utm=[566407., 4213095]
    
    c=grid_harmonics.select_cells_nearest(utm)
    cc=grid_harmonics.cells_center()
    dist = utils.dist(cc[c],utm)
    print(f"Site {site} will get harmonics from point {dist:.1f} m away")
    omegas=harmonics.omegas.values
    h_comps = harmonics.stage_harmonics.isel(face=c)
    u_comps = harmonics.u_harmonics.isel(face=c)
    v_comps = harmonics.v_harmonics.isel(face=c)
    
    h_m2 = h_comps.sel(component='M2')
    u_m2 = u_comps.sel(component='M2')
    v_m2 = v_comps.sel(component='M2')
    
    # Can we choose principal direction solely from M2 harmonics?
    # I'm sure this can be done directly, but to avoid introducing
    # new bugs, fabricate one cycle of h,u,v and then extract principal
    # theta.
    t_test=np.linspace(0,2*np.pi,50)
    # Follow sign convention of harm_decomp.recompose:
    h_test=h_m2.values[0]*np.cos(t_test - h_m2.values[1])
    u_test=u_m2.values[0]*np.cos(t_test - u_m2.values[1])
    v_test=v_m2.values[0]*np.cos(t_test - v_m2.values[1])
    theta=utils.principal_theta(np.c_[u_test,v_test],h_test,
                                positive='flood')
    # report in compass direction, but theta is radians in math convention.
    print(f"Site {site}: principal flood direction {(90-180*theta/np.pi)%360:.1f} degTrue")
    
    # I think the model runs in UTC, so harmonics are referenced to UTC
    t_utc = (time_pst + np.timedelta64(8,'h') - harmonics.t_ref.values) / np.timedelta64(1,'s')
    h_pred=harm_decomp.recompose(t_utc,h_comps.values, omegas)
    u_pred=harm_decomp.recompose(t_utc,u_comps.values, omegas)
    v_pred=harm_decomp.recompose(t_utc,v_comps.values, omegas)
    u_flood_pred=u_pred*np.cos(theta) + v_pred*np.sin(theta) 

    if np.any(np.isnan(h_pred)):
        breakpoint()
    return h_pred,u_flood_pred

#%%

# Get local wind:

# Would be nice to go straight from the csv and station coordinates,
# with something that approximates natural neighbor interpolation
# to remain consistent with Allie's fields.


# prepare inputs:
#wind_data_dir="../Data_Forcing/WindAK"
wind_data_dir="/richmondvol1/google/SFEI_Wind"
wind_obs_dir=os.path.join(wind_data_dir,"Compiled_Hourly_10m_Winds","data")

assert os.path.exists(wind_obs_dir),f"Update compiled wind data path in {__file__}"

def load_obs_wind(year):
    fn=os.path.join(wind_obs_dir,f"SFB_hourly_wind_and_met_data_{year}.nc")
    if not os.path.exists(fn):
        print(f"No wind data for year={year}")
        return None
    wind_ds=xr.open_dataset(fn)
    wind_ds=wind_ds.rename_vars({'time':'julian_day'})
    
    # Times are julian day, PST. Wind in m/s
    t_pst=np.datetime64(f"{year}-01-01") + np.timedelta64(86400,'s')*(wind_ds.julian_day-1.0)
    wind_ds['time']=('time',),t_pst.values
    
    ll=np.c_[ wind_ds.longitude.values, wind_ds.latitude.values]
    xy=proj_utils.mapper('WGS84','EPSG:26910')(ll)
    wind_ds['x_utm']=('station',),xy[:,0]
    wind_ds['y_utm']=('station',),xy[:,1]
    return wind_ds


def get_local_wind(site,utm,time_pst):
    """
    Given a utm point [x,y] and array of datetime64 in PST,
    return u and v, each length of time_pst, with the NN
    interpolated wind field (from Allie).
    """
    time_pst = np.asarray(time_pst) # drop Series wrapper
    
    u_years=[]
    v_years=[]

    t_remaining=time_pst
    while len(t_remaining):
        dt=utils.to_datetime(t_remaining[0])
        year=dt.year
        next_year=datetime.datetime(year=year+1,month=1,day=1)
        sel=t_remaining<np.datetime64(next_year)

        t_year=t_remaining[sel]
        t_remaining=t_remaining[~sel]
        
        u_result=np.zeros(len(t_year),np.float64)
        v_result=np.zeros(len(t_year),np.float64)
        
        wind_ds=load_obs_wind(year)
        if wind_ds is None:
            # Fill with nan
            u_result[:]=np.nan
            v_result[:]=np.nan
        else:
            wind_xy=np.c_[ wind_ds.x_utm, wind_ds.y_utm ]
        
            # wind time is already in PST
            # but it suffers from some roundoff, so we can't use
            # searchsorted directly.
            t_idxs=utils.nearest(wind_ds.time.values, t_year)
            
            t_err = wind_ds.time.values[t_idxs] - t_year
            tol=np.timedelta64(2,'h')
            invalid = (t_err<-tol) | (t_err>tol)
            
            u10=wind_ds.u10.values # time,station
            v10=wind_ds.v10.values # 
            
            for i,t_idx in enumerate(t_idxs):
                if invalid[i]:
                    u_result[i]=np.nan
                    v_result[i]=np.nan
                    continue
                valid = np.isfinite(u10[t_idx,:]+v10[t_idx,:])
                valid_xy=wind_xy[valid]
        
                weights=interp_nn.nn_weights_memo(valid_xy,utm)
                u_result[i] = (weights*u10[t_idx,valid]).sum()
                v_result[i] = (weights*v10[t_idx,valid]).sum()
            
        u_years.append(u_result)
        v_years.append(v_result)
    u_final=np.concatenate(u_years)
    v_final=np.concatenate(v_years)
    return u_final, v_final


#%%
def get_global_ts(time_pst,source_time,source_value):
    result = np.full(len(time_pst),np.nan)
    _,iA,iB = np.intersect1d(time_pst,source_time,return_indices=True)
    result[iA] = source_value[iB]
    return result

#def get_wind(site=None,utm=None,time_pst=None):
#    return get_global_ts(time_pst=time_pst,
#                         source_time=wnd.ts_pst,source_value=wnd.spd)

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

    # Make sure this spans the OutputStart/End interval
    # Expand rawdf to reflect the period of data *and* the output.
    # i.e. this is going to be a file both used to train the GAM, and to evaluate it
    # for the output period. in some cases like Carquinez Bridge that's a bit wasteful
    # since there is a long gap between the training data and the output data, but
    # not worth the complexity.
    pieces=[]
    if OutputStart<rawdf.ts_pst.values[0]:
        pieces.append(pd.DataFrame([dict(ts_pst=OutputStart)]))
    pieces.append(rawdf)
    if OutputEnd>rawdf.ts_pst.values[-1]:
        pieces.append(pd.DataFrame([dict(ts_pst=OutputEnd)]))
    rawdf=pd.concat(pieces)
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
    
    # rawdf['wind'] = get_wind(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['tdvel'] = get_tide_velocity(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['wl'] = get_tide_elevation(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['storm'] = get_trib_flow(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['delta'] = get_delta_flow(site=site,utm=utm,time_pst=rawdf.ts_pst) 

    h,u = get_local_tides( site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['h_tide_local']= h
    rawdf['u_tide_local']= u
    
    wind_u,wind_v=get_local_wind(site=site,utm=utm,time_pst=rawdf.ts_pst)
    rawdf['wind_u_local']=wind_u
    rawdf['wind_v_local']=wind_v    
    wind_speed=np.sqrt(wind_u**2 + wind_v**2)
    rawdf['wind_spd_local']=wind_speed

    # and a boxcar from -4 hours to current sample, with TLC at the
    # start of the sequence and make sure it's the preceding samples,
    # not centered.
    def antecedent(x,winsize=5):
        y=filters.lowpass_fir(x,winsize,window='boxcar',
                              mode='full')[:-(winsize-1)]
        assert len(y)==len(x)
        return y
    rawdf['wind_u_4h_local']=antecedent(wind_u)
    rawdf['wind_v_4h_local']=antecedent(wind_v)
    # for resuspension, I think it's a bit better to average speed than
    # velocity.
    rawdf['wind_spd_4h_local']=antecedent(wind_speed)

    rawdfs.append(rawdf)
    station_dfs[site] = rawdf
   
dest_dir=gam_common.data_dir
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for site in station_dfs:
    df=station_dfs[site]
    valid=df['ssc_mgL'].notnull().values
    for required in ['wind_u_local','tdvel','storm','delta']:
        valid=valid & df[required].notnull()
    print(f"Site {site}: {valid.sum()} / {len(valid)}  ({100*valid.sum()/len(valid):.1f}%) valid")
    #df_valid=df[valid]
    #df_valid.to_csv(os.path.join(dest_dir,f'model-inputs-valid-{site}.csv'))
    df.to_csv(os.path.join(dest_dir,f'model-inputs-{site}.csv'))
    
