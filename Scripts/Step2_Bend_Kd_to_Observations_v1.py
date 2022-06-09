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
from stompy import filters
from matplotlib.dates import date2num
import pickle
from dotmap import DotMap
from stompy import utils

OutputTimeStep = 1 # hours, must be at least 1 hour and no more than 24 hours

SmoothShift_Window = 30 # days, for smoothing offset/shift time-series
win = int(np.timedelta64(SmoothShift_Window,'D')/np.timedelta64(OutputTimeStep,'h'))

Example_WY = 2017 # an example water year for plotting a shorter window

# Analysis sites
#sites = ['Mallard_Island']
sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge',
         'Channel_Marker_01','Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']

# Input Kd data path 
input_ext = '_SSCandKd_Filled_2009_to_2018.csv'
dir_input = r'../Data_Constructed_SSC_and_Kd_RH'

dir_matchref = r'..'
file_matchref = os.path.join(dir_matchref,'Match_Cruise_to_HFsite_for_Kd_Bending.xlsx')

dir_cruise = r'../Data_Cruise'
file_cruisessc = os.path.join(dir_cruise,'Curated_SiteSpec_SSCandKD_2_to_36.p')

dir_output = r'../Data_Kd_Shifted_RH'

dir_figs = r'../Figures_KdOutput_vs_KdObserved_RH'

# %%

#Load cruise data, but handling is site-specific
cruise = pickle.load(open(file_cruisessc,'rb'))

# Load cruise match table
matchref = pd.read_excel(file_matchref)


# %%

for d in [dir_output,dir_figs]:
    if not os.path.exists(d):
        os.makedirs(d)


from matplotlib import gridspec

def plot_gam_error_post_shift(data,num=None,lowpass=None,invert=True):
    """
    num: figure number
    lowpass: None for no filter, or cutoff in hours.
    invert: plot as 4/Kd.  Otherwise plot as Kd.
    """
    # Sample gam vs observed calculation:
    pred=data.kd_PropShiftedobs
    obs =data.kd_PropShiftedgam
    ts_pst=data.ts_pst
    
    if invert:
        pred=4./pred
        obs=4./obs
    
    if lowpass is not None:
        winsize=int(round(lowpass/OutputTimeStep))
        pred=filters.lowpass_fir(pred,winsize)[::winsize//2]
        obs =filters.lowpass_fir(obs, winsize)[::winsize//2]
        ts_pst=ts_pst[::winsize//2]

    valid=np.isfinite(obs * pred)    
    errors=pred-obs # data.kd_PropShiftedgam - data.kd_PropShiftedobs
    
    if np.isfinite(errors).sum()<10:
        # probably the output period doesn't overlap the station
        return None

    if invert:
        qty="4/K$_d$"
    else:
        qty="K$_d$"
    
    errors=errors[np.isfinite(errors)]
    
    fig=plt.figure(num=num)
    fig.set_size_inches((9,4),forward=True)
    fig.clf()
    gs=gridspec.GridSpec(2,5)
    
    ax_hist=fig.add_subplot(gs[0,:-2])
    ax_ts=fig.add_subplot(gs[1,:-2])
    ax_scat=fig.add_subplot(gs[:,-2:])
    
    ax_hist.hist(errors,bins=200)
    ax_hist.text(0.01,0.9,f"{qty} error\nprop shifted\ngam$-$obs",
                 transform=ax_hist.transAxes,va='top')
    # Add some metrics:
    rmse=np.sqrt(np.mean(errors**2))
    txt=f"{qty} RMSE: {rmse:.2f}"
    ax_hist.text(0.95,0.9,txt,transform=ax_hist.transAxes,
                 ha='right',va='top')
        
    if 1:             
        from matplotlib import colors
        if lowpass:
            nbins=30
        else:
            nbins=100
        ax_scat.hist2d(obs,pred,
                       norm=colors.LogNorm(vmin=1,clip=True),
                       alpha=1,cmap='magma_r',
                       bins=2*[np.linspace(0,6,nbins)])

    if 1: # IQ line plot
        # Steal code from RHStep1
        valid=np.isfinite(pred+obs)
        pval=pred[valid]
        oval=obs[valid]
        
        if lowpass is not None:
            breaks=np.percentile(oval,np.linspace(0,100,10))
        else:            
            breaks=np.percentile(oval,np.linspace(0,100,50))
        breaks[-1]*=1.1 # avoid anybody over the top
        breaks[0]*=0.9 # or under the bottom
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
        ax_scat.plot(bin_centers,bin_lows,'k-',lw=2.5)
        ax_scat.plot(bin_centers,bin_lows,'r-')
        ax_scat.plot(bin_centers,bin_highs,'k-',lw=2.5)
        ax_scat.plot(bin_centers,bin_highs,'r-')        
    
    ax_hist.yaxis.set_visible(0)
    ax_ts.plot(ts_pst, obs, label='obs')
    ax_ts.plot(ts_pst, pred, label='gam')
    ax_ts.legend(loc='upper left',frameon=False)
    ax_ts.set_ylabel(qty)
    ax_scat.set_xlabel(f'{qty} obs')
    ax_scat.set_ylabel(f'{qty} gam')
    ax_scat.plot([0,6],[0,6],'g-',lw=0.75)
    ax_scat.axis([0,6,0,6])
    fig.text(0.5,0.97,site,va='top',ha='center')
    fig.subplots_adjust(wspace=0.5,bottom=0.15)
    plt.draw()
    plt.pause(0.01)
    return fig

for site in sites:
    inp = pd.read_csv(os.path.join(dir_input,site+input_ext))
    
    data = DotMap()
    data.ts_pst = pd.to_datetime(inp['ts_pst']).to_numpy()
    data.kd = inp['kd'].to_numpy()
    data.kd_gam = inp['pred_kd'].to_numpy()
    data.flag = inp['flag'].to_numpy()
    data.kd_obs = np.where(data.flag==1,data.kd,np.nan)
    
    iSite = matchref['Site'] == site
    if ',' in str(matchref['CruiseStations'][iSite].values[0]):
        matchsites = tuple(map(int,matchref['CruiseStations'][iSite].values[0].split(',')))
    else:
        matchsites = (matchref['CruiseStations'][iSite].values[0],)
    
    # start with an overshoot time-series; so we can mesh with both the model fit and with the prediction data
    cruise_ts = data.ts_pst # DBG: this is coming up empty
    cruise_kd_set = np.ones((len(cruise_ts),len(matchsites)))*np.nan

    j = 0
    for n in matchsites: # loop over cruise sites associated with this SSC site
        site_ts = pd.Series(cruise[n].ts_pst).dt.round(str(OutputTimeStep)+'h').to_numpy()
        _,iA,iB = np.intersect1d(cruise_ts,site_ts,return_indices=True)
        cruise_kd_set[iA,j] = cruise[n].Kd[iB] # array of cruise Kd
        j = j+1
        
    cruise_kd = np.nanmean(cruise_kd_set,axis=1) # collapse cruise_ssc data into site-averages when time steps overlap
    
    out = {}
    out['ts_pst'] = data.ts_pst
    out['Kd'] = np.round(data.kd,2)

    for src_data in ['','obs','gam']:
        if src_data=='':
            kd_in=data.kd
        elif src_data=='obs':
            kd_in=data.kd_obs
        elif src_data=='gam':
            kd_in=data.kd_gam
        ovm_prop = cruise_kd / kd_in # modeled/measure proportion (at times when there is cruise data)
        ovm_diff = cruise_kd - kd_in # modeled - measured difference (at times when there is cruise data)
        
        iCruise = ~np.isnan(cruise_kd) # where there is cruise data
        ovm_prop_filled = np.interp(date2num(cruise_ts),
                                    date2num(cruise_ts[iCruise]),
                                    ovm_prop[iCruise])
        ovm_diff_filled = np.interp(date2num(cruise_ts),
                                    date2num(cruise_ts[iCruise]),
                                    ovm_diff[iCruise])
        
        ovm_prop_smooth = (pd.Series(ovm_prop_filled)
                           .rolling(window=win,center=True,min_periods=1)
                           .mean().to_numpy())
        ovm_diff_smooth = (pd.Series(ovm_diff_filled)
                           .rolling(window=win,center=True,min_periods=1)
                           .mean().to_numpy())
        
        data['kd_PropShifted'+src_data] = kd_in * ovm_prop_smooth
        data['kd_DiffShifted'+src_data] = kd_in + ovm_diff_smooth
        out['Kd_PropShifted'+src_data] = np.round(data.kd_PropShifted,2)
        out['Kd_DiffShifted'+src_data] = np.round(data.kd_DiffShifted,2)
    
    ### Output the data    
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
    
    fig=plot_gam_error_post_shift(data)
    if fig is not None: # may fail if no observed data
        fig.savefig(os.path.join(dir_figs,site+'_Kd_gam_shifted_error.png'),dpi=150)
        
    fig=plot_gam_error_post_shift(data,lowpass=60)
    if fig is not None: # may fail if no observed data
        fig.savefig(os.path.join(dir_figs,site+'_Kd_gam_shifted_error_lp60.png'),dpi=150)

#%%


#plot_gam_error_post_shift(data)
