# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:10:32 2022

Test skill of original python-based GAM
@author: rusty
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from stompy import utils, filters, memoize

from pygam import LinearGAM, s, l, te
from sklearn.linear_model import LinearRegression
import RH_fetch_and_waves as waves
import pickle

#%%

data_dir="../DataFit00"

sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']


dir_gamfigs="../Figures_GAM_dev"
if not os.path.exists(dir_gamfigs):
    os.makedirs(dir_gamfigs)
    
#%%

with open('global-model.pkl','rb') as fp:
    ggam=pickle.load(fp)
# ggam.pred_vars: fields to pass to ggam.predict
# ggam.dep_var: variable that is predicted

#%%
from gam_common import antecedent, envelope, grid, parse_predictor


g=grid()

@memoize.memoize(lru=2)
def src_data(site):
    src = pd.read_csv(os.path.join(data_dir,f"model-inputs-{site}.csv"),
                      parse_dates=['ts_pst'])
    
    # spring-neap indicator:
    src['u_rms']=np.sqrt( filters.lowpass_fir(src['u_tide_local']**2,60) )
    # Had been integer year, now decimal years.
    src['year'] = ((src.ts_pst.values - np.datetime64("2000-01-01"))
                   /(np.timedelta64(int(365.2425*86400),'s')))
    src['doy'] = src.ts_pst.dt.dayofyear
    src['tide_hour'] = utils.hour_tide(utils.to_dnum(src.ts_pst.values),
                                       u=src['u_tide_local'])
    src['wind_spd_ante']=antecedent(src['wind_spd_local'],30)
    src['wind_u_ante']=antecedent(src['wind_u_local'],30)
    src['wind_v_ante']=antecedent(src['wind_v_local'],30)
    
    utm=np.r_[ src.utm_e.values[0], src.utm_n.values[0] ]
    W=waves.fetch_limited_waves(utm,g,
                                src['wind_u_local'],src['wind_v_local'],
                                eta=0.75)
    src['wave_stress']=W['tau_w']
    src['wave_stress_ante']=antecedent(W['tau_w'],3)
    
    # Evaluate the global model for this station
    # Needs wl_rms as a global spring-neap indicator
    src['wl_rms']=np.sqrt( filters.lowpass_fir(src['wl']**2, 60 ))
    
    Xglobal=src[ggam.pred_vars]
    valid=np.all( Xglobal.notnull().values, axis=1)
    gpred=np.full(len(src),np.nan)
    gpred[valid]=ggam.predict(Xglobal.values[valid,:])
    src['glbl_'+ggam.dep_var]=gpred
    
    return src

#%%

if 0:# testing
    site="Mare_Island_Causeway"
    src=src_data(site)    
    # First, double check that data are coming in reasonably.
    for i, fld in enumerate(src.columns):
        if fld.startswith('Unnamed') or fld in ['turb','flags','site','utm_e','utm_n',
                                                'along','across','src']:
            continue
        plt.figure(i).clf()
        plt.plot(src.ts_pst,src[fld],label=fld)
        plt.legend(loc='upper left')
    
#%%


def rmse(a,b):
    return np.sqrt(np.mean( (a-b)**2))


#%%

# Notes regarding other formulas tried:
# RMSE train: 27.89  RMSE_test 30.54
#pred_vars = ['wind_spd_local','tdvel','wl','storm','delta','usgs_lf']

# RMSE train: 27.76 RMSE_test 30.46
# pred_vars = ['wind_spd_4h_local','tdvel','wl','storm','delta','usgs_lf']

# RMSE_train: 27.93  RMSE_test=30.98
#pred_vars = ['wind','tdvel','h_tide_local','storm','delta','usgs_lf']

# RMSE_train=27.47  RMSE_test=29.87
# pred_vars = ['wind','u_tide_local','wl','storm','delta','usgs_lf']

# RMSE_train=27.29  RMSE_test=30.03 def overfit
#pred_vars = ['wind_u_local','wind_v_local','wind_spd_local',
#             'tdvel', 'u_tide_local','wl','storm','delta','usgs_lf']

# 27.94 / 30.51. i.e. exactly 0.01 mgL better.
#src['storm_12h']=antecedent(src['storm'],12)
# pred_vars = ['wind','tdvel','wl','storm_12h','delta','usgs_lf']

# RMSE_train: 27.58  RMSE_test:32.66
#pred_vars = ['wind','tdvel','wl','storm','delta','usgs_lf','year']

# RMSE_train: 27.79, 30.85. overfit.
# pred_vars = ['wind','tdvel','wl','storm','delta','usgs_lf','pre_gap']

# RMSE_train: 27.52  RMSE_test=30.09
#pred_vars = ['wind','tdvel','wl','storm','delta','usgs_lf','tide_hour']

# RMSE_train=27.37 RMSE_test=30.09
#pred_vars = ['wind','tdvel','wl','storm','delta','usgs_lf','tide_hour','doy']

# RMSE_train: 27.06 RMSE_test=29.66
#pred_vars = ['wind','u_tide_local','wl','storm','delta','usgs_lf','tide_hour','doy']

# RMSE train: 27.03 RMSE_test=29.85
#src['storm_10d']=antecedent(src['storm'],240)
#pred_vars = ['wind','u_tide_local','wl','storm_10d','delta','usgs_lf','tide_hour','doy']

# RMSE_train: 26.89 RMSE_test=29.42
#pred_vars = ['wind_spd_4h_local','u_tide_local','wl','storm_12h','delta','usgs_lf','tide_hour','doy']

# RMSE_train: 26.90 RMSE_test=29.42: same, but dropped storm influence.
# can get a little better, even for test, with more splines.
# pred_vars = ['wind_spd_4h_local','u_tide_local','wl','delta','usgs_lf','tide_hour','doy']

# Scanning for best antecedent wind speed.
# 30 hours. train: 26.61  test=29.15
# Scanning for best antecedent storm flow: best is no storm flow.
# pred_vars = ['wind_spd_ante','u_tide_local','wl','delta','usgs_lf','tide_hour','doy']

# RMSE_train: 26.81 test: 29.36, but with a lot more DOFs.
#pred_vars = ['wind_spd_4h_local','u_tide_local','wl','storm_12h','delta','usgs_lf','tide_hour','doy']
#formula=  (   s(0)                  + s(1)    + s(2) + s(3)  +    s(4)   + s(5)   + s(6)  +    s(7)  
#             + te(0,6) )
             
# train: 26.85, test 29.39. slightly worse than with interaction with wind.
#pred_vars = ['wind_spd_4h_local','u_tide_local','wl','storm_12h','delta','usgs_lf','tide_hour','doy']
#formula=  (   s(0)                  + s(1)    + s(2) + s(3)  +    s(4)   + s(5)   + s(6)  +    s(7)  
#             + te(3,6) )
    
#%%

def ssc_to_zphotic(ssc):
    # Just use a representative fit with nice values
    # to get the basic idea.
    Kd=0.25*ssc**(2./3)
    return 4.5/Kd
def zphotic_to_ssc(zph):
    Kd=4.5/zph
    return (Kd/0.25)**(3./2)
    
recs=[]

# what are we trying to predict:
#dep_var='ssc_mgL'
#err_var='ssc_mgL'
dep_var='log_ssc_mgL'
err_var='log_ssc_mgL'

#dep_var='zphotic' 
#err_var='zphotic' # variable to use for error reporting

for site_i,site in enumerate(sites): # ['Dumbarton_Bridge']:
    print(f"Site: {site}")
    src=src_data(site)
    
    for v in [dep_var,err_var]:
        if v not in src:
            if v=='zphotic':
                src[v]=ssc_to_zphotic(src['ssc_mgL'])
            elif v=='log_ssc_mgL':
                src[v]=np.log10(src['ssc_mgL'].clip(1.0))
            else:
                raise Exception("Unknown dependent variable " + v)

    # SPB testing before implementing fetch
    # this gets about the same RMSE test as before. Just using 30h 
    # antecedent wind is almost as good.

    # predictors=[
    #     # 93/98
    #     #("te(wind_v_3h,wind_u_3h) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),
        
    #     # 95/99
    #     #("s(wave_stress_ante) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),

    #     # 96/99 
    #     #("s(wave_stress) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),

    #     # 97, 100
    #     #("s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),

    #     # 95, 99
    #     #("te(wave_stress_ante, u_tide_local)"
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " + s(delta)"),
        
    #     # 90/99
    #     #("te(wind_v_3h,wind_u_3h) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta) + s(doy)"),

    #     # 88/88
    #     #("te(wind_v_3h,wind_u_3h) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta) + s(year)"),

    #     # 86/74
    #     #("te(wind_v_3h,wind_u_3h) "
    #     # " + te(tide_hour,year) + s(h_tide_local)" 
    #     # " + te(tide_hour, u_rms)"),

    #     # 88/80
    #     #("s(wave_stress_ante) "
    #     # " + te(tide_hour,year) + s(h_tide_local)" 
    #     # " + te(tide_hour, u_rms)"),

    #     # --- Switching to log(ssc) ---
    #     # 0.28 / 0.43
    #     #("te(wind_v_3h,wind_u_3h) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),
       
    #     # 0.29 / 0.44
    #     #("s(wave_stress_ante) "
    #     # " + s(tide_hour) + s(h_tide_local)" 
    #     # " +s(u_tide_local) + s(delta)"),
        
    #     # 0.24 / 0.99
    #     #("te(wave_stress_ante,year) "
    #     # " + te(tide_hour, year) "
    #     # " + te(h_tide_local, year) " 
    #     # " + te(u_tide_local, year) "
    #     # " + te(delta,year)"),

    #     # Seems that most anything with te( x, year)
    #     # or s(year) does very poorly on test data.
    #     (
    #      "s(u_rms) "
    #      " + te(wind_v_3h,wind_u_3h) "
    #      " + s(wave_stress_ante) "
    #      " + s(u_tide_local)"
    #      ),
    #     ]
    
    predictors = [
        # original model
        #"s(wind) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)",
        
        # The model I would expect. Has best R2, but RMSE about
        # the same
        #( "te(tide_hour,wl_rms) "
        #  "+l(glbl_log10_ssc_mgL) "
        #  " + te(tide_hour,wave_stress_ante) "
        #  " + te(tide_hour,storm)"
        #),

        # variants
        # ( "te(tide_hour,wl_rms) "
        #   " + te(tide_hour,wave_stress_ante) "
        #   " + te(tide_hour,storm)"
        # ),

        # ( "s(tide_hour) + s(wl_rms) + s(u_tide_local) "
        #   " + te(tide_hour,wave_stress_ante) "
        #   " + te(tide_hour,storm)"
        # )
        
        # Kitchen sink model - slightly better than the assumed 
        # model.
        # ( "te(tide_hour,wl_rms) "
        #   "+l(glbl_log10_ssc_mgL) "
        #   " + te(tide_hour,wave_stress_ante) "
        #   " + te(tide_hour,storm)"
        #   " + s(delta,constraints=flatends) "
        #   " + s(usgs_lf,constraints=['monotonic_inc',flatends])"
        #   " + s(storm,constraints=flatends)"
        # ),
        
        ( "te(tide_hour,wl_rms) "
          "+l(glbl_log10_ssc_mgL) "
          " + te(tide_hour,wave_stress_ante) "
          " + te(tide_hour,storm)"
          " + s(delta,constraints=flatends) "
          " + s(usgs_lf,constraints=['monotonic_inc',flatends])"
          " + s(storm,constraints=flatends)"
        ),
        

    ]
        
    src['wind_u_3h']=antecedent(src['wind_u_local'],5)
    src['wind_v_3h']=antecedent(src['wind_v_local'],5)
    
    for predictor in predictors: 
        rec=dict(site=site)
        recs.append(rec)
                   
        gam_params={} #dict(n_splines=8)
    
        # mimic R style formulas to make it easier to test variations:
        rec['predictor']=predictor
        
        pred_vars,formula = parse_predictor(predictor)
            
        # RMSE train: 27.95  RMSE_test 30.52
        #pred_vars = ['wind','tdvel','wl','storm','delta','usgs_lf']
        
           
        # train: 26.04  test 28.98 
        # if using local water level: train 25.93, test: 29.03
        #pred_vars = ['wind_spd_ante','u_tide_local','h_tide_local','delta','usgs_lf','tide_hour','doy','u_rms']
        #formula=  (   s(0)                  + s(1) + s(2)           + s(3) +  s(4)             + s(6)    
        #             + te(5,7) )
        
        # train: 26.17  test: 29.01
        #pred_vars = ['wind_spd_ante','u_tide_local','wl','delta','usgs_lf','tide_hour','doy','u_rms']
           
        
        X = src[pred_vars].values
        Y = src[dep_var]
        
        iXgood = ~(np.isnan(X).any(axis=1))
        iYgood = ~np.isnan(Y)
        iGood =  iXgood & iYgood
        xGood = X[iGood,:]
        yGood = Y[iGood]
        
        # basic train/test 
        N=len(xGood)
        Ntrain=int(0.8*N)
        Ntest=N-Ntrain
        xGood_train=xGood[:Ntrain]
        xGood_test =xGood[Ntrain:]
        yGood_train=yGood[:Ntrain]
        yGood_test =yGood[Ntrain:]
        
        t_train=src['ts_pst'].values[iGood][:Ntrain]
        t_test =src['ts_pst'].values[iGood][Ntrain:]
        
        if Ntrain<100:
            # In particuar, Mare Island Causeway has bad local tide info
            print(f"Site {site}  variables: {', '.join(pred_vars)}: insufficient data.")
            continue
        # dial down the weights to discourage wiggly fits
        weights=1e-2 * np.ones_like(yGood_train)
        gam = (LinearGAM(formula,**gam_params)
               .fit(xGood_train,yGood_train,weights=weights))    
        #print(gam.summary())
        # calculate RMSE against training and test data, see how that varies with
        # LinearGAM parameters
        pred_train=gam.predict(xGood_train)
        pred_test =gam.predict(xGood_test)

        y_train=yGood_train
        y_test =yGood_test
        
        if err_var==dep_var:
            to_err_var=lambda x: x
        elif dep_var=='ssc_mgL' and err_var=='zphotic':
            to_err_var=ssc_to_zphotic
        elif dep_var=='zphotic' and err_var=='ssc_mgL':
            to_err_var=zphotic_to_ssc
        else:
            raise NotImplementedError("too lazy")        
            
        y_train=to_err_var(y_train)
        y_test =to_err_var(y_test)
        pred_train = to_err_var(pred_train)
        pred_test  = to_err_var(pred_test) 
        
        rmse_train = rmse(y_train,pred_train)
        rmse_test  = rmse(y_test,pred_test)
    
        rec['rmse_train']=rmse_train
        rec['rmse_test'] =rmse_test
        rec['std_train'] = np.std(y_train)
        rec['std_test'] = np.std(y_test)
        rec['std'] = np.std(np.r_[y_train,y_test])
        rec['n_train'] = Ntrain
        rec['n_test'] = Ntest
        rec['n']=N
        rec['pseudoR2']=gam.statistics_['pseudo_r2']['explained_deviance']
        # This isn't a very good metric -- we want to know how well the 
        # new period is predicted, but R here is blind to scale/offset 
        # errors which are very real concerns.
        rec['testR']=np.corrcoef(pred_test,y_test)[0,1]
        # Better metric is normalized RMSE
        rec['nrmse_train']=rmse_train/rec['std_train']
        rec['nrmse_test']=rmse_test/rec['std_test']

        
        print(f"  Formula: {predictor}")
        print(f"    RMSE_train={rmse_train:.2f} RMSE_test={rmse_test:.2f}")
        
        if 1: # Plot GAM fits
            fig, axs = plt.subplots(1,len(gam.terms)-1,num=50+i,clear=1)
            fig.set_size_inches((14,3.5),forward=True)
            
            for i, ax in enumerate(axs):
                if gam.terms[i].istensor:
                    XX = gam.generate_X_grid(term=i,meshgrid=True)
                    Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                    extent=[ XX[1].min(), XX[1].max(), XX[0].min(), XX[0].max()]
                    ax.imshow(Z,aspect='auto',extent=extent,origin='lower')
                    label=" x ".join([pred_vars[t['feature']]
                                      for t in gam.terms[i].info['terms']])
                else:
                    XX = gam.generate_X_grid(term=i)
                    feature=gam.terms[i].info['feature']
                    ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX))
                    ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
                    label=pred_vars[gam.terms[i].info['feature']]
                
                if 1: # include stats for the term:
                    idx = gam.terms.get_coef_indices(i)
                    edof = gam.statistics_['edof_per_coef'][idx].sum()
                    pval=gam.statistics_['p_values'][i]
                    label+=f"\nEDoF={edof:.1f}" 
                    if pval>1e-10:
                        label+=f" p<={pval:.1e}"
                    else:
                        pass # not informative. 
                ax.set_title(label,fontsize=9)
            axs[0].set_ylabel(dep_var)
            txts=[site,
                  f"RMSE train: {rmse_train:.2f}",
                  f"RMSE test: {rmse_test:.2f}",
                  f"std.dev: {np.std(yGood):.2f}",
                  f"pseudo R$^2$: {rec['pseudoR2']:.3f}",
                  f"AIC: {gam.statistics_['AIC']:.3e}",
                  "params:",str(gam_params),
                  ]
            fig.text(0.01, 0.85,"\n".join(txts), va='top',fontsize=10)
            fig.subplots_adjust(right=0.99,left=0.2)
            #fig.savefig(os.path.join(dir_gamfigs,site+'_gam_fits.png')) 

all_fits=pd.DataFrame(recs)

#%%
if 1:
    save=True
    
    if dep_var=='ssc_mgL' and err_var=='ssc_mgL':
        var_label=''
    else:
        var_label=f'-{dep_var}-{err_var}'
    
    plt.figure(20).clf()
    fig,ax=plt.subplots(num=20)
    
    sns.barplot(x='site', y='rmse_test',  data=all_fits, hue='predictor', ci=None)
    plt.setp(ax.get_xticklabels(),rotation=90)
    fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
    ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)  
    save and fig.savefig(os.path.join(dir_gamfigs,f'ssc-compare-formulas{var_label}-rmse.png'),dpi=200)

    plt.figure(21).clf()
    fig,ax=plt.subplots(num=21)
    sns.barplot(x='site', y='pseudoR2',  data=all_fits, hue='predictor', ci=None)
    plt.setp(ax.get_xticklabels(),rotation=90)
    fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
    ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)
    save and fig.savefig(os.path.join(dir_gamfigs,f'ssc-compare-formulas{var_label}-pseudoR2.png'),dpi=200)

    plt.figure(22).clf()
    fig,ax=plt.subplots(num=22)
    sns.barplot(x='site', y='nrmse_test',  data=all_fits, hue='predictor', ci=None)
    plt.setp(ax.get_xticklabels(),rotation=90)
    fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
    ax.axhline(1.0,color='k',lw=0.5,ls='--')
    ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)

    save and fig.savefig(os.path.join(dir_gamfigs,f'ssc-compare-formulas{var_label}-nRMSE.png'),dpi=200)

#%%
# Things to try
#  - focus on Dumbarton, what drives the data? mostly an error related to USGS lf.
#  - allow for fits between stations.
#  - tidal average beforehand.

# check other storm flows:
from stompy.io.local import usgs_nwis

if 0:
    ds_coyote=usgs_nwis.nwis_dataset(11172175,
                                     np.datetime64('1999-02-01'),
                                     np.datetime64('2019-01-01'),
                                     products=[60],cache_dir='cache')
    
    ds_guad= usgs_nwis.nwis_dataset(11169025,
                                    np.datetime64('2002-06-01'),
                                    np.datetime64('2019-01-01'),
                                    products=[60],cache_dir='cache')
    
    Qcoyote=ds_coyote.to_dataframe().resample('H')['stream_flow_mean_daily'].mean()
    Qguad  =ds_guad.to_dataframe().resample('H')['stream_flow_mean_daily'].mean()

#%%

plot_vars=[err_var,'wind_u_local','wind_v_local',
           'u_tide_local','wl','u_rms','delta','usgs_lf']

# Commune with the data a bit
plt.figure(1000).clf()
fig,axs=plt.subplots(len(plot_vars),1,sharex=True,num=1000)

for plot_var,ax in zip(plot_vars,axs):
    ax.plot(src['ts_pst'].values[iGood],
            src[plot_var].values[iGood], label=plot_var )
    ax.set_ylabel(plot_var)

axs[0].plot(t_train,pred_train)
axs[0].plot(t_test,pred_test)

# these are in cfs, but still scale a bit
#axs[6].plot(Qcoyote.index.values,2*Qcoyote,label='coyote')
#axs[6].plot(Qguad.index.values,2*Qguad,label='guad')
#axs[6].legend(loc='upper left')
            
 
# Wind-events seem to have an effect that can last for some days
# spring-neap appears important
# there is an event of sorts Dec 2017, but no apparent driver. 
# maybe stormwater is wrong? 
