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

@memoize.memoize(lru=15)
def src_data(site):
    src = pd.read_csv(os.path.join(data_dir,f"model-inputs-{site}.csv"),
                      parse_dates=['ts_pst'])
    
    # spring-neap indicator:
    src['u_rms']=np.sqrt( filters.lowpass_fir(src['u_tide_local']**2,60) )
    # Had been integer year, now decimal years.
    src['year'] = ((src.ts_pst.values - np.datetime64("2000-01-01"))
                   /(np.timedelta64(int(365.2425*86400),'s')))
    src['doy'] = src.ts_pst.dt.dayofyear
    # u_tide_local is positive for flood.
    # hour_tide is 0 at low slack and 6 at high slack
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

# gap-aware tidal average.  test plots at end
def to_dense(t_dense,t_sparse,x_sparse):
    x_dense=np.full(len(t_dense),np.nan)
    idxs=np.searchsorted(t_dense,t_sparse)
    x_dense[idxs] = x_sparse
    return x_dense

def densify_and_lowpass(t_test, pred_test, y_test):
    winsize=60 # 60 hours
    dt=np.timedelta64(1,'h')
    t_dense=np.arange(t_test.min(),t_test.max()+dt,dt)
    
    pred_dense=to_dense(t_dense,t_test,pred_test)
    y_dense=to_dense(t_dense,t_test,y_test)
    pred_dense_lp=filters.lowpass_fir(pred_dense,winsize,nan_weight_threshold=0.75)
    y_dense_lp   =filters.lowpass_fir(y_dense,   winsize,nan_weight_threshold=0.75)
    return t_dense,pred_dense_lp,y_dense_lp

def tavg_metrics(t_test, pred_test, y_test):
    t_dense,pred_lp,y_lp = densify_and_lowpass(t_test,pred_test,y_test)
    valid=np.isfinite(pred_lp*y_lp)
    pred=pred_lp[valid]
    y = y_lp[valid]
    
    rec=dict(tavg_rmse=rmse(pred,y))
    return rec

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

fit_sites=sites
#fit_sites=['San_Mateo_Bridge']

test_frac=0.2
test_chunks=[0,1,2,3,4]

for site_i,site in enumerate(fit_sites): # enumerate(sites): # ['Dumbarton_Bridge']:
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

    predictors = [
        # Useful to compare to:
        # original model
        # Bad at Alviso, kind of bad at Dumbarton. Best at Carquinez.
        "s(wind) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)",
        
        # The model I would expect. Has best R2, but RMSE about
        # the same
        #( "te(tide_hour,wl_rms) "
        #  "+l(glbl_log10_ssc_mgL) "
        #  " + te(tide_hour,wave_stress_ante) "
        # " + te(tide_hour,storm)"
        #),
        
        # slightly better to include flatends for storm
        # USGS_LF is a wash -- some stations better with it,
        # some stations better without it.
        
        # WIND
        # te(tide_hour,wave_stress_ante)
        # te(wind_u_3h,wind_v_3h)
        # te(wind_u_ante,wind_v_ante) -- best, by small margin.
        # te(wind, tide_hour)

        # GLOBAL MODEL
        # "+l(glbl_log10_ssc_mgL) "
        # "+s(glbl_log10_ssc_mgL) "
        # "+te(glbl_log10_ssc_mgL, tide_hour) " -- best by small margin
        # same, but drop te(tide_hour,wl_rms)
        # ignore global model -- worst by large margin.

        # STORM FLOWS
        # s(storm,constraints=['monotonic_inc',flatends])
        # te(storm,tide_hour)
        # te(storm,wl) # substantially better.
        # omit storm
        #
        # Dumbarton is a tricky but important one.
        #  looks like it does have a long response to storm
        #  flows. Maybe lagged a day, and lasting for 20 days?
        #  antecedent and envelope with various parameters did
        # not help

        # SPRING-NEAP
        #  te(tide_hour,wl_rms) - pretty much same as using u_rms and simpler
        #  s(wl_rms) 
        #  te(tide_hour,u_rms) maybe the best, but there's not much difference
        #  s(u_rms)
        #  te(wl_rms,year) - this is the worst choice for Benicia, CM17,
        #    but the best choice for CM9, Dumbarton. Probably has to do with
        #    extrapolating too much over years.

        # YEAR
        #  just using the global trend
        #  s(year,constraints=flatends)
        #     This one is interesting. CM9 and Dumbarton get better.
        #     CM17 gets much worse.
        #     Given the danger of overfitting, drop it.
        
        # DELTA
        #  s(delta,constraints=flatends)
        #  te(delta,year) -- creates winners and losers. 
        #    CM9 and Dumbarton improve, but more sites get worse.
        #  s(delta,constraints='monotonic_inc') -- same or better
        #    than flatends, every where except SMB.
        #  omit - only slightly worse.

        # U TIDE
        # omit
        # s(tdvel) - slightly better than omitting.
        # te(tdvel,year) - generally the same or worse, except CM9 and DMB.
        # te(u_tide_local,year) - very similar, to te(tdvel, year), maybe slightly
        #   worse.
        # s(u_tide_local) -- very similar to s(tdvel), maybe slightly worse
        #    but it's in the noise.
        # s(tdvel,constraints='concave') - worse. forces concave down.
        # s(tdvel,constraints='convex') - same no constraint
        
        # Running best:
        ( "te(tide_hour,wl_rms) "
          " + te(glbl_log10_ssc_mgL, tide_hour) "
          " + te(wind_u_ante,wind_v_ante) "
          " + s(delta,constraints='monotonic_inc') "
          " + te(storm,wl)"
          " + s(tdvel)"
        ),
        
        # Per station year now that we test over multiple chunks.
        ( "te(tide_hour,wl_rms) "
          " + te(glbl_log10_ssc_mgL, tide_hour) "
          " + s(year, constraints=flatends) "
          " + te(wind_u_ante,wind_v_ante) "
          " + s(delta,constraints='monotonic_inc') "
          " + te(storm,wl)"
          " + s(tdvel)"
        ),
        
        # And a more extensive local year fit instead of the
        # global.
        ( "te(tide_hour,wl_rms) "
          " + s(year, n_splines=128, constraints=flatends) "
          " + te(wind_u_ante,wind_v_ante) "
          " + s(delta,constraints='monotonic_inc') "
          " + te(storm,wl)"
          " + s(tdvel)" ),

        # Are we really doing anything? This one to keep
        # us honest.      
        "intercept"
    ]
        

    # Note that u_tide_local is positive-flood
    src['wind_u_3h']=antecedent(src['wind_u_local'],5)
    src['wind_v_3h']=antecedent(src['wind_v_local'],5)

    for predictor in predictors:                    
        gam_params={} #dict(n_splines=8)
    
        pred_vars,formula = parse_predictor(predictor)
              
        all_vars=pred_vars+[dep_var,'ts_pst']    
        # check nan in pandas with notnull() to get robust handling of
        # non-float types.
        df_valid=src[ np.all(src[all_vars].notnull().values,axis=1) ]
        N=len(df_valid)

        for test_chunk in test_chunks:
            rec=dict(site=site)
            recs.append(rec)
            rec['predictor']=predictor
            rec['chunk']=test_chunk
            
            # Select the test vs train datasets
            test_start=int(N*test_frac*test_chunk)
            test_end  =min(N,int(N*test_frac*(1+test_chunk)))
            
            is_test=np.zeros(N,bool)
            is_test[test_start:test_end] = True

            df_train = df_valid.iloc[~is_test,:]
            df_test  = df_valid.iloc[is_test,:]
            
            # Fit the model                            
            Ntrain=len(df_train)
            Ntest=len(df_test)
            xGood_train = df_train[pred_vars].values
            yGood_train = df_train[dep_var].values
            xGood_test = df_test[pred_vars].values
            yGood_test = df_test[dep_var].values    
            
            t_train=df_train['ts_pst'].values
            t_test =df_test['ts_pst'].values 
            
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
            if np.std(pred_test) > 0.0:
                rec['testR']=np.corrcoef(pred_test,y_test)[0,1]
            else:
                rec['testR']=np.nan
            rec['AIC']=gam.statistics_['AIC']
            rec['GCV']=gam.statistics_['GCV']
            
            # Better metric is normalized RMSE
            rec['nrmse_train']=rmse_train/rec['std_train']
            rec['nrmse_test']=rmse_test/rec['std_test']
            
            rec.update( tavg_metrics(t_test, pred_test, y_test))            
            
            def term_label(ti):
                if gam.terms[ti].istensor:
                    return " x ".join([pred_vars[t['feature']]
                                      for t in gam.terms[ti].info['terms']])
                elif gam.terms[ti].isintercept:
                    return "intcpt"
                else:
                    return pred_vars[gam.terms[ti].info['feature']]  
            def term_edof(ti):
                idx = gam.terms.get_coef_indices(ti)
                return gam.statistics_['edof_per_coef'][idx].sum()
            
            for ti,term in enumerate(gam.terms):
                rec[term_label(ti)+" edf"]=term_edof(ti)
            
            print(f"  Formula: {predictor}")
            print(f"    chunk: {test_chunk}")            
            print(f"    RMSE_train={rmse_train:.2f} RMSE_test={rmse_test:.2f}")
            
            if 0: # Plot GAM fits
                fig, axs = plt.subplots(1,len(gam.terms)-1,num=50+site_i,clear=1)
                fig.set_size_inches((14,3.5),forward=True)
                
                for i, ax in enumerate(axs):
                    if gam.terms[i].istensor:
                        XX = gam.generate_X_grid(term=i,meshgrid=True)
                        Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                        extent=[ XX[1].min(), XX[1].max(), XX[0].min(), XX[0].max()]
                        ax.imshow(Z,aspect='auto',extent=extent,origin='lower')
                    else:
                        XX = gam.generate_X_grid(term=i)
                        feature=gam.terms[i].info['feature']
                        ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX))
                        ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
                        
                    label=term_label(i)
                    
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
                      f"std.dev: {np.std(df_valid[dep_var]):.2f}",
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
    
    all_fits['log10AIC']=np.log10(all_fits['AIC'])
    
    for plot_i,plot_var in enumerate(['rmse_test','pseudoR2','nrmse_test','log10AIC',
                                      'tavg_rmse']):
        fig,ax=plt.subplots(num=20+plot_i,clear=1)
        fig.set_size_inches([10.5,4.75],forward=True)

        sns.barplot(x='site', y=plot_var,  data=all_fits, hue='predictor')
        plt.setp(ax.get_xticklabels(),rotation=90)
        fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
        ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)  
        if save:
            fig.savefig(os.path.join(dir_gamfigs,
                                     f'ssc-compare-formulas{var_label}-{plot_var}.png'),
                        dpi=200)


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

plot_vars=[err_var] + pred_vars

# Commune with the data a bit
plt.figure(1000).clf()
fig,axs=plt.subplots(len(plot_vars),1,sharex=True,num=1000)

for plot_var,ax in zip(plot_vars,axs):
    ax.plot(df_valid['ts_pst'],
            df_valid[plot_var], label=plot_var )
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

#%%

if 0: # dev plots for tidal average
    # Gap-aware lowpass 
    t_dense,pred_dense_lp,y_dense_lp=densify_and_lowpass(t_test,pred_test,y_test)   

    # just for plotting
    pred_dense=to_dense(t_dense,t_test,pred_test)
    y_dense=to_dense(t_dense,t_test,y_test)
    
    fig,ax=plt.subplots(1,1,num=1,clear=1,sharex=True)
    ax.set_title(site)
    ax.plot( t_dense, pred_dense, label='Predicted',alpha=0.5)
    ax.plot( t_dense, y_dense, label='Observed',alpha=0.5)
    ax.plot( t_dense, pred_dense_lp, label='Predicted LP')
    ax.plot( t_dense, y_dense_lp, label='Observed LP')
    ax.legend()
    # good range for SMB
    ax.axis((17985.9, 18073.147, 0.590, 2.6637))
    
