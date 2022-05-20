# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:10:32 2022

Test skill and utility of fitting across stations.

End result is a pygam model saved to 'global-model.pkl'
which can be used to generate an additional correlate for
station models.
@author: rusty
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from stompy import utils, filters, memoize

from pygam import LinearGAM, s, l, te
import pygam
from sklearn.linear_model import LinearRegression
import six
import RH_fetch_and_waves as waves

import textwrap

#%%

data_dir="../DataFit00"

sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']


dir_gamfigs="../Figures_GAM_dev"
if not os.path.exists(dir_gamfigs):
    os.makedirs(dir_gamfigs)
    
#%%
from gam_common import antecedent, envelope, grid, parse_predictor

g=grid()

#%%
six.moves.reload_module(waves)
@memoize.memoize(lru=2)
def src_data(site):
    src = pd.read_csv(os.path.join(data_dir,f"model-inputs-{site}.csv"),
                      parse_dates=['ts_pst'])
    
    # spring-neap indicator:
    src['u_rms']=np.sqrt( filters.lowpass_fir(src['u_tide_local']**2,60) )
    
    # similar, but waterlevel and global. mostly just care that it's global,
    # and I trusty wl a bit more than tdvel
    src['wl_rms']=np.sqrt( filters.lowpass_fir(src['wl']**2, 60 ))
    
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
    
    return src

#%%

dfs=[]

for site in sites:
    print(site)
    dfs.append( src_data(site) )
    
#%%

src=pd.concat(dfs)

    
#%%

def rmse(a,b):
    return np.sqrt(np.mean( (a-b)**2))

#%%

import gam_common
six.moves.reload_module(gam_common)
from gam_common import antecedent, envelope, grid, parse_predictor

def ssc_to_zphotic(ssc):
    # Just use a representative fit with nice values
    # to get the basic idea.
    Kd=0.25*ssc**(2./3)
    return 4.5/Kd
def zphotic_to_ssc(zph):
    Kd=4.5/zph
    return (Kd/0.25)**(3./2)
    
recs=[]

#train_frac=0.8
#split='global'
#split='site'
train_frac=1.0 # for feeding into station models fit on everything.
split=None

# what are we trying to predict:
#dep_var='ssc_mgL'
#err_var='ssc_mgL'
dep_var='log10_ssc_mgL'
err_var='log10_ssc_mgL'

#dep_var='zphotic' 
#err_var='zphotic' # variable to use for error reporting


for v in [dep_var,err_var]:
    if v not in src:
        if v=='zphotic':
            src[v]=ssc_to_zphotic(src['ssc_mgL'])
        elif v=='log10_ssc_mgL':
            src[v]=np.log10(src['ssc_mgL'].clip(1.0))
        else:
            raise Exception("Unknown dependent variable " + v)

year_domain=[src['year'].min(), src['year'].max()]

# Long term estuary wide fit:
predictors=[
    # R2 0.123. RMSE 0.33/0.33
    # "s(year)",
    
    # R2 0.427, RMSE 0.27/0.36
    #"s(year) + s(along)"
    
    # R2=0.467, RMSE 0.26/0.40
    #"te(year,along)"

    # R2=0.492  0.25/0.30    
    #"s(year,n_splines=128) + s(along,n_splines=12)"

    # R2=0.490 0.25/0.31
    # "te(year,doy,n_splines=[22,6]) + s(along,n_splines=12)"

    # R2=0.480 RMSE 0.26 (no train/test)
    # these are all about the same.
    # "s(year,n_splines=128) + f(site_idx)"
    # "s(year,n_splines=128) + f(site_idx, penalties='l2')"
    # "s(year,n_splines=128) + f(site_idx, penalties=None)"
    
    # Factors do not support by=x.
    
    # Other long-term factors? u_rms varies across sites, so
    # there is some ambiguity and the s(along) smooth becomes wonky
    #"s(year,n_splines=128) + s(along) + s(u_rms)"

    # Back to training/testing
    # R2 0.555, RMSE=0.23 / 0.29
    #"s(year,n_splines=128) + s(along) + s(wl_rms)"

    # R2 0.555, RMSE=0.23 / 0.29. Just looks slightly more physical
    # R2 0.555, RMSE=0.23 / 1.0. when split within sites
    "s(year, n_splines=128, constraints=flatends) + s(along) + s(wl_rms, constraints='monotonic_inc')"

    # At least show the year spline over the full period.
    # this helps test RMSE some (0.55), but the ends of the smooth are poorly
    # constrained.
    #f"s(year,n_splines=128,edge_knots={year_domain}) + s(along) + s(wl_rms,constraints='monotonic_inc')"
    
]

def add_site_idx(df):
    # assign the site indexes in same order as along to ease interpretation
    # Do this after selecting the training set in order to have
    # sequential numbers
    site_2_along = df.groupby('site').along.first().to_frame()
    site_order=np.argsort(np.argsort(site_2_along.along.values))
    site_2_along['idx']=site_order
    site_idx=site_2_along.loc[src['site'],'idx'].values
    df['site_idx']=site_idx # np.searchsorted(src['site'].unique(), src['site'])
    
add_site_idx(src)
    
src['wind_u_3h']=antecedent(src['wind_u_local'],5)
src['wind_v_3h']=antecedent(src['wind_v_local'],5)

def split_sites(grp):
    Ngrp=len(grp)
    Ngrp_train=int(train_frac*Ngrp)
    dfs_train.append(grp.iloc[:Ngrp_train])
    dfs_test.append(grp.iloc[Ngrp_train:])
    return Ngrp_train


for predictor in predictors: 
    rec=dict(site=site)
    recs.append(rec)
               
    gam_params={} # dict(n_splines=64)

    # mimic R style formulas to make it easier to test variations:
    rec['predictor']=predictor
    
    pred_vars,formula = parse_predictor(predictor)

    if any([isinstance(t,pygam.terms.FactorTerm) for t in formula]):
        # Could be more smarter and make sure that the same set
        # of factors are present in training and testing. 
        print("Factor term -- use all data for training")
        train_frac=1.0
    else:
        train_frac=0.8

    
    all_vars=pred_vars+[dep_var,'ts_pst']    
    # check nan in pandas with notnull() to get robust handling of
    # non-float types.
    df_valid=src[ np.all(src[all_vars].notnull().values,axis=1) ]
    N=len(df_valid)
    if split=='global':
        Ntrain=int(train_frac*N)
        Ntest =N - Ntrain
        
        df_train = df_valid.iloc[:Ntrain,:]
        df_test  = df_valid.iloc[Ntrain:,:]
                
    elif split=='site': # split train/test per station
        dfs_train = []
        dfs_test  = []
        df_valid.groupby('site').apply(split_sites)
        df_train=pd.concat(dfs_train)
        df_test=pd.concat(dfs_test)
    elif split is None:
        # like split by site, but train on everything.
        dfs_train = []
        dfs_test  = []
        df_valid.groupby('site').apply(split_sites)
        df_train=df_valid
        df_test=pd.concat(dfs_test)
    else:
        raise Exception(f"Bad value for split: {split}")
        
    if 1:
        print("Training data:")
        print(df_train.groupby('site')['ts_pst'].apply( lambda grp: [grp.min(),grp.max()]))
        print()
        print("Test data")
        print(df_test.groupby('site')['ts_pst'].apply( lambda grp: [grp.min(),grp.max()]))
        
    Ntrain=len(df_train)
    Ntest=len(df_test)
    xGood_train = df_train[pred_vars].values
    yGood_train = df_train[dep_var].values
    xGood_test = df_test[pred_vars].values
    yGood_test = df_test[dep_var].values

    yGood=np.concatenate([yGood_train,yGood_test])     
                
    t_train=df_train['ts_pst'].values
    t_test =df_test['ts_pst'].values
    
    if Ntrain<100:
        # In particuar, Mare Island Causeway has bad local tide info
        print(f"Site {site}  variables: {', '.join(pred_vars)}: insufficient data.")
        continue
    gam = LinearGAM(formula,**gam_params).fit(xGood_train,yGood_train)

    #print(gam.summary())
    # calculate RMSE against training and test data, see how that varies with
    # LinearGAM parameters
    pred_train=gam.predict(xGood_train)
    try:
        pred_test =gam.predict(xGood_test)
    except ValueError:
        print("Could not evaluate test set -- maybe a f(site) term?")
        pred_test = None
        
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
    pred_train = to_err_var(pred_train)
    rmse_train = rmse(y_train,pred_train)
    
    if pred_test is None:
        pred_test=np.nan*y_test
    else:           
        y_test =to_err_var(y_test)
        
    pred_test  = to_err_var(pred_test)     
    rmse_test  = rmse(y_test,pred_test)
    pred_test = np.nan*y_test
        
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
        plt.figure(50).clf()
        fig, axs = plt.subplots(1,len(gam.terms)-1,num=50)
        fig.set_size_inches((14,3.5),forward=True)
        axs=np.atleast_1d(axs)

        for i, ax in enumerate(axs):
            if gam.terms[i].istensor:
                XX = gam.generate_X_grid(term=i,meshgrid=True)
                Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                extent=[ XX[1].min(), XX[1].max(), XX[0].min(), XX[0].max()]
                ax.imshow(Z,aspect='auto',extent=extent,origin='lower')
                label=" x ".join([pred_vars[t['feature']]
                                  for t in gam.terms[i].info['terms']])
            else:
                npoints=max(100,5*gam.terms[i].n_coefs)
                XX = gam.generate_X_grid(term=i,n=npoints)
                feature=gam.terms[i].info['feature']
                # expand range to include test and train:
                full_min=min( xGood_train[:,i].min(), xGood_test[:,i].min())
                full_max=max( xGood_train[:,i].max(), xGood_test[:,i].max())
                XX[:,i] = np.linspace(full_min,full_max,XX.shape[0])
                    
                ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX))
                ax.plot(XX[:, feature], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
                label=pred_vars[gam.terms[i].info['feature']]
                if gam.terms[i].by is not None:
                    label+=f" by {pred_vars[gam.terms[i].by]}" 
            
            ax.set_title(label,fontsize=9)
        axs[0].set_ylabel(dep_var)
        txts=[site,
              f"RMSE train: {rmse_train:.2f}",
              f"RMSE test: {rmse_test:.2f}",
              f"std.dev: {np.std(yGood):.2f}",
              f"pseudo R$^2$: {rec['pseudoR2']:.3f}",
              "params:",str(gam_params),
              ]
        txts.append('TERMS')
        term_wrapped= "\n".join( ["\n".join(textwrap.wrap(term.strip(),width=35,
                                                          subsequent_indent='   '))
                         for term in predictor.split('+')])
        txts.append(term_wrapped)
        fig.text(0.01, 0.85,"\n".join(txts), va='top',fontsize=10)
        fig.subplots_adjust(right=0.99,left=0.2)
        #fig.savefig(os.path.join(dir_gamfigs,site+'_gam_fits.png')) 

all_fits=pd.DataFrame(recs)

#%%

# Can the GAM be pickled or similar? Yes
import pickle

gam.pred_vars=pred_vars
gam.dep_var=dep_var
gam.err_var=err_var

with open('global-model.pkl','wb') as fp:
    pickle.dump(gam,fp)
    
with open('global-model.pkl','rb') as fp:
    gam2=pickle.load(fp)

test1=gam.predict(xGood_test)
test2=gam2.predict(xGood_test)

assert np.allclose(test1,test2)

