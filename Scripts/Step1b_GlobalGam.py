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
import fetch_and_waves as waves
import pickle

import textwrap

#%%

dir_gamfigs="../Figures_GAM_dev"
if not os.path.exists(dir_gamfigs):
    os.makedirs(dir_gamfigs)
    
#%%
from gam_common import data_dir,sites, antecedent, envelope, grid, parse_predictor

g=grid()

#%%

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

recs=[]

train_frac=1.0 # for feeding into station models fit on everything.
split=None

src['log10_ssc_mgL']=np.log10(src['ssc_mgL'].clip(1.0))

year_domain=[src['year'].min(), src['year'].max()]

# Long term estuary wide fit:
predictor="s(year, n_splines=128, constraints=flatends) + s(along) + s(wl_rms, constraints='monotonic_inc')"

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
    
pred_vars,formula = parse_predictor(predictor)
    
all_vars=pred_vars+[dep_var,'ts_pst']    
# check nan in pandas with notnull() to get robust handling of
# non-float types.
df_valid=src[ np.all(src[all_vars].notnull().values,axis=1) ]
N=len(df_valid)
df_train=df_valid

if 1:
    print("Training data:")
    print(df_train.groupby('site')['ts_pst'].apply( lambda grp: [grp.min(),grp.max()]))

Ntrain=len(df_train)
xGood_train = df_train[pred_vars].values
yGood_train = df_train[dep_var].values
t_train=df_train['ts_pst'].values

gam = LinearGAM(formula,**gam_params).fit(xGood_train,yGood_train)

#print(gam.summary())
# calculate RMSE against training and test data, see how that varies with
# LinearGAM parameters

pred_train=gam.predict(xGood_train)
rmse_train=rmse(yGood_train,pred_train)

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
          f"std.dev: {np.std(yGood_train):.2f}",
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
    fig.savefig(os.path.join(dir_gamfigs,'global_gam_fits.png')) 


#%%

# Can the GAM be pickled or similar? Yes

gam.pred_vars=pred_vars
gam.dep_var=dep_var
gam.err_var=err_var

with open('global-model.pkl','wb') as fp:
    pickle.dump(gam,fp)
    
with open('global-model.pkl','rb') as fp:
    gam2=pickle.load(fp)

test1=gam.predict(xGood_train)
test2=gam2.predict(xGood_train)

assert np.allclose(test1,test2)
with open('global-model.txt','wt') as fp:
    fp.write(gam.summary())

          
