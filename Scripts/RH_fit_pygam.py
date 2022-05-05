# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:10:32 2022

Test skill of original python-based GAM
@author: rusty
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from stompy import utils, filters, memoize

from pygam import LinearGAM, s, l, te
from sklearn.linear_model import LinearRegression
#%%

data_dir="../DataFit00"

sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']


dir_gamfigs="../Figures_GAM_dev"
if not os.path.exists(dir_gamfigs):
    os.makedirs(dir_gamfigs)
    
#%%

@memoize.memoize(lru=2)
def src_data(site):
    src = pd.read_csv(os.path.join(data_dir,f"model-inputs-{site}.csv"),
                      parse_dates=['ts_pst'])
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

# helper methods to fabricate a derived predictors
def antecedent(x,winsize):
    if winsize==1: return x
    return filters.lowpass_fir(x,winsize,nan_weight_threshold=0.01,
                               window='boxcar',mode='full')[:-(winsize-1)]

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

def parse_predictor(predictor):
    """
    geek way to turn mgcv-like formula:
    's(wind_spd_ante) + s(tdvel) + s(wl) + s(storm) + s(delta) 
     + s(usgs_lf) + te(delta,storm) + s(wl,nsplines=5)'
    into a list of variables and numbered terms suitable for pygam.
    ['wind_spd_ante', 'tdvel', 'wl', 'storm', 'delta', 'usgs_lf']
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + te(4, 3) + s(2, nsplines=5)
    and returns the actual terms expression.
    """
    import ast
    class Visitor(ast.NodeTransformer):
        def process(self,predictor):
            node=ast.parse(predictor)
            self.pred_vars=[]
            self.xformed=self.visit(node.body[0])
            # string, that could probably just be eval'd
            self.formula=ast.unparse(self.xformed)
            self.terms = eval(self.formula)
            return self.pred_vars,self.terms
        def visit_Call(self,node):
            args=node.args
            new_args=[]
            for arg in args:
                var_name=arg.id
                if var_name not in self.pred_vars:
                    self.pred_vars.append(var_name)
                idx=self.pred_vars.index(var_name)
                # transformed AST:
                new_arg=ast.Constant(value=idx)
                new_args.append(new_arg)                
            return ast.Call(node.func,
                            new_args, # node.args,
                            node.keywords)
    return Visitor().process(predictor)

if 0: # test example
    predictor="s(wind_spd_ante) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)"
    pred_vars,formula = parse_predictor(predictor)
    
    print(predictor)
    print(pred_vars)
    print(formula)
    
#%%
sites = ['Alcatraz_Island','Alviso_Slough','Benicia_Bridge','Carquinez_Bridge','Channel_Marker_01',
         'Channel_Marker_09','Channel_Marker_17','Corte_Madera_Creek','Dumbarton_Bridge',
         'Mallard_Island','Mare_Island_Causeway','Point_San_Pablo','Richmond_Bridge','San_Mateo_Bridge']


recs=[]

for site in sites:
    print(f"Site: {site}")
    src=src_data(site)
    
    # spring-neap indicator:
    src['u_rms']=np.sqrt( filters.lowpass_fir(src['u_tide_local']**2,60) )
    src['year'] = src.ts_pst.dt.year - 2000
    src['doy'] = src.ts_pst.dt.dayofyear
    src['tide_hour'] = utils.hour_tide(utils.to_dnum(src.ts_pst.values),
                                       u=src['u_tide_local'])
    src['wind_spd_ante']=antecedent(src['wind_spd_local'],30)
    
    
    predictors=["s(wind) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)",
                
                "s(wind_spd_ante) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)",
                
                # never helps the test skill.
                #("s(wind_spd_ante) + s(u_tide_local) + s(h_tide_local) "
                #"  + s(delta) + s(usgs_lf) + te(tide_hour,u_rms) + s(doy)"),
                
                ("s(wind_spd_ante) + s(u_tide_local) + s(h_tide_local) "
                 "  + s(delta) + s(usgs_lf) + s(tide_hour) + s(u_rms) + s(doy)"),
                
                ("s(wind_spd_ante) + s(u_tide_local) + s(h_tide_local) + s(storm)"
                 "  + s(delta) + s(usgs_lf) + s(tide_hour) + s(u_rms) + s(doy)")
                ]
                
                
    for predictor in predictors: 
        rec=dict(site=site)
        recs.append(rec)
                   
        gam_params=dict(n_splines=8)
    
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
        Y = src['ssc_mgL']
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
        gam = LinearGAM(formula,**gam_params).fit(xGood_train,yGood_train)
    
        #print(gam.summary())
        # calculate RMSE against training and test data, see how that varies with
        # LinearGAM parameters
        pred_train=gam.predict(xGood_train)
        pred_test =gam.predict(xGood_test)
        rmse_train = rmse(yGood_train,pred_train)
        rmse_test  = rmse(yGood_test,pred_test)
    
        rec['rmse_train']=rmse_train
        rec['rmse_test'] =rmse_test
        rec['std_train'] = np.std(yGood_train)
        rec['std_test'] = np.std(yGood_test)
        rec['std'] = np.std(yGood)
        rec['n_train'] = Ntrain
        rec['n_test'] = Ntest
        rec['n']=N
        rec['pseudoR2']=gam.statistics_['pseudo_r2']['explained_deviance']
        rec['testR']=np.corrcoef(pred_test,yGood_test)[0,1]
        

        print(f"  Formula: {predictor}")
        print(f"    RMSE_train={rmse_train:.2f} RMSE_test={rmse_test:.2f}")
        
        if 0: # Plot GAM fits
            plt.figure(50).clf()
            fig, axs = plt.subplots(1,len(gam.terms)-1,num=50)
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
                
                ax.set_title(label,fontsize=9)
            axs[0].set_ylabel('SSC (mg/L)')
            txts=[site,
                  f"RMSE train: {rmse_train:.2f}",
                  f"RMSE test: {rmse_test:.2f}",
                  f"std.dev: {np.std(yGood):.2f}",
                  "params:",str(gam_params),
                  ]
            fig.text(0.01, 0.85,"\n".join(txts), va='top',fontsize=10)
            fig.subplots_adjust(right=0.99,left=0.2)
            #fig.savefig(os.path.join(dir_gamfigs,site+'_gam_fits.png')) 

all_fits=pd.DataFrame(recs)

#%%

import seaborn as sns

plt.figure(20).clf()
fig,ax=plt.subplots(num=20)

sns.barplot(x='site', y='rmse_test',  data=all_fits, hue='predictor', ci=None)
plt.setp(ax.get_xticklabels(),rotation=90)
fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)

fig.savefig(os.path.join(dir_gamfigs,'ssc-compare-formulas-rmse.png'),dpi=200)


plt.figure(21).clf()
fig,ax=plt.subplots(num=21)
all_fits['testR2']=all_fits['testR']**2 * np.sign(all_fits['testR'])
sns.barplot(x='site', y='testR2',  data=all_fits, hue='predictor', ci=None)
plt.setp(ax.get_xticklabels(),rotation=90)
fig.subplots_adjust(bottom=0.5,top=0.98,right=0.97)
ax.legend(loc='upper left',bbox_to_anchor=[-0.15,-0.65],fontsize=7.5)

fig.savefig(os.path.join(dir_gamfigs,'ssc-compare-formulas-testR.png'),dpi=200)


#%%
# Things to try
#  - antecedent wind vector
#  - focus on low SSC data points.  maybe inverse? or fit to 4.5/Kd?
#  - antecedent tidal velocity, maybe squared or cubed.
#  - allow for fits between stations.

plot_vars=['ssc_mgL','wind_u_local','wind_v_local',
           'u_tide_local','wl','u_rms','delta','usgs_lf']

# Commune with the data a bit
plt.figure(1000).clf()
fig,axs=plt.subplots(len(plot_vars),1,sharex=True,num=1000)

for plot_var,ax in zip(plot_vars,axs):
    ax.plot(src['ts_pst'].values[iGood],
            src[plot_var].values[iGood] );
    ax.set_ylabel(plot_var)

axs[0].plot(t_train,pred_train)
axs[0].plot(t_test,pred_test)

 
# Wind-events seem to have an effect that can last for some days
# spring-neap appears important
# there is an event of sorts Dec 2017, but no apparent driver. 
# maybe stormwater is wrong? 
