# -*- coding: utf-8 -*-
"""
Created on Fri May  6 04:45:52 2022

@author: rusty
"""
import six
from stompy.grid import unstructured_grid
from stompy import filters
from stompy import utils
import gam_common
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(gam_common)
from gam_common import antecedent, grid, data_dir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.optimize import root_scalar


# Fetch limited wave methods

# Shore Protection Manual is found to be least accurate of several.
# https://www.sciencedirect.com/science/article/abs/pii/037838399290012J
# They reason that friction adjustments and stability adjustments in the
# SPM are not justified. Sverdrup-Munk-B? performs better.

# DWAQ uses a combination of Groen and Dorrestein 1976 for equilibrium
#  wave height, a growth limit from Nelson 1983, and bed shear stress
#  from Soulsby 1997 or Van Rijn 1993.
#  inputs are depth, fetch, U10, gravity, predicts wave height, period,
# and bed stress



#%%


def fetch_precalc(utm,g,degree_res,degree_smooth=30, eta=None,
                  min_fetch=10.0):
    """
    For a given location utm=[x,y], use node-centered bathy
    in UnstructuredGrid g.nodes['depth'] (positive up), to 
    calculate fetch for all directions.
    
    Directions are discretized by degree_res, and smoothed 
    over degree_smooth.
    
    eta: a scalar elevation. If specified, the fetch calculation
      will stop if it encounters a cell with this elevation.
      Otherwise all cells are counted up to the shoreline.
      Note this eta is *not* used to adjust the bed elevation. 
    
    Returns a pandas DataFrame with 
     vecx, vecy: unit vector in the direction away from utm
       note that this is the opposite the relevant wind-direction
       since you want the fetch looking 'upwind'.
     theta_deg: corresponding angle in degrees. Note that this is 
       is in math convention, so CCW from the +x direction.
     mean_z_bed (mean bed elevation along the fetch)
     z_bed (bed elevation at utm)
     fetch (distance from UTM to grid boundary, meters).
    """
    ntheta=int(np.ceil(360/degree_res))
    # center of each direction bin in degrees, math convention. 
    theta_deg=np.linspace(0,360,1+ntheta)[:-1]
    
    vecs=np.c_[np.cos(theta_deg*np.pi/180.0),
              np.sin(theta_deg*np.pi/180.0)]
    
    cell_elev=g.interp_node_to_cell(g.nodes['depth'])
    
    recs=[]
    for theta,vec in zip(theta_deg,vecs):
        rec=dict(vecx=vec[0],vecy=vec[1],theta_deg=theta)
        
        path=g.select_cells_along_ray(x0=utm,vec=vec)
        if path is None:
            # point is outside the grid
            print("Site location is outside grid. No fetch or waves")
            return None            
        elevs=cell_elev[path['c']]
        if eta is not None:
            dry=elevs>eta
            if np.any(dry):
                stop=np.argmax(dry).clip(1) # include at least 1 cell
                path=path[:stop]
                elevs=elevs[:stop]                
            
        rec['mean_z_bed']=elevs.mean()
        rec['z_bed']=elevs[0]
        # possible to end up with 0 fetch, which causes problems later.
        # clip to something small
        rec['fetch']=utils.dist(utm,path['x'][-1]).clip(min_fetch)
        recs.append(rec)
        
    df=pd.DataFrame(recs)
    
    if degree_smooth>degree_res:
        win=int(degree_smooth/degree_res)
        def smooth(x):
            N=len(x)
            triple=np.concatenate([x,x,x])
            triple_lp=filters.lowpass_fir(triple,win)
            return triple_lp[N:2*N]
        df['z_bed']=smooth(df['z_bed'].values)
        df['fetch']=smooth(df['fetch'].values)
    return df

def fetch_limited_waves(utm,g,wind_u,wind_v,eta=0.0,degree_res=2,degree_smooth=30):
    """
    utm: [x,y] location for the calculation.
    g: UnstructuredGrid, assumed to have node elevations in nodes['depth'], assumed positive up.
    wind_u, wind_v: vectors of wind velocity. assumed to be in m/s, adjusted to 10m,
     and giving the direction of the wind velocity (i.e. velocity convention, not
     where the wind is coming from).
    degree_res, degree_smooth: fetch is precalculated at the given resolution and
      smoothed over the given number of degrees.                                            
                                                    
    eta: water level, relative to the same datum as the grid bathy. can be a scalar
      or time series vector. Note that wetting/drying is not accounted for over time.
      Fetch is precalculated based on mean(eta). If eta is given as an array and
      a specific eta[idx] causes some part of the fetch to become dry this 
      (i) will not decrease the fetch and (ii) elevation above the water surface
      will decrease the average depth (but a negative result will be clipped at 0.10).
      
    returns pd.DataFrame with
      wind_u, wind_v
      h: average water depth
      Hsig, Tsig, Lsig: significant wave parameters
      F: fetch in meters
      
    If the location is not in the grid a dataframe will be returned but all
    wave-related values will be zero. That might not be the best.
    
    The formulation here comes the D-WAQ manual.
    """
    # calculate fetch and average depth for each u/v combination.
    wind_u=np.asarray(wind_u)
    wind_v=np.asarray(wind_v)
    df=pd.DataFrame()
    df['wind_u']=wind_u # do this before replacing nan's
    df['wind_v']=wind_v #

    # Note that we want to look "upwind", so flip wind vector
    invalid=np.isnan(wind_u) | np.isnan(wind_v)
    if np.any(invalid):
        # otherwise trips FP exceptions along the way
        wind_u=np.where(invalid,0.001,wind_u)
        wind_v=np.where(invalid,0.001,wind_v)

    precalc=fetch_precalc(utm,g,degree_res=degree_res,degree_smooth=degree_smooth,
                          eta=np.mean(eta))
    if precalc is None:
        # Not sure what the best thing to do is here.
        print("No fetch information available. Will punt with zeros")
        for fld in ['Hsig','Tsig','F','Lsig','Uorb','tau_w']:
            df[fld]=0.0
            df[fld].loc[invalid] = np.nan
        return df
        
    pre_theta=precalc['theta_deg']
            
    thetas=(180./np.pi * np.arctan2(-wind_v,-wind_u))%360
    Ntheta=len(pre_theta)
    idx=(np.round(thetas/degree_res) % Ntheta).astype(np.int32)
    F=precalc['fetch'].values[idx]
    mean_h=(eta - precalc['mean_z_bed'].values[idx]).clip(0.10)
    h=(eta - precalc['z_bed'].values[idx]).clip(0.10)
    
    # avoid div by zero.
    U10=np.sqrt(wind_u**2+wind_v**2)
    calm=U10<0.1
    U10[calm]=0.1
    
    grav=9.81
    hstar=grav*mean_h/U10**2 # DWAQ not clear on h vs mean_h
    Fstar=grav*F/U10**2
    # Follow DWAQ formulations:
    m0=2*np.pi
    m1=0.45
    m2=0.37
    m3=0.763
    m4=0.365
    k0=0.24
    k1=0.015
    k2=0.0345
    k3=0.710
    k4=0.855
    tanh_k3h=np.tanh(k3*hstar**m3)
    Hstar = k0 * tanh_k3h*np.tanh( (k1*Fstar**m1)/tanh_k3h)
    tanh_k4h=np.tanh(k4*hstar**m4)
    Tstar = m0 * tanh_k4h*np.tanh( (k2*Fstar**m2)/tanh_k4h)
    
    H=Hstar * U10**2/grav
    T=Tstar * U10/grav
    
    # Limit H to 0.55*h
    limit=H>0.55*mean_h # or should this be h?
    if np.any(limit):
        # supposed to also limit T to its value when H hit 0.55h.
        # solve for Fstar_limit:
        Hstar_limit = grav*(0.55*mean_h[limit])/U10[limit]**2
        # this is reporting errors -- 
        Fstar_limit = (tanh_k3h/k1 * np.arctanh( Hstar_limit/(k0*tanh_k3h[limit]) ))**(1./m1)    
        Tstar_limit = m0 * tanh_k4h[limit]*np.tanh( (k2*Fstar_limit**m2)/tanh_k4h[limit])   

        print(f"{limit.sum()} of {len(limit)} heights limited")
        H[limit] = 0.55*mean_h[limit]
        T[limit]=(Tstar_limit * U10[limit]/grav)
    else:
        print("No heights limited")
    
    df['h']=h
    df['mean_h']=mean_h    
    df['Hsig']=H
    df['Tsig']=T
    df['F']=F
    
    def calc_T_resid(L,T,h):
        # coth = 1/tanh
        return T-np.sqrt( 2*np.pi/grav * L / np.tanh(2*np.pi*h/L))
    def calc_L(row):
        bracket=[1e-2,100]
        # very low winds will have miniscule waves and the root finder
        # will fail.  just report a small upper bound in those cases.
        bvals=[calc_T_resid(b,row.Tsig,row.mean_h) for b in bracket]
        if bvals[0]*bvals[1] >= 0:
            if bvals[0]<0:
                return bracket[0]
            else:
                raise Exception("Failed to bracket length, but it's not small")
        result=root_scalar(calc_T_resid,args=(row.Tsig,row.mean_h),bracket=bracket)
        if result.converged:
            return result.root
        else:
            raise Exception("Root finder failed to converge")
            return np.nan
    df['Lsig'] = df.apply(calc_L,axis=1)
    
    add_wave_bed_stress(df)
    
    # Replace invalid entries with nan
    for fld in ['Hsig','Tsig','F','Lsig','Uorb','tau_w']:
        df[fld].loc[invalid] = np.nan
        # fetch could be nonzero, but often calm means 0 wind,
        # so direction is not defined and fetch meaningless.
        df[fld].loc[calm] = 0.0    
    
    return df

def add_wave_bed_stress(df,k_s=1e-3):
    """
    Given a DataFrame as computed by fetch_limited_waves, calculate
    the corresponding bed stress.
    k_s: Nikuradse roughness  (m)
    returns the same DataFrame, with fields Uorb and tau_w
    """
    k_s=np.asarray(k_s)
    rho=1000 # kg/m3
    # here use the local depth h, not fetch-averaged mean_h
    # Note that Lsig is wrong in the DWAQ manual, and should be 
    # in the denominator (verified against linear wave theory in 
    # Kundu).                  
    # clip is to avoid overflow in sinh
    Uorb=( (np.pi*df.Hsig.values) 
          / 
          (df.Tsig.values
           *np.sinh( (2*np.pi*df.h.values/df.Lsig.values).clip(0,100))))
    # semi orbital excursion
    A=Uorb*df.Tsig.values/(2*np.pi)
    # RH: just a guess. seems that if the excursion is less than 10% the
    # grainsize, things aren't going to be realistic even at that scale.
    r=(A/k_s).clip(0.1) 
    f_w=0.237 * r**(-0.52) # Soulsby 1997
    tau_w=0.25 * rho * f_w * Uorb**2 
    df['Uorb']=Uorb
    df['tau_w']=tau_w
    return df

#%%

if 0: 
    # Testing / dev
    site="Channel_Marker_09" # fairly high errors now, and highly wind-affected
    
    src = pd.read_csv(os.path.join(data_dir,f"model-inputs-{site}.csv"),
                      parse_dates=['ts_pst'])
        
    src['wind_spd_ante']=antecedent(src['wind_spd_local'],4)
    src['wind_u_ante']=antecedent(src['wind_u_local'],4)
    src['wind_v_ante']=antecedent(src['wind_v_local'],4)
    
    g=grid()
    
    zoom=(539622., 566601., 4200807., 4224239)
    
    plt.figure(1).clf()
    fig,ax=plt.subplots(num=1)
    
    g.contourf_node_values(g.nodes['depth'],np.linspace(-10,3,50),
                           cmap='turbo',extend='both')
    
    utm=np.r_[src['utm_e'].values[0],
              src['utm_n'].values[0]] 
    ax.plot(src['utm_e'].values[:1],
            src['utm_n'].values[:1],
            'go')
    ax.text(utm[0],utm[1],site)
    ax.axis(zoom)
    
    g.plot_cells(mask=[28011],color='r',alpha=0.4)
     
    utm=np.r_[src['utm_e'].values[0],
              src['utm_n'].values[0]] 
    
    df=fetch_limited_waves(utm,g,src['wind_u_local'],src['wind_v_local'],
                           eta=0.75)
    
    if 1: # qualitative look:
        # tau_w looks inverted -- only substantial when wave state is near
        # zero.
        # Uorb likewise seems almost clamped to zero when wind is significant.
        fig,axs=plt.subplots(3,1,sharex=True,num=2,clear=1)
        
        axs[0].plot(src.ts_pst, src['wind_u_local'], label='wind_u')
        axs[0].plot(src.ts_pst, src['wind_v_local'], label='wind_v')
        axs[1].plot(src.ts_pst, df['F']/1000, label='fetch_km')
        axs[1].plot(src.ts_pst, np.sqrt(src.wind_u_local**2 + src.wind_v_local**2),
                    label='Wind mag')
        axs[2].plot(src.ts_pst, df['Hsig'],label='Hsig')
        axs[2].plot(src.ts_pst, df['Lsig'],label='Lsig')
        axs[2].plot(src.ts_pst, df['Tsig'],label='Tsig')
        axs[2].plot(src.ts_pst, 10*df['tau_w'],label='10*tau_w')
        #axs[2].plot(src.ts_pst, 10*df['Uorb'],label='10*Uorb')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[2].legend(loc='upper right')

    if 0:
        zoom=(539622., 566601., 4200807., 4224239)
        
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)
        
        g.contourf_node_values(g.nodes['depth'],np.linspace(-10,3,50),
                               cmap='turbo',extend='both')
        #g.plot_edges(color='k',alpha=0.5,lw=0.4)
        ax.plot([utm[0]], [utm[1]], 'go')
        #g.plot_cells(mask=path['c'],color='r',alpha=0.4)
        #g.plot_cells(mask=cells,color='b',alpha=0.3)
        
        for idx,row in fetch_precalc(utm, g, 2.0,degree_smooth=30).iterrows():
            seg=np.array( [utm,
                           utm+row['fetch']*np.r_[ row['vecx'],row['vecy']]])
            ax.plot(seg[:,0], seg[:,1], color='k',lw=0.7)
            mid=seg.mean(axis=0)
            ax.text(mid[0],mid[1],f"{row['z_bed']:.1f}")
        
        ax.axis('equal')
        ax.axis(zoom)
            



