# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:48:04 2022

@author: rusty
"""

import glob, os
from stompy.spatial import field
import matplotlib.pyplot as plt
from matplotlib import colors,gridspec
from scipy.ndimage import binary_dilation,generate_binary_structure
import pandas as pd
import numpy as np
import xarray as xr

#%%
from stompy.grid import unstructured_grid
g=unstructured_grid.UnstructuredGrid.read_dfm("../Grid/wy2013c_waqgeom.nc",cleanup=True)

#%%
img_dir="../../Sentinel2"

fns=glob.glob(os.path.join(img_dir,"*TSM.tif"))
fns.sort()

#%%
good_fns=[]

for fn in fns:
    print(f"File: {fn}")
    fld=field.GdalGrid(fn)
    # basic culling
    invalid=(fld.F>100)
    fld.F[invalid]=np.nan
    
    n_valid=np.isfinite(fld.F).sum()
    n_total=fld.F.size
    f_valid=n_valid/float(n_total)
    print(f"Fraction valid: {f_valid:.3f}")
    if f_valid< 0.075: # arbitrary
        continue
    good_fns.append(fn)
    if 0:
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)
        ax.set_adjustable('datalim')
        i=fld.plot(cmap='turbo')
        i.set_clim([0,100])
        g.plot_edges(color='k',alpha=0.3,lw=0.6)
        plt.show()
        plt.pause(1.0)
    del fld

# yields 167 "good" files.
#%% 

# clip to grid, downsample by factor of 8, see where we stand.
poly=g.boundary_polygon()

def expand(F):
    for _ in range(5):
        F=binary_dilation(F)
    return F

def clean_and_downsample(fld,factor=8):
    in_bay_mask=fld.polygon_mask(poly)
    fld.F[~in_bay_mask]=np.nan
    
    inv_mask=field.SimpleGrid(F=expand(fld.F>100),
                              extents=fld.extents)
    fld.F[inv_mask.F]=np.nan
    
    crop=[535000,593000,4139745,4199100]
    fld=fld.crop(crop)
    
    fld=fld.downsample(factor,method='ma_mean')
    fld.F = fld.F.filled(np.nan)
    return fld

#%%

# Is there any metadata?
# Not really.
# But looks like it almost always comes over at 1900
# which is presumably UTC, so 11:00am PST, consistent with
# the design spec of 10:30 local mean solar time

stack_fn='image-stack-16.nc'
if True: # not os.path.exists(stack_fn):
    # Build stack
    recs=[]
    
    for fn in good_fns:
        fld=field.GdalGrid(fn)
        fclean=clean_and_downsample(fld,factor=16)
        date_str=os.path.basename(fn)[:10]
        t_pst=np.datetime64(date_str)+np.timedelta64(11,'h')
        rec=dict(t_pst=t_pst,F=fclean.F,fn=fn)
        recs.append(rec)    
    
    stack=pd.DataFrame(recs)

    # xarray actually a better candidate here.
    ds=xr.Dataset()
    ds['time']=('scene',),stack.t_pst.values
    
    full_stack=np.array( list(stack.F.values) )
    
    ds['tsm']=('scene','row','column'), full_stack
    ds['filename']=('scene',),stack.fn.values
    ds['extents']=('four'),fld.extents
    if os.path.exists(stack_fn):
        os.unlink(stack_fn)
    ds.to_netcdf(stack_fn)
else:
    ds=xr.open_dataset(stack_fn)
    full_stack=ds['tsm'].values
    ds.close()

#%%

if 0:
    plt.figure(1).clf()
    fig,ax=plt.subplots(num=1)
    ax.set_adjustable('datalim')
    i=fld.plot(cmap='turbo')
    i.set_clim([0,100])
    g.plot_edges(color='k',alpha=0.3,lw=0.6)
    plt.show()
    
# 60k pixels, but a pretty solid field has only 18k valid
# pixels.
#%%

# For the existing analysis, it's about 1.8M points per station, for
# something like 10 stations.

# This analysis is 10M points total.

# What's possible with this dataset?

# Try super simple PCA

ravel_dim=full_stack.reshape((167,-1))
n_valid_per_pixel=np.isfinite(ravel_dim).sum(axis=0)
enough_data=n_valid_per_pixel>20

valid_ravel_dim=ravel_dim[:,enough_data]

temporal_mean=np.nanmean(valid_ravel_dim,axis=0)
temporal_std=np.nanstd(valid_ravel_dim,axis=0)

valid_ravel_standard=(valid_ravel_dim-temporal_mean)/temporal_std
valid_standard_ma=np.ma.masked_invalid(valid_ravel_standard)

#%%
# Always a gamble whether the order is correct.
# but in this case, I want the more expensive order.
# This gives C ~ [npixels,npixels]
C=np.ma.cov(valid_standard_ma.T)

# How painful is this going to be? a few minutes?
U,s,V=np.linalg.svd(C.data,full_matrices=False)

#%%

def redim(evec=None,component=None,scale=True):
    if evec is None:
        evec=U[:,component]
    if scale:
        evec=evec*temporal_std
    unravel=np.zeros(ravel_dim.shape[1])
    unravel[enough_data]=evec
    unravel[~enough_data]=np.nan
    restack_dim=unravel.reshape(full_stack.shape[1:])
    
    evec_field=field.SimpleGrid(extents=ds.extents.values,
                                F=restack_dim)
    return evec_field
#%% 
fig_dir="../Figures_RS_PCA/base"

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
def show_field(fld,num,label,title='',symm=True):
    plt.figure(num).clf()
    fig,ax=plt.subplots(num=num)
    fig.set_size_inches((5,5),forward=True)
    if symm:
        kw=dict(cmap='coolwarm',norm=colors.CenteredNorm())
    else:
        kw=dict(cmap='turbo')
    im=fld.plot(ax=ax,interpolation='nearest',**kw)
    ax.text(0.5,0.9,title,transform=ax.transAxes)
    plt.colorbar(im,label=label)
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    fig.tight_layout()
    return fig

mean_ssc=redim(temporal_mean,scale=False)
mean_ssc.F[mean_ssc.F==0.0]=np.nan # cheat on unmasking
fig=show_field(mean_ssc,1,label='(mg/l)',title='Mean TSM',
               symm=False)
fig.savefig(os.path.join(fig_dir,'pca-mean_tsm.png'))

for eig in range(10):
    num=2+eig
    fig=show_field(redim(component=eig),num,label="EOF weight (-)",
                   title=f"EOF {eig}",symm=True)
    fig.savefig(os.path.join(fig_dir,f'pca-eof{eig}-weight.png'))
    
#%%

from statsmodels.multivariate.pca import PCA

#p=PCA(valid_ravel_dim,ncomp=10,missing="fill-em",method='nipals')
#fig_dir="../Figures_RS_PCA/statsmodels00"

p=PCA(valid_ravel_dim,ncomp=10,standardize=False,missing="fill-em",method='nipals')
fig_dir="../Figures_RS_PCA/statsmodels-nostandard"  

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
fig=plt.figure(100)
fig.clf()
fig,ax=plt.subplots(num=100)
p.plot_scree(ax=ax)
fig.savefig(os.path.join(fig_dir,'scree.png'))
    
fig=plt.figure(101)
fig.clf()
fig,ax=plt.subplots(num=101)
p.plot_rsquare(ax=ax)
fig.savefig(os.path.join(fig_dir,'rsquare.png'))

#%%

def fig_component(fld,pc,num,label='weight',title='PCA Component'):
    fig=plt.figure(num)
    fig.clf()
    fig.set_size_inches((5,7),forward=True)
    gs=gridspec.GridSpec(4,1)
    ax=fig.add_subplot(gs[:-1,:])
    kw=dict(cmap='coolwarm',norm=colors.CenteredNorm())
    im=fld.plot(ax=ax,interpolation='nearest',**kw)
    ax.text(0.5,0.9,title,transform=ax.transAxes)
    plt.colorbar(im,label=label)
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    
    ax_t=fig.add_subplot(gs[-1,:])
    ax_t.plot(ds.time.values, pc)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

for comp in range(p.loadings.shape[1]):
    num=200+comp
    fig=fig_component(redim(evec=p.loadings[:,comp]),
                            pc=p.factors[:,comp],
                            num=num,label="EOF weight (-)",
                            title=f"Component {comp}")
    fig.savefig(os.path.join(fig_dir,f'pca-eof{comp}-weight.png'))

#%%
mean_ssc=redim(temporal_mean,scale=False)
# mean_ssc.F[mean_ssc.F==0.0]=np.nan # cheat on unmasking
fig=show_field(mean_ssc,num=220,label='(mg/l)',title='Mean TSM',
               symm=False)
fig.savefig(os.path.join(fig_dir,'mean_tsm.png'))

std_ssc=redim(temporal_std,scale=False)
fig_std=show_field(std_ssc,num=221,label=r'$\sigma$ (mg/l)',
                   title='stddev(TSM)',symm=False)
fig_std.savefig(os.path.join(fig_dir,'stddev_tsm.png'))

#%%

# Demo idea of using PCA to drive extrapolation.

sel_xy=np.array([[588760.8929277118, 4151544.290169862],
 [584418.2143757662, 4156064.2209076015],
 [570947.0482554451, 4169180.8826563354],
 [551626.560003932, 4207290.10260198],
 [571925.1303026669, 4176067.253166216]])

loading_fields=[]
for comp in range(p.loadings.shape[1]):
    loading_fields.append( redim(evec=p.loadings[:,comp]))

Ftotal=np.stack( [f.F for f in loading_fields]).transpose(1,2,0)
pca_fld=field.SimpleGrid(extents=loading_fields[0].extents,F=Ftotal)
station_coords=pca_fld.interpolate(sel_xy,interpolation='nearest')

D_L2s=[]
for station_i in range(len(sel_xy)):
    station_coord=station_coords[station_i]
    
    D=Ftotal - station_coord
    D_L2=np.sqrt((D**2).sum(axis=2))
    D_L2s.append(D_L2)

Dmat=np.stack(D_L2s,axis=2) # rows,cols,stations.
weights=(Dmat.clip(1e-4)**-2)
weights=weights / weights.sum(axis=2)[...,None]

fig=plt.figure(200)
fig.clf()
fig.set_size_inches([9,3.5])
fig,axs=plt.subplots(1,len(sel_xy),num=200)
for i,ax in enumerate(axs):
    fld=field.SimpleGrid(extents=loading_fields[0].extents,
                         F=weights[:,:,i])    
    img=fld.plot(ax=ax,vmin=0,vmax=1,cmap='turbo')
    ax.plot([sel_xy[i,0]],
            [sel_xy[i,1]],
            'k+')
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    
fig.subplots_adjust(top=0.99,bottom=0.15,left=0.01,right=0.99,
                    wspace=0.05)
cax=fig.add_axes([0.4,0.13,0.2,0.03])
plt.colorbar(img,cax=cax,label='Interpolation weight',
             orientation='horizontal')

fig.savefig(os.path.join(fig_dir,f'interp_pca_weights-{len(sel_xy)}stations.png'),dpi=150)