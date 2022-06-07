"""
Extract spatially variable tidal harmonics for stage and velocity from DFM output**
converted from hydro_harmonics.ipynb

Caveats:
  - map output, at least at a glance, is daily. May be worth hitting the DWAQ output to get greater time resolution.
"""

import xarray as xr
from stompy.grid import unstructured_grid, multi_ugrid
import os
from stompy import utils
import stompy.model.delft.waq_scenario as waq
from stompy import harm_decomp
import six
import numpy as np

# Use DWAQ output to get reasonable time resolution
hydro_run = '/hpcvol2/open_bay/Hydro_model/Full_res/WY2013/wy2013c/DFM_DELWAQ_wy2013c/wy2013c.hyd'
hydro = waq.HydroFiles(hyd_path=hydro_run)

class DepthAverager(object):
    """
    Package up methods and some caching related to extracting depth-averaged
    hydro data from a Hydro object.
    Can extract stage and cell-centered velocity vectors.
    """
    memmap=True # where possible memory map files
    
    def __init__(self,hydro):
        self.hydro = hydro
        self.g=hydro.grid()
        self.prepare()

    def prepare(self):
        # Verify 3D layout and gather static geometry
        self.hydro.infer_2d_elements() # hydro.seg_to_2d_element[segID] = 2D cell ID
        self.n_seg=hydro.n_seg
        self.n_elt=hydro.n_2d_elements
        self.n_layer=1+hydro.seg_k.max()

        # If it's dense output it's easier to sum to 2D.
        assert np.all(self.hydro.seg_to_2d_element == (np.arange(self.n_seg) % self.hydro.n_2d_elements))

        # no existing, direct way to get velocities.
        # what about surface elevation?
        self.plan_areas = self.hydro.planform_areas().data

        bottom_depth = self.hydro.bottom_depths().data

        # Make sure those don't vary by layer
        assert np.all(np.diff( self.plan_areas.reshape([self.n_layer,self.n_elt]),axis=0)==0.0)
        assert np.all(np.diff( bottom_depth.reshape([self.n_layer,self.n_elt]),axis=0)==0.0)

        self.z_bed=bottom_depth[:self.n_elt]
        self.area=self.plan_areas[:self.n_elt]

        self.prepare_velocity()
    def prepare_velocity(self):
        # And matrix for velocity:        
        self.hydro.infer_2d_links()

        # with the domain merging, links are not nicely organized, so we can't
        # just reshape and sum.
        # at least make sure we don't have to worry about sign
        for idx,grp in utils.enumerate_groups(self.hydro.exch_to_2d_link['link']):
            sgn=self.hydro.exch_to_2d_link['sgn'][grp]
            assert np.all(sgn[0]==sgn)
        M=self.g.interp_perot_matrix()
        L2E=self.hydro.flowlink_to_edge(self.g)
        self.Mlink2vel=M.tocsr().dot(L2E) # now M can operate directly on flow link values.
        
    def calc_stage(self,t_sec,min_depth=0.01):
        """
        Calculate stage across all water columns (elements) for a given
        time in seconds. 
        min_depth: when depth is below this threshold stage is set to nan.
        """
        vol3d = self.hydro.volumes(t_sec,memmap=self.memmap)
        vol2d = vol3d.reshape([self.n_layer,self.hydro.n_2d_elements]).sum(axis=0)
        depth=vol2d/self.area
        eta=self.z_bed + depth
        eta[depth<min_depth]=np.nan
        return eta
    
    # Now for velocity
    def calc_velocity(self,t_sec):
        exchA=self.hydro.areas(t_sec,memmap=self.memmap)
        flow =self.hydro.flows(t_sec,memmap=self.memmap)

        linkA=np.bincount(hydro.exch_to_2d_link['link'],weights=exchA[:hydro.n_exch_x])
        linkQ=np.bincount(hydro.exch_to_2d_link['link'],weights=flow[:hydro.n_exch_x])
        linkU=linkQ/np.where(linkQ==0,1,linkA) # avoid divzero

        cellU=self.Mlink2vel.dot(linkU).reshape([-1,2])
        return cellU


davg=DepthAverager(hydro)

g=hydro.grid()

# Ntime=len(hydro.t_secs)
# Ntime=min(Ntime,48*60) # 2 months
# t_secs=hydro.t_secs[:Ntime]
# 
# # All the data for a single cell:
# c_time=np.datetime64(hydro.time0) + t_secs*np.timedelta64(1,'s')
# c_h=np.zeros(Ntime,np.float64)
# c_U=np.zeros((Ntime,2),np.float64)
# 
# # With memmap, about 29/5 seconds, or 200/5 seconds if data is already 
# # in RAM.
# for ti,t_sec in utils.progress(enumerate(t_secs)):
#     stage=davg.calc_stage(t_sec)
#     vel = davg.calc_velocity(t_sec)
#     c_h[ti]=stage[c]
#     c_U[ti,:]=vel[c,:]
#     
# # unix epoch for reference
# t_ref=np.datetime64('1970-01-01 00:00')
# t=(c_time-t_ref)/np.timedelta64(1,'s')
# 
# h_comps,omegas=harm_decomp.decompose(t, c_h, omegas='select')
# u_comps=harm_decomp.decompose(t, c_U[:,0],omegas=omegas)
# v_comps=harm_decomp.decompose(t, c_U[:,1],omegas=omegas)
# 
# h_pred=harm_decomp.recompose(t,h_comps, omegas)
# u_pred=harm_decomp.recompose(t,u_comps, omegas)
# v_pred=harm_decomp.recompose(t,v_comps, omegas)




# Streaming version
ds=xr.Dataset()
Ntime=len(hydro.t_secs)
t_secs=hydro.t_secs[:Ntime]
t_ref=np.datetime64('1970-01-01 00:00')
ds['t_ref']=(),t_ref

c_time=np.datetime64(hydro.time0) + t_secs*np.timedelta64(1,'s')
# unix epoch for reference
t=(c_time-t_ref)/np.timedelta64(1,'s')

omegas=harm_decomp.select_omegas(t.max() - t.min())
omegas=np.r_[omegas, 0.0] # DC offset

ds['omegas']=('component',), omegas
ds['omegas'].attrs['units'] = 'rad s-1'

print(f"Will be fitting with {len(omegas)} components")
Ainv = harm_decomp.getAinv(t,omegas) # 2*ncomps, ntimes

h_comps_strm = np.zeros( (g.Ncells(),2*len(omegas)), np.float64)
u_comps_strm = np.zeros( (g.Ncells(),2*len(omegas)), np.float64)
v_comps_strm = np.zeros( (g.Ncells(),2*len(omegas)), np.float64)

# With memmap, about 29/5 seconds, or 500/5 seconds if data is already 
# in RAM.
# slows to 100/5 seconds with Ainv product over 50k cells.
for ti,t_sec in utils.progress(enumerate(t_secs)):
    stage=davg.calc_stage(t_sec)
    vel = davg.calc_velocity(t_sec)
    # BUG: check for drying.
    
    # [Nc,2*omegas]  ~  (2*omegas,1) * Nc
    h_comps_strm[:] += (Ainv[:,ti][None,:] * stage[:,None])
    u_comps_strm[:] += (Ainv[:,ti][None,:] * vel[:,0][:,None])
    v_comps_strm[:] += (Ainv[:,ti][None,:] * vel[:,1][:,None])


for quant,comps in [('stage',h_comps_strm),
                    ('u',u_comps_strm),
                    ('v',v_comps_strm)]:
    comps=comps.reshape( (g.Ncells(),len(omegas),2))
    combined=np.zeros_like(comps) # [Nc, omega, {amp,phase} ]
    combined[:,:,0] = np.sqrt( comps[:,:,0]**2 + comps[:,:,1]**2 )
    combined[:,:,1] = np.arctan2( comps[:,:,1], comps[:,:,0] )
        
    ds[quant+"_harmonics"] = ('face','component','amp_phase'), combined

ds['component']=('component',),harm_decomp.omegas_to_names( ds.omegas.values )
ds['amp_phase']=('amp_phase',),['amp','phase']
g.write_to_xarray(ds) # embed grid

out_dir="../Data_HydroHarmonics"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
ds.to_netcdf(os.path.join(out_dir,'harmonics-wy2013.nc'))


if 0: # verify the streaming, full grid version:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    h_pred_strm_full=harm_decomp.recompose(t,ds.stage_harmonics.isel(face=c).values,ds.omegas.values)

    t_np=utils.unix_to_dt64(t)

    fig,axs=plt.subplots(2,1,sharex=True)

    axs[0].plot(t_np,c_h,label='data')
    axs[0].plot(t_np,h_pred,label='orig decomp')
    axs[0].plot(t_np,h_pred_strm_full,label='streaming full-grid decomp',ls='--')


    axs[1].plot(t_np,c_U[:,0],label='U data')
    axs[1].plot(t_np,c_U[:,1],label='V data')

    axs[1].plot(t_np,u_pred,label='U orig decomp')
    axs[1].plot(t_np,v_pred,label='V orig decomp')

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')

    fig.autofmt_xdate()


    # And check on spatial variability:
    comp='M2'

    fig,axs=plt.subplots(1,2,figsize=(11,5.5))
    plt.setp(axs,adjustable='datalim')

    M2_amp=ds['stage_harmonics'].sel(component=comp,amp_phase='amp').values
    M2_phi=ds['stage_harmonics'].sel(component=comp,amp_phase='phase').values
    collA=g.plot_cells(values=M2_amp,ax=axs[0],cmap='turbo')
    collP=g.plot_cells(values=M2_phi,ax=axs[1],cmap='hsv')
    plt.colorbar(collA,ax=axs[0],label=f'{comp} ampl.',orientation='horizontal')
    plt.colorbar(collP,ax=axs[1],label=f'{comp} phase',orientation='horizontal')

    for ax in axs:
        ax.axis('off')
    fig.tight_layout()

    # not really setup for plotting from script -- go back to notebook,
    # save plots, or run interactively.
    
