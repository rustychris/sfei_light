# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:24:27 2022

@author: rusty
"""

import numpy as np
from shapely import geometry
from stompy import utils

# create an along/across channel coordinate system.
thalweg=np.array([[ 588380., 4147386.],[ 586723., 4147059.],
                  [ 584652., 4146406.], [ 582892., 4147168.], [ 580097., 4149563.],
                  [ 578440., 4150651.], [ 576784., 4152066.], [ 574610., 4153590.],
                  [ 573160., 4154897.], [ 570779., 4157292.], [ 566948., 4159469.],
                  [ 564049., 4161972.], [ 560425., 4166871.], [ 559080., 4171116.],
                  [ 558044., 4177104.], [ 555663., 4183200.], [ 553178., 4186574.],
                  [ 551936., 4191037.], [ 549554., 4195065.], [ 548312., 4198875.],
                  [ 549347., 4203229.], [ 552660., 4207801.], [ 556077., 4210632.],
                  [ 560529., 4212156.], [ 564463., 4213135.], [ 569019., 4213026.],
                  [ 571711., 4212373.], [ 572953., 4210632.], [ 574713., 4209978.],
                  [ 576887., 4211067.], [ 580615., 4212373.], [ 583410., 4213135.],
                  [ 587137., 4213135.], [ 590657., 4212700.], [ 595006., 4211502.],
                  [ 598422., 4212373.], [ 600493., 4214006.], [ 602253., 4213788.],
                  [ 604945., 4213026.]])

chan_geom=geometry.LineString(thalweg)

def utm_to_aa(utm,chan_geom=chan_geom):
    """
    map utm coordinates to along-channel, across channel
    """
    pnts = [geometry.Point(xy) for xy in utm]
    alongs=[chan_geom.project(pnt) for pnt in pnts]
    on_axis=np.array([chan_geom.interpolate(along).coords[0]
                      for along in alongs])
    eps=10.0
    on_axis_eps=np.array([chan_geom.interpolate(along+eps).coords[0]
                         for along in alongs])
    tan=on_axis_eps - on_axis
    norm=tan[:,::-1]*np.r_[-1,1]
    norm = norm/ utils.mag(norm)[:,None]
    across=((utm-on_axis)*norm).sum(axis=1)
    return np.c_[alongs,across]