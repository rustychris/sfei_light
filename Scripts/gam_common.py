# -*- coding: utf-8 -*-
"""
Created on Fri May  6 04:50:09 2022

@author: rusty
"""
from stompy import filters
import numpy as np
import scipy as sp
from pygam import LinearGAM, s, l, te, f, penalties, intercept

data_dir="../DataFit00"

def grid():
    from stompy.grid import unstructured_grid
    return unstructured_grid.UnstructuredGrid.read_dfm("../Grid/wy2013c_waqgeom.nc",cleanup=True)

# helper method to fabricate a derived predictors
def antecedent(x,winsize):
    if winsize==1: return x
    return filters.lowpass_fir(x,winsize,nan_weight_threshold=0.01,
                               window='boxcar',mode='full')[:-(winsize-1)]

def envelope(x,rise_win=1.0,decay_win=1.0):
    """
    akin to envelope follower.
    Decay response to x. Exponential filter with distinct rates 
    for rise vs. decay.
    Missing data reverts to mean.
    """
    mean=np.nanmean(x)
    x=np.where(np.isnan(x),mean,x)
    
    result=np.zeros(len(x),np.float64)
    y=result[0]=x[0]
    for i in range(1,len(result)):
        delta=x[i]-y
        # if delta is positive, y rises to input over rise_win samples
        # if delta is negative, y falls to input over decay_win samples
        y += (1./rise_win)*delta.clip(0,np.inf) + (1./decay_win)*delta.clip(-np.inf,0)
        result[i] = y
    return result
        

def flatends(n, coef):
    """
    Builds a penalty matrix for P-Splines with continuous features.
    Penalizes ends not being flat

    Parameters
    ----------
    n : int
        number of splines
    coef : array-like
        coefficients of the feature function
    increasing : bool, default: True
        whether to enforce monotic increasing, or decreasing functions
    Returns
    -------
    penalty matrix : sparse csc matrix of shape (n,n)
    """
    if n != len(coef.ravel()):
        raise ValueError('dimension mismatch: expected n equals len(coef), '\
                         'but found n = {}, coef.shape = {}.'\
                         .format(n, coef.shape))

    if n==1:
        # no penalty for constant functions
        return sp.sparse.csc_matrix(0.)
    if n<3:
        # is this possible?
        return sp.sparse.csc_matrix(0.)
        
    C=sp.sparse.dok_matrix((n,n))
    C[0,0]=1
    C[0,2]=-1
    C[2,0]=-1
    C[2,2]=1
    C[n-3,n-3]=1
    C[n-1,n-3]=-1
    C[n-3,n-1]=-1
    C[n-1,n-1]=1

    # I thought slope would just be set by the last two
    # entries, but something about the spline basis requires that 
    # that the last and third to last entries must be equal, and
    # the intervening entry doesn't affect the extrapolated slope.
    # coefr=coef.ravel()
    # diags=np.zeros(len(coefr)-1,np.float64)    
    # diags[0]=1
    # diags[-1]=1
    # mask = sp.sparse.diags(diags)
    
    # derivative = 1
    # D = penalties.sparse_diff(sp.sparse.identity(n).tocsc(), n=derivative) * mask
    # return D.dot(D.T).tocsc()
    return C.tocsc()

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
        def var_idx(self,name):
            if name not in self.pred_vars:
                self.pred_vars.append(name)
            return self.pred_vars.index(name)
        def visit_Call(self,node):
            args=node.args
            new_args=[]
            for arg in args:
                idx=self.var_idx(arg.id)
                # transformed AST:
                new_arg=ast.Constant(value=idx)
                new_args.append(new_arg)
            new_kws=[]
            for kw in node.keywords:
                if kw.arg=='by':
                    # something like f(foo,by=bar)
                    idx=self.var_idx(kw.value.id)
                    new_kws.append(ast.keyword(kw.arg,ast.Constant(value=idx)))
                else:
                    new_kws.append(kw)

            return ast.Call(node.func,
                            new_args, # node.args,
                            new_kws) # node.keywords
    return Visitor().process(predictor)

if 0: # test example
    predictor="s(along,by=wave_stress) + s(wind_spd_ante) + s(tdvel) + s(wl) + s(storm) + s(delta) + s(usgs_lf)"
    pred_vars,formula = parse_predictor(predictor)
    
    print(predictor)
    print(pred_vars)
    print(formula)
    