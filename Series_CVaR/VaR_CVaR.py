#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:33:36 2022

@author: philipwallace
"""

import pandas as pd 
import numpy as np 



import cvxpy as cp


def CVaR_opt(X, alpha, lam=0.,
                short=True, renormalize=True, norm=2, max_weight=1.0, threshold = 1e-2, upper = 0.25):
    #Determine shape of distributed estimates
    n, d = X.shape
    
    #initialize weights and target (loss) variable from the CVaR obj function
    w = cp.Variable(d)
    t = cp.Variable(1)
    #defined the CVaR Obj function
    obj = t + cp.sum(cp.maximum(- X @ w - t, 0.))/(1-alpha)/n
    #optional lamda parameter that was not used
    if lam>0.: obj += lam * cp.norm(w,1)
    #define the optimization problem 
    objective = cp.Minimize(obj)
    #add in vector definiting constraints, being not short and having a diverseification parameter 
    constraints = [cp.norm(w, norm) <= 1.,]
    constraints += [w<=max_weight]
    
    if not short:
        constraints += [w>=0]
    #define problem for solver
    prob = cp.Problem(objective, constraints)
    #run solver
    result = prob.solve(solver='ECOS', verbose=False)
    w = np.array(w.value)

    #RENORMALIZE and enforce shorting constraint
    if np.sum(w**2)>1e-4:
        if lam>0.:
            w[np.abs(w)<np.max(np.abs(w))/1e3] = 0.
        if not short:
            w[w<0.] = 0.
        if renormalize:
            w = w / np.sum(np.abs(w))
    else:
        w = np.zeros(d)

    #enforce minimum ownsership
    for i in range(len(w)):
        if w[i] < threshold:
            w[i] = 0
        else:
            pass
    #normalize
    w = w / np.sum(np.abs(w))
    
    return w
    
    
