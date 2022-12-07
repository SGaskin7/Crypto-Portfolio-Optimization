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
                short=True, renormalize=True, norm=2, threshold = 1e-2, upper = 0.25):

    n, d = X.shape
    
    w = cp.Variable(d)
    t = cp.Variable(1)
    
    obj = t + cp.sum(cp.maximum(- X @ w - t, 0.))/(1-alpha)/n
    if lam>0.: obj += lam * cp.norm(w,1)
    objective = cp.Minimize(obj)
    constraints = [cp.norm(w, norm) <= 1.,]
    if not short:
        constraints += [w>=0]
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver='ECOS', verbose=False)
    w = np.array(w.value)

    if np.sum(w**2)>1e-4:
        if lam>0.:
            w[np.abs(w)<np.max(np.abs(w))/1e3] = 0.
        if not short:
            w[w<0.] = 0.
        if renormalize:
            w = w / np.sum(np.abs(w))
    else:
        w = np.zeros(d)

    for i in range(len(w)):
        if w[i] < threshold:
            w[i] = 0
        else:
            pass
    w = w / np.sum(np.abs(w))
    
    return w
    
    
