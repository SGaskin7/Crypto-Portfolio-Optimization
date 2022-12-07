#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 06:08:12 2022

@author: philipwallace
"""
from Series_CVaR.time_series import * 
from Series_CVaR.VaR_CVaR import *

""""""

"""
Options:
    Simulation Method:
        Historical
        Short-Term Normal 
        Short Term T
        
        ARIMA
        ARIMA-GARCH
        ARFIMA-FIGARCH
        
    Look Back Period 
    Look Forward Period 
    Number of Paths to Generate 

Example params:
"""

look_back = 60 
look_forward = 1 
num_paths = 100

method = 'ARIMA-GARCH'
'''
Historical 
st-Normal
st-T
ARIMA
ARFIMA-FIGARCH
'''
frac_dif_d = 0.5
error_dist = 'Normal'
threshold = 1e-2
frac_dif = False

'''

data, prices, returns = generate_data() to generate data, this utility is from the time series file

generates a returns dict in the correct format for this function, based on the Kaggle data sets

set a for loop to iterate through every date you want in the format of the dataset only

call: paths = simulate(*args) with returns dict from above as the returns_market argument with whichever date you are on and the other params.

call portfolio = CVaR_opt(paths, alpha. *args)

this will generate a portfolio based on the parameters selected 

'''

def simulate(returns_market, date, look_back, look_forward, num_paths, method, error_dist = "Normal", threshold = 1e-2, frac_dif_d = 0.5):
    
    last_n = last_n_days_market(look_back, returns_market, date)
    
    sim_returns = pd.DataFrame(columns = last_n.columns, data = np.zeros((num_paths, len(last_n.columns))))
    
    for name in last_n.columns:
        
        returns = last_n[name].fillna(method = 'ffill').dropna().to_numpy()
    
        if method in set(['Historical', 'Normal', 'studentst']):
            paths = short_term_return(returns, num_paths, look_forward, dist = method)
        
        if method in set(['ARIMA', 'ARIMA-GARCH', 'ARFIMA-FIGARCH']):
             
            if method=='ARIMA':
                garch = False 
            else:
                garch = True
                
            if method == 'ARFIMA-FIGARCH':
                frac_dif = True
            else:
                frac_dif = False 
            
            paths = arima_garch(returns, num_paths, look_forward, garch = garch,frac_dif = frac_dif, error_dist = error_dist, d = frac_dif_d, threshold = threshold)
        sim_returns[name] = paths.T
        
    return sim_returns
            


        
