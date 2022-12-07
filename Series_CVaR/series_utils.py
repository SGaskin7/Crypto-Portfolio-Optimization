#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:21:55 2022

@author: philipwallace
"""
import pmdarima as pmd
import statsmodels 
import pandas as pd 
import numpy as np 
from scipy import stats 
import arch
from arch.__future__ import reindexing


#TIME SERIES CLASS FOR FITTING
class Auto_Arima():
    def __init__(self, df,
                 seasonal=False, information_criterion='aic'):
        self.returns = df
        self.seasonal = seasonal
        self.information_criterion = information_criterion
        self.model = pmd.auto_arima(
            self.returns,
            seasonal=self.seasonal,
            information_criterion=self.information_criterion,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            scoring='mse'
        )
        
    #GET MODEL RESIDUALS
    def residuals(self):
        
        return self.model.arima_res_.resid
    
    #PREDICT ONE STEP IN THE FUTURE AND RETURN A SIMULATED VALUE 
    def one_step_prediction(self):
        predicted, conf_int = self.model.predict(n_periods=1, return_conf_int=True, alpha = 0.05)
        
        var = self.get_var(predicted.tolist()[0], np.asarray(conf_int).tolist(), 0.05)
        
        sim = np.random.normal(loc=predicted.tolist()[0], scale = var)
        
        return sim
            
    #ADD THE MOST RECENT SIMULATION TO THE SERIES AND REFIT
    def update(self, x):
        self.model.update(x)
        return self.model.arima_res_.resid[-1]
    
    #GET THE VARIANCE OF THE PREDICITON BASED ON CONFIDENCE INTERVAL
    def get_var(self, mu, conf_int, alpha = 0.05):
        z_alpha = stats.norm.ppf(1-alpha)
        var = ((conf_int[0][1] - mu)/z_alpha)**2
        return var 
    #PREDICT VALUES AND VARINACE N DAYS IN THE FUTURE
    
    def n_step_prediction(self, days):
        predicted, conf_int = self.model.predict(n_periods=days , return_conf_int=True, alpha = 0.05)
        
        mean = np.sum(predicted)
        
        conf = days*conf_int[0][1]
        
        var = self.get_var2(mean, conf, 0.05)
        
        return predicted, var
    #DIFFERENT VERSION OF FINDING THE VARIANCE
    def get_var2(self, mu, conf, alpha = 0.05):
        z_alpha = stats.norm.ppf(1-alpha)
        var = ((conf - mu)/z_alpha)**2
        return var 
        
        
 

'''
can include other regressors if needed 

vol options:
    GARCH
    FIGARCH
    
dist_options for how the error terms are distributed:
    normal
    gaussian 
    t
    studentst
    skewstudent
    skewl
'''
#class to fit a garch model and interact with
class Auto_Garch():
    def __init__(self, residuals, p=1, o =0, q=1, vol = 'GARCH', dist = 'Normal'): #can also set the dist to normal
        self.resids = residuals
        self.p = p
        self.o = o
        self.q = q
        self.power = 2.0
        self.vol = vol 
        self.dist = dist
        self.model = arch.arch_model(residuals,
                                p= self.p,q= self.q,o= self.o,
                                power = self.power, 
                                dist = self.dist,
                                vol = self.vol, 
                                rescale=False
                                )
        
        self.fit_model = self.model.fit(disp='off', show_warning=False)
        
    #predict one step in the future and retun a simulated value    
    def one_step_prediction(self):
        
        model_fit = self.model.fit(update_freq=1,disp ='off', show_warning = False)
        pred = model_fit.forecast(horizon = 1)
        mean = pred.mean['h.1'].iloc[-1]
        variance = pred.variance['h.1'].iloc[-1]
        
        if self.dist == 'Normal':
            sim = np.random.normal(loc = mean, scale = variance)
        
        return sim 
    
    #update and refit the model based on a new observation
    
    def update(self, x):
        np.append(self.resids, x)
        self.model = arch.arch_model(self.resids,
                                p= self.p,q= self.q,o= self.o,
                                power = self.power, 
                                dist = self.dist,
                                vol = self.vol,
                                rescale = False)
        return None
    
    #predict and sim n steps in the future
    def n_step_prediction(self, days):
        model_fit = self.model.fit(update_freq=1,disp ='off', show_warning = False)
        pred = model_fit.forecast(horizon = days)
        mean = days*pred.mean[f'h.{days}'].iloc[-1]
        variance =(days**2)*pred.variance[f'h.{days}'].iloc[-1]
        
        return mean, variance 
        
        
#Function used for fraction differncing...do not use it is not appropritae for short term series 
#will throw an error becuase of return lengths         
def fracDiff(series, d, threshold = 1e-5):
    #compute weights using function above
    weights = findWeights_FFD(d, len(series), threshold)
    width = len(weights) - 1
    
    df = {}
    #for each series to be differenced, apply weights to appropriate prices and save 
    for name in series.columns:
        
        #forward fill through unavailable prices and create a temporary series to hold values
        curr_series = series[name]
        df_temp = np.zeros((curr_series.shape[0]-width))
        
        #loop through all values that fall into range to be fractionally differenced
        for iloc1 in range(width, curr_series.shape[0]):
            
            #set values for first and last time-series point to be used in current pass of fractional
                #difference
            loc0 = curr_series.index[iloc1-width]
            loc1 = curr_series.index[iloc1]
            
            #make sure current value is valid
            if not np.isfinite(curr_series.loc[loc1]):
                continue
            
            #dot product of weights with values from first and last indices
            df_temp[loc1]= np.dot(weights.T, curr_series.loc[loc0:loc1])[0] #0,0
            
        df[name] = df_temp.copy(deep=True)
    df = pd.concat(df, axis=1)
            
    return df

#utility from above, not used
def findWeights_FFD(d, length, threshold):
    #set first weight to be 1 and k to be 1
    w, k = [1.], 1
    w_curr = 1
    
    #while we still have more weights to process, do the following:
    while(k < length):
        
        w_curr = (-w[-1]*(d-k+1))/k
        
        #if the current weight is below threshold, exit loop
        if(abs(w_curr) <= threshold):
            
            break
            
        #append coefficient to list if it passes above threshold condition
        w.append(w_curr)
        
        #increment k
        k += 1
        
    #make sure to convert it into a numpy array and reshape from a single row to a single
    #column so they can be applied to time-series values easier
    w = np.array(w[::-1]).reshape(-1,1)
    
    return w
        
        
       
