#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:03:33 2022

@author: philipwallace
"""
import os 
import pandas as pd 
import numpy as np 
import pmdarima as pmd
import arch
import statsmodels
from Series_CVar.series_utils import *
from scipy import stats
import holidays 
from datetime import datetime
#from math import stats

#General Hyperparameters 

# Utility function to remove non trading days from the data frame to make SPY agree with Crypto
def RemoveNonBusinessDays(data : pd.DataFrame) -> pd.DataFrame :
    exclusion_list = ['Saturday', 'Sunday']
    YEARS_LIST = ['2014', '2015', '2016', '2017', '2018', '2019', '20202', '2021', '2022', '2023']
    
    mask = (pd.to_datetime(data['Date']).dt.dat_name().isin(exclusion_list)) | (data['Date'].df.date.isin(holidays.NYSE(years = YEARS_LIST)))
    data = data.loc[~mask]
    
    
    return data 

#Function to read all the return data into data types
def generate_data():
    
    data_dict = {}
    returns_dict = {}
    prices_dict = {}

    
    for files in os.listdir('Datasets'):
        coin = files.split('.')[0].split('_')[1]
        temp_df = pd.read_csv(f'Datasets/{files}')
        data_dict[coin] = pd.read_csv(f'Datasets/{files}')
        prices = temp_df[['Date', 'High', 'Low', 'Open', 'Close']]
        prices_dict[coin] = prices
        returns = np.diff(np.log(prices[['High', 'Low', 'Open', 'Close']].to_numpy()), axis = 0)
        df = pd.DataFrame(columns = ['Date','High', 'Low', 'Open', 'Close'], data = np.zeros((len(prices)-1, 5)))
        df.Date = prices["Date"].apply(lambda x : x.split(' ')[0])
        df[['High', 'Low', 'Open', 'Close']] = returns 
        returns_dict[coin] = df 
        returns_dict[coin] = RemoveNonBusinessDays(returns_dict[coin])
        
        
    return data_dict, prices_dict, returns_dict

#Function to take a series of returns and an index and return the previous n days 
def last_n_days(n : int = 30, returns : np.array = None, index : int = None) -> np.array :
    '''
    if returns == None or index == None:
        print('Needs argument')
        return
    '''
    try:
        return returns[(index - n):index, :]
    
    except:
        print(f'Needs at least {n} days of data')
 
#Function to take a series of returns and a date and return the previous n days of returns
def last_n_days_market(n : int = 30, returns_dict : dict = None, date : str = None) -> pd.DataFrame :
    
    
    
    df = pd.DataFrame(columns = sorted(returns_dict.keys()), data = np.zeros([n, len(returns_dict.keys())]) )
    
    for key in sorted(returns_dict.keys()):
        
        returns = returns_dict[key]
        
        
        start_index = returns.index[returns['Date'] == date][0]
        
        df[key] = returns_dict[key].iloc[start_index - n : start_index].Close.to_numpy()
        
    return df
        

#Utility function to simulate paths for normal and historical
def short_term_return(returns: np.array = None, num_paths: int = 10000, days: int = 30, dist : str = 'Normal') -> np.array:
    '''
    if type(returns) != np.array:
        print('None numpy data makes this function sad :(')
        return None
    '''
    if dist == 'Normal':
        mean = np.mean(returns)
        variance = np.var(returns)
    
        paths = np.zeros(num_paths)
    
        for i in range(days):
            paths += np.random.normal(loc = mean, scale = variance, size = num_paths)
        
        paths = np.sort(paths)
    
        return paths 
    
    if dist == 'Historical':
        
        paths = np.zeros(num_paths)
        
        for i in range(days):
            paths += np.random.choice(returns, size = num_paths)
            
        paths = np.sort(paths)

        return paths   
    
    if dist == "studentst":
        
        paths = np.zeros(num_paths)
        
        dist = stats.t.fit(returns)
        for i in range(days):
            paths += dist.rvs(size = 1000)
        paths = np.sort(paths)
        
        return paths 
    

#Utility function to simulate returns using the ARIMA-GARCH Methodology
def arima_garch(returns: np.array = None, num_paths: int = 100, days: int = 5, garch : bool = True, frac_dif : bool = False, error_dist = 'Normal', d : float = 0.5, threshold : float = 1e-8) -> np.array:
    
    paths = np.zeros(num_paths)
        
    for i in range(1):
        ret_path = np.zeros(days)
        
        #DO NOT USE, NOT APPROPRIATE FOR SHORT TERM LOOKBACK PERIODS
        if frac_dif:
            diff_returns = fracDiff(returns, d, threshold)
            arima = Auto_Arima(diff_returns, information_criterion='aic')
            residuals = arima.residuals()
            garch = Auto_Garch(residuals, vol = 'FIGARCH', dist = error_dist)
            
        #ARIMA GAVE POOR RESULTS DO USE THIS EITHER
        else:
        
            arima = Auto_Arima(returns,
                               seasonal=False, information_criterion='aic')
        
        #FITS THE ARIMA RESIDUALS TO A GARCH MODEL TO SIMULATE THE ERROR TERMS
        if garch and not frac_dif:
        
            residuals = arima.residuals()
            garch = Auto_Garch(residuals, dist = error_dist)
 
      
        #GETS THE ESTIMATION MEANS AND VARIANCE
        mean, var = arima.n_step_prediction(days)
        error, e_var = garch.n_step_prediction(days)
         
        paths = np.random.normal(loc = mean, scale = var, size = num_paths) + np.random.normal(loc = error, scale = e_var, size = num_paths)
            
    
        
    return np.sort(paths)
            
            
            
            
    
        
        

