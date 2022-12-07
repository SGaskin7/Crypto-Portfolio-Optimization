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
from Series_CVaR.series_utils import *
from scipy import stats
import holidays 
from datetime import datetime
#from math import stats

#General Hyperparameters 
look_back = 60 
look_forward = 30
coin = 'Bitcoin'
num_paths = 10000

def RemoveNonBusinessDays(data : pd.DataFrame) -> pd.DataFrame :
    exclusion_list = ['Saturday', 'Sunday']
    YEARS_LIST = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    mask = (pd.to_datetime(data['Date']).dt.day_name().isin(exclusion_list)) | (pd.to_datetime(data['Date']).dt.date.isin(holidays.NYSE(years = YEARS_LIST).keys()))
    data = data.loc[~mask]
    
    
    return data 


def generate_data():
    
    data_dict = {}
    returns_dict = {}
    prices_dict = {}
    asset_list = ['Bitcoin',
                'BinanceCoin',
                'Ethereum',
                'Cardano',
                'Cosmos',
                'ChainLink',
                'Crypto.com Coin',
                'Dogecoin',
                'EOS',
                'Ethereum',
                'Iota',
                'Litecoin',
                'Monero',
                'NEM',
                'Tether',
                'Solana',
                'Stellar',
                'Tether',
                'Tron',
                'WrappedBitcoin',
                'XRP',
                'SPY']

    
    for files in os.listdir('FactorResearch/backtesting/Series_CVaR/Datasets'):
        
        coin = files.split('.')[0].split('_')[1]
        temp_df = pd.read_csv(f'FactorResearch/backtesting/Series_CVaR/Datasets/{files}')
        if 'SPY' not in coin:
            temp_df = temp_df.drop(temp_df[~temp_df.Name.isin(asset_list)].index)
            if len(temp_df) == 0:
                continue

        data_dict[coin] = pd.read_csv(f'FactorResearch/backtesting/Series_CVaR/Datasets/{files}')
        prices = temp_df[['Date', 'High', 'Low', 'Open', 'Close']]
        prices_dict[coin] = prices
        returns = np.diff(np.log(prices[['High', 'Low', 'Open', 'Close']].to_numpy()), axis = 0)
        df = pd.DataFrame(columns = ['Date','High', 'Low', 'Open', 'Close'], data = np.zeros((len(prices)-1, 5)))
        df.Date = prices["Date"].apply(lambda x : x.split(' ')[0])
        df[['High', 'Low', 'Open', 'Close']] = returns
        returns_dict[coin] = df 
        returns_dict[coin] = RemoveNonBusinessDays(returns_dict[coin])
        
        
    return data_dict, prices_dict, returns_dict


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
        
def last_n_days_market(n : int = 30, returns_dict : dict = None, date : str = None) -> pd.DataFrame :
    
    
    
    df = pd.DataFrame(columns = sorted(returns_dict.keys()), data = np.zeros([n, len(returns_dict.keys())]) )
    
    for key in sorted(returns_dict.keys()):
        
        returns = returns_dict[key].reset_index()
        
        start_index = returns.index[returns['Date'] == date][0]
        df[key] = returns.iloc[(start_index - n) : start_index].Close.to_numpy()

        
    return df
        
        
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
    

def arima_garch(returns: np.array = None, num_paths: int = 100, days: int = 5, garch : bool = True, frac_dif : bool = False, error_dist = 'Normal', d : float = 0.5, threshold : float = 1e-8) -> np.array:
    
    paths = np.zeros(num_paths)
        
    for i in range(1):
        ret_path = np.zeros(days)
        
        if frac_dif:
            diff_returns = fracDiff(returns, d, threshold)
            arima = Auto_Arima(diff_returns, information_criterion='aic')
            residuals = arima.residuals()
            garch = Auto_Garch(residuals, vol = 'FIGARCH', dist = error_dist)
            
        else:
        
            arima = Auto_Arima(returns,
                               seasonal=False, information_criterion='aic')
        
        if garch and not frac_dif:
        
            residuals = arima.residuals()
            garch = Auto_Garch(residuals, dist = error_dist)
 
      
         
        mean, var = arima.n_step_prediction(days)
        error, e_var = garch.n_step_prediction(days)
         
        paths = np.random.normal(loc = mean, scale = var, size = num_paths) + np.random.normal(loc = error, scale = e_var, size = num_paths)
            
    
        
    return np.sort(paths)
            
            
            
            
    
        
        

