from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import holidays
from sklearn.model_selection import train_test_split
import pandas as pd

start_date, end_date = '2020-10-07', '2021-07-05'
TEST_DATASET_SIZE = 0.01
YEARS_LIST = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
# CVAR
# alpha = 0.05

asset_list = ['Bitcoin',
            'Binance Coin',
            'Ethereum',
            'Cardano',
            'Cosmos',
            'ChainLink',
            'CryptocomCoin',
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

equity_factors_list = ['Mkt-RF', 'SMB', 'HML']

# Adding momentum factor here
# Look at residuals after regression
# Panic copula
crypto_factors_list = {
    'big_vs_small': {
        'var_name': 'Marketcap',
        'need_compute': True
    },
    'vol_high_vs_low': {
        'var_name': 'Volume',
        'need_compute': True
    },
    'price_high_vs_low': {
        'var_name': 'Close',
        'need_compute': True
    },
    'daily_variation': {
        'var_name': 'DailyVariation',
        'need_compute': True
    },
    'crypto_mkt': {
        'need_compute': False
    },
    'momentum': {
        'need_compute': False
    }
}

def GetStartEnd(date, lookback):
    # start_date = Date-lookback
    start_date = datetime.combine((datetime.fromisoformat(date) - relativedelta(months=lookback)).date(), datetime.max.time())

    # end_date = date+1
    end_date = datetime.combine((datetime.fromisoformat(date) + timedelta(days=1)).date(), datetime.max.time())

    return start_date, end_date

def RemoveNonBusinessDays(data):
    # Remove holidays and weekends
    exclusion_list = ['Saturday', 'Sunday']
    mask = (pd.to_datetime(data['Date']).dt.day_name().isin(exclusion_list)) | (data['Date'].dt.date.isin(holidays.NYSE(years=YEARS_LIST)))
    data = data.loc[~mask]

    return data


def PriceMAPrediction(data, date, lookback):
    start_date, end_date = GetStartEnd(date, lookback)
    data = RemoveNonBusinessDays(data)

    # Train model from date-lookback to date+1
    mask = (data['Date'] > start_date) & (data['Date'] <= end_date)
    data_recent = data.loc[mask]

    # Store Predictions
    ret_pred = {}

    symbols = list(data.Symbol.unique())
    for sym in symbols:
        data_temp = data_recent.loc[data_recent['Symbol'] == sym]
        data_temp = data_temp.drop_duplicates()
        
        if len(data_temp)<=0:
            continue
        
        data_temp = data_temp.sort_values('Date')
        
        # Incl Columns
        data_temp = data_temp[['Date', 'Symbol', 'PctReturn']]

        # calc moving avg of last 7 days
        data_temp['PriceMA'] = data_temp['PctReturn'].rolling(window=7).mean().shift(1)

        data_temp = data_temp.dropna()
        import time

        # Last 30 day moving average return regression fit
        X_last30 = data_temp['PriceMA'].to_numpy().reshape(-1, 1)
        Y = data_temp['PctReturn'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_last30, Y, test_size=TEST_DATASET_SIZE, random_state=42)
        reg = LinearRegression().fit(X_train, y_train)

        ret_pred[sym] = reg
    
    return ret_pred

    # dates_by_sym = data_recent.groupby('Symbol')['Date'].unique()

    # d_list = None
    # incl_sym = []  
    # for sym, d in dates_by_sym.iteritems():
    #     print(d_list)
    #     if d_list.all():
    #         if d==d_list:
    #             incl_sym.append(sym)
    #     else:
    #         incl_sym.append(sym)
    #         d_list = d
    # print(incl_sym)

def FactorModelPrediction(data, date, lookback, crypto_factors):
    start_date, end_date = GetStartEnd(str(date), lookback)
    data = RemoveNonBusinessDays(data)

    # Train model from date-lookback to date+1
    mask = (data['Date'] > start_date) & (data['Date'] <= end_date)
    data_recent = data.loc[mask]

    # Store Predictions
    ret_pred = {}

    symbols = list(data.Symbol.unique())
    for sym in symbols:
        if sym == 'SPY':
            factors_list = equity_factors_list
        else:
            factors_list = crypto_factors.keys()

        data_temp = data_recent.loc[data_recent['Symbol'] == sym]
        data_temp = data_temp.drop_duplicates()
        
        if len(data_temp)<=0:
            continue
        
        # Add t-1 Factor
        for factor in factors_list:
            data_temp['{}_return_previous'.format(factor)] = data_temp['{}_return'.format(factor)].shift(1)

        data_temp = data_temp.sort_values('Date')
        
        # Incl Columns
        data_temp = data_temp[['Date', 'Symbol', 'PctReturn'] + ['{}_return_previous'.format(factor) for factor in factors_list]]

        data_temp = data_temp.dropna()

        # Last 30 day moving average return regression fit
        X_factor = data_temp[['{}_return_previous'.format(x) for x in factors_list]].to_numpy()
        Y = data_temp['PctReturn'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_factor, Y, test_size=TEST_DATASET_SIZE, random_state=42)
        reg = LinearRegression().fit(X_train, y_train)

        ret_pred[sym] = reg
    
    return ret_pred
from sklearn import tree
def DecisionTreePrediction(data, date, lookback, crypto_factors):
    start_date, end_date = GetStartEnd(str(date), lookback)
    data = RemoveNonBusinessDays(data)

    # Train model from date-lookback to date+1
    mask = (data['Date'] > start_date) & (data['Date'] <= end_date)
    data_recent = data.loc[mask]


    # Store Predictions
    ret_pred = {}

    symbols = list(data.Symbol.unique())
    for sym in symbols:
        if sym == 'SPY':
            factors_list = equity_factors_list
        else:
            factors_list = crypto_factors.keys()

        data_temp = data_recent.loc[data_recent['Symbol'] == sym]
        data_temp = data_temp.drop_duplicates()
        
        if len(data_temp)<=0:
            continue
        
        # Add t-1 Factor
        for factor in factors_list:
            data_temp['{}_return_previous'.format(factor)] = data_temp['{}_return'.format(factor)].shift(1)

        data_temp = data_temp.sort_values('Date')
        
        # Incl Columns
        data_temp = data_temp[['Date', 'Symbol', 'PctReturn'] + ['{}_return_previous'.format(factor) for factor in factors_list]]

        data_temp = data_temp.dropna()

        # Last 30 day moving average return regression fit
        X_factor = data_temp[['{}_return_previous'.format(x) for x in factors_list]].to_numpy()
        Y = data_temp['PctReturn'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_factor, Y, test_size=TEST_DATASET_SIZE, random_state=42)
        reg = tree.DecisionTreeRegressor().fit(X_train, y_train)

        ret_pred[sym] = reg
    
    return ret_pred


def generate_date_list(data, start, end):
    data = RemoveNonBusinessDays(data)
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)

    # Train model from start_date to date
    mask = (data['Date'] >= start) & (data['Date'] <= end)
    data = data.loc[mask]
    return data.Date.apply(lambda x: x.date()).unique().tolist()


# Robust MVO
import numpy as np
from scipy.stats import chisquare
from scipy.stats import gmean
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
import pandas as pd
options['show_progress'] = False
options['feastol'] = 1e-9

def RTMVO(mu,Q,x0, lamda, max_weight):
    # Penalty on Turnover (very sensitive)
    c = 0
    # Penalty on variance
    lambd = lamda
    # Pentalty on returns
    rpen = 1
    # Max weight of an asset
    max_weight = max_weight
    # between 0% and 200%
    turnover = 2
    #size of uncertainty set
    ep = 2

    T = np.shape(mu)[0]
    Theta = np.diag(np.diag(Q))/T
    sqrtTh = np.diag(matrix(np.sqrt(Theta)))
    n = len(Q)

    # Make Q work for abs value
    Q = matrix(np.block([[Q, np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]))

    # A and B
    b1 = np.ones([1,1])
    b2 = x0


    A = matrix(np.block([[np.ones(n), c * np.ones(n), -c * np.ones(n)], [np.eye(n), np.eye(n), -np.eye(n)]]))
    b = np.concatenate((b1,b2))

    # G and h
    G = matrix(0.0, (6 * n + 1, 3 * n))
    h = opt.matrix(0.0, (6 * n + 1, 1))
    for k in range(3 * n):
        # xi > 0 constraint
        G[k, k] = -1
    # xi > max_weight
        G[k + 3 * n, k] = 1
        h[k + 3 * n] = max_weight
    for k in range(2 * n):
        # sum dwi+ + dwi- < turnover
        G[6 * n, k + n] = 1

    h[6 * n] = turnover

    quad = lambd*Q

    r = matrix(np.block([rpen*np.array(mu) - ep*sqrtTh, -c * np.ones(2*n)]))

    return np.transpose(np.array(qp(matrix(quad), -1*matrix(r), matrix(G), matrix(h), matrix(A), matrix(b))['x'])[0:n])[0].tolist()

import numpy as np
from scipy.stats import chisquare
from scipy.stats import gmean
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
import pandas as pd


def geo_mean(returns):
    geo = []
    n = len(returns.transpose())
    for i in range(0,n):
        geo = geo + [np.exp(np.log(rets[:,i]+1).mean())-1]
    return geo

def quad_opt_func(Q,n):
    
    return 2*40*Q

def lin_opt_func(r,n):
    if r == False:
        return np.zeros([n,1])
    else:
        return r
    
def inequality_constraints(n, max_weight):
    
    # Inequality Constraint
    # Expected Return Over 0.0035
    # G1 = np.identity(n)
    # h1 = np.zeros([n,1])*max_weight
    
    G = -1*np.identity(n)
    h = np.zeros([n,1])

    G2 = np.identity(n)
    h2 = np.ones([n,1])*max_weight

    G = np.concatenate((G,G2),axis=0)
    h = np.concatenate((h,h2),axis=0)

    # # Lower Bound on Each Element!
    # G2 = -1*np.identity(n)
    # h2 = np.zeros([n,1])

    # #Concat all Answers
    # # G = G2
    # G = np.concatenate((G1,G2),axis=0)
    # # h = h2
    # h = np.concatenate((h1,h2),axis=0)
    
    return G,h

def equality_constraints(n):
    
    # Equality Constraint
    # Weight sum is 1
    A1 = np.ones([1,n])
    b1 = np.ones([1,1])
    
    
    #Concat All Equality
    A = A1#np.concatenate((A1),axis=0)
    b = b1#np.concatenate((b1),axis=0)
    
    return A,b


def MVO(mu,Q,x0, lamda, max_weight):
    
    #NOTE: X0 is not used yet but it will be for transaction costs
    # Number of Assets
    n = len(Q)
    # ----- Constraints -----------
    # Equality Constraint
    A,b = equality_constraints(n)
    # Inequality Constraint
    G,h = inequality_constraints(n, max_weight)
    # --- Quadtratic Optimization Function --------
    #quad = 2*Q
    quad = quad_opt_func(Q,n)
    # ------ Linear Optimization Function ---------
    #r = np.zeros([n,1])
    r = lin_opt_func(mu,n)
    
    #------- Random Robust Stuff -------
    T = np.shape(mu)[0]
    Theta = np.diag(np.diag(Q))/T
    sqrtTh = matrix(np.diag(matrix(np.sqrt(Theta))))
    ep = 1.96
    
    # ------------ Optimize! --------
    sol = qp(lamda*matrix(quad), -1*matrix(r) + ep*sqrtTh, matrix(G), matrix(h), matrix(A), matrix(b))['x']
    #sol = qp(matrix(quad), matrix(r), None, None, matrix(A), matrix(b))#['x']
    return sol

def SharpeRatio(returns, rf):
    port_ret = (returns[len(returns)-1] - returns[0])/returns[0]
    return (port_ret - rf)/np.std(returns)

# import pandas as pd
# Data = pd.read_csv('data/ProcessedData.csv')
# Data['Date'] = pd.to_datetime(Data['Date'])
# print(Data.columns)
# print(FactorModelPrediction(Data, '2021-01-03', 3, crypto_factors_list))




