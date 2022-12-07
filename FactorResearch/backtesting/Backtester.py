import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import *
import warnings
from Series_CVaR.simulate import *
from Series_CVaR.time_series import *
from Series_CVaR.VaR_CVaR import *

warnings.filterwarnings("ignore")

# All of the important functions are defined in util.py (ex: optimizers, model predictors, etc)
# All cvar functions are defined in series_CVaR
class Backtester():
    def __init__(self, predictor, optimizer, alpha=None, max_weight=None, l=None, lookback=3):
        self.lookback = lookback # Months
        self.ReturnPrediction = predictor
        self.Optimizer = optimizer
        self.PortfolioValue = 1e6 # $
        self.Results = {}
        self.lookforward = 1 # Days, rebalance everyday
        self.num_paths = 20 # How many paths generated for cvar
        self.alpha = alpha # For CVaR
        self.lamda = l # For MVO/RMVO
        self.max_weight = max_weight # For all

    # Import Data
    def LoadData(self):
        self.Data = pd.read_csv('FactorResearch/backtesting/data/ProcessedData.csv')
        self.Data['Date'] = pd.to_datetime(self.Data['Date'])
    
    # This function returns the predictors and target for factor models
    def FactorModelFormatData(self,date, sym):
        if sym == 'SPY':
            factors_list = equity_factors_list
        else:
            factors_list = crypto_factors_list.keys()
        
        data_today = self.Data.loc[(self.Data['Date'].dt.date == date)]
        X = data_today[['Symbol', 'Date'] + ['{}_return_previous'.format(x) for x in factors_list]]
        return X
    
    # This function returns the predictors and target for return moving average model
    def PriceMAFormatData(self, date, sym):
        self.Data['PriceMA'] = self.Data['PctReturn'].rolling(window=7).mean().shift(1)
        data_today = self.Data.loc[(self.Data['Date'].dt.date == date)]
        X = data_today[['Symbol', 'Date', 'PriceMA']]

        return X
    
    # This function gets the optimal allocation based on the predictor and optimizer chosen
    def GetOptimalAllocation(self, date, x0):
        print("-------")
        print("Running for {}".format(date))
        # Get Q
        start_date, end_date = GetStartEnd(str(date), self.lookback)
        data = RemoveNonBusinessDays(self.Data)
        mask = (data['Date'] > start_date) & (data['Date'] <= end_date)
        data_recent = data.loc[mask]

        Q_data = []
        Q_sym = []
        
        # Add returns for covariance matrix
        for sym in list(data_recent['Symbol'].unique()):
            data_sym = data_recent.loc[data_recent['Symbol']==sym]
            historical_rets = np.array(list(data_sym[['PctReturn']].dropna()['PctReturn']))

            Q_data.append(historical_rets)
            Q_sym.append(sym)
        
        Q_data = np.array(Q_data)
        Q = np.cov(Q_data)

        # Get mu
        mu_by_sym = {}

        factors_list = crypto_factors_list

        # predictors here is a dictionary mapping the key to its model object returned by sklearn
        if self.ReturnPrediction in ['Factor Model']:
            predictors = FactorModelPrediction(self.Data, date, self.lookback, factors_list)
        elif self.ReturnPrediction == 'Price MA':
            predictors = PriceMAPrediction(self.Data, str(date), self.lookback)
        elif self.ReturnPrediction == 'Decision Tree':
            predictors = DecisionTreePrediction(self.Data, date, self.lookback, factors_list)
        elif self.ReturnPrediction == 'CVaR':
            data_dict, prices_dict, returns_dict = generate_data()
            paths = simulate(returns_dict, str(date), self.lookback*15, self.lookforward, self.num_paths, self.Optimizer, error_dist = "Normal", threshold = 1e-2, frac_dif_d = 0.5)
            paths = paths.to_numpy()

            holdings = CVaR_opt(paths, self.alpha, lam=0.,
                short=False, renormalize=True, norm=2, threshold = 1e-2, upper = 0.25)
            
            holdings = holdings/np.sum(holdings)

        if self.ReturnPrediction in ['Factor Model', 'Price MA', 'Decision Tree']:
            mu_sym_list = []
            for sym in sorted(predictors.keys()):
                pred = predictors[sym]
                if self.ReturnPrediction in ['Factor Model', 'Decision Tree']:
                    X = self.FactorModelFormatData(date, sym)
                elif self.ReturnPrediction == 'Price MA':
                    X = self.PriceMAFormatData(date, sym)

                mu_sym_list.append(sym)
                X_sym = X.loc[X['Symbol'] == sym]
                X_sym = X_sym.drop(['Symbol', 'Date'], axis=1)
                mu_sym = pred.predict(X_sym)
                mu_by_sym[sym] = mu_sym

            mu = np.array(list(mu_by_sym.values()))

            # Pass into the optimizer
            if self.Optimizer == 'RMVO':
                holdings = RTMVO([x[0] for x in list(mu)],Q,x0, self.lamda, self.max_weight)
            elif self.Optimizer == 'MVO':
                holdings = MVO([x[0] for x in list(mu)],Q,x0,  self.lamda, self.max_weight)

        return matrix(holdings)

    def RunBacktesting(self, start_date, end_date):
        # Filter data between start and end date
        dates = generate_date_list(self.Data, start_date, end_date)
        
        n = len(list(self.Data['Symbol'].unique()))
        x0 = matrix(np.ones(n)*1/n)

        for date in dates:
            res = {
                'holdings': {},
                'pnl': 0
            }

            curr_value = 0

            # Update value of portfolio
            if len(list(self.Data['Symbol'].unique())) != len(x0):
                raise Exception("Sorry, no numbers below zero")
            
            i=0
            for sym, holding in zip(sorted(list(self.Data['Symbol'].unique())), x0):
                growth = self.Data.loc[(self.Data['Symbol'] == sym) & (self.Data['Date'].dt.date == date)]['PctReturn'].values
                
                # If we have price for today
                if len(growth) != 0 and (not np.isnan(x0[i]).any()):
                    growth=growth[0]
                    # Example:
                    # port: 1.5M, BTC holding 50%, btc 5%
                    # SPY holding 50%, spy 10%
                    # (1.5M*0.5*1.05) + (1.5*0.5*1.1)
                    curr_value += holding*self.PortfolioValue*(1+growth)
                i+=1

            self.PortfolioValue = curr_value
                
            res['pnl'] = self.PortfolioValue

            x0 = self.GetOptimalAllocation(date, x0)

            for sym, holding in zip(list(self.Data['Symbol'].unique()), x0):
                res['holdings'][sym] = holding
            
            self.Results[str(date)] = res
        return self.Results

# Use this code to run cross-validation test
def RunCrossVal():
    prefix = 'big_test'

    for pred_methodology in ['Factor Model', 'Price MA', 'Decision Tree']:
        for opt_methodology in ['RMVO', 'MVO']:
            for mw in np.arange(0.2, 1.2, 0.2):
                for lambda_val in [0.5, 1, 5, 10, 50, 100]:
                    print('mw_{}_lambda_{}_pred_{}_opt_{}_alpha_{}_cvarmethod_{}'.format(mw, lambda_val, pred_methodology, opt_methodology, 0, 'na'))
                    backtest = Backtester(predictor=pred_methodology, optimizer=opt_methodology, alpha=0, max_weight=mw, l=lambda_val)
                    backtest.LoadData()
                    backtest.RunBacktesting(start_date, end_date)
                    pnl_data_fm = []

                    for date in list(backtest.Results.keys()):
                        pnl_data_fm.append(backtest.Results[date]['pnl'])

                    import json
                    f = open('FactorResearch/backtesting/data/results-{}.json'.format(prefix), 'r+')
                    data = json.load(f)
                    f.close()
                    f = open('FactorResearch/backtesting/data/results-{}.json'.format(prefix), 'w')
                    data['mw_{}_lambda_{}_pred_{}_opt_{}_alpha_{}_cvarmethod_{}'.format(mw, lambda_val, pred_methodology, opt_methodology, 0, 'na')] = backtest.Results
                    json.dump(data, f)
                    f.close()

    pred_methodology = 'CVaR'
    prefix = 'big_test'

    for cvar_method in ['ARIMA-GARCH','Historical', 'Normal']:
        for alpha_val in [0.01, 0.05, 0.1, 0.5]:
            for mw in np.arange(0.2, 1.2, 0.2):
                print('mw_{}_lambda_{}_pred_{}_opt_{}_alpha_{}_cvarmethod_{}'.format(mw, 0, 'na', 'cvar', alpha_val, cvar_method))
                backtest = Backtester(predictor=pred_methodology, optimizer=cvar_method, alpha=alpha_val, max_weight=mw, l=0)
                backtest.LoadData()
                backtest.RunBacktesting(start_date, end_date)
                pnl_data_fm = []

                for date in list(backtest.Results.keys()):
                    pnl_data_fm.append(backtest.Results[date]['pnl'])
                
                import json
                f = open('FactorResearch/backtesting/data/results-{}.json'.format(prefix), 'r+')
                data = json.load(f)
                f.close()
                f = open('FactorResearch/backtesting/data/results-{}.json'.format(prefix), 'w')
                data['mw_{}_lambda_{}_pred_{}_opt_{}_alpha_{}_cvarmethod_{}'.format(mw, 0, 'na', 'cvar', alpha_val, cvar_method)] = backtest.Results
                json.dump(data, f)
                f.close()
