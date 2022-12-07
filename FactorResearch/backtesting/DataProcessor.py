import pandas as pd 
import numpy as np

from util import asset_list, crypto_factors_list, start_date, end_date, equity_factors_list

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.Data = None
        self.sym_list = None

    def LoadData(self):
        df_list = []

        for asset in asset_list:
            df = pd.read_csv(self.data_path.format('coin_{}.csv'.format(asset)))
            if asset == 'SPY':
                df['Symbol'] = asset
                df['Marketcap'] = 0
                df['SNo'] = 0
                df['Name'] = 'SPY'
                df['Date'] = pd.to_datetime(df.Date.astype(str) + ' 23:59:59')
            else:
                df['Adj Close'] = df['Close']
            df_list.append(df)
        
        data = df_list[0]
        df_list = df_list[1:]

        for df in df_list:
            data = data.append(df)
        
        self.Data = data
    
    def SortData(self, by):
        self.data = self.Data.sort_values(by,
              ascending = [True, True])

    def AppendPercentageReturn(self):
        # Calculate percentage return by symbol
        self.sym_list = list(self.Data.Symbol.unique())
        return_dfs = []

        # Iterate through every asset, create a new df with Symbol, Date, Percentage Return
        for sym in self.sym_list:
            print('Generating Returns for {}'.format(sym))
            d_pct_return = self.Data.loc[self.Data['Symbol'] == sym]
            d_pct_return = d_pct_return.drop_duplicates(subset=['Date'])

            pct_return_series = d_pct_return.Close.pct_change()
            date_series = d_pct_return.Date
            sym_series = pd.Series([sym]*len(date_series))

            Data = {
                'Date': date_series,
                'Symbol': sym_series,
                'PctReturn': pct_return_series
            }

            d = pd.DataFrame(Data)

            return_dfs.append(d)
        data_return = return_dfs[0]
        return_dfs = return_dfs[1:]

        for df in return_dfs:
            data_return = data_return.append(df)
        
        self.Data = self.Data.merge(data_return, on=['Date', 'Symbol'], how='left')

    def AddDailyVariation(self):
        self.Data['DailyVariation'] = (self.Data['High']-self.Data['Low'])/self.Data['Low']

    def ConvertToDatetime(self, colName):
        self.Data[colName] = pd.to_datetime(self.Data[colName])

    def AppendCryptoFactors(self):
        for factor, det in crypto_factors_list.items():
            if not det['need_compute']:
                continue
            
            from datetime import datetime, timedelta
            from dateutil.relativedelta import relativedelta
            # Get data between start and end date
            # mask = (self.Data['Date'] > datetime.combine((datetime.fromisoformat(start_date) - relativedelta(months=3)).date(), datetime.max.time())) & (self.Data['Date'] <= end_date)

            # data_recent = self.Data.loc[mask]
            data_recent = self.Data
            # Get average FACTOR by symbol
            average_by_factor = data_recent.groupby('Symbol')[det['var_name']].mean().sort_values()
            average_by_factor = average_by_factor.reset_index()

            # Get top 10, bottom 10 symbols sorted by factor
            half_index = (len(average_by_factor)//2)
            bottom_factor = list((average_by_factor.iloc[:half_index,:]).Symbol)
            top_factor = list((average_by_factor.iloc[half_index:,:]).Symbol)

            # Get average return by FACTOR, date
            data_recent['TopOrBottom'] = np.where(data_recent['Symbol'].isin(bottom_factor), 'Bottom', 'Top')

            return_factor_portfolios = data_recent.groupby(['Date', 'TopOrBottom'])['PctReturn'].mean().reset_index()
            return_factor_portfolios = return_factor_portfolios.pivot(index='Date', columns='TopOrBottom', values='PctReturn')
            return_factor_portfolios['{}_return'.format(factor)] = return_factor_portfolios['Top'] - return_factor_portfolios['Bottom']

            # Save to dictionary
            crypto_factors_list[factor]['return'] = return_factor_portfolios.reset_index()[['Date', '{}_return'.format(factor)]]
    
        # Add crypto market factor
        self.Data = data_recent

        def my_agg(x):
            names = {'crypto_mkt_return': (x['Marketcap'] * x['PctReturn']).sum()/x['Marketcap'].sum()}
            return pd.Series(names, index=['crypto_mkt_return'])

        # Add momentum factor
        data_imp = self.Data[['Date', 'Symbol', 'PctReturn']]   
        data_imp.set_index('Date', inplace=True)
        df_rolling = data_imp.groupby('Symbol').rolling(7).mean().shift(1)
        df_rolling.reset_index(inplace=True)
        df_rolling = df_rolling.rename(columns={'PctReturn': 'momentum_return'})
        self.Data = self.Data.merge(df_rolling, on=['Date', 'Symbol'], how='left')
        self.Data = self.Data.drop_duplicates(subset=['Date', 'Symbol'])

        self.Data = self.Data.merge(self.Data.groupby("Date").apply(my_agg).reset_index(), on='Date', how='left')

        # Add factor returns to price data
        for factor, det in crypto_factors_list.items():
            if det['need_compute']:
                self.Data = self.Data.merge(det['return'], on='Date', how='left')


    def AppendTMinus1Factors(self):
        dfs = []
        for sym in self.sym_list:
            data_temp = self.Data.loc[self.Data['Symbol'] == sym]
            data_temp = data_temp.drop_duplicates()

            if len(data_temp)<=0:
                continue
            
            for factor in list(crypto_factors_list.keys()) + (equity_factors_list):
                print(factor)
                data_temp['{}_return_previous'.format(factor)] = data_temp['{}_return'.format(factor)].shift(1)
            
            data_temp = data_temp.sort_values('Date')
            data_temp['PriceMA'] = data_temp['PctReturn'].rolling(window=30, closed= "left").mean()

            dfs.append(data_temp)

        self.DataWithFactors = pd.concat(dfs)

        
        return self.DataWithFactors
    
    def EquityFactors(self):
        factors = pd.read_csv(self.data_path.format('FF_3factor.CSV'))
        factors['Date'] = pd.to_datetime(factors['Date'], format="%Y%m%d")
        factors['Date'] = pd.to_datetime(factors.Date.astype(str) + ' 23:59:59')

        self.EquityFactors_df = factors

    def ProcessData(self):
        # Sort by date, Symbol
        self.SortData(by=['Date', 'Symbol'])

        # Add Percentage Return
        self.AppendPercentageReturn()

        # Add Daily Variation Factor
        self.AddDailyVariation()

        # Convert Date column to datetime
        self.ConvertToDatetime('Date')

        # Add Cyrpto Factors
        self.AppendCryptoFactors()

        #Add Equity Factors
        self.EquityFactors()
        self.Data = self.Data.merge(self.EquityFactors_df, on='Date', how='left')
        
        # Add t-1 Factors
        self.AppendTMinus1Factors()

        # Convert last column to datetime
        self.DataWithFactors['Date'] = pd.to_datetime(self.DataWithFactors['Date'])

        # Remove certain assets
        self.DataWithFactors = self.DataWithFactors.drop(self.DataWithFactors[~self.DataWithFactors.Name.isin(asset_list)].index)
        print(self.DataWithFactors.Symbol.unique())

    def SaveProcessedData(self, filename, df):
        df.to_csv(self.data_path.format(filename))

path = 'FactorResearch/backtesting/data/{}'
Processor = DataProcessor(path)

Processor.LoadData()
Processor.ProcessData()
Processor.SaveProcessedData('ProcessedData.csv', Processor.DataWithFactors)



