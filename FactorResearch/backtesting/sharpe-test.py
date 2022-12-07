import pandas as pd
import numpy as np

data = pd.read_csv('FactorResearch/backtesting/data/ProcessedData.csv')
spy = data.loc[data['Symbol'] == 'SPY']
returns = spy['PctReturn'].to_numpy()
returns = returns[~np.isnan(returns)]
rf=0.02
print((np.mean(returns)-rf)/np.std(returns))
