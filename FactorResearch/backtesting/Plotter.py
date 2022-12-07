import json
from matplotlib import pyplot as plt

f = open('FactorResearch/backtesting/data/results.json')
data = json.load(f)
f.close()

pnl_data_fm = []
pnl_data_pma = []

for date in list(data['FactorModel'].keys()):
    pnl_data_fm.append(data['FactorModel'][date]['pnl'])

for date in list(data['PriceMA'].keys()):
    pnl_data_pma.append(data['PriceMA'][date]['pnl'])

print(pnl_data_fm)
plt.plot(list(data['FactorModel'].keys()), pnl_data_fm)
plt.plot(list(data['PriceMA'].keys()), pnl_data_pma)
plt.legend(['Factor Model', '7 Day Average Price'])

plt.savefig('FactorResearch/backtesting/data/results/pnl.png')


