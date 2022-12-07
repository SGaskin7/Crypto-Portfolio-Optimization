import json
from matplotlib import pyplot as plt
import datetime as dt
import pandas as pd
import seaborn as sns
import sys
import SGCharts
from distinctipy import distinctipy
import matplotlib.ticker as mtick

def GUI_Chart(GUI, state):
    print('Loading Data....')
    clrs = distinctipy.get_colors(18)


    ch = SGCharts.Chart('Model',ncols = 2,figsize=[12,4],col_ratios=[1,3],margin=0.15)

    ch.ax = ch.axes[0]

    dates = list(GUI.keys())
    pnl = []
    for date in dates:
        pnl = pnl + [GUI[date]['pnl']]


    df = pd.DataFrame(pnl,dates,columns=['AUM']).reset_index()
    df['date'] = pd.to_datetime(df['index'])
    #df = df[df.date<'2021-02-07']

    #df = pd.DataFrame(accum,dates,columns=['AUM'])
    initial_val = df['AUM'].iloc[0]
    df['Returns'] = ((df['AUM']/initial_val) - 1)
    df['Vol'] = df['Returns'].rolling(min(15, len(df))).std()
    #df = df.dropna()
    #display(df.head(2))

    #dates_dt = [dt.datetime.strptime(x,'%Y-%m-%d') for x in df.index]
    #print(df['date'])
    ch.ax.plot(df['date'],df['Returns'],color=SGCharts.colors['nbf_blue'])
    ch.ax.fill_between(df['date'],df['Returns']-df['Vol'],df['Returns']+df['Vol'],color=SGCharts.colors['nbf_blue2'])

    ch.ax.set_ylabel('Return')
    fmt = '%.00f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ch.ax.yaxis.set_major_formatter(yticks)
    ch.rotate_xticks(45)

    import numpy as np
    #-------- Sharpe ------
    sharpe = round(((pnl[-1]/1000000) - 1)/np.std(((np.array(pnl)/1000000) - 1)),3)
    print(dates)
    # print(pnl)
    ch.ax.set_title('Sharpe = {}'.format(sharpe))


    ch.ax = ch.axes[1]
    dates = list(GUI.keys())
    assets = list(GUI[dates[2]]['holdings'].keys())

    #accum = []
    #for date in dates:
    #    holding = data[data_factor][date]['holdings'][stock]
    #    accum = accum + [holding]

    color_cnt = 0
    old = [0]*len(dates)
    for stock in assets:
        accum = []
        for date in dates:
            #print(stock)
            #print(date)
            holding = GUI[date]['holdings'][stock]
            accum = accum + [holding]
        ch.ax.bar(dates,accum,bottom=old,color = clrs[color_cnt],label=stock)
        old = [x+y for x,y in zip(old,accum)]
        color_cnt = color_cnt + 1

    xticks = []
    for i in range(0,len(dates),max(round(len(dates)/10), 1)):
        xticks = xticks + [dates[i]]

    ch.ax.set_xticks(xticks)
    ch.rotate_xticks(45)
    #ch.ax.set_title(data_factor)
    ch.ax.set_ylabel('Holding (%)')
    ch.render(spines=['top','bottom'],legend_columns=8)

    if state['Optimizer'] in ['MVO','RMVO']:
        ch.ax.set_title('Optimizer: {opt} | Predictor: {pred} | Weight Cap: {weight}'.format(opt=state['Optimizer'],pred=state['Predictor'],weight=str(state['Weight'])))

    else:
        ch.ax.set_title('Optimizer: {opt} | Alpha: {alpha}'.format(opt=state['Optimizer'],alpha=state['alpha']))



    ch.render(spines=['left','bottom'],loc='below',legend_title='Assets')

    ch.ax.legend(bbox_to_anchor=(1.15,1),title='Assets')
    #ch.fig.savefig('GUI.png'.format(model=data_factor))#,bbox_inches="tight")
    ch.fig.savefig('GUI/GUI.png',bbox_inches='tight',dpi=150,figsize=[12,4])
    return 'GUI.png'
