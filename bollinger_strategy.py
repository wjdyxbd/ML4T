"""MC2-P2: bollinger_strategy."""

import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
from marketsim import compute_portvals




def bollinger_strategy(symbol, start_date, end_date):
    dates=pd.date_range(start_date,end_date)
    dates_2=pd.date_range('2008-02-01','2008-05-31')
    df_data=get_data(symbol,dates)
    df_data=df_data.dropna(axis=0)
    df_data['20d_ma'] = pd.rolling_mean(df_data[symbol], window=20)
    df_data['Bol_upper'] = pd.rolling_mean(df_data[symbol], window=20) + 2* pd.rolling_std(df_data[symbol], 20, min_periods=20)
    df_data['Bol_lower'] = pd.rolling_mean(df_data[symbol], window=20) - 2* pd.rolling_std(df_data[symbol], 20, min_periods=20)
    df_data_2=df_data.ix[dates_2]
    df_data_2=df_data_2.dropna(axis=0)
    plot=df_data.plot(y=symbol+['20d_ma','Bol_upper','Bol_lower'],color=['b','y','g','g'])
    plot.legend(['IBM', 'SMA','Bollinger Bands'], loc='best')
    df_data['Sell']=False
    df_data['Buy']=False
    df_data['End']=False
    df_data['portfolio']=0
    for index,date in enumerate(df_data.index):
        if index > 19 and index < len(df_data.index)-1:
            if (df_data.ix[df_data.index[index-1]]['Bol_upper']<df_data.ix[df_data.index[index-1]][symbol[0]]) and (df_data.ix[date]['Bol_upper']>df_data.ix[date][symbol[0]]) and (df_data.ix[date]['20d_ma']<df_data.ix[date][symbol[0]]) and (df_data.ix[df_data.index[index-1],'portfolio']==0):
                df_data.ix[date.date(),'Sell']=True
                df_data.ix[date.date(),'portfolio']=-100
            elif (df_data.ix[df_data.index[index-1]]['Bol_lower']>df_data.ix[df_data.index[index-1]][symbol[0]]) and (df_data.ix[date]['Bol_lower']<df_data.ix[date][symbol[0]]) and (df_data.ix[date]['20d_ma']>df_data.ix[date][symbol[0]]) and df_data.ix[df_data.index[index-1],'portfolio']==0:
                df_data.ix[date.date(),'Buy']=True
                df_data.ix[date.date(),'portfolio']=100
            elif (df_data.ix[df_data.index[index-1]]['portfolio'] > 0 and df_data.ix[date][symbol[0]] > df_data.ix[date]['20d_ma']) or (df_data.ix[df_data.index[index-1]]['portfolio'] < 0 and df_data.ix[date][symbol[0]] < df_data.ix[date]['20d_ma']):
                df_data.ix[date.date(),'portfolio']=0
                if df_data.ix[df_data.index[index-1]]['portfolio'] > 0:
                    df_data.ix[date.date(),'End']='Sell'
                else:
                    df_data.ix[date.date(),'End']='Buy'
            else:
                df_data.ix[date.date(),'portfolio']=df_data.ix[df_data.index[index-1],'portfolio']
    sell_dates = df_data.index[df_data['Sell']==True]
    buy_dates = df_data.index[df_data['Buy']==True]
    end_dates = df_data.index[df_data['End']!=False]
    ymin, ymax = plot.get_ylim()
    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1, color='r')
    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1, color='g')
    plot.vlines(x=end_dates, ymin=ymin, ymax=ymax-1, color='black')
    plt.show()
    df_order=pd.DataFrame(index=df_data.index)
    df_order['Symbol']= None
    df_order['Order']= None
    df_order['Shares']= None
    for date in df_data.index:
        if df_data.ix[date.date()]['Sell']==True or df_data.ix[date.date()]['End']=='Sell':
            df_order.ix[date.date()]['Order']='Sell'
            df_order.ix[date.date()]['Shares']=100
            df_order.ix[date.date(),'Symbol']=symbol[0]
        elif df_data.ix[date]['Buy']==True or df_data.ix[date]['End']=='Buy':
            df_order.ix[date.date()]['Order']='Buy'
            df_order.ix[date.date()]['Shares']=100
            df_order.ix[date.date()]['Symbol']=symbol[0]
    df_order=df_order.dropna(axis=0)
    df_order.to_csv(path_or_buf='/Users/chen/Downloads/ml4t/mc2_p2/orders/orders_mc2_p2.csv',index_label='Date')
    return df_order
def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    start_val = 10000
    symbol = ["IBM"]
    bollinger_strategy(symbol, start_date,end_date)
    orders_file = os.path.join("orders","orders_mc2_p2.csv")
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    print portvals
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)


    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    #Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    #Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")



if __name__ == "__main__":
    test_run()
