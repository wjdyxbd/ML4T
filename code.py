"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os
import csv
import math
import matplotlib.pyplot as plt

from marketsim import compute_portvals
from LinRegLearner import LinRegLearner
from KNNLearner import KNNLearner
from util import get_data, plot_data, symbol_to_path
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def get_symbol_list(df_data):
    """
        Create SymbolList
    """
    symbol_list=[]
    for item in reader:
        if item[0][0]>='0' and item[0][0]<='9':
            symbol_list.append(item[1])
    symbol_list=list(set(symbol_list))
    return symbol_list

def momentum(df_data,n):
    df_data=df_data.copy()
    df_momentum=df_data
    df_momentum[n:]=(df_data[n:]/df_data[:-n].values)-1
    return df_momentum
def bollinger_band(df_data,n):
    df_data=df_data.copy()
    symbol=df_data.columns.values
    df_data_nd_ma = pd.rolling_mean(df_data, window=n)
    df_data_bb_value = (df_data-df_data_nd_ma)/(2*pd.rolling_std(df_data[symbol], n, min_periods=n))
    df_data_bb_value.columns=['bb_value']
    return df_data_bb_value
def volatility(df_data,n):
    df_data=df_data.copy()
    symbol=df_data.columns.values
    df_data['volatility']=pd.rolling_std(df_data[symbol], n, min_periods=n)
    return df_data[['volatility']]
def create_train_data(data,vector_n):
    X_1=momentum(data,vector_n[0])
    X_2=bollinger_band(data,vector_n[1])
    X_3=volatility(data,vector_n[2])
    df_data=X_1.copy()
    df_data.columns=['X_1']
    df_data['X_2']=X_2.copy()
    df_data['X_3']=X_3.copy()
    n=max(vector_n)
#    df_data=df_data/df_data.mean(axis=0)
    df_return=five_day_return(data)
    df_data['Y']=pd.DataFrame(df_return,index=X_1.index)
    df_data['price']=data
    return df_data[n-1:-5]
def five_day_return(df_data):
    df_data=df_data.copy()
    df_return=np.array(df_data)
    df_return=np.squeeze(df_return)
    df_return[:-5]=(df_return[5:]/df_return[:-5])-1
    return df_return
def wrap_up(symbol,start_date,end_date,out=False):
    dates=pd.date_range(start_date,end_date)
    data=get_data(symbol,dates, addSPY=False)
    data=data.dropna()
    vector_n=(5,5,5)
    df_data=create_train_data(data,vector_n)
    trainX=np.array(df_data[['X_1','X_2','X_3']])
    trainY=np.array(df_data[['Y']])
    
    dates_test=pd.date_range('2010-01-01','2010-12-31')
    test_data=get_data(symbol,dates_test,addSPY=False)
    test_data=test_data.dropna()
    df_test_data=create_train_data(test_data,vector_n)
    testX=np.array(df_test_data[['X_1','X_2','X_3']])
    testY=np.array(df_test_data[['Y']])
    testY=testY[:,0]
    
    
    
    learner = KNNLearner(3)
#    learner = LinRegLearner()
    learner.addEvidence(trainX, trainY) # train it
    
    # evaluate bin sample
    if out==False:
        predY = learner.query(trainX) # get the predictions
        df_data['predY']=predY
        return df_data
    else:
        predY = learner.query(testX)
        df_test_data['predY']=predY
        return df_test_data

def trading_dates(df_data,symbol):
    df_data['portfolio']=0
    df_data['Sell']=False
    df_data['Buy']=False
    df_data['End']=False
    for index,date in enumerate(df_data.index):
        if index < len(df_data.index):
            if (df_data.ix[df_data.index[index]]['predY']<-0.01) and (df_data.ix[df_data.index[index-1]]['portfolio']==0) and index < len(df_data.index)-5:
                df_data.ix[date.date(),'Sell']=True
                df_data.ix[date.date(),'portfolio']=-100
                df_data.loc[df_data.index[index+5],'End']='Buy'
                df_data.loc[df_data.index[index+5],'portfolio']=0
            elif df_data.ix[df_data.index[index]]['predY']>0.01 and df_data.ix[df_data.index[index-1]]['portfolio']==0 and index < len(df_data.index)-5:
                df_data.ix[date.date(),'Buy']=True
                df_data.ix[date.date(),'portfolio']=100
                df_data.loc[df_data.index[index+5],'End']='Sell'
                df_data.loc[df_data.index[index+5],'portfolio']=0
            elif df_data.ix[date.date(),'End']==False:
                df_data.ix[date.date(),'portfolio']=df_data.ix[df_data.index[index-1],'portfolio']
    sell_dates = df_data.index[df_data['Sell']==True]
    buy_dates = df_data.index[df_data['Buy']==True]
    end_dates = df_data.index[df_data['End']!=False]
    df_order=pd.DataFrame(index=df_data.index)
    df_order['Symbol']= None
    df_order['Order']= None
    df_order['Shares']= None
    for date in df_data.index:
        if df_data.ix[date.date()]['Sell']==True or df_data.ix[date.date()]['End']=='Sell':
            df_order.ix[date.date()]['Order']='Sell'
            df_order.ix[date.date()]['Shares']=100
            df_order.ix[date.date(),'Symbol']=symbol
        elif df_data.ix[date]['Buy']==True or df_data.ix[date]['End']=='Buy':
            df_order.ix[date.date()]['Order']='Buy'
            df_order.ix[date.date()]['Shares']=100
            df_order.ix[date.date()]['Symbol']=symbol
    df_order=df_order.dropna(axis=0)
    df_order.to_csv(path_or_buf='/Users/chen/Downloads/ml4t/mc3_p2/orders/orders_mc3_p2.csv',index_label='Date')
    return sell_dates,buy_dates,end_dates

def test_run():
    """Driver function."""
    #plot1.1
    symbol=['ML4t-399']
    start_date='2008-01-01'
    end_date='2009-12-30'
    df_data=wrap_up(symbol,start_date,end_date)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])
    

    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['trainY']=df_data['price']*(df_data['Y']+1)
    plot=df_data.plot(y=['trainY','predY','price'],color=['green','red','blue'])
    plt.title('Training Y/Price/Predicted Y plot for ML4T-399')
    plt.legend(loc=1)
    plt.show()


    #plot 1.2
    symbol=['IBM']
    start_date='2008-01-01'
    end_date='2009-12-30'
    df_data=wrap_up(symbol,start_date,end_date)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])

    
    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['trainY']=df_data['price']*(df_data['Y']+1)
    plot=df_data.plot(y=['trainY','predY','price'],color=['green','red','blue'])
    plt.title('Training Y/Price/Predicted Y plot for IBM')
    plt.show()

    #plot 2 in
    symbol=['ML4t-399']
    start_date='2008-01-01'
    end_date='2009-12-30'
    df_data=wrap_up(symbol,start_date,end_date)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])
    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['trainY']=df_data['price']*(df_data['Y']+1)

    plot=df_data.plot(y=['trainY','predY','price'],color=['yellow','blue','purple'])
    ymin, ymax = plot.get_ylim()
    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1, color='r')
    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1, color='g')
    plot.vlines(x=end_dates, ymin=ymin, ymax=ymax-1, color='black')
    plt.title('Sine Data In Sample Entries/Exits')
    plt.legend(loc=1)
    plt.show()

    #plot 3 in
    start_val=10000
    orders_file = os.path.join("orders","orders_mc3_p2.csv")
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

    #plot 4 out
    symbol=['ML4t-399']
    start_date='2008-01-01'
    end_date='2009-12-30'
    df_data=wrap_up(symbol,start_date,end_date,out=True)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])

    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['Y']=df_data['price']*(df_data['Y']+1)
    plot=df_data.plot(y=['Y','predY','price'])

    ymin, ymax = plot.get_ylim()
    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1, color='r')
    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1, color='g')
    plot.vlines(x=end_dates, ymin=ymin, ymax=ymax-1, color='black')
    plt.title('Sine Data Out of Sample Entries/Exits')
    plt.show()


    #plot 5 out
    start_date='2010-01-01'
    end_date='2010-12-30'
    start_val=100000
    orders_file = os.path.join("orders","orders_mc3_p2.csv")
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

    #plot 6 in
    symbol=['IBM']
    start_date='2008-01-01'
    end_date='2009-12-30'
    df_data=wrap_up(symbol,start_date,end_date)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])
    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['trainY']=df_data['price']*(df_data['Y']+1)
    plot=df_data.plot(y=['trainY','predY','price'])
    ymin, ymax = plot.get_ylim()
    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1, color='r')
    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1, color='g')
    plot.vlines(x=end_dates, ymin=ymin, ymax=ymax-1, color='black')
    plt.title('IBM Data In Sample Entries/Exits')
    plt.show()
##
    #plot 7 in
    start_val=10000
    orders_file = os.path.join("orders","orders_mc3_p2.csv")
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
##


    #plot 8 out
    symbol=['IBM']
    start_date='2008-01-01'
    end_date='2009-12-31'
    df_data=wrap_up(symbol,start_date,end_date,out=True)
    sell_dates,buy_dates,end_dates=trading_dates(df_data,symbol[0])
    df_data['predY']=df_data['price']*(df_data['predY']+1)
    df_data['Y']=df_data['price']*(df_data['Y']+1)
    plot=df_data.plot(y=['Y','predY','price'])
    ymin, ymax = plot.get_ylim()
    plot.vlines(x=sell_dates, ymin=ymin, ymax=ymax-1, color='r')
    plot.vlines(x=buy_dates, ymin=ymin, ymax=ymax-1, color='g')
    plot.vlines(x=end_dates, ymin=ymin, ymax=ymax-1, color='black')
    plt.title('IBM Data Out Sample Entries/Exits')
    plt.show()


#
    #plot 9 out
    start_date='2010-01-01'
    end_date='2010-12-31'
    start_val=100000
    orders_file = os.path.join("orders","orders_mc3_p2.csv")
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
