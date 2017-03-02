"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os
import csv

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data





def get_date_list(orders_file):
    """
    Create DateList
    """
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    date_list=[]
    for line in reader:
        if line[0][0]>='0' and line[0][0]<='9':
            date_list.append(line[0])
    date_list=list(set(date_list))
    date_list.sort()
    return date_list


def get_single_date(item_1):
    """
    get the date of 1 single trade order
    """
    if item_1[0][0]>='0' and item_1[0][0]<='9':
        temp_date=item_1[0]
    else:
        return 'Line Error'
    return temp_date

def get_date_range(start_date,end_date):
    dates=pd.date_range(start_date,end_date)
    date=[]
    for time in dates:
        date.append(time.date())
    return date



def get_symbol_list(orders_file):
    """
    Create SymbolList
    """
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    symbol_list=[]
    for item in reader:
        if item[0][0]>='0' and item[0][0]<='9':
            symbol_list.append(item[1])
    symbol_list=list(set(symbol_list))
    return symbol_list


def create_price_matrix(orders_file, start_date, end_date):
    """
    Create PriceMatrix
    """
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    dates = pd.date_range(start_date, end_date)
    symbols=get_symbol_list(orders_file)
    prices_all = get_data(symbols, dates)
    prices=prices_all[symbols]
    prices['Cash']=1.0
    return prices

def create_trade_matrix(orders_file, start_date, end_date, start_val):
    """
    Create TradeMatrix
    """
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    df_price=create_price_matrix(orders_file,start_date,end_date)
    symbols=get_symbol_list(orders_file)
    symbols_original=list(symbols)
    symbols.append('Cash')
    df=pd.DataFrame(0.0, index=df_price.index, columns=symbols)
    df['Cash'][df_price.index[0]]=1.0*start_val
    for item in reader:
        if item[0][0]>='0' and item[0][0]<='9':
            if get_single_date(item) in df_price.index:
                if item[2]=='Buy':
                    df[item[1]][get_single_date(item)]+=int(item[3])
                    df['Cash'][get_single_date(item)]-=int(item[3])*df_price[item[1]][get_single_date(item)]
                else:
                    df[item[1]][get_single_date(item)]-=int(item[3])
                    df['Cash'][get_single_date(item)]+=int(item[3])*df_price[item[1]][get_single_date(item)]
            
                df_trade_temp=df.copy()
                df_trade_temp=df_trade_temp.cumsum()
                df_trade_temp=df_trade_temp.abs()
                df_price_temp=df_price.copy()
                df_trade_temp=df_trade_temp.drop('Cash',1)
                df_price_temp=df_price_temp.drop('Cash',1)
                df_trade_temp=df_trade_temp*df_price_temp
                df_trade_temp['holding']=df_trade_temp.sum(axis=1)
                
                
                df_trade_temp_de=df.copy()
                df_trade_temp_de=df_trade_temp_de.cumsum()
                df_trade_temp_de[symbols_original]*=df_price_temp
                df_trade_temp_de['value']=df_trade_temp_de.sum(axis=1)
                if (1.0*df_trade_temp['holding'][get_single_date(item)]/df_trade_temp_de['value'][get_single_date(item)])>2.0:
                    if item[2]=='Buy':
                        df[item[1]][get_single_date(item)]-=int(item[3])
                        df['Cash'][get_single_date(item)]+=int(item[3])*df_price[item[1]][get_single_date(item)]
                    else:
                        df[item[1]][get_single_date(item)]+=int(item[3])
                        df['Cash'][get_single_date(item)]-=int(item[3])*df_price[item[1]][get_single_date(item)]

    return df
    

def create_holding_matrix(orders_file, start_date, end_date, start_val):
    """
    Create holding matrix
    """
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    dates = pd.date_range(start_date, end_date, freq='D')
    symbols=get_symbol_list(orders_file)
    df_trade=create_trade_matrix(orders_file, start_date, end_date, start_val)
    df_price=create_price_matrix(orders_file,start_date,end_date)
    df=df_trade
    df=df.cumsum()
    return df
def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here
    dates=pd.date_range(start_date,end_date)
    df_price=create_price_matrix(orders_file,start_date,end_date)
    df_holding=create_holding_matrix(orders_file,start_date,end_date, start_val)
    df_stock_value=df_holding.dot(df_price.transpose())
    df_portvals=pd.DataFrame(list(np.diagonal(df_stock_value)), index=df_price.index, columns=['Portfolio Value'])
    return df_portvals



def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-14'
    end_date = '2011-12-14'
    orders_file = os.path.join("orders", "orders2.csv")
    reader=csv.reader(open(orders_file,'rU'),delimiter=',')
    start_val = 1000000
    
    
    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    
    
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
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

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")



if __name__ == "__main__":
    test_run()
