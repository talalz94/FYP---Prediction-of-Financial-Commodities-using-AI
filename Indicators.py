from pylab import *
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf


def MACD(data,stock):
#data   = pd.read_csv('data.csv')

#stock  = Sdf.retype(data)

    signal = stock['macds']        # Your signal line
    macd   = stock['macd']         # The MACD that need to cross the signal line
    #                                              to give you a Buy/Sell signal
    listLongShort = ["No data"]    # Since you need at least two days in the for loop

    for i in range(1, len(signal)):
        #                          # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
        #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
        #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

            
def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    ave = pd.stats.moments.rolling_mean(price,length)
    sd = pd.stats.moments.rolling_std(price,length)
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)





def Datapull(Stock):
    try:
        df = (pd.io.data.DataReader(Stock,'google',start='01/01/2010'))
        return df
        print 'Retrieved', Stock
        time.sleep(5)
    except Exception, e:
        print 'Main Loop', str(e)


def RSIfun(price, n=14):
    delta = price['Close'].diff()
    #-----------
    dUp=
    dDown=

    RolUp=pd.rolling_mean(dUp, n)
    RolDown=pd.rolling_mean(dDown, n).abs()

    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    return rsi
