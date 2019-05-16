import numpy as np
import unicodedata
import importlib
import pandas as pd

#My python file
import DataFramer

def preSentimentFramer(stock):

    df_main = DataFramer.loadData(stock)

    df_main['Prices'] = df_main['5. adjusted close']#.apply(np.int64)
    df_main['DATE'] = df_main.index
    df_main = df_main[['DATE','Prices', 'Headlines']]

    sf = df_main

    priceList = []

    for i in range (0,len(sf)):
            if i == 0:
                priceList.append(0)
            else:
                priceList.append((sf['Prices'][i]) - (sf['Prices'][i-1]))

    myList = list(np.around(np.array(priceList),1))
    sf['Price-Change'] = myList
    

    #sf.loc[sf['Price-Change'] == 0, 'Change'] = 2
    sf.loc[sf['Price-Change'] >= 0,  'Change'] = 1
    sf.loc[sf['Price-Change'] < 0,  'Change'] = 0
    sf['Change'][0] = 0
    sf = sf.iloc[1:]
    return sf

def frameAll():

    sf = pd.DataFrame()
    g=[]
    c=[]
    dt = []
    pr = []
    
    for i in ['GOOGL','MSFT','FB','AMZN']:

        df = preSentimentFramer(i)

##        if i == 'GOOGL':
##            df = df[:200]

        for index, row in df.iterrows():
            for i in row['Headlines'].split('$#%'):
                if len(i) > 9:
                    g.append(i)
                    c.append(row['Change'])
                    dt.append(row['DATE'])
                    pr.append(row['Prices'])
    
    sf['Headline'] = g
    sf['Change'] = c
    sf['DATE'] = dt
    sf['Prices'] = pr

    return sf
        
