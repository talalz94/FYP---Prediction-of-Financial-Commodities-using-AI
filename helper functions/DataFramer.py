import pandas as pd
import pickle
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators





def loadData(stock):

    file = []
    
    if stock == 'MSFT':
        file = ['mic']
    if stock == 'GOOGL':
        file = ['data1','data2']
    if stock == 'AMZN':
        file = ['amzn']
    if stock == 'FB':
        file = ['fb1','fb2']
        
    
    ts = TimeSeries(key='UJIVC8A85XPTQULK', output_format='pandas')

    data, meta_data = ts.get_daily_adjusted(symbol=stock, outputsize='full')

    st = data.ix['2010-01-01':"2019-01-04"]
    st.index = pd.to_datetime(st.index)

    a = []
    c = []

    df = loadArticles(file)
    
    for i in st.index:

        a = []
        for j in df.index:

            if (i > j):

                a.append(df[j])
                df = df.iloc[1:]

        b = ''.join(a)


        c.append(b)
        

    st['Headlines'] = c

    return st
    
#Main Function, Returns dataFrame with articles and dates column
def loadArticles(file):
    if len(file) == 1:
    #Combining all the pickle files if more than one
        allArticles = load_obj(file[0]) #+ load_obj('data2')

    elif len(file) > 1:
        allArticles = load_obj(file[0]) + load_obj(file[1])

    headlines = []
    dates =[]
    snip=[]
    df = pd.DataFrame()

    data = parseArticles(allArticles)

    for i in data:
        headlines.append((i['headline'] + '. '+ i['snippet']+ '$#%'))
        dates.append(i['date'])

    df['Date'] = dates
    df['headline'] = headlines
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.drop_duplicates(subset='headline', keep='last')
    sf = df.groupby('Date', sort=True)['headline'].apply(''.join)

    return sf


# Loading the pickle file from Scrapper
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)





    


       

    

def parseArticles(data):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    pagec=10000000
    articlec=0
    for x in data:
        if (len(list(x.keys())) == 3):
            #print(len(x))
            #print(x)
            for i in x['response']['docs']:
                #print(articlec)
                dic = {}
                #dic['id'] = i['_id']
        ##        if i['abstract'] is not None:
        ##            dic['abstract'] = i['abstract'].encode("utf8")
                dic['headline'] = i['headline']['main']#.encode("utf8")
                dic['snippet'] = i['snippet']

                #dic['desk'] = i['news_desk']
                dic['date'] = i['pub_date'][0:10] # cutting time of day.  
                news.append(dic)
                
                articlec += 1
            pagec += 1
        

    return(news)





