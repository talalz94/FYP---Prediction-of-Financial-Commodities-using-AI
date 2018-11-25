from pandas_datareader import data
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
#from newsapi import NewsApiClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from datetime import datetime, timedelta
from ast import literal_eval
import nltk
import re
#import twitter
#from TwitterSearch import *
import csv
import random
import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from nytimesarticle import articleAPI



sys.__stdout__ = sys.stdout

#ts = TimeSeries(key='UJIVC8A85XPTQULK')
# Get json object with the intraday data and another with the call's metadata
#data, meta_data = ts.get_intraday('GOOGL')

##ts = TimeSeries(key='UJIVC8A85XPTQULK', output_format='pandas')
##data, meta_data = ts.get_intraday(symbol='GOOGL',interval='1min', outputsize='full')
#data.plot()
#plt.title('Intraday Times Series for the MSFT stock (1 min)')
#plt.show()

##ti = TechIndicators(key='UJIVC8A85XPTQULK', output_format='pandas')
##data, meta_data = ti.get_bbands(symbol='MSFT', interval='60min', time_period=60)
##data.plot()
##plt.title('BBbands indicator for MSFT stock (60 min)')
##plt.show()

#data.to_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\project\stockprice.csv''', encoding='utf-8')

##newsapi = NewsApiClient(api_key='a0633fcc7fad4ece898ec1f51160e835')
##
##dd = newsapi.get_everything(q='bitcoin', sources = 'mashable',
##                                      from_parameter='2018-3-15',
##                                      to='2018-3-18',
##			      language='en',  sort_by='relevancy')
##df =[d['title'] for d in dd['articles']]
##
###twitter
##tapi = TwitterSearch(consumer_key='AaL9sWow5fXqHdGbwNnu9TJS9',
##                  consumer_secret='1bV7jkWgFbZ8vwjeKtDxoHf0c0uQQniXXy3ve3AhD4Hdoh8rsi',
##                  access_token='456495793-4Ea8uMt1lAjx1p5EZdl8yrui7qffKPUrSljauczs',
##                  access_token_secret='tycC4mCPQDZreqyGGXyIXXW0CNom5FyunwN2O8Xvqmxzs')
##
##api = twitter.Api(consumer_key='AaL9sWow5fXqHdGbwNnu9TJS9',
##                  consumer_secret='1bV7jkWgFbZ8vwjeKtDxoHf0c0uQQniXXy3ve3AhD4Hdoh8rsi',
##                  access_token_key='456495793-4Ea8uMt1lAjx1p5EZdl8yrui7qffKPUrSljauczs',
##                  access_token_secret='tycC4mCPQDZreqyGGXyIXXW0CNom5FyunwN2O8Xvqmxzs')

#text processing
stop = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
cat = []
wl = WordNetLemmatizer()
ss = SnowballStemmer('english')

    

###################################

def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        dic = {}
        #dic['id'] = i['_id']
##        if i['abstract'] is not None:
##            dic['abstract'] = i['abstract'].encode("utf8")
        dic['headline'] = i['headline']['main'].encode("utf8")
        dic['snippet'] = i['snippet']

        #dic['desk'] = i['news_desk']
        dic['date'] = i['pub_date'][0:10] # cutting time of day.  
        news.append(dic)
    return(news)


def get_articles(date,query):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    all_articles = []
    for i in range(0,240): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        articles = api.search(q = query,
               fq = {'source':['Reuters','AP', 'The New York Times']},
               begin_date = date + '0101',
               end_date = date + '1231',
               sort='oldest',
               page = str(i))
        articles = parse_articles(articles)
        all_articles = all_articles + articles
    return(all_articles)


global startDate, articlesListx, priceListx

api = articleAPI('10e395709e384996a152bc796f21d20f')
articleList = []
articleList2 = []
articleList3 = []
allArticles = []
allArticlesx = []
dateList = []
df = pd.DataFrame()
sf = pd.DataFrame()
dff = pd.DataFrame()
mf = pd.DataFrame()
articles = []
articleinfo = []
tempList = []
dict1 = {}
data2 = []
articlesListx=[]
priceListx = []


def getDate(date):

    date1 = datetime.strptime(date, '%Y-%m-%d')
    return date1

def getData(stock,st,fn):

    
    global data2, dff, data

    ts = TimeSeries(key='UJIVC8A85XPTQULK', output_format='pandas')

    data, meta_data = ts.get_weekly_adjusted(symbol=stock)
    #
    datad, meta_datad = ts.get_intraday('GOOGL')
    #
    date1 = data.index.values[1]
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime('2010-01-08', '%Y-%m-%d')
    data2 = data
    if (date1 >= date2):
        startDate = date1 - timedelta(days=7)
        print(datetime.strftime(startDate, '%Y-%m-%d'))
        data2 = data2.ix[datetime.strftime(startDate, '%Y-%m-%d'):'2010-01-24']

    else:
        startDate = date2 - timedelta(days=7)
        print(datetime.strftime(startDate, '%Y-%m-%d'))
        data2 = data2.ix[datetime.strftime(startDate, '%Y-%m-%d'):'2010-01-24']


    
    dff = pd.DataFrame(index= data2.index)
    dff['Price'] = data2['4. close']

    startDate = datetime.strftime(startDate, '%Y%m%d')

    print(startDate)

    getArticles(startDate,st,fn)



def getArticles(sDate,st,fn):
    
    for i in range (st,fn):
        articles = api.search(q="google", fq = {'headline':'google'},
                          
                          begin_date=sDate,
			  end_date="20180124",
                          sort="oldest",
                          page = i,
                          )
        articleinfo = parse_articles(articles)

        framer(articles)
        allArticlesx.append(articles)
             
        print(i)
        
    updateDataFrame()

def textprocessor(text):

    csd = ''.join([i for i in text if not i.isdigit()])
    csd = csd.split()
    csd = [x.strip(' ') for x in csd]  
    csd = [ss.stem(wl.lemmatize((re.sub('[^A-Za-z0-9]+', '', i)))) for i in csd if i not in stop]
    csd = [x for x in csd if x]
    return csd
    
datel = []
def framer(articles):

    articleinfo = parse_articles(articles)

    
    for i in articleinfo:
        info = textprocessor(i.get('snippet'))
        date = i.get('date')
        datel.append(date)
        #print(date)
        if (info not in articleList) and (len(info) > 4):
            articleList.append(info)
            dateList.append(date)

def updateDataFrame():

    global df ,mf, df1, df2, df3, sf
    
    #df['Articles'] = pd.DataFrame({'Article': articleList})
    #df['Date'] = pd.DataFrame({'Date': dateList})

    df['Articles'] = articleList
    df['Date'] = dateList

    df1 = df
    #Sorting by Date
    df['Date'] =pd.to_datetime(df.Date)
    dff.index =pd.to_datetime(dff.index)
    df.sort_values(by='Date')

    df2 = df

    df['Articles'] = df.Articles.apply(tuple)
    df = df.groupby('Date').agg(lambda x: x.tolist())
    df['Articles'] = df.Articles.apply(tuple)
    
    df3 = df
    
    df.index = df.index.strftime( "%Y-%m-%d")
    dff.index = dff.index.strftime( "%Y-%m-%d")
    del dateList[:]

   
    for i in dff.index:
        articleList2 = []
        
        for j in df.index:
            
            
            if (i > j):
                
                
                articleList2.append(df['Articles'].loc[j])
                if (len(df.index) > 0 ):
                    df = df.drop(df.index[[0]])


        articleList3.append(articleList2)
        dateList.append(i)

        
                

    #mf['Articles'] = pd.DataFrame({'Article': articleList3})
    #mf['Date'] = pd.DataFrame({'Date': dateList})

    mf['Articles'] = articleList3
    mf['Date'] = dateList
    
    
    mf['Price'] = dff['Price'].values
    mf.set_index('Date', inplace=True)
    #mf.to_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\project\stockData.csv''', sep='\t')     
    #car = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\project\stockData.csv''', sep='\t', index_col=0)

    sf = mf

    dd=[]
    for i in range (0,len(sf)):
        if i == 0:
            dd.append(0)
        else:
            dd.append((sf['Price'][i]) - (sf['Price'][i-1]))

    myList = list(np.around(np.array(dd),1))
    sf['Price-Change'] = myList
    #sf = sf.iloc[1:]

    sf.loc[sf['Price-Change'] == 0, 'Change'] = 2
    sf.loc[sf['Price-Change'] > 0,  'Change'] = 1
    sf.loc[sf['Price-Change'] < 0,  'Change'] = 0

    sf = sf.iloc[1:]
    sf.to_csv(r'''C:\Users\Talal\Desktop\Talal\6TH SEMESTER\AI\project\stockDataTest.csv''', sep='\t')     


    
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec    
###############################################################################################changes
def getArticles2(sDate):

    for i in range (1):
        articles = api.search(q="google", fq = {'headline':'google'},
                          
                          begin_date=sDate,
			  end_date="20180321",
                          sort="oldest",
                          page = i,
                          )
        
        
        framer2(articles)
        print(i)
        
    #updateDataFrame()


def runModel():

    
    # defining training and test data set
    global x_train,tweet_w2v, tfidf, y_test, x_test, model, y_train, dataSet, model
    
    articlesListx = []
    priceListx = []
    

    dataSet = pd.read_csv(r'''C:\Users\Talal\Desktop\Talal\6TH SEMESTER\AI\project\stockData.csv''', sep='\t', index_col=0)
    #dataPredict = dataSet.ix[-3]
    #dataSet = dataSet.ix[:-3]
    dataPredict = dataSet.ix[257:]
    #dataSet = dataSet.ix[:257]
    x=0
    
    for i in dataSet['Articles']:
        for j in literal_eval(i):
            for k in j:
                #print(k)
                articlesListx.append(k)
                priceListx.append(int(dataSet['Change'][x]))
        x=x+1

    
            
    c = list(zip(articlesListx, priceListx))
    random.shuffle(c)
    articlesListx, priceListx = zip(*c)

    #print('Article List Length: ')
    #print(len(articlesListx))
    #print(priceListx)

    
    x_train, x_test, y_train, y_test = train_test_split(articlesListx, priceListx, test_size=0.4)

    

    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')

    y_test = [1 if x==1 else 0 for x in y_test]
    y_train = [1 if x==1 else 0 for x in y_train]

    #print(x_train)
    #print(x_test)

    
    tweet_w2v = Word2Vec(size=200,min_count=2)
    tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
    tweet_w2v.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)

    #tweet_w2v.build_vocab(x_train)
    #tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)

    #output_notebook()
    #plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    #tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    #x_axis_type=None, y_axis_type=None, min_border=1)    
    #word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]
    #tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    #tsne_w2v = tsne_model.fit_transform(word_vectors)
    #tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    #tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=2)
    matrix = vectorizer.fit_transform([x.words for x in x_train])

    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_train))])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_test))])
    test_vecs_w2v = scale(test_vecs_w2v)
    
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.models import model_from_json

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=200))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.fit(train_vecs_w2v, y_train, epochs=100, batch_size=32, verbose=2)
    
    #score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
    score, acc = model.evaluate(test_vecs_w2v[:int(len(test_vecs_w2v)/2)], y_test[:int(len(y_test)/2)], batch_size=128, verbose=2)

    print ('\nValidation: '+  'Score - ' + str(score) + '       '+ 'Accuracy - ' + str(acc))

    score, acc = model.evaluate(test_vecs_w2v[int(len(test_vecs_w2v)/2):], y_test[int(len(y_test)/2):], batch_size=128, verbose=2)

    print ('\nTest: '+  'Score - ' + str(score) + '       '+ 'Accuracy - ' + str(acc))

    # Saving Model

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("\n Saved model to disk")

    #Preparing Prediction

##    #print(literal_eval(dataPredict['Articles']))
##    dataPredictList = []
##    
##    for i in dataPredict['Articles'][86:]:
##        for j in literal_eval(i):
##            for k in j:
##                dataPredictList.append(k)
##
##    
##    x_predict = labelizeTweets(dataPredictList, 'PREDICT')
##    #print(x_predict)
##    #vectorizer = TfidfVectorizer(analyzer=lambda x: x, max_df=0.95)
##    #matrix = vectorizer.fit_transform([x.words for x in x_predict])
##
##    #tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
##    
##
##    predict_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_predict))])
##    predict_vecs_w2v = scale(predict_vecs_w2v)
##    
##    prediction = model.predict_classes(predict_vecs_w2v)
##    #print('Prediction: ')
##    #print(prediction)
##
##    predictions(prediction)



##def predictions(pred):
##    
##    pred = [item for sublist in pred for item in sublist]
##
##    onesCount = pred.count(1)
##    zerosCount = pred.count(0)
##
##    if (onesCount > zerosCount):
##        print('Stock Price Will Increase for the upcoming weekly close from: '+ str(dataSet['Price'][-1])+ '$')
##
##    elif (onesCount < zerosCount):
##        print('Stock Price Will Decrease for the upcoming weekly close from: ' + str(dataSet['Price'][-1])+ '$')
##
##    else:
##        print('Stock Price Will remain about the same as previous week, for the upcoming weekly close at ' + str(dataSet['Price'][-1])+ '$')
##    print(list(pred))

def labelizeTweets(tweets, label_type):
        labelized = []
        for i,v in tqdm(enumerate(tweets)):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized


#Run after RunModel() and feed dataPredict['Articles'][86:]
def predict(inputArticles):

    dataPredictList = []
    
    for i in inputArticles:
        for j in literal_eval(i):
            for k in j:
                dataPredictList.append(k)

    
    x_predict = labelizeTweets(dataPredictList, 'PREDICT')
    

    predict_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_predict))])
    predict_vecs_w2v = scale(predict_vecs_w2v)
    
    pred = model.predict_classes(predict_vecs_w2v)
    
    
    pred = [item for sublist in pred for item in sublist]

    onesCount = pred.count(1)
    zerosCount = pred.count(0)

    if (onesCount > zerosCount):
        print('Stock Price Will Increase for the upcoming weekly close from: '+ str(dataSet['Price'][-1])+ '$')

    elif (onesCount < zerosCount):
        print('Stock Price Will Decrease for the upcoming weekly close from: ' + str(dataSet['Price'][-1])+ '$')

    else:
        print('Stock Price Will remain about the same as previous week, for the upcoming weekly close at ' + str(dataSet['Price'][-1])+ '$')
    print(list(pred))

    

    
def loadModel():

    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.models import model_from_json
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

