from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import unicodedata
import importlib

#My python file
import DataFramer

df_main = DataFramer.loadData()

df_main['Prices'] = df_main['5. adjusted close'].apply(np.int64)
df_main = df_main[['Prices', 'Headlines']]

df = df_main[['Prices']].copy()

df["compound"] = ''
df["neg"] = ''
df["neu"] = ''
df["pos"] = ''


sid = SentimentIntensityAnalyzer()
for date, row in df_main.T.iteritems():
    #try:
    sentence = unicodedata.normalize('NFKD', df_main.loc[date, 'Headlines'])
    ss = sid.polarity_scores(sentence)
    df.set_value(date, 'compound', ss['compound'])
    df.set_value(date, 'neg', ss['neg'])
    df.set_value(date, 'neu', ss['neu'])
    df.set_value(date, 'pos', ss['pos'])
##    except TypeError:
##        print (df_main.loc[date, 'Headlines'])
##        print (date)
