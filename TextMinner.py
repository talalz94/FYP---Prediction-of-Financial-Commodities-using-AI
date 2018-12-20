from nytimesarticle import articleAPI
import twitter
from TwitterSearch import *

#sys.__stdout__ = sys.stdout

api = articleAPI('10e395709e384996a152bc796f21d20f')


a = []

def Scrapper(x):

    global a
    for i in range (x):
        articles = api.search(q="google", fq = {'headline':'google'},
                          
                          begin_date='20100124',
			  end_date="20180124",
                          sort="oldest",
                          page = i,
                          )

        print(i)
        a.append(articles)

    #Saving the articles
    with open('your_file.txt', 'w',encoding='utf-8') as f:
        for item in a:
            f.write("%s\n" % item)


def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        dic = {}
        dic['headline'] = i['headline']['main'].encode("utf8")
        dic['snippet'] = i['snippet']

        dic['date'] = i['pub_date'][0:10] # cutting time of day.  
        news.append(dic)
    return(news)


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

#def twitterMiner()

    
