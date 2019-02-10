from newsapi import NewsApiClient
import pickle 

newsapi = NewsApiClient(api_key='a0633fcc7fad4ece898ec1f51160e835')

a = newsapi.get_everything(q='google',
                                      from_param='2019-01-10',
                                      to='2019-01-12',
                                      language='en',
                                      page=19)



allarticles = []

def ga():
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    
    global allarticles
    
    for i in range (1,21):
        articles = newsapi.get_everything(q='google',
                                      from_param='2019-01-10',
                                      to='2019-02-10',
                                      language='en',
                                      page=i)

        allarticles.append(articles)
        print(i)

    save_obj(allarticles, 'data2')

    


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



