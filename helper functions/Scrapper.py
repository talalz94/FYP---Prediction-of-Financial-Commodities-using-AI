from nytimesarticle import articleAPI
import pickle


#api = articleAPI('10e395709e384996a152bc796f21d20f')
api = articleAPI('9EQdIlsHNcIufg3QonXbpewFaNRltc1X')
allarticles = []

def ga():
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    
    global allarticles
    articles = {}
    
    
    for i in range (87):
        fkey = True
        while (fkey):
            
            articles = api.search(q="facebook", fq = {'headline':'facebook'},
                              
                              begin_date="20150930",
                              end_date="20190101",
                              sort="oldest",
                              page = i,
                              )
            if 'fault' not in articles.keys():
                fkey = False
            

            

        allarticles.append(articles)
        
        print(i)

    save_obj(allarticles, 'fb2')


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



