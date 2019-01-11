from nytimesarticle import articleAPI
import pickle


api = articleAPI('10e395709e384996a152bc796f21d20f')
allarticles = []

def ga():
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    
    global allarticles
    
    for i in range (64):
        articles = api.search(q="google", fq = {'headline':'google'},
                          
                          begin_date="20150313",
			  end_date="20190101",
                          sort="oldest",
                          page = i,
                          )

        allarticles.append(articles)
        print(i)

    save_obj(allarticles, 'data2')


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



