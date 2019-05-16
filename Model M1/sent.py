classes=['positive','negative']
raw_positive_doc_master = open('positive.txt', 'r')
raw_negative_doc_master = open('negative.txt', 'r')
positive={}
negative={}
master_structure={'positive':positive,'negative':negative}
# Function to Clean the corpu
def clean_corpus(corpus):#Function to clean the corpus
    b = "*_=][{}1234567890!&@#$,.\";:'?/-|+"
    corpus=corpus.lower()# Change to lowercase
    for i in range(0,len(b)):
        corpus=corpus.replace(b[i]," ")
    return " ".join(corpus.split())
positive_documents =raw_positive_doc_master.read().split("\n")
negative_documents = raw_negative_doc_master.read().split("\n")

def count_frequency(doc):
	vector={}
	words=doc.split()
	for word in words:
		if(word not in vector):
			vector[word]=1
		else:
			vector[word]+=1
	return vector

def tokens(dict_):
	tokens=0
	for key in dict_:
		tokens+=dict_[key]
	return tokens

normalized_positive_docs=[]
for doc in positive_documents:
	if(doc!=''):
		normalized_positive_docs.append(clean_corpus(doc))
normalized_negative_docs=[]
for doc in negative_documents:
	if(doc!=''):
		normalized_negative_docs.append(clean_corpus(doc))
#Each line can be considered as individual document
#represent Document in a vector form , so that later look can be made easier
master_structure['positive']['docs']=normalized_positive_docs
master_structure['negative']['docs']=normalized_negative_docs

#Crete Vocabulary Array from normalised positive and negative docs
vocabulary=[]
_map={}
for doc in normalized_negative_docs+normalized_positive_docs:
	words=doc.split()
	for word in words:
		if word not in _map:
			vocabulary.append(word)
			_map[word]=1


# Store the Prior Probability in Master_structure
master_structure['positive']['prior_probablity']=len(master_structure['positive']['docs'])/float(len(master_structure['positive']['docs'])+len(master_structure['negative']['docs']))
master_structure['negative']['prior_probablity']=len(master_structure['negative']['docs'])/float(len(master_structure['positive']['docs'])+len(master_structure['negative']['docs']))

def get_likely_hood(dict_,word):
	if word not in dict_:
		return dict_['unknown']
	else:
		return dict_[word]
def get_count(dict_,word):
	if word not in dict_:
		return 0
	else:
		return dict_[word]
# Create Mega Document for Computing Likelyhood 
# Create a document vector for showing frequencies of each word in the master docs
mega_positive_doc=clean_corpus(open('positive.txt', 'r').read())
mega_negative_doc=clean_corpus(open('negative.txt', 'r').read())
mega_positive_doc_vector=count_frequency(mega_positive_doc)
mega_negative_doc_vector=count_frequency(mega_negative_doc)
total_token_mega_positive_doc=tokens(mega_positive_doc_vector)
total_token_mega_negative_doc=tokens(mega_negative_doc_vector)

likely_hood_probablity_words_positive={}
likely_hood_probablity_words_negative={}
for word in vocabulary:
	#Find frequency of word in mega_positive_doc
	nk_positive=get_count(mega_positive_doc_vector,word)#mega_positive_doc_vector[word]
	alpha=1# Add One smoothing(Laplace smoothing for not seen words) 
	likely_hood_positive=(nk_positive+alpha)/float((total_token_mega_positive_doc+alpha*(len(vocabulary)+1)))
	likely_hood_probablity_words_positive[word]=likely_hood_positive
	likely_hood_probablity_words_positive['unknown']=(alpha)/float((total_token_mega_positive_doc+alpha*(len(vocabulary)+1)))
	nk_negative=get_count(mega_negative_doc_vector,word)#=mega_positive_doc_vector[word]
	likely_hood_negative=(nk_negative+alpha)/float((total_token_mega_negative_doc+alpha*(len(vocabulary)+1)))
	likely_hood_probablity_words_negative[word]=likely_hood_negative
	likely_hood_probablity_words_negative['unknown']=(alpha)/float((total_token_mega_negative_doc+alpha*(len(vocabulary)+1)))

headline = raw_input("Give your headline : ")
#headline="discards concept"
#print likely_hood_probablity_words_positive
normalised_headline=clean_corpus(headline)


positive_probability= master_structure['positive']['prior_probablity']
negative_probability= master_structure['negative']['prior_probablity']
words=normalised_headline.split()
prod_positive=1
prod_nagative=1
for word in words:
	prod_positive*=get_likely_hood(likely_hood_probablity_words_positive,word)
	prod_nagative*=get_likely_hood(likely_hood_probablity_words_negative,word)
	#print get_likely_hood(likely_hood_probablity_words_negative,word)
positive_probability=positive_probability*prod_positive
negative_probability=negative_probability*prod_nagative

print positive_probability,negative_probability
if(positive_probability>negative_probability):
	print "POSITIVE headline"
else:
	print "NEGATIVE headline"







