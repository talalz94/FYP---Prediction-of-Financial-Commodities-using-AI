import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import argparse
import SentimentAnalyzerFramer as saf
import tensorflow as tf
from keras import backend as K
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.porter import PorterStemmer
#from keras_lstm_model_builder import KerasLstmModelBuilder
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import matplotlib.pyplot as plt
import operator
import numpy as np
from keras.utils.vis_utils import plot_model
import os
#import train_lstm as tls

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


_USE_PRE_TRAINED_GLOVE_EMBEDDING = True
porter = PorterStemmer()
##
##print('Loading Dataset ... \n')
##df = saf.frameAll()
###df = df.sample(frac=1)
##
##def word2Num(df):
##    c=0
##    lst1 = []
##    lst = []
##    for i in df['Headline']:
##
##        lst = lst + word_tokenize(i)
##
##    
##    
##    table = str.maketrans('', '', string.punctuation)
##    stemmed = [porter.stem(word) for word in lst]
##    lowercase = [element.lower() for element in stemmed]
##    stripped = [w.translate(table) for w in lowercase if
##                len(w) > 1]
##        
##        
##    print('done')
##    counts = Counter(stripped)
##    print('done2')
##
##    sx = sorted(counts.items(), key=operator.itemgetter(1))
##
##    sx = sx[::-1]
##
##    cnt=1
##    c={}
##    for i in sx:
##
##        c[i[0]] = (i[1],cnt)
##
##        cnt +=1 
##
##        
##    
##    return c
##
##c = word2Num(df)  
##print('Dataset Loaded ... \n')
##
##def converter(s):
##
##    lst =[]
##    x = word_tokenize(s)
##    table = str.maketrans('', '', string.punctuation)
##    stemmed = [porter.stem(word) for word in x]
##    lowercase = [element.lower() for element in stemmed]
##    stripped = [w.translate(table) for w in lowercase if
##                len(w) > 1]
##
##    for i in stripped:
##
##        if i in c:
##            lst.append(c[i][1])
##
##        else:
##
##            lst.append(0)
##
##    
##
##    return lst
##
##            
##
##       
##
##
##def dd():
##    df["c"] = ""
##
##    c=0
##    for i in df['Headline']:
##
##        g = converter(i)
##        #print('g: ',)
##        df.at[c,'c'] = g
##
##        c+=1
##
##def prep():
##
##    
##    data_x = []
##    
##    for i in df['c']:
##
##        data_x.append(i)
##
##    data_y = []
##
##    for i in df['Change']:
##
##        data_y.append(int(i))
##
##    return data_x, data_y
##
##    
##
##
##    
##
_DEFAULT_GLOVE_DATA_FILE = "data/glove.6B.100d.txt"
_DEFAULT_PRE_TRAINED_EMBEDDING_DIM = 100 # for glove pre-trained embedding
_DEFAULT_EMBEDDING_DIM = 128
##dd()
##
##def recall_m(y_true, y_pred):
##        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
##        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
##        recall = true_positives / (possible_positives + K.epsilon())
##        return recall
##
##def precision_m(y_true, y_pred):
##        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
##        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
##        precision = true_positives / (predicted_positives + K.epsilon())
##        return precision
##
##def f1_m(y_true, y_pred):
##    precision = precision_m(y_true, y_pred)
##    recall = recall_m(y_true, y_pred)
##    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class KerasLstmModelBuilder:

    def __init__(self, word_index=None):
        if word_index is not None:
            self._use_glove_embedding = True
            self._word_index = word_index
            self._embedding_index = {}

            with open(_DEFAULT_GLOVE_DATA_FILE) as glove_data:
                for word_vec in glove_data:
                    values = word_vec.split()
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    self._embedding_index[word] = vec

            print("Loaded {} word vectors from GloVe".format(len(self._embedding_index)))
        else:
            self._use_glove_embedding = False
            self._word_index = None
            self._embedding_index = None

    def build_lstm_with_embedding_model(self, max_num_words, max_seq_len):

##        from keras.preprocessing.sequence import pad_sequences
##        max_features = 20000
##        maxlen = 80 
##        x,y = prep()
##
##        x_train = x[:5500]
##        y_train = y[:5500]
##        x_test = x[5400:]
##        y_test = y[5400:]
##
##        #eturn x_train, y_train, x_test, y_test
##        
##        #print("Number of training sequences is {} \n".format(len(x_train)))
##        #print("Number of testing sequences is {} \n".format(len(x_test)))
##
##        #print("Padding the sequences ...\n")
##        x_train = pad_sequences(x_train, maxlen=maxlen)
##        x_test = pad_sequences(x_test, maxlen=maxlen)
        
        
        if self._use_glove_embedding:
            # prepare embedding matrix
            embedding_num_words = min(max_num_words, len(self._embedding_index) + 1)
            embedding_matrix = np.zeros((embedding_num_words, _DEFAULT_PRE_TRAINED_EMBEDDING_DIM))

            for word, i in self._word_index.items():
                embedding_vector = self._embedding_index.get(word)
                if i < max_num_words:
                    if embedding_vector is None:
                        # Words not found in embedding index will be all-zeros.
                        pass
                    else:
                        embedding_matrix[i] = embedding_vector

            sequence_input = Input(shape=(max_seq_len,), dtype='int32')

            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = Embedding(embedding_num_words, _DEFAULT_PRE_TRAINED_EMBEDDING_DIM,
                                        weights=[embedding_matrix], input_length=max_seq_len, trainable=False)

            embedded_seq = embedding_layer(sequence_input)
            lstm_seq = LSTM(_DEFAULT_PRE_TRAINED_EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5)(embedded_seq)
            output = Dense(1, activation='sigmoid')(lstm_seq)

            model = Model(sequence_input, output)
            

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))

##            loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, batch_size=16)
##            print(loss, accuracy, f1_score, precision, recall)

            return model#,x_train, y_train,x_test, y_test

        else:
            # Training Embedding and LSTM
            sequence_input = Input(shape=(max_seq_len,), dtype='int32')

            embedded_seq = Embedding(max_num_words, _DEFAULT_EMBEDDING_DIM)(sequence_input)
            lstm_seq = LSTM(_DEFAULT_EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5)(embedded_seq)
            output = Dense(1, activation='sigmoid')(lstm_seq)

            model = Model(sequence_input, output)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            #model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))

            #loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, batch_size=16)

            #print(loss, accuracy, f1_score, precision, recall)

            return model#,x_train, y_train,x_test, y_test


