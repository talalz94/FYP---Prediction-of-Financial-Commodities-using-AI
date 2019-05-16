import argparse
import SentimentAnalyzerFramer as saf
import tensorflow as tf
from keras import backend as K
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.porter import PorterStemmer
from keras_lstm_model_builder import KerasLstmModelBuilder
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import matplotlib.pyplot as plt
import operator
import numpy as np
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


_USE_PRE_TRAINED_GLOVE_EMBEDDING = True
porter = PorterStemmer()
df = pd.DataFrame()
print('Loading Dataset ... \n')
df = saf.frameAll()
#df = df.sample(frac=1)

def word2Num(df):
    c=0
    lst1 = []
    lst = []
    for i in df['Headline']:

        lst = lst + word_tokenize(i)

    
    
    table = str.maketrans('', '', string.punctuation)
    stemmed = [porter.stem(word) for word in lst]
    lowercase = [element.lower() for element in stemmed]
    stripped = [w.translate(table) for w in lowercase if
                len(w) > 1]
        
        
    print('done')
    counts = Counter(stripped)
    print('done2')

    sx = sorted(counts.items(), key=operator.itemgetter(1))

    sx = sx[::-1]

    cnt=1
    c={}
    for i in sx:

        c[i[0]] = (i[1],cnt)

        cnt +=1 

        
    
    return c

c = word2Num(df)  
print('Dataset Loaded ... \n')

def converter(s):

    lst =[]
    x = word_tokenize(s)
    table = str.maketrans('', '', string.punctuation)
    stemmed = [porter.stem(word) for word in x]
    lowercase = [element.lower() for element in stemmed]
    stripped = [w.translate(table) for w in lowercase if
                len(w) > 1]

    for i in stripped:

        if i in c:
            lst.append(c[i][1])

        else:

            lst.append(0)

    

    return lst

            

       


def dd():
    df["c"] = ""

    c=0
    for i in df['Headline']:

        g = converter(i)
        #print('g: ',)
        df.at[c,'c'] = g

        c+=1

def prep():

    
    data_x = []
    
    for i in df['c']:

        data_x.append(i)

    data_y = []

    for i in df['Change']:

        data_y.append(int(i))

    return data_x, data_y

    


    

def train_lstm_model_with_imdb_review(batch_size, epoches):

    
    # prepare the imdb data
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)

    print('Loading imdb review data ...\n')

    
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    
    x,y = prep()

    x_train = x[:5500]
    y_train = y[:5500]
    x_test = x[5450:]
    y_test = y[5450:]

    #eturn x_train, y_train, x_test, y_test
    
    #print("Number of training sequences is {} \n".format(len(x_train)))
    #print("Number of testing sequences is {} \n".format(len(x_test)))

    print("Padding the sequences ...\n")
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    #print("Training Sequence Shape: {} \n".format(x_train.shape))
    #print("Testing Sequence Shape: {} \n".format(x_test.shape))

    #word_index = imdb.get_word_index()
    #print("Number of Words in imdb dataset is {} \n".format(len(word_index)))

    

    # build the model
    #lstm_model_builder = KerasLstmModelBuilder(word_index) if _USE_PRE_TRAINED_GLOVE_EMBEDDING else KerasLstmModelBuilder()
    lstm_model_builder = KerasLstmModelBuilder()

    model = lstm_model_builder.build_lstm_with_embedding_model(max_features, maxlen)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print('Start training...\n')
    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))
    #score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    #loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, batch_size=batch_size)
    #print('Test score:', score)
    #print('Test accuracy:', acc)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    

    return model


class CleanText(BaseEstimator, TransformerMixin):
   
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##
parser = argparse.ArgumentParser(description="Training LSTM Model for Sentiment Analysis with Keras")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")
dd()


if __name__ == '__main__':
    parsed_args = parser.parse_args()
    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Training the lstm model
    m = train_lstm_model_with_imdb_review(batch_size, 10)

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
