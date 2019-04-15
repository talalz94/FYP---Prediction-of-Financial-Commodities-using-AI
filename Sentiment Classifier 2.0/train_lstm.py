import argparse

import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
#from keras.datasets import imdb
from sklearn.base import BaseEstimator, TransformerMixin

from keras_lstm_model_builder import KerasLstmModelBuilder

_USE_PRE_TRAINED_GLOVE_EMBEDDING = True





##def train_lstm_model_with_imdb_review(batch_size, epoches):
##    # prepare the imdb data
##    max_features = 20000
##    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
##
##    print('Loading imdb review data ...\n')
##    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
##    print("Number of training sequences is {} \n".format(len(x_train)))
##    print("Number of testing sequences is {} \n".format(len(x_test)))
##
##    print("Padding the sequences ...\n")
##    x_train = pad_sequences(x_train, maxlen=maxlen)
##    x_test = pad_sequences(x_test, maxlen=maxlen)
##    print("Training Sequence Shape: {} \n".format(x_train.shape))
##    print("Testing Sequence Shape: {} \n".format(x_test.shape))
##
##    word_index = imdb.get_word_index()
##    print("Number of Words in imdb dataset is {} \n".format(len(word_index)))
##
##    # build the model
##    lstm_model_builder = KerasLstmModelBuilder(word_index) if _USE_PRE_TRAINED_GLOVE_EMBEDDING else KerasLstmModelBuilder()
##    model = lstm_model_builder.build_lstm_with_embedding_model(max_features, maxlen)
##
##    print('Start training...\n')
##    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoches, validation_data=(x_test, y_test))
##    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
##    print('Test score:', score)
##    print('Test accuracy:', acc)


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


parser = argparse.ArgumentParser(description="Training LSTM Model for Sentiment Analysis with Keras")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Training the lstm model
    train_lstm_model_with_imdb_review(batch_size, epochs)

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
