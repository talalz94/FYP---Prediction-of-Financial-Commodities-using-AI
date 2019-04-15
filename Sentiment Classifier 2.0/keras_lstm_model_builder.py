import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

_DEFAULT_GLOVE_DATA_FILE = "data/glove.6B.100d.txt"
_DEFAULT_PRE_TRAINED_EMBEDDING_DIM = 100 # for glove pre-trained embedding
_DEFAULT_EMBEDDING_DIM = 128


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

            return model

        else:
            # Training Embedding and LSTM
            sequence_input = Input(shape=(max_seq_len,), dtype='int32')

            embedded_seq = Embedding(max_num_words, _DEFAULT_EMBEDDING_DIM)(sequence_input)
            lstm_seq = LSTM(_DEFAULT_EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5)(embedded_seq)
            output = Dense(1, activation='sigmoid')(lstm_seq)

            model = Model(sequence_input, output)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            return model


