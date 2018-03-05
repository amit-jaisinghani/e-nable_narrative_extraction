from Preprocessing import *

import numpy
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)
# path for glove directory
GLOVE_DIR = "/home/amit/PycharmProjects/eNable_Narrative_Extraction/data/"
# embedding dimension i.e. glove dimension which we used is 100
EMBEDDING_DIM = 100

# load eNable dataset into 3 sets - train(80%), validate(20%) and test(20%)
data_set = loadData()
train, validate, test = getTrainTest(data_set)

# tokenize training set
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_set['content'])
word_index = tokenizer.word_index

# truncate and pad input sequences
max_review_length = 1000
X_train = sequence.pad_sequences(train['x_term'], maxlen=max_review_length)
X_validate = sequence.pad_sequences(validate['x_term'], maxlen=max_review_length)
X_test = sequence.pad_sequences(test['x_term'], maxlen=max_review_length)

# Loaded glove 100d embeddings in embeddings_index
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#  computing embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
          input_length=max_review_length, trainable=False))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, train['y_term'], validation_data=(X_validate, validate['y_term']), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, test['y_term'], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
