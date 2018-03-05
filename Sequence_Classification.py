from Preprocessing import *

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# load eNable dataset into 3 sets - train(80%), validate(20%) and test(20%)
data_set = loadData()
train, validate, test = getTrainTest(data_set)

# truncate and pad input sequences
max_review_length = 1000
X_train = sequence.pad_sequences(train['x_term'], maxlen=max_review_length)
X_validate = sequence.pad_sequences(validate['x_term'], maxlen=max_review_length)
X_test = sequence.pad_sequences(test['x_term'], maxlen=max_review_length)

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(train['x_term'].shape[0], 1000, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, train['y_term'], validation_data=(X_validate, validate['y_term']), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, test['y_term'], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
