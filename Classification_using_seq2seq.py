import sys
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from Model import get_model
from Preprocessing import get_train_validate_test_dataset


def get_encoded_padded_content(tokenizer, content, max_length):
    # integer encode the documents
    encoded_docs = tokenizer.texts_to_sequences(content)
    # print(encoded_docs)

    # pad documents to a max length of 4 words
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # print(padded_docs)
    return padded_docs


# main script
LATENT_DIM = 100
BATCH_SIZE = 10
EPOCHS = 4
max_encoder_seq_length = 1000
num_decoder_tokens = 8

# fetch data sets
training_dataset, validate_dataset, test_dataset = get_train_validate_test_dataset(max_encoder_seq_length)

# tokenize all content
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_dataset['content'])
tokenizer.fit_on_texts(validate_dataset['content'])
tokenizer.fit_on_texts(test_dataset['content'])
num_encoder_tokens = len(tokenizer.word_index) + 1

# fetch encoder input data and decoder input data
encoder_input_data = get_encoded_padded_content(tokenizer, training_dataset['content'], max_encoder_seq_length)
decoder_input_data = labels = np.expand_dims(np.vstack(training_dataset["labels"].values), 1);

model, encoder_model, decoder_model = get_model(num_encoder_tokens, num_decoder_tokens, tokenizer, sys.argv[1],
                                                LATENT_DIM)

# Compile & run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([encoder_input_data, decoder_input_data], decoder_input_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)