import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from Model import get_model
from Preprocessing import get_train_validate_test_dataset


def tokenize_content(content):
    t = Tokenizer()
    t.fit_on_texts(content)
    number_of_tokens = len(t.word_index) + 1
    return number_of_tokens, t


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
BATCH_SIZE = 1
EPOCHS = 4
MAX_LENGTH = 1000
num_decoder_tokens = 10
max_decoder_seq_length = 11

training_dataset, validate_dataset, test_dataset = get_train_validate_test_dataset(MAX_LENGTH)

num_encoder_tokens, training_dataset_tokenizer = tokenize_content(training_dataset['content'])
training_padded_docs = get_encoded_padded_content(training_dataset_tokenizer, training_dataset['content'], MAX_LENGTH)

decoder_input_data = np.zeros((len(training_dataset), num_decoder_tokens, max_decoder_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(training_dataset), num_decoder_tokens, max_decoder_seq_length), dtype='float32')

for i, inp in enumerate(training_dataset['labels']):
    for t, char in enumerate(inp):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if char == 0:
            decoder_input_data[i, t, 0] = 1
        else:
            decoder_input_data[i, t, t+1] = char
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1] = decoder_input_data[i, t]


# print(training_dataset['labels'].shape)
# print(decoder_input_data.shape)
# print(decoder_target_data.shape)
model, encoder_model, decoder_model = get_model(num_encoder_tokens, max_decoder_seq_length, training_dataset_tokenizer, sys.argv[1],
                                                LATENT_DIM)

# Compile & run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([training_padded_docs, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)
