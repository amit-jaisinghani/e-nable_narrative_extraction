import sys
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot

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
BATCH_SIZE = 5
EPOCHS = 4
max_encoder_seq_length = 1000
num_decoder_tokens = 8

# fetch data sets
training_dataset, test_dataset = get_train_validate_test_dataset(max_encoder_seq_length)

# tokenize all content
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_dataset['content'])
tokenizer.fit_on_texts(test_dataset['content'])
num_encoder_tokens = len(tokenizer.word_index) + 1

# fetch encoder input data and decoder input data
encoder_input_data = get_encoded_padded_content(tokenizer, training_dataset['content'], max_encoder_seq_length)
decoder_input_data = np.expand_dims(np.vstack(training_dataset["labels"].values), 1)

# print(encoder_input_data[0])
# print(decoder_input_data[0])

model, encoder_model, decoder_model = get_model(num_encoder_tokens, num_decoder_tokens, tokenizer, sys.argv[1],
                                                LATENT_DIM)

# Compile & run training
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr = 0.001), metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([encoder_input_data, decoder_input_data], decoder_input_data,
          batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

encoder_input_test_data = get_encoded_padded_content(tokenizer, test_dataset['content'], max_encoder_seq_length)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

    print("Result")
    print(output_tokens[0][0])


for seq_index in range(len(encoder_input_test_data)):
    # # Take one sequence (part of the training set)
    # # for trying out decoding.
    input_seq = encoder_input_test_data[seq_index: seq_index + 1]
    decode_sequence(input_seq)
