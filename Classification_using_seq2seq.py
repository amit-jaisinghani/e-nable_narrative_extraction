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


def get_decoder_inputs(label_data, num_decoder_tokens):
    decoder_input_data = np.zeros((len(label_data), 2, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(label_data), 2, num_decoder_tokens), dtype='float32')
    for i, (input_data, target_data) in enumerate(zip(decoder_input_data, decoder_target_data)):
        input_data[0][0] = 1
        input_data[1] = label_data[i]

        target_data[0] = label_data[i]
        target_data[1][9] = 1

    return decoder_input_data, decoder_target_data


# main script
max_encoder_seq_length = 500
num_decoder_tokens = 10

# fetch data sets
training_dataset, test_dataset = get_train_validate_test_dataset(max_encoder_seq_length)

# tokenize all content
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_dataset['content'])
tokenizer.fit_on_texts(test_dataset['content'])
num_encoder_tokens = len(tokenizer.word_index) + 1

# fetch encoder input data and decoder input data
encoder_input_data = get_encoded_padded_content(tokenizer, training_dataset['content'], max_encoder_seq_length)
decoder_input_data, decoder_target_data = get_decoder_inputs(training_dataset["labels"].values, num_decoder_tokens)

# print(encoder_input_data[0])
# print(decoder_input_data[0])

model, encoder_model, decoder_model = get_model(num_encoder_tokens, num_decoder_tokens, tokenizer, sys.argv[1],
                                                encoder_input_data, decoder_input_data, decoder_target_data)


encoder_input_test_data = get_encoded_padded_content(tokenizer, test_dataset['content'], max_encoder_seq_length)
decoder_input_test_data, decoder_target_test_data = get_decoder_inputs(test_dataset["labels"].values, num_decoder_tokens)


def decode_sequence(seq_input):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(seq_input)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, 0] = 1
    print("Input", target_seq)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    count = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        print("Output tokens:", output_tokens)
        # Sample a token
        sampled_token = np.rint(output_tokens)

        print("Sampled:", sampled_token)
        count = count + 1
        # Exit condition: either hit max length
        # or find stop character.
        if sampled_token[0, 0, 9] > 0 or count > 1:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = sampled_token

        # Update states
        states_value = [h, c]


for seq_index in range(len(encoder_input_test_data)):
    # # Take one sequence (part of the training set)
    # # for trying out decoding.
    input_seq = encoder_input_test_data[seq_index: seq_index + 1]
    print("For Input", seq_index, ":")
    decode_sequence(input_seq)
    print("")
