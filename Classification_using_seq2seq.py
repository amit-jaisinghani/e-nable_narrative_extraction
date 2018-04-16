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
BATCH_SIZE = 1
EPOCHS = 4
max_encoder_seq_length = 1000

target_characters = dict()
target_characters['\t'] = 0
target_characters['\n'] = 1
target_characters['Report'] = 2
target_characters['Device'] = 3
target_characters['Delivery'] = 4
target_characters['Progress'] = 5
target_characters['becoming_member'] = 6
target_characters['attempt_action'] = 7
target_characters['Activity'] = 8
target_characters['Other'] = 9

rv_target_characters = dict()
rv_target_characters[0] = '\t'
rv_target_characters[1] = '\n'
rv_target_characters[2] = 'Report'
rv_target_characters[3] = 'Device'
rv_target_characters[4] = 'Delivery'
rv_target_characters[5] = 'Progress'
rv_target_characters[6] = 'becoming_member'
rv_target_characters[7] = 'attempt_action'
rv_target_characters[8] = 'Activity'
rv_target_characters[9] = 'Other'

training_dataset, validate_dataset, test_dataset = get_train_validate_test_dataset(max_encoder_seq_length)
num_decoder_tokens = 11
max_decoder_seq_length = max(len(label) for label in training_dataset['labels'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_dataset['content'])
tokenizer.fit_on_texts(validate_dataset['content'])
tokenizer.fit_on_texts(test_dataset['content'])
num_encoder_tokens = len(tokenizer.word_index) + 1

encoder_input_data = get_encoded_padded_content(tokenizer, training_dataset['content'], max_encoder_seq_length)
decoder_input_data = np.zeros((len(training_dataset), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(training_dataset), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, inp in enumerate(training_dataset['labels']):

    for t, char in enumerate(inp):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_characters[char]] = 1
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1] = decoder_input_data[i, t]

# print(training_dataset['labels'][0])
# print(decoder_input_data[0])
# print(decoder_target_data[0])

model, encoder_model, decoder_model = get_model(num_encoder_tokens, num_decoder_tokens, tokenizer, sys.argv[1],
                                                LATENT_DIM)

# Compile & run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)


def decode_sequence(input_seq):
    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_characters['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = rv_target_characters[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# Now Preparing test the data
validate_posts = get_encoded_padded_content(tokenizer, validate_dataset['content'], max_encoder_seq_length)

model_output = []
for seq_index in range(len(validate_dataset['content'])):
    input_seq = validate_posts[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence([input_seq])
    decoded_sentence = " ".join(decoded_sentence.split("\t"))
    print(decoded_sentence)
    model_output.append(decoded_sentence[0])
