from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import os
import time


# open and read text file to be learned
# fname = 'facebook_data_taiyipan_shuffled.txt' # swap this text file for anything you like
# text = open(fname, mode = 'r', encoding = 'utf-8').read()
# print('Training text: {}'.format(fname))
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')

print('Length of training text: {} chars'.format(len(text)))

# unique chars
vocab = sorted(set(text))
print('Unique chars: {}'.format(len(vocab)))

# process text
# vectorize text
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# build model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                 batch_input_shape = [batch_size, None]),
        tf.compat.v1.keras.layers.CuDNNLSTM(rnn_units,
                                            return_sequences = True,
                                            stateful = True,
                                            recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.compat.v1.keras.layers.CuDNNLSTM(rnn_units,
                                            return_sequences = True,
                                            stateful = True,
                                            recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# create a new model and load weights
checkpoint_path = 'trained_weights/shakespeare'
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(checkpoint_path)
model.build(tf.TensorShape([1, None]))
model.summary()

# prediction loop
def generate_text(model, start_string, temp):
    # evaluation step: generate text using the learned model

    # number of chars to generate
    num_generate = 10000

    # convert start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    # expand dimension to accommodate batch dimension
    input_eval = tf.expand_dims(input_eval, 0)

    # empty string to store results
    text_generated = []

    # low temperature for more predictable text
    # high temperature for more surprising text
    temperature = temp

    # here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # use categorical distribution to predict the word returned by model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()

        # pass predicted word as next input to model
        # along with previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # add generated text to result
        text_generated.append(idx2char[predicted_id])

    # return result string
    return start_string + ''.join(text_generated)

# generate example text
# for temp in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}:
#     print('------------------{}------------------'.format(temp))
#     print(generate_text(model, start_string = 'JULIET: ', temp = temp))
#     print('---------------------------------------------------------------')

outfile = open('shakespeare_output.txt', 'w')
output = generate_text(model, start_string = 'JULIET:\n', temp = 0.6)
outfile.write(output)
outfile.close()













#
