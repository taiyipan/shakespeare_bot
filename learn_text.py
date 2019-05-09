from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import os
import time

# open and read text file to be learned
# fname = 'facebook_data_taiyipan_shuffled.txt' # swap this text file for anything you like
# text = open(fname, mode = 'r', encoding = 'utf-8').read()
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')

# print('Training text: {}'.format(fname))
print('Length of training text: {} chars'.format(len(text)))

# unique chars
vocab = sorted(set(text))
print('Unique chars: {}'.format(len(vocab)))

# process text
# vectorize text
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# max length sentence
seq_length = 100
examples_per_epoch = len(text) // seq_length

# create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

# convert chars to sequences
sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

# for each sequence, duplicate and shift to form input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# create training batches
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

print(dataset)

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

model = build_model(
    vocab_size = len(vocab),
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
    batch_size = BATCH_SIZE
)

# restore weights
checkpoint_path = 'trained_weights/shakespeare'
model.load_weights(checkpoint_path)

# try the model
# shape of output
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')

model.summary()

# define loss
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print('Prediction shape: ', example_batch_predictions.shape, ' # (batch_size, sequence_length, vocab_size)')
print('scalar_loss:      ', example_batch_loss.numpy().mean())

# compile
model.compile(optimizer = 'adam', loss = loss)

# configure checkpoints
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    save_weights_only = True
)

# train model
EPOCHS = 40

history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])
