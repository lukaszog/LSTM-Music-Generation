#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import *

tf.enable_eager_execution()
import numpy as np
import os
import time
import functools

path_to_file = 'chopin/chopin.abc'

text = open(path_to_file).read()


print("Dlugosc: ", len(text))
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}

text_as_int = np.array([char2idx[c] for c in text])

idx2char = np.array(vocab)
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))

seq_length = 100

examples_per_epoch = len(text) // seq_length


char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# In[50]:


for input_example, target_example in dataset.take(2):
    print(input_example)
    print(target_example)
    # for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #    print("Step {:4d}".format(i))
    #    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


vocab_size = len(vocab)

embedding_dim = 256
rnn_units = 1024

LSTM = functools.partial(
    tf.keras.layers.LSTM, recurrent_activation='sigmoid')


LSTM = functools.partial(LSTM,
                         return_sequences=True,
                         recurrent_initializer='glorot_uniform',
                         stateful=True
                         )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim,
                  batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)

model.summary()
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

def compute_loss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = compute_loss(target_example_batch, example_batch_predictions)


print("scalar_loss:      ", example_batch_loss.numpy().mean())

optimizer = tf.train.AdamOptimizer()
checkpoint_dir = './traning_checkpoints_abc'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
import util_abc


history = []
plotter = util_abc.PeriodicPlotter(sec=1, xlabel='Iterations', ylabel='Loss')

for epoch in range(5):
    hidden = model.reset_states()
    custom_msg = util_abc.custom_progress_text("Loss: %(loss)2.2f")
    bar = util_abc.create_progress_bar(custom_msg)
    for inp, target in bar(dataset):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            '''TODO: feed the current input into the model and generate predictions'''
            predictions = model(inp)  # TODO
            '''TODO: compute the loss!'''
            loss = compute_loss(target, predictions)  # TODO

        # Now, compute the gradients and try to minimize
        '''TODO: complete the function call for gradient computation'''
        grads = tape.gradient(loss, model.trainable_variables)  # TODO
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update the progress bar!
        history.append(loss.numpy().mean())
        custom_msg.update_mapping(loss=history[-1])
        plotter.plot(history)

    # Update the model with the changed weights!
    model.save_weights(checkpoint_prefix.format(epoch=epoch))
model.save("model_abc.hf5")

# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# In[ ]:
