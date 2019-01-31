import matplotlib

matplotlib.use('Agg')

import numpy as np
import pickle
from bitarray import bitarray
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras.layers import Activation
from keras.optimizers import *
from keras.activations import *
from keras.layers.advanced_activations import *
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os
import utils
import matplotlib.pyplot as plt
from keras import regularizers

SEQ_LEN = 10

results = utils.create_results_dir()

data = pickle.load(open("ballada.n", "rb"))

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(np.array(data))
input_data, output_data = utils.prepare_seq(data, SEQ_LEN)

X = np.array(input_data)
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = np.array(output_data).reshape((len(output_data), 1))

print(X[0:100])

print(X.shape)
print(X)
print(y.shape)
print(y)

model = Sequential()
model.add(LSTM(
    64,
    input_shape=(X.shape[1], SEQ_LEN),
    return_sequences=False, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)
))
model.add(Dropout(0.3))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

callbacks_list = utils.model_callbacks(results)

utils.save_model_to_json(model, results)
utils.logging('Model saved to file: {}/{}'.format(results, 'model.json'))
history = model.fit(X, y,
                    callbacks=callbacks_list,
                    validation_split=0.33,
                    epochs=50,
                    batch_size=64,
                    verbose=1,
                    )

utils.generate_final_plots(history, results)
