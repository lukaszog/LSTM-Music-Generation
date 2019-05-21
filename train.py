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

SEQ_LEN = 100

results = utils.create_results_dir()

data = pickle.load(open("dataset/tpd_classical.digits", "rb"))
data = np.array(data)
print(data.shape)
print("Rozmiar danych: ", len(data))
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(np.array(data))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.astype(np.int64).reshape(-1, 1))

input_data, output_data = utils.prepare_seq(data, SEQ_LEN)

# print(input_data[0:1][0][0])

print("Ilosc sekwencji: ", len(input_data))
print("Shape: ", input_data.shape)
# trainX = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

# print(test_data.shape)
# test_data = test_data.reshape((input_data[0].shape[0], 1, input_data[0].shape[1]))
# print(test_data)

X = input_data


# X = X.reshape((input_data[0].shape[0], 1, input_data[0].shape[1]))
# y = np.array(output_data).reshape((len(output_data), 1))
y = output_data

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
print(X.shape)
print(X[0])
print(X[1])
print(X[2])

# print(X.shape)
# print(y.shape)

model = Sequential()
model.add(LSTM(
    256,
    input_shape=(1, SEQ_LEN),
    return_sequences=True, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)
))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

callbacks_list = utils.model_callbacks(results)

utils.save_model_to_json(model, results)
utils.logging('Model saved to file: {}/{}'.format(results, 'model.json'))
history = model.fit(X, y,
                    callbacks=callbacks_list,
                    validation_split=0.33,
                    epochs=50,
                    batch_size=128,
                    verbose=1
                    )

utils.generate_final_plots(history, results)
