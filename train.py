import matplotlib

matplotlib.use('Agg')

import numpy as np
import pickle
from bitarray import bitarray
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding, Bidirectional
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

SEQ_LEN = 3

results = utils.create_results_dir()

data = pickle.load(open("dataset/folk_music_clean.digits", "rb"))
data = np.array(data)
print(data.shape)
print("Rozmiar danych: ", len(data))
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(np.array(data))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.astype(np.int64).reshape(-1, 1))
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
print(data[0:10])
input_data, output_data = utils.prepare_seq(data, SEQ_LEN)

# print(input_data[0:1][0][0])

print("Ilosc sekwencji: ", len(input_data))
print("Shape: ", input_data.shape)
# trainX = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

# print(test_data.shape)
# test_data = test_data.reshape((input_data[0].shape[0], 1, input_data[0].shape[1]))
# print(test_data)

X = input_data


# exit()

# X = X.reshape((input_data[0].shape[0], 1, input_data[0].shape[1]))
# y = np.array(output_data).reshape((len(output_data), 1))
y = output_data

# (batch_size, time_steps, seq_len)
# (902898, 3, 1)
# X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
print(X.shape)
print(X[0])
print(X[1])
print(X[2])

# print(X.shape)
# print(y.shape)

train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
# reshape into X=t and Y=t+1
trainX, trainY = utils.prepare_seq(train, SEQ_LEN)
testX, testY = utils.prepare_seq(test, SEQ_LEN)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add((LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True,
)))
# model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
# model.add(Activation('relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
callbacks_list = utils.model_callbacks(results)

utils.save_model_to_json(model, results)
utils.logging('Model saved to file: {}/{}'.format(results, 'model.json'))
history = model.fit(trainX, trainY,
                    callbacks=callbacks_list,
                    validation_data=(testX, testY),
                    epochs=50,
                    batch_size=512,
                    verbose=1,
                    shuffle=True
                    )

utils.generate_final_plots(history, results)
