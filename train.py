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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os
import utils
import matplotlib.pyplot as plt
from keras import regularizers

SEQ_LEN = 50

results = utils.create_results_dir()

data = pickle.load(open("dataset/folk_music_803_tune15_order.digits", "rb"))
data = np.array(data)
# data = data[0:100000]
import random
# data = []
# for i in range(0, 900000):
#     data.append(random.uniform(1, 50000))
# data = np.array(data)
# print(data.shape)
print("Rozmiar danych: ", len(data))
data = data.reshape(-1, 1)
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
input_data, output_data = utils.create_dataset(data, SEQ_LEN)

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


# print(X.shape)
# print(y.shape)
# data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# SEQ_LEN = 3
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
# reshape into X=t and Y=t+1
trainX, trainY = utils.create_dataset(data, SEQ_LEN)
testX, testY = utils.create_dataset(data, SEQ_LEN)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print(trainX[0:5])


model = Sequential()
model.add((LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False,
                 kernel_regularizer=regularizers.l2(0.01),
                # activity_regularizer=regularizers.l1(0.01)
)))
# model.add(LSTM(4))
# model.add(Dropout(0.5))
model.add(Dense(1))
# model.add(Activation('linear'))
model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
callbacks_list = utils.model_callbacks(results)

utils.save_model_to_json(model, results)
utils.logging('Model saved to file: {}/{}'.format(results, 'model.json'))
history = model.fit(trainX, trainY,
                    callbacks=callbacks_list,
                    validation_split=0.20,
                    # validation_data=(testX, testY),
                    epochs=100,
                    batch_size=1,
                    verbose=1,
                    shuffle=True
                    )
# x_input = np.array([70, 80, 90])
# x_input = x_input.reshape((1, 3, 1))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
utils.generate_final_plots(history, results)
