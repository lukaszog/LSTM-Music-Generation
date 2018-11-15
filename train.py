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


results = utils.create_results_dir()

input_data = pickle.load(open("input", "rb"))
output_data = pickle.load(open("output", "rb"))

X = np.array(input_data)
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = np.array(output_data).reshape((len(output_data), 1))

print(X.shape)
print(X)
print(y.shape)
print(y)

model = Sequential()
model.add(LSTM(
    128,
    input_shape=(X.shape[1], 4),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('relu'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

callbacks_list = utils.model_callbacks(results)

history = model.fit(X, y,
                    callbacks=callbacks_list,
                    validation_split=0.33,
                    epochs=100,
                    batch_size=1,
                    verbose=1,
                    )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("acc_history.png")
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("history_loss.png")
