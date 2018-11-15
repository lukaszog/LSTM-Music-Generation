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
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

data = data.reshape((1, len(data), 1))

print(data.shape)
print(data)

a = np.arange(10).reshape(1, 10, 1)
print(a)

input_data = pickle.load(open("input", "rb"))
output_data = pickle.load(open("output", "rb"))

# print(input_data)
# exit()

# input_n = len(set(input_data))
# output_n = len(set(output_data))

X = np.array(input_data)
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = np.array(output_data).reshape((len(output_data), 1))



print(X.shape)
print(X)
print(y.shape)
print(y)



HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100
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
#
# model.summary()
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

callbacks_list = utils.model_callbacks(results)

history = model.fit(X, y,
                    # validation_data=(X_test, y_test),
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
