import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

data = pickle.load(open("dataset/tpd_classical.digits", "rb"))
data_notes = pickle.load(open("dataset/tpd_classical.notes", "rb"))
# print(data)
data = np.array(data, dtype=np.int64).astype(np.int64)
# data = data[0:1000]
# print(data_notes[30:50])
# print(data[0:100])
print(data_notes[0:5])
data = data[0:5]
print(data)
plt.hist(data)
plt.title('Przed normalizacją')
plt.show()
exit()
data = data.reshape(1, -1)
scaler = MinMaxScaler(feature_range=(0, 1))
note_data = scaler.fit_transform(data.astype(np.int64))
plt.hist(note_data)
plt.title('Po normalizacji')
plt.show()