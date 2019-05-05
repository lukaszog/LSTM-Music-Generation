import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

data = pickle.load(open("data_raw", "rb"))

print(data)
data = np.array(data)
print(data[0:100])


plt.hist(data)
plt.title('Przed normalizacją')
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
note_data = scaler.fit_transform(data)
plt.hist(note_data)
plt.title('Po normalizacji')
plt.show()