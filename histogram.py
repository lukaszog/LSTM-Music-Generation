import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from collections import Counter


data = pickle.load(open("dataset/folk_music.digits", "rb"))
data_notes = pickle.load(open("dataset/folk_music.notes", "rb"))
# print(data)
data = np.array(data, dtype=np.int64).astype(np.int64)
plt.boxplot(data)
plt.show()

clean_data = data[data < 60000]
plt.boxplot(clean_data)
plt.show()
pickle.dump(clean_data, open("dataset/folk_music_clean.bits", "wb"))

# A = Counter(data)
# for key, cnts in list(A.items()):   # list is important here
#     if cnts < 100:
#         del A[key]

# data = data[0:1000]
# print(data_notes[30:50])
# print(data[0:100])
print(Counter(data))
plt.hist(clean_data)
plt.title('')
plt.show()
exit()
data = data.reshape(1, -1)
scaler = MinMaxScaler(feature_range=(0, 1))
note_data = scaler.fit_transform(data.astype(np.int64))
plt.hist(note_data)
plt.title('Po normalizacji')
plt.show()