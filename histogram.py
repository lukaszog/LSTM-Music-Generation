import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from collections import Counter
from scipy import stats


data = pickle.load(open("dataset/folk_music.digits", "rb"))
# data_notes = pickle.load(open("dataset/folk_music.notes", "rb"))
# print(Counter(data_notes))
data = np.array(data, dtype=np.int64).astype(np.int64)
plt.boxplot(data)
plt.show()
remove_data_dict = {49224: 93, 49296: 93, 8194: 92, 8196: 92, 33026: 92, 33284: 92, 33281: 91, 49168: 89, 49160: 89, 34820: 85, 51204: 82, 33056: 82, 33344: 82, 51264: 81, 33794: 81, 49665: 78, 18688: 78, 16528: 74, 16456: 74, 33824: 72, 50178: 72, 49408: 68, 49664: 68, 49668: 67, 49410: 67, 49280: 63, 49220: 61, 34824: 60, 49186: 59, 50208: 57, 49728: 50, 49440: 50, 18436: 45, 49156: 43, 49154: 43, 49184: 41, 49218: 39, 49284: 36, 17410: 35, 26624: 34, 51232: 34, 32778: 33, 32788: 33, 32960: 31, 32864: 31, 25600: 28, 49169: 27, 35328: 26, 34048: 26, 17416: 26, 50432: 26, 51712: 26, 18448: 26, 18440: 25, 18496: 22, 33088: 22, 33408: 22, 16960: 22, 32897: 22, 49732: 21, 16418: 21, 50192: 21, 16452: 21, 49442: 21, 32808: 20, 32848: 20, 16672: 20, 51344: 18, 49162: 18, 50248: 18, 49172: 18, 51201: 17, 19456: 17, 26688: 17, 16642: 16, 16900: 16, 52224: 15, 16897: 15, 26880: 15, 18944: 14, 49696: 14, 17440: 14, 49281: 14, 49217: 14, 17664: 14, 49424: 14, 49161: 14, 34817: 14, 26628: 13, 26656: 12, 32777: 12, 49192: 11, 26752: 11, 49232: 11, 16912: 11, 16648: 11, 25664: 11, 16704: 10, 32773: 10, 51202: 10, 17024: 10, 16449: 10, 34848: 10, 16516: 9, 16450: 9, 42048: 9, 41986: 9, 25602: 9, 32833: 9, 16513: 9, 43136: 9, 43012: 9, 35840: 8, 16928: 8, 32774: 8, 32780: 8, 16656: 8, 25632: 8, 16401: 7, 33025: 7, 33282: 7, 49666: 7, 26632: 7, 16576: 7, 16480: 7, 10496: 6, 49792: 6, 24832: 6, 49472: 6, 25088: 6, 49153: 6, 25089: 6, 33288: 5, 32786: 5, 49672: 5, 18464: 5, 49409: 5, 33028: 5, 16389: 4, 43072: 4, 16404: 4, 16394: 4, 49170: 4, 32898: 4, 26626: 4, 16393: 4, 49288: 4, 33040: 4, 33312: 4, 8193: 4, 50688: 3, 43010: 3, 49744: 3, 16464: 3, 16424: 3, 32804: 3, 49412: 3, 49448: 3, 49185: 3, 49157: 3, 9280: 2, 50180: 2, 33808: 2, 33864: 2, 24864: 2, 17536: 2, 49920: 2, 34960: 2, 49200: 2, 25616: 2, 26625: 2, 8832: 2, 34850: 2, 10368: 2, 11264: 2, 8512: 2, 33152: 2, 25152: 2, 49248: 2, 49233: 1, 10752: 1, 8264: 1, 35080: 1, 35112: 1, 49418: 1, 34304: 1, 25216: 1, 17088: 1, 19072: 1, 51716: 1, 16736: 1, 17728: 1, 50434: 1, 51240: 1, 33940: 1, 9472: 1, 33536: 1, 33796: 1, 24896: 1, 17424: 1, 8336: 1, 49684: 1}
clean_data = data[data < 60000]
plt.boxplot(clean_data)
plt.show()
pickle.dump(clean_data, open("dataset/folk_music_clean.digits", "wb"))
print(Counter(clean_data))

remove = remove_data_dict.keys()

for r in remove:
    clean_data = clean_data[clean_data!=r]

print(Counter(clean_data))
print(clean_data.shape)
clean_data = clean_data.reshape(-1, 1)
print(clean_data[0:200])
# print(data_notes[0:5])
# A = Counter(data)
# for key, cnts in list(A.items()):   # list is important here
#     if cnts < 100:
#         del A[key]

# data = data[0:1000]
# print(data_notes[30:50])
# print(data[0:100])
# print(Counter(data))
pickle.dump(clean_data, open("dataset/folk_music_remove_small_values.digis", "wb"))

plt.plot(clean_data)
plt.show()

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