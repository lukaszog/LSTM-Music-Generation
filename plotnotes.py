import matplotlib.pyplot as plt
import pickle
import numpy as np

data = pickle.load(open("dataset/folk_music_803.digits", "rb"))
print(len(data))
data = data[0:103664]

plt.plot(data)
plt.show()