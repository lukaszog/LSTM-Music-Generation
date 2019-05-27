from sklearn.preprocessing import MinMaxScaler
import numpy as np
data = np.array([[536936448],
 [33619968],
 [536903680],
 [37133056],
 [33685504],
 [134479872],
 [16908288],
 [2147516416],
 [33816576],
 [4325376]])

print(data)
scaler = MinMaxScaler(feature_range=(0, 1))
print(scaler.fit_transform((data)))

X = scaler.fit_transform(np.array(data))

inverse = scaler.inverse_transform(X)
print(np.array(inverse,dtype='int'))