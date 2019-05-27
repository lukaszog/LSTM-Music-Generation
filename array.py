import numpy as np

a = np.array([23, 21, 31, 1, 23, 11, 10])
remove = [2,10, 20]
for r in remove:
    a = a[a > r]

print(a)