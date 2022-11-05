import numpy as np

x = np.empty(0)
x = np.append(x, [1, 2.4]).reshape((1,2))
print(np.abs(x[:,0] - 1))
print('')

x = np.vstack((x, [2, 4.4]))
print(x)
