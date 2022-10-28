import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
x = x.reshape((x.shape[0], 1))

plt.plot(x)
plt.show()
