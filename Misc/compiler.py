import pandas as pd
import numpy as np

x = np.array([1, 1, 1])
y = np.array([2, 2, 2])

h = np.hstack((x, y))

print(type(h))
print(h.shape[0])
