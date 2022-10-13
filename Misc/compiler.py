import pandas as pd
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([1, 2, 3])
x3 = np.array([1, 2, 3])

x = [x1, x2, x3]

x = np.array(x)

print(x)

print(np.sum(x, axis=0))

y = np.array([[1,1,1],[1,1,1],[1,1,1]])

print(x + y)

