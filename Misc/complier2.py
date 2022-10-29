import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

x0 = 0 * np.ones(3)
x1 = 1 * np.ones(3)
x2 = 2 * np.ones(3)

print(x0)
print(x0.T)
print(x0.reshape((-1,1)).T)
print(x0.reshape((-1,1)).T[0])

from sklearn.preprocessing import MinMaxScaler
Mm = MinMaxScaler(feature_range=(0,1))
data_scaled = Mm.fit_transform(data.reshape((-1,1))).T[0]
