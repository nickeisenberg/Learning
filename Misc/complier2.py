import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

x0 = 0 * np.ones(3)
x1 = 1 * np.ones(3)
x2 = 2 * np.ones(3)

from sklearn.preprocessing import MinMaxScaler

data = np.array([*range(10)]) 
print(data)
Mm = MinMaxScaler(feature_range=(0,1))
# data_scaled = Mm.fit_transform(data.reshape((-1,1))).T[0]
data_scaled = Mm.fit_transform(data.reshape((-1,1)))
print(data_scaled)
print(data_scaled.reshape((-1)))

