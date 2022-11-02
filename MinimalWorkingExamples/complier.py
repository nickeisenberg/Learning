import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

Mm = MinMaxScaler(feature_range=(0,1))

data=np.array([1,2,3,4]).reshape((-1,1))
data_scaled = Mm.fit_transform(data)

print(Mm.data_min_)

