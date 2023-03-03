import numpy as np
from keras.utils import timeseries_dataset_from_array

x = np.random.normal(size=(1000, 10))

inds = np.arange(0, 1000, 1)[:-5]
target_inds = np.arange(0, 1000, 1)[9:]

inds.shape
target_inds.shape

x_5_inds = timeseries_dataset_from_array(
        inds,
        targets=target_inds,
        sequence_length=5,
        sequence_stride=1,
        batch_size=None)

for xx in x_5_inds:
    print(xx)
    break

x_5 = []
x_5_targets = []
for ind, tar in x_5_inds:
    x_5.append(ind)
    x_5_targets.append(tar)
x_5 = np.array(x_5)
x_5_targets = np.array(x_5_targets)

x_5[:4]
x_5_targets[:4]

x_5[-4:]
x_5_targets[-4:]
