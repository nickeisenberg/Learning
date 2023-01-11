import numpy as np


class Stats:

    def __init__(self, data=None):
        self.data = data

    def mean_with_weights(self, weights=None):
        weighted_array = self.data * weights
        return np.mean(weighted_array)


array = np.array([1, 2, 3])
weights = np.array([.1, .7, .2])

sts = Stats(data=array)
weighted_mean = sts.mean_with_weights(weights=weights)

print(array.mean())
print(weighted_mean)

