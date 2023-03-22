import numpy as np

class example:
    def __init__(self):
        self.list = []
    def function(self, data):
        mean = self.mean(data)
        self.list.append(mean)
        return mean
    def mean(self, data):
        return np.mean(data)

ex = example()

ex.function(data=[1, 2, 3])
ex.function(data=[1, 2, 9])
ex.function(data=[1, 2, 99])

ex.list
