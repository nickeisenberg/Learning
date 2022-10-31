import numpy as np
from scipy.spatial.distance import cdist

x0 = np.ones((2,2))
x1 = np.ones((2,2)) * np.array([1, 2]).reshape((-1,1))

print(x0)
print(x1)

print(cdist(x0, x1, metric='euclidean'))
