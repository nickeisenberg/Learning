import numpy as np
from scipy.spatial import distance

x = np.array([[1,1],[2,2]])
print(distance.cdist(x, x))
