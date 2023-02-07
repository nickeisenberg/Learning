import pandas as pd
import numpy as np

inds = np.cumsum(np.ones(5))
data = np.cumsum(np.ones((5, 2)), axis=0)

np.random.seed(1)
np.random.shuffle(inds)
inds

np.random.seed(1)
np.random.shuffle(data)
data

df = pd.DataFrame(data=data, index=inds).sort_index()

data = df.values

inds = df.index.values

data

inds
