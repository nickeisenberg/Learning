import pandas as pd
import numpy as np

data = np.cumsum(np.ones((3, 3)), axis=0)

df = pd.DataFrame(data)

df_sample = df.sample(frac=1 / 3)

