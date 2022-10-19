
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot, show, subplot

x = np.random.normal(0, 1, 5)
y = np.random.normal(0,4,5)

z = [np.hstack((x,y))]
print(z)

df = pd.DataFrame(z)

print(df)
