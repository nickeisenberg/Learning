import pandas as pd
import numpy as np

print('--------------------------------------------------')

x = []
shot0 = pd.Series(np.array([1,2,3]),
                 index=['metric1', 'metric2', 'metric3'],
                 name='shot0')
shot1 = pd.Series(np.array([1,2]),
                 index=['metric1', 'metric3'],
                 name='shot1')
x.append(shot0)
x.append(shot1)

print('--------------------------------------------------')
print(pd.DataFrame(x))

print('--------------------------------------------------')
print(pd.DataFrame(x).T)

