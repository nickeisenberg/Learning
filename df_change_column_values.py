import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.array([[1,2,3],[1,4,22],[2,43,3],[2,333,444]])

df = pd.DataFrame(x, columns=['col1','col2','col3'])

times = []
for time in df['col1'].values:
    if time not in times:
        times.append(time)

section = []
for time in times:
    section.append(df.loc[df['col1'] == time].values)

df['z'] = np.zeros(len(df.index.values))

print(df)

success_indicator = []
for time in times:
    scores = df.loc[df['col1'] == time]['col2'].values

    if np.sum(scores) == 6:
        for i in range(len(scores)):
            success_indicator.append(1)
    else:
        for i in range(len(scores)):
            success_indicator.append(0)

df['z'] += np.array(success_indicator)



print(df)
