import plotly.express as px
import numpy as np
import pandas as pd
from copy import deepcopy

time = np.linspace(0, 1, 1000)
bm_paths = np.sqrt(time[1] - time[0]) * np.random.normal(size=(3, 999))
bm_paths = np.hstack((np.zeros((3, 1)), bm_paths))
bm_paths = np.cumsum(bm_paths, axis=1)
df = pd.DataFrame(
        data=np.vstack((time, bm_paths)).T,
        columns=['time', 'col1', 'col2', 'col3'])

df_stack = []
for i, ind in enumerate(np.arange(10, 1001, 1)):
    df_i = deepcopy(df.iloc[: ind])
    df_i['anim'] = np.ones(df_i.shape[0]).astype(int) * i
    df_stack.append(df_i)
df_stack = pd.concat(df_stack)

fig = px.line(df_stack, x='time', y=['col1', 'col2', 'col3'],
              animation_frame='anim',
              range_x=[0, 1], range_y=[-2, 2])

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 25

fig.show()

