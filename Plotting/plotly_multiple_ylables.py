import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

time1 = np.array([1, 2, 3])
time2 = np.array([2, 3, 4])

data = lambda x: x + 1
data1 = data(time1)
data2 = data(time2)

fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(
        go.Scatter(x=time1, y=data1),
        secondary_y=True,)
fig.add_trace(
        go.Scatter(x=time2, y=data2),
        secondary_y=False)
fig.update_layout(
        title_text='double y axis'
        )
fig.update_xaxes(
        title_text='x-axis')
fig.update_yaxes(
        title_text='<b>primary</b> y-axis title', secondary_y=False
        )
fig.update_yaxes(
        title_text='<b>seconday</b> y-axis title', secondary_y=True
        )
fig.show()
