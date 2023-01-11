import plotly.graph_objects as go
import numpy as np

time = np.linspace(0, 1, 1000)
data1 = np.sin(2 * np.pi * 3 * time)
data2 = np.cos(2 * np.pi * 7 * time)

fig = go.Figure(
        [go.Scatter(x=time, y=data1),
         go.Scatter(x=time, y=data2)]
        )

_ = fig.update_layout(title_text='title')
_ = fig.update_xaxes(title_text='x-axis')
_ = fig.update_yaxes(title_text='y-axis')

fig.show()


