import numpy as np
import plotly.graph_objects as go

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(x=x, y=y))
_ = fig.add_trace(
        go.Scatter(x=x, y=y,
                   line=dict(color='red'),
                   showlegend=False)
        )
fig.show()
