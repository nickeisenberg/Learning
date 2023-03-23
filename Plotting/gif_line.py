import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
y = np.sqrt(5 / 1000) * np.cumsum(np.random.normal(size=999))
y = np.hstack((0, y))

frames = [go.Frame(data=[go.Scatter(x=x[: 500], y=y[:500])]),
          go.Frame(data=[go.Scatter(x=x, y=y)])]

frames = []
for i in range(101):
    frames.append(go.Frame(data=[go.Scatter(x=x[: 10 * i], y=y[: 10 * i])]))
fig = go.Figure(
    data=[go.Scatter(x=x, y=y)],
    layout=go.Layout(
        xaxis=dict(range=[0, 5], autorange=False),
        yaxis=dict(range=[-5, 5], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=frames
)

fig.show()

