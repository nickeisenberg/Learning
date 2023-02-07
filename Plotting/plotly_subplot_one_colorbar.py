import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

#--------------------------------------------------
no_sigs = 9
cmap = plt.get_cmap('viridis', no_sigs)
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
plotly_cmap = matplotlib_to_plotly(cmap, 9)

data = np.cumsum(np.sqrt(1 / 1000) * np.random.normal(size=(9, 1000)), axis=1)

plots = []
for i in range(1, 4):
    for j in range(1, 4):
        plots.append((i, j))
subplot_fig = make_subplots(3, 3)
norm = mplc.Normalize(vmin=0, vmax=no_sigs)
for i, p in enumerate(plots):
    _  = subplot_fig.add_trace(
            go.Scatter(x=np.linspace(0, 1, 1000),
                       y=data[i],
                       showlegend=False,
                       line={'color': f'rgba{cmap(norm(i))}'}),
                       row=p[0], col=p[1])
colorbar_trace = go.Scatter(x=[None],
                            y=[None],
                            mode='markers',
                            marker=dict(
                                colorscale=plotly_cmap, 
                                showscale=True,
                                cmin=-5,
                                cmax=5,
                                colorbar=dict(thickness=15, tickvals=[-5, 5],
                                              ticktext=['', ''],
                                              title='Shot Number',
                                              outlinewidth=0)
                             ),
                            hoverinfo='none',
                            showlegend=False
                            )
_ = subplot_fig.add_trace(colorbar_trace)
subplot_fig.show()
