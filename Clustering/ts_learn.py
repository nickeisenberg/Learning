import plotly.graph_objects as go
import numpy as np
import tslearn.clustering as tsc
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import norm

# Come up with some test signals 
time = np.linspace(0, 1, 1000)
data = []
for i in range(50):
    if i <= 25:
        std = np.random.uniform(.08, .18, 1)[0]
        scale = np.random.uniform(.9, 1.1, 1)[0]
        dial = np.random.uniform(.9, 1.1, 1)[0]
        data.append(
                scale * np.sin(dial * 2 * np.pi * time * 3) +
                np.random.normal(scale=std, size=1000)
                )
    else:
        std = np.random.uniform(.08, .18, 1)[0]
        scale = np.random.uniform(.9, 1.1, 1)[0]
        dial = np.random.uniform(.9, 1.1, 1)[0]
        data.append(
                scale * np.sin(dial * 2 * np.pi * time * 3) +
                np.random.normal(scale=std, size=1000)
                )
data = np.array(data)

data_inds = np.array([*range(data.shape[0])])
np.random.shuffle(data_inds)
data = data[data_inds]

data[:25, :] *= np.exp(time) / 1.8
data[25:, :] *= np.exp(-time)

data_inds = np.array([*range(data.shape[0])])
np.random.shuffle(data_inds)
data = data[data_inds]
#--------------------------------------------------

# plot the generated data
fig = go.Figure()
for d in data[:10]:
    _ = fig.add_trace(
            go.Scatter(
                y=d)
            )
fig.show()
#--------------------------------------------------

# use tslearn to cluster the data
# dtw takes forever to run
clusters = tsc.TimeSeriesKMeans(
        n_clusters=5,
        metric='dtw'
        ).fit_predict(data)

rows = 2
cols = 3
plots = []
for i in  range(1, rows + 1):
    for j in range(1, cols + 1):
        plots.append((i, j))
subplot_titles = ["" for i in range(rows * cols)]
subplot_titles[-1] = 'Unclustered data'
fig = make_subplots(rows=rows, cols=cols,
                    subplot_titles=subplot_titles)
for i, d in zip(clusters, data):
    _ = fig.add_trace(
            go.Scatter(y=d),
            row=plots[i][0],
            col=plots[i][1]
            )
for d in data:
    _ = fig.add_trace(
            go.Scatter(y=d),
            row=2, col=3)
_ = fig.update_layout(
        title={'text': "tslearn's kmeans time series clustering"}
        )
fig.show()
#--------------------------------------------------
