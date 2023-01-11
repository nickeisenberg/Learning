import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.signal import find_peaks, peak_widths
from sklearn.cluster import KMeans
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

time = np.linspace(-1, 1, 1000)
delta = time[1] - time[0]

no_of_groups = 5
group_size = 30
group_centers = [0, .1, -.1, .1, .2]
group_scales = [.15, .18, .1, .09, .11]
# group_heights = np.random.normal(1, .01, size=3)
group_heights = [1, .9, 1.0, 1.1, 1.2]

center_perts = {}
scale_perts = {}
for i in range(no_of_groups):
    center_perts[i] = np.random.normal(0, .01, group_size)
    scale_perts[i] = np.random.normal(0, .01, group_size)

groups = []
for i in range(no_of_groups):
    mu = group_centers[i]
    std = group_scales[i]
    data = []
    for mu_pert, std_pert in zip(center_perts[i], scale_perts[i]):
        mu_new = mu + mu_pert
        std_new = std + std_pert
        signal = np.array(norm(mu_new, std_new).pdf(time))
        signal /= np.max(signal)
        signal *= group_heights[i]
        signal += np.sqrt(delta * .2) * np.random.normal(0, 1, size=time.shape[0])
        data.append(signal)
    groups.append(pd.DataFrame(
        data=data, index=np.array([i for k in range(group_size)])
        )
                  )

groups_df = pd.concat(groups, ignore_index=False)

plots = []
for inp in groups_df.values:
    plots.append(
            go.Scatter(
                x=time, y=inp)
            )
fig = go.Figure(plots)
_ = fig.update_layout(showlegend=False)
fig.show()

group_identifiers = []
centered_plots = []
for signal in groups_df.values:
    sig_peaks = find_peaks(signal,
                           rel_height=.99,
                           prominence=.5)
    sig_height = signal[sig_peaks[0][0]]
    sig_width = peak_widths(signal,
                            peaks=sig_peaks[0],
                            rel_height=.5)[0][0] * delta
    group_identifiers.append((sig_height, sig_width))
    centered_plots.append(signal[sig_peaks[0][0] - 300: sig_peaks[0][0] + 300])
group_identifiers = np.array(group_identifiers)
centered_plots = np.array(centered_plots)

plots_centered = []
for sig in centered_plots:
    plots_centered.append(
            go.Scatter(
                x=np.linspace(0, sig.shape[0] - 1, sig.shape[0]),
                y=sig))
fig = go.Figure(plots_centered)
_ = fig.update_layout(showlegend=False)
fig.show()

scatter_id = []
scatter_id.append(go.Scatter(
    x=group_identifiers[:, 0], y=group_identifiers[:, 1], mode='markers',
    marker={'size': 15}))
fig = go.Figure(scatter_id)
_ = fig.update_layout(showlegend=False)
_ = fig.update_xaxes(title={'text': 'Peak height'})
_ = fig.update_yaxes(title={'text': 'Peak width at half-max'})
fig.show()

kmeans = KMeans(n_clusters=no_of_groups).fit(group_identifiers)
labels = kmeans.labels_

scatter_id_labeled = []
for i, label in enumerate(labels):
    scatter_id_labeled.append(
            go.Scatter(
                x=[group_identifiers[i][0]], y=[group_identifiers[i][1]],
                mode='markers', marker_color=DEFAULT_PLOTLY_COLORS[label],
                marker={'size': 15}))
fig = go.Figure(scatter_id_labeled)
_ = fig.update_layout(showlegend=False)
_ = fig.update_xaxes(title={'text': 'Peak height'})
_ = fig.update_yaxes(title={'text': 'Peak width at half-max'})
fig.show()

labeled_groups = []
for label, signal in zip(labels, groups_df.values):
    labeled_groups.append(
            pd.Series(data=signal, name=label))
labeled_groups_df = pd.DataFrame(labeled_groups)

labeled_groups_centered = []
for label, signal in zip(labels, centered_plots):
    labeled_groups_centered.append(
            pd.Series(data=signal, name=label))
labeled_groups_centered_df = pd.DataFrame(labeled_groups_centered)

plots_labeled = []
for k, v in zip(labeled_groups_df.index, labeled_groups_df.values):
    color = DEFAULT_PLOTLY_COLORS[k]
    plots_labeled.append(
            go.Scatter(x=time, y=v,
                       mode='lines',
                       marker_color=color)
            )
fig = go.Figure(plots_labeled)
_ = fig.update_layout(showlegend=False)
fig.show()

plots_labeled_centered_one_plot=[]
for k, v in zip(labeled_groups_centered_df.index, labeled_groups_centered_df.values):
    color = DEFAULT_PLOTLY_COLORS[k]
    plots_labeled_centered_one_plot.append(
            go.Scatter(x=time, y=v,
                       mode='lines',
                       marker_color=color)
            )
fig = go.Figure(plots_labeled_centered_one_plot)
_ = fig.update_layout(showlegend=False)
fig.show()

plots_labeled_centered = {}
for label in list(set(labels)):
    plots_labeled_centered[label] = []

for k, v in zip(labeled_groups_centered_df.index,
                labeled_groups_centered_df.values):
    color = DEFAULT_PLOTLY_COLORS[k]
    plots_labeled_centered[k].append(
            go.Scatter(x=time, y=v,
                       mode='lines',
                       marker_color=color)
            )

for k, v in plots_labeled_centered.items():
    fig = go.Figure(plots_labeled_centered[k])
    fig.show()

plots_for_subplots = []
plots_for_subplots += [plots_labeled_centered_one_plot]
for k, p in plots_labeled_centered.items():
    plots_for_subplots += [plots_labeled_centered[k]]

fig = make_subplots(rows=2, cols=3, shared_yaxes=True)
for i in range(1, 7):
    row, col = int(np.ceil(i / 3)), int(i - 3 * (np.ceil(i / 3) - 1))
    for plot in plots_for_subplots[i - 1]:
        _ = fig.add_trace(plot,
                row=row, col=col)
_ = fig.update_layout(
        title={'text': 'Time series clustering',
               'x': .5},
        showlegend=False)
fig.show()


