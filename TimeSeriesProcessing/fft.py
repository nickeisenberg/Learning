import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# set up the spectrogram
def spectrogram(signal, period_len=16):
    freqs = np.fft.rfftfreq(period_len, d=1 / period_len)
    len_thresh = signal.shape[0] // 16
    signal = signal[: period_len * len_thresh].reshape((-1, 16))
    ffts = np.abs(np.fft.rfft(signal, axis=1))
    return freqs, ffts

def plot_specrogram_ptly(freqs, ffts, show=False):
    scatter_pairs = []
    for t, fft in enumerate(ffts):
        for freq, fft_val in zip(freqs, fft):
            scatter_pairs.append([t, freq, fft_val])
    scatter_pairs = np.array(scatter_pairs)
    scatter_df = pd.DataFrame(data=scatter_pairs,
                              columns=['time', 'freq', 'fft'])
    fig = px.scatter(scatter_df,
                  x='time',
                  y='freq',
                  color='fft',
                  color_continuous_scale=px.colors.sequential.Viridis)
    _ = fig.update_layout(
            title={'text': 'Spectrogram'})
    if show:
        return fig.show()
    else:
        return fig

def plot_spectrogram_mpl(freqs, ffts, show=False):
    t = np.linspace(0, ffts.shape[0] - 1, ffts.shape[0])
    xx, yy = np.meshgrid(t, freqs)
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, ffts.T,
                  shading='auto',
                  norm=colors.Normalize(ffts.min(), ffts.max()),
                  cmap=plt.cm.viridis)
    if show:
        return plt.show()
    else:
        return fig, ax

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

xx, yy = np.meshgrid(x, y)

#--------------------------------------------------

# test the functions
time =np.linspace(0, 10000, 10001)
data = np.sin(1 / 2 * np.pi * time)

freqs, ffts = spectrogram(data)

fig, ax = plot_spectrogram_mpl(freqs, ffts)
plt.show()

#  view the data
fft = np.abs(np.fft.rfft(data))
freqs = np.fft.rfftfreq(time.shape[0])

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fft,
            )
        )
_ = fig.update_xaxes(
        title={
            'text': 'text',
            })
_ = fig.update_yaxes(
        title={
            'text': 'text',
            })
_ = fig.update_layout(
        title={
            'text': 'text',
            })
fig.show()
