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
    len_thresh = signal.shape[0] // period_len
    signal = signal[: period_len * len_thresh].reshape((-1, period_len))
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

#--------------------------------------------------

# generate some data
time =np.linspace(0, 1, 100000)
desired_freqs_0 = np.linspace(1, 10, 250)
desired_freqs_1 = desired_freqs_0[::-1]
desired_freqs = np.hstack((desired_freqs_1, desired_freqs_0))
desired_freqs = np.hstack((desired_freqs, desired_freqs))

data = []
for i, t in enumerate(time[::100]):
    time_i = time[i: i + 100]
    a, b = time_i[0], time_i[-1]
    d_freq = desired_freqs[i]
    data.append(np.sin((2 * np.pi * d_freq / (b - a)) * (time_i - a)))
data = np.array(data).reshape(-1)

# plot the spectrogram
freqs, ffts = spectrogram(data, period_len=100)
fig, ax = plot_spectrogram_mpl(freqs, ffts)

plt.show()

# slow way for the spectrogram. Wont work well with large data. But I believe
# spect will be needed to extracrt the frequency data.
spect = []
for i, fft in enumerate(ffts):
    for freq, f in zip(freqs, fft):
        spect.append([time[::100][i], freq, f])
spect = np.array(spect)

plt.scatter(spect[:,0], spect[:,1], c=spect[:,2])
plt.show()

# need to find a way to extract this line.
arg_maxes = np.argmax(ffts, axis=1)
freq_max=freqs[arg_maxes]

plt.scatter(spect[:,0], spect[:,1], c=spect[:,2])
plt.plot(time[::100], freq_max, color='black', linewidth=2)
plt.show()

