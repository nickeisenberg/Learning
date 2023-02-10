import torch
import torch.nn as nn
from keras.utils import timeseries_dataset_from_array as tsd
import plotly.graph_objects as go
from math import pi

# create a signal to encode and decode.
time = torch.linspace(0, 1, 1000)
signal = torch.sin(2 * np.pi * time * 4)
signal += torch.sqrt(torch.tensor(1 / 1000)) * torch.randn(1000)

# view the signal
fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
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

# create the dataset for training.
dataset_keras = tsd(data=signal,
                    targets=None,
                    sequence_stride=1,
                    sequence_length=10,
                    batch_size=None)

dataset_torch = torch.empty(10)
for d in dataset_keras:
    dataset_torch = torch.vstack((dataset_torch, torch.tensor(d.numpy())))
dataset_torch = dataset_torch[1:,:]

# define the model
class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
                nn.LSTM(10, 64),
                nn.LSTM(64, 32),
                nn.LSTM(32, 1),



