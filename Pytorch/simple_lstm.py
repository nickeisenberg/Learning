import torch
from torch import nn

input_size = 20

class lstm_nn(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.reshape((-1, 1))
        print(x.shape)
        x, _ = nn.LSTM(1, 128)(x)
        print(x.shape)
        x, _ = nn.LSTM(128, 64)(x)
        print(x.shape)
        x = x[-1, :].reshape(-1)
        print(x.shape)
        x = x.repeat(input_size, 1)
        print(x.shape)
        x, _ = nn.LSTM(64, 64)(x)
        print(x.shape)
        x, _ = nn.LSTM(64, 128)(x)
        print(x.shape)
        x = nn.Linear(input_size * 128, input_size)(x.reshape(-1))
        print(x.shape)

input = torch.randn(1, input_size)
output = lstm_nn()(input)

