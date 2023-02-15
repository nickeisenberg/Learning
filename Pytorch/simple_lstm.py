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
        X, D = nn.LSTM(64, 128)(x)
        print(X.shape)
        x = nn.Linear(input_size * 128, input_size)(X.reshape(-1))
        print(x.shape)
        return x, X, D

input = torch.randn(1, input_size)
output, X, D = lstm_nn()(input)

input.shape
D[0].shape
X.shape

# I am pretty sure that D[0] is the same thing as return_sequences=False
# from keras
X[-1] - D[0]
