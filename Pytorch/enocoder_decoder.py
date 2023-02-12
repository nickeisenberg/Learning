import torch
import torch.nn as nn
from keras.utils import timeseries_dataset_from_array as tsd
import plotly.graph_objects as go
from math import pi

# create a signal to encode and decode.
time = torch.linspace(0, 1, 1000)

signal = torch.sin(2 * pi * time * 4)
signal += torch.sqrt(torch.tensor(1 / 1000)) * torch.randn(1000)

test_sig = torch.sin(2 * pi * time * 4)
test_sig += torch.sqrt(torch.tensor(1 / 1000)) * torch.randn(1000)

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
seq_len = 50

dataset_keras = tsd(data=signal,
                    targets=None,
                    sequence_stride=1,
                    sequence_length=seq_len,
                    batch_size=None)
dataset_torch = torch.empty(seq_len)
for d in dataset_keras:
    dataset_torch = torch.vstack((dataset_torch, torch.tensor(d.numpy())))
dataset_torch = dataset_torch[1:,:]

testset_keras = tsd(data=signal,
                    targets=None,
                    sequence_stride=1,
                    sequence_length=seq_len,
                    batch_size=None)
testset_torch = torch.empty(seq_len)
for d in testset_keras:
    testset_torch = torch.vstack((testset_torch, torch.tensor(d.numpy())))
testset_torch = testset_torch[1:,:]

# define the model
class lstm_nn(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 128)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(128, 64)
        self.relu2 = nn.ReLU()
        self.lstm3 = nn.LSTM(64, 64)
        self.relu3 = nn.ReLU()
        self.lstm4 = nn.LSTM(64, 128)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(seq_len * 128, seq_len)

    def forward(self, x):
        x = x.reshape((-1, 1))
        x, _ = self.lstm1(x)
        x = self.relu1(x)
        x, _ = self.lstm2(x)
        x = self.relu2(x)
        x = x[-1, :].reshape(-1)
        x = x.repeat(seq_len, 1)
        x, _ = self.lstm3(x)
        x = self.relu3(x)
        x, _ = self.lstm4(x)
        x = self.relu4(x)
        x = x.reshape(-1)
        x = self.linear1(x)
        return x

model = lstm_nn().to('cpu')

loss_fn = nn.L1Loss()
accuracy = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataset, model, loss_fn, optimizer, epoch):
    size = dataset.shape[0]
    model.train()
    count = 1
    for inp, out in zip(dataset, dataset):
        pred = model(inp)
        loss = loss_fn(pred, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} percent complete: {count / size}")
        print(f"loss: {loss:>7f}")
        count += 1

def test(dataset, model, loss_fn, accuracy):
    size = dataset.shape[0]
    model.eval()
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for inp, out in zip(testset_torch, testset_torch):
            inp, out = inp.to('cpu'), out.to('cpu')
            pred = model(inp)
            test_loss.append(loss_fn(pred, out))
            test_acc.append(max(0, 1 - accuracy(pred, out)))
            print(f"Average accuracy: {torch.tensor(test_acc).mean().item()}")
            print(f"Average loss: {torch.tensor(test_loss).mean().item()}")
    return test_loss, test_acc

epochs = 5
test_loss = []
test_acc = []
for t in range(epochs):
    print(f"Epoch {t + 1}\n--------------------")
    train(dataset_torch, model, loss_fn, optimizer, epoch=t)
    # l, a = test(testset_torch, model, loss_fn, accuracy)
    # test_loss.append(l)
    # test_acc.append(a)
print('DONE')

testset_torch.shape

sub = dataset_torch[::20]

guess_signal = []
for inp in sub:
    with torch.no_grad():
        guess_signal.append(model(inp))

guess_signal = torch.vstack(guess_signal).reshape(-1)

fig = go.Figure()
_ = fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            )
        )
_ = fig.add_trace(
        go.Scatter(
            x=time,
            y=guess_signal,
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
