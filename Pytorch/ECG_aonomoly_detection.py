import pandas as pd
import numpy as np
import arff as a2p
from scipy.io.arff import loadarff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from copy import deepcopy
import torch
from torch import nn

device = 'cpu'

path = '/Users/nickeisenberg/GitRepos/OFFLINE_DIRECTORY/Datasets/ECG5000/'

train = loadarff(path + 'ECG5000_TRAIN.arff')
test = loadarff(path + 'ECG5000_TEST.arff')

# combine the test and training data into one data frame and then shuffle
# the data.
np.random.seed(1)
df_train = pd.DataFrame(train[0])
df_test = pd.DataFrame(test[0])
df = pd.concat((df_train, df_test)).sample(frac=1)

df_train.shape
df_test.shape
df.shape

df_heart = deepcopy(df[df.columns.values[: -1]]).reset_index(drop=True)
df_target = deepcopy(
        pd.DataFrame(df['target'].astype(int))).reset_index(drop=True)

df_heart.head()
df_target.head()

# Each row in the dataframe is a single heartbeat
fig = go.Figure()
for i in range(10):
    _ = fig.add_trace(
            go.Scatter(
                y=df_heart.iloc[i].values.astype(float),
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

# Name the possible classes
# the target column specifies which class the heartbeat falls into.
class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']
normal = 1

# We can see how many of each target we have
val_counts = df_target['target'].value_counts()
val_counts.index = class_names
val_counts_df = pd.DataFrame(val_counts)

fig = go.Figure()
_ = fig.add_trace(
        go.Bar(
            x=val_counts_df.index,
            y=val_counts_df['target']
            )
        )
_ = fig.update_xaxes(
        title={
            'text': 'Class of hearbeat',
            })
_ = fig.update_yaxes(
        title={
            'text': 'amount from each class',
            })
fig.show()

# Lets view a smoothed average of each class

class_avgs = {}
for i in range(1, 6):
    df_target_i_inds = df_target.loc[df_target.target == i].index
    df_heart_i = deepcopy(df_heart.loc[df_target_i_inds].values)
    avg = df_heart_i.mean(axis=0)
    smooth_avg = pd.Series(avg).rolling(window=10).mean().iloc[10 - 1:]
    class_avgs[i] = smooth_avg.values

nrow, ncol = 2,3 
rcs = []  # row and column subplot pairs
for i in range(1, nrow + 1):
    for j in range(1, ncol + 1):
        rcs.append((i, j))
fig = make_subplots(rows=nrow, cols=ncol,
                    subplot_titles=class_names)
for i, rc in enumerate(rcs[: 5]):
    _ = fig.add_trace(
            go.Scatter(
                y=class_avgs[i + 1],
                ),
            row=rc[0], col=rc[1]
            )
    _ = fig.update_xaxes(
            title={
                'text':'',
                },
            row=rc[0], col=rc[1])
    _ = fig.update_yaxes(
            title={
                'text':'',
                },
            row=rc[0], col=rc[1])
_ = fig.update_layout(
        title={
            'text':'',
            })
fig.show()

# Now we can start with getting the training set for the autoencoder
# We will train the autoecoder on the normal heartbeats.

df_heart_normal = deepcopy(
        df_heart.loc[df_target.target == 1]
        )
df_heart_normal_targets = deepcopy(
        df_target.loc[df_target.target == 1]
        )

df_heart_normal.shape
df_heart_normal_targets.shape

# We can combine all the other anomalies into one 'anomalies' dataframe'
df_heart_anamoly = deepcopy(
        df_heart.loc[df_target.target != 1]
        )
df_heart_anomaly_targets = deepcopy(
        df_target.loc[df_target.target != 1]
        )

df_heart_anamoly.shape
df_heart_anomaly_targets.shape

# Now we can split the normal heart rates into train, val and testing groups.
def train_val_test(df:pd.DataFrame, tr_v_split=(.7, .15)):
    inds = df.index.values
    train_upper = int(len(inds) * tr_v_split[0])
    train_inds = inds[: train_upper]
    val_upper = train_upper + int(len(inds) * tr_v_split[1])
    val_inds = inds[train_upper: val_upper]
    test_inds = inds[val_upper: ]
    train_df = df.loc[train_inds]
    val_df = df.loc[val_inds]
    test_df = df.loc[test_inds]
    return train_df, val_df, test_df

train_df, val_df, test_df = train_val_test(df_heart_normal)

df_heart_normal.shape
train_df.shape
val_df.shape
test_df.shape

# we need to convert these dataframes into tensors. We want the shape to be
# seq_len x no of features. 

#-This way forces the batchsize to be one----------
def make_dataset(df, batch_size=1):
    signals = df.values
    dataset = [
            torch.tensor(s, dtype=torch.float32).unsqueeze(1) for s in signals
            ]
    nsig, sig_len, nfeat = torch.stack(dataset).shape
    return dataset, sig_len, nfeat

train_dataset, sig_len, nfeat = make_dataset(train_df)

# make the validation and a normal test and anaomaly test dataset
val_dataset, _, _ = make_dataset(val_df)
test_normal_dataset, _, _ = make_dataset(test_df)
test_anomaly_dataset, _, _ = make_dataset(df_heart_anamoly)
#--------------------------------------------------

#-This way allows for custom batch sizes-----------
# We need float32, It needs to be specified. LSTM layers require this
train_arr = deepcopy(torch.tensor(train_df.values, dtype=torch.float32))
val_arr = deepcopy(torch.tensor(val_df.values, dtype=torch.float32))
test_norm_arr = deepcopy(torch.tensor(test_df.values, dtype=torch.float32))
test_anom_arr = deepcopy(
        torch.tensor(df_heart_anamoly.values, dtype=torch.float32))

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(
        train_arr, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(
        val_arr, batch_size=batch_size)
test_normal_dataloader = torch.utils.data.DataLoader(
        test_norm_arr, batch_size=batch_size)
test_anomaly_dataloader = torch.utils.data.DataLoader(
        test_anom_arr, batch_size=batch_size)

for d in test_anomaly_dataloader:
    print(d.unsqueeze(-1).shape)
    print(d.dtype)
    break
#--------------------------------------------------

# Now we will make an encoder class. 
# This has been updated to handle a different batch size
class Encoder(nn.Module):
    def __init__(self, sig_len, nfeat, batch_size, embedding_dim=64):
        super().__init__()
        self.sig_len = sig_len
        self.nfeat = nfeat
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden_dim = embedding_dim * 2
        self.rnn1 = nn.LSTM(
                input_size=self.nfeat,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True)
        self.rnn2 = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.embedding_dim,
                num_layers=1,
                batch_first=True)
    def forward(self, x):
        # x = x.reshape((1, self.sig_len, self.nfeat))
        x = x.unsqueeze(-1)
        print(x.shape)
        x, _ = self.rnn1(x)
        print(x.shape)
        x, (hidden_n, _) = self.rnn2(x)
        print(x.shape)
        print(hidden_n.shape)
        # return hidden_n.reshape((self.nfeat, self.embedding_dim))
        return hidden_n.reshape((self.batch_size, self.embedding_dim))

#--------------------------------------------------
xx = train_dataset[0]
xx.shape
xx.dtype

xx = torch.randn((10, 140, 1))
XX1 = nn.LSTM(input_size=1, batch_first=True, hidden_size=128)(xx)
XX1[0].shape
XX2 = nn.LSTM(input_size=128, batch_first=True, hidden_size=64)(XX1[0])
XX2[1][0].shape


en = Encoder(sig_len=140, nfeat=1, batch_size=batch_size)
for inp in train_dataloader:
    inp.shape
    xxx = en.forward(inp)
    xxx.shape
    break
#--------------------------------------------------

# now we need to write an decoder class
class Decoder(nn.Module):
    def __init__(self, sig_len, input_dim=64, nfeat=1):
        super().__init__()
        self.sig_len = sig_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.nfeat = nfeat
        self.rnn1 = nn.LSTM(
                input_size=input_dim,
                hidden_size=input_dim,
                num_layers=1,
                batch_first=True)
        self.rnn2 = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True)
        self.output_layer = nn.Linear(
                self.hidden_dim, nfeat)
    def forward(self, x):
        x = x.repeat(self.sig_len, self.nfeat)
        x = x.reshape((self.nfeat, self.sig_len, self.input_dim))
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.reshape((self.sig_len, self.hidden_dim))
        output = self.output_layer(x) 
        return output

#--------------------------------------------------
xx = torch.randn((1, 140, 64))

XX1 = nn.LSTM(input_size=64, hidden_size=64)(xx)
XX2 = nn.LSTM(input_size=64, hidden_size=128)(XX1[0])
XX3 = XX2[0].reshape((140, 128))
XX4 = nn.Linear(128, 1)(XX3)
XX4.shape

decoder = Decoder(sig_len=140)
xx = torch.randn((1, 64))
XX = decoder.forward(xx)
XX.shape
#--------------------------------------------------

# We can combine the Encode and Decode layers into one class

class Autoencoder(nn.Module):
    def __init__(self, sig_len, nfeat, embedding_dim=64):
        super().__init__()
        self.encode = Encoder(sig_len, nfeat, embedding_dim).to(device)
        self.decode = Decoder(sig_len, embedding_dim, nfeat).to(device)
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

#--------------------------------------------------
xx = train_dataset[0]
autoencoder = Autoencoder(140, 1, 64)
XX = autoencoder(xx)
print(f'input shape {xx.shape}')
print(f'output shape {XX.shape}')
#--------------------------------------------------

# Now we can define the model and train it.
model = Autoencoder(sig_len, nfeat, 128)
model = model.to(device)

def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        train_losses = []
        count = -1
        for sig in train_dataset:
            count += 1
            print(f'percent complete for epoch {epoch}:')
            print(f'{count / len(train_dataset)}')

            optimizer.zero_grad()

            sig = sig.to(device)
            sig_pred = model(sig)

            loss = loss_fn(sig_pred, sig)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for sig in val_dataset:
                sig = sig.to(device)
                sig_pred = model(sig)

                loss = loss_fn(sig_pred, sig)
                val_losses.append(loss)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # monitor the validation loss 
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history

#--------------------------------------------------
fun = nn.L1Loss(reduction='sum').to(device)

x = torch.tensor([1, 2, 3])
y = torch.tensor([1, 2, 2])

type(fun(x, y).item())
#--------------------------------------------------

# I trained the model on google colabs GPU and saved it.
model, history = train_model(model, train_dataset, val_dataset, n_epochs=150)

# reload the saved model
model_path = '/Users/nickeisenberg/GitRepos/Python_Notebook/Pytorch/Models/'
loaded_model = Autoencoder(sig_len, nfeat, 128)
loaded_model.load_state_dict(torch.load(
    model_path + 'ECG_model_state_dict.pth',
    map_location=torch.device('cpu')))

# Create a predict function
def predict(model, dataset):
    preds, losses = [], []
    loss_fn = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for i, sig in enumerate(dataset):
            print(f'percent complete: {i / len(dataset)}')
            sig = sig.to(device)
            sig_pred = loaded_model(sig)
            loss = loss_fn(sig_pred, sig)
            preds.append(sig_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return preds, losses

_, train_losses = predict(loaded_model, train_dataset)

fig = go.Figure()
_ = fig.add_trace(
        go.Histogram(
            x=train_losses,
            histnorm='probability'))
fig.show()

# Set a threshold for a pass or fail for our normal test set.
thresh = 26

test_preds, test_pred_losses = predict(loaded_model, test_normal_dataset)

fig = go.Figure()
_ = fig.add_trace(
        go.Histogram(
            x=test_pred_losses,
            histnorm='probability'))
fig.show()

test_normal_dataset

len(train_dataset)
len(val_dataset)
len(test_normal_dataset)

correct = np.sum(np.array(test_pred_losses) < thresh)
print(correct)
