from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import random

data = MNIST('/Users/nickeisenberg/GitRepos/DataSets_local/MNIST/imgs')

train_ims, train_labs = data.load_training()
train_ims = torch.stack([
    torch.tensor(
        im, dtype=torch.uint8
    ).reshape((28, 28)) / 255 for im in train_ims
])

def vectorize(idx):
    vec = torch.zeros(10)
    vec[idx] = 1
    return vec

train_labs = torch.stack([vectorize(id) for id in train_labs])

class ImageDataset(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(1, -1)
        self.dense1 = nn.Linear(28 * 28, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return self.softmax(x)

model = Model()

random.seed(1)
dataset_inds = np.arange(len(train_ims))
random.shuffle(dataset_inds)

train_dataset = ImageDataset(
    train_ims[dataset_inds[:50000]], train_labs[dataset_inds[:50000]]
)

val_dataset = ImageDataset(
    train_ims[dataset_inds[50000:]], train_labs[dataset_inds[50000:]]
)

train_dataloader = DataLoader(train_dataset, 32)
val_dataloader = DataLoader(val_dataset, 32)

optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
loss_fn = nn.CrossEntropyLoss()

def train_one_epoch(epoch_index, dataloader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        ims, labs = data

        optimizer.zero_grad()

        guess = model(ims)

        loss = loss_fn(guess, labs)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

EPOCHS = 20
best_vloss = 1e6

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch + 1}')

    _ = model.train()
    avg_loss = train_one_epoch(epoch, dataloader=train_dataloader)


    _ = model.eval()

    running_v_loss = 0.0
    with torch.no_grad():
        for i, v_data in enumerate(val_dataloader):
            v_ims, v_labs = v_data
            v_guess = model(v_ims)
            v_loss = loss_fn(v_guess, v_labs)
            running_v_loss += v_loss

    avg_vloss = running_v_loss / (i + 1)

    print(f'LOSS Train-{avg_loss} Val-{avg_vloss}')
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model'
        torch.save(model.state_dict(), model_path)

model.eval()
for i, (im, lab) in enumerate(val_dataloader):
    # print(torch.argmax(lab, axis=1))
    # print(torch.argmax(model(im), axis=1))
    same = torch.argmax(model(im), axis=1) - torch.argmax(lab, axis=1)
    print((32 - torch.where(same != 0)[0].size()[0]) / 32)
    if i == 20:
        break
