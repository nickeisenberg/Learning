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
    ).reshape((1, 28, 28)) / 255 for im in train_ims
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

random.seed(1)
dataset_inds = np.arange(len(train_ims))
random.shuffle(dataset_inds)

train_dataset = ImageDataset(
    train_ims[dataset_inds[:50000]], train_labs[dataset_inds[:50000]]
)

val_dataset = ImageDataset(
    train_ims[dataset_inds[50000:]], train_labs[dataset_inds[50000:]]
)

train_dataset[0][0].shape

train_dataloader = DataLoader(train_dataset, 32)
val_dataloader = DataLoader(val_dataset, 32)

latent_dim = 2

class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(7 * 7 * 64, 16)
        self.z_mean_ = nn.Linear(16, latent_dim)
        self.z_log_var_ = nn.Linear(16, latent_dim)

    def forward(self, input_img):
        x = self.conv1(input_img)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = nn.ReLU()(x)
        z_mean = self.z_mean_(x)
        z_log_var = self.z_log_var_(x)
        return z_mean, z_log_var

class Sampler(nn.Module):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super().__init__()

    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        epsilon = torch.randn((batch_size, self.latent_dim))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(latent_dim, 16)
        self.linear2 = nn.Linear(16, 7 * 7 * 64)
        self.convt1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.convt2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv1 = nn.Conv2d(
            1, 1, kernel_size=3, padding=1
        )

    def forward(self, latent_input):
        x = self.linear1(latent_input)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.shape[0], 7, 7, 64).permute(0, 3, 1, 2)
        x = self.convt1(x)
        x = nn.ReLU()(x)
        x = self.convt2(x)
        x = nn.ReLU()(x)
        decoded_image = self.conv1(x)
        decoded_image = nn.Sigmoid()(decoded_image)
        return decoded_image

encoder = Encoder(2)
sampler = Sampler(2)
decoder = Decoder(2)

