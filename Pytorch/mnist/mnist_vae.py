from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import random

#|%%--%%| <k7PS0uYOHY|uqWwStcYEn>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#|%%--%%| <uqWwStcYEn|p2j73JdljP>

data = MNIST('/Users/nickeisenberg/GitRepos/DataSets_local/MNIST/imgs')

train_ims, train_labs = data.load_training()
train_ims = torch.stack([
    torch.tensor(
        im, dtype=torch.uint8
    ).reshape((1, 28, 28)) / 255 for im in train_ims
])
train_labs = torch.tensor(train_labs)

train_ims.size()

#|%%--%%| <p2j73JdljP|ipMv8PFi0O>

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
    train_ims[dataset_inds[:50000]], train_ims[dataset_inds[:50000]]
)

val_dataset = ImageDataset(
    train_ims[dataset_inds[50000:]], train_ims[dataset_inds[50000:]]
)

train_dataloader = DataLoader(train_dataset, 32)
val_dataloader = DataLoader(val_dataset, 32)

#|%%--%%| <ipMv8PFi0O|Exb9JYmqm4>

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

class VAE(nn.Module):

    def __init__(self, encoder, sampler, decoder, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.sampler = Sampler(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, inputs):
        zmean, zlogvar = self.encoder(inputs)
        samps = self.sampler(zmean, zlogvar)
        recon = self.decoder(samps)
        return recon, (zmean, zlogvar)

def rmse_loss(input, target, batched=True):
    if batched:
        return torch.mean(torch.sum((input - target) ** 2, axis=(1, 2, 3)))
    else:
        return torch.mean((input - target) ** 2)

def kl_regularization(z_mean, z_log_var, batched=True):
    kl_reg = -0.5
    kl_reg *= (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    if batched:
        return torch.mean(torch.sum(kl_reg, axis=-1))
    else:
        return torch.mean(kl_reg)

#|%%--%%| <Exb9JYmqm4|yTDNqaKKyM>

vae = VAE(Encoder, Sampler, Decoder, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=.001)

def train_one_epoch(dataloader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        imgs, _ = data

        optimizer.zero_grad()

        recon_imgs, (zmean, zlogvar) = vae(imgs)

        recon_loss = rmse_loss(recon_imgs, imgs)
        kl_loss = kl_regularization(zmean, zlogvar)

        total_loss = recon_loss + kl_loss

        total_loss.backward()

        optimizer.step()

        # Track the losses
        running_loss += total_loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

#|%%--%%| <yTDNqaKKyM|aUGAVHFWXA>
        
EPOCHS = 30
best_vloss = 1e6

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch + 1}')

    _ = vae.train()
    avg_loss = train_one_epoch(dataloader=train_dataloader)

    _ = vae.eval()
    running_v_loss = 0.0
    with torch.no_grad():
        for i, v_data in enumerate(val_dataloader):
            v_ims, _ = v_data
            v_guess, (v_zmean, v_zlogvar) = vae(v_ims)
            v_loss_recon = rmse_loss(v_guess, v_ims)
            running_v_loss += v_loss_recon
    
    avg_vloss = running_v_loss / (i + 1)

    print(f'LOSS Train: {avg_loss} Val: {avg_vloss}')
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'mnist_vae.torch'
        torch.save(vae.state_dict(), model_path)

