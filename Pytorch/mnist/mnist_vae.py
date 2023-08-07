from keras.datasets import mnist

(train_ims, train_labs), (test_ims, test_labs) = mnist.load_data()

#|%%--%%| <k7PS0uYOHY|8O1yEnaEck>

from mnist import MNIST

data = MNIST('/Users/nickeisenberg/GitRepos/DataSets_local/MNIST/imgs')
train_ims, train_labs = data.load_training()

#|%%--%%| <8O1yEnaEck|HwMJ9yB9AX>

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import random


#|%%--%%| <HwMJ9yB9AX|uqWwStcYEn>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#|%%--%%| <uqWwStcYEn|p2j73JdljP>

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

train_dataloader = DataLoader(train_dataset, 64)
val_dataloader = DataLoader(val_dataset, 64)

#|%%--%%| <ipMv8PFi0O|Exb9JYmqm4>

latent_dim = 2

class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1 * 1 * 64, 16)
        self.z_mean_ = nn.Linear(16, latent_dim)
        self.z_log_var_ = nn.Linear(16, latent_dim)

    def forward(self, input_img):
        x = self.conv1(input_img)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
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
        self.linear2 = nn.Linear(16, 1 * 1 * 64)
        self.convt1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.convt2 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.convt3 = nn.ConvTranspose2d(
            16, 8, kernel_size=4, stride=2, padding=1, output_padding=0
        )
        self.convt4 = nn.ConvTranspose2d(
            8, 1, kernel_size=4, stride=2, padding=1, output_padding=0
        )

    def forward(self, latent_input):
        x = self.linear1(latent_input)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.shape[0], 1, 1, 64).permute(0, 3, 1, 2)
        x = self.convt1(x)
        x = nn.ReLU()(x)
        x = self.convt2(x)
        x = nn.ReLU()(x)
        x = self.convt3(x)
        x = nn.ReLU()(x)
        decoded_image = self.convt4(x)
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


loss_fn = nn.BCELoss()

def kl_regularization(z_mean, z_log_var):
    kl_reg = -0.5
    kl_reg *= (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    return torch.sum(kl_reg)

# ims = nn.Sigmoid()(torch.randn((4, 1, 28, 28)))
# tars = nn.Sigmoid()(torch.randn((4, 1, 28, 28)))
# zm = torch.randn((4, 2))
# zv = torch.randn((4, 2))
# 
# loss_fn(ims, tars)
# kl_regularization(zm, zv)

#|%%--%%| <Exb9JYmqm4|yTDNqaKKyM>

vae = VAE(Encoder, Sampler, Decoder, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=.001)

# im = train_ims[0]
# im = im.unsqueeze(0)
# im.shape
# recon = vae(im)[0][0][0].detach().numpy()
# plt.imshow(recon)
# plt.show()

def train_one_epoch(dataloader):
    total_loss = 0.
    running_loss = 0.
    counter = 1
    for i, data in enumerate(dataloader):
        counter = i
        imgs, _ = data
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon_imgs, (zmean, zlogvar) = vae(imgs)
        recon_loss = loss_fn(recon_imgs, imgs)
        kl_loss = kl_regularization(zmean, zlogvar)
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss += loss.item()
        if i % 80  == 0:
            last_loss = running_loss / 80 # loss per batch
            print(f'batch {i / 80} loss: {last_loss}')
            running_loss = 0.
    return total_loss / counter

for d in train_dataloader:
    print(type(d[0]))
    break

#|%%--%%| <yTDNqaKKyM|Atekkf6qiX>
        
EPOCHS = 10
best_vloss = 1e6

for epoch in range(EPOCHS):

    print(f'Epoch: {epoch + 1}')

    _ = vae.train()
    avg_loss = train_one_epoch(dataloader=train_dataloader)

    _ = vae.eval()
    running_v_loss = 0.0
    counter = 1
    with torch.no_grad():
        for i, v_data in enumerate(val_dataloader):
            counter = i
            v_ims, _ = v_data
            v_ims = v_ims.to(device)
            v_guess, (v_zmean, v_zlogvar) = vae(v_ims)
            v_loss_recon = loss_fn(v_guess, v_ims)
            running_v_loss += v_loss_recon
    
    avg_vloss = running_v_loss / counter

    print(f'LOSS Train: {avg_loss} Val: {avg_vloss}')
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'mnist_vae.torch'
        torch.save(vae.state_dict(), model_path)

#|%%--%%| <Atekkf6qiX|aUGAVHFWXA>

vae_loaded = VAE(Encoder, Sampler, Decoder, 2)

vae_loaded.load_state_dict(torch.load('mnist_vae.torch'))

im = train_ims[200].unsqueeze(0)

recon = vae_loaded(im)[0].detach().numpy()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(im[0][0])
ax[1].imshow(recon[0][0])
plt.show()
