import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import skimage.io as io

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#|%%--%%| <w4aBwJDoRX|keMTMEb2sT>

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

root = '/Users/nickeisenberg/GitRepos/DataSets_local/' 
# training set and train data loader
totalset = torchvision.datasets.CelebA(
    root=root, split='train', download=True, transform=transform
)

# direct from directory
root = '/Users/nickeisenberg/GitRepos/DataSets_local/celeba/'
totalset = torchvision.datasets.ImageFolder(root, transform)

train_size = int(len(totalset) * .8)
val_size = len(totalset) - train_size

trainset, valset = random_split(totalset, (train_size, val_size))

train_dataloader = DataLoader(
    trainset, batch_size=64, shuffle=True
)
val_dataloader = DataLoader(
    valset, batch_size=64, shuffle=True
)

#|%%--%%| <keMTMEb2sT|LNhyjPJAgt>

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 32, 4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            32, 64, 4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            64, 128, 4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            128, 128, 4, stride=2, padding=1
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 4 * 4, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = nn.LeakyReLU(.02)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(.02)(x)
        x = self.conv3(x)
        x = nn.LeakyReLU(.02)(x)
        x = self.conv4(x)
        x = nn.LeakyReLU(.02)(x)
        x = self.flatten(x)
        x = nn.Dropout(.2)(x)
        x = self.linear(x)
        des_p_value = nn.Sigmoid()(x)
        return des_p_value

class Generator(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(latent_dim, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 128 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(
            128, 128, 4, stride=2, padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            128, 64, 4, stride=2, padding=1
        )
        self.conv3 = nn.ConvTranspose2d(
            64, 32, 4, stride=2, padding=1
        )
        self.conv4 = nn.ConvTranspose2d(
            32, 3, 4, stride=2, padding=1
        )

    def forward(self, latent_inputs):
        x = self.linear1(latent_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.reshape((-1, 128, 4, 4))
        x = self.conv1(x)
        x = nn.LeakyReLU(.2)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(.2)(x)
        x = self.conv3(x)
        x = nn.LeakyReLU(.2)(x)
        x = self.conv4(x)
        x = nn.Sigmoid()(x)
        return x

#|%%--%%| <LNhyjPJAgt|X0c1TILNRb>

latent_dim = 2
discriminator = Discriminator().to(device)
generator = Generator(latent_dim).to(device)

loss_fn = nn.BCELoss()

optim_d = torch.optim.Adam(discriminator.parameters(), lr=.0001, )
optim_g = torch.optim.Adam(generator.parameters(), lr=.0001)

#|%%--%%| <X0c1TILNRb|9BG1XYDcSq>

def train_one_epoch(
        train_dataloader=train_dataloader,
        latent_dim=2
    ):

    running_loss_d = 0.
    running_loss_g = 0.
    
    counter = 1
    for i, t_data in enumerate(train_dataloader):
        counter += 1

        t_ims, _ = t_data.to(device)
        batch_size = t_ims.shape[0]
        
        # Discriminator
        optim_d.zero_grad()

        latents = torch.randn((batch_size, latent_dim)).to(device)
        fake_ims = generator(latents).to(device)

        combined_ims = torch.vstack((t_ims, fake_ims)).to(device)
        combined_labels = torch.hstack(
            (
                torch.ones(batch_size).to(device), 
                torch.zeros(batch_size).to(device)
            )
        ).reshape((-1, 1)).to(device)
        combined_labels += .05 * torch.rand((batch_size * 2, 1)).to(device)

        d_loss = loss_fn(discriminator(combined_ims), combined_labels)
        d_loss.backward()
        running_loss_d += d_loss.item()
        optim_d.step()
        
        # Generator 
        optim_g.zero_grad()

        latents = torch.randn((batch_size, latent_dim)).to(device)
        fake_ims = generator(latents).to(device)
        fake_labels = torch.ones((batch_size, 1)).to(device)

        g_loss = loss_fn(discriminator(fake_ims), fake_labels)
        g_loss.backward()
        running_loss_g += g_loss.item()
        optim_g.step()

        if counter % 50 == 0:
            print(f'Batch {i} Disciriminator loss: {running_loss_d / counter}')
            print(f'Batch {i} Generator loss: {running_loss_g / counter}')
            running_loss_d = 0.
            running_loss_g = 0.
            counter = 0.

    return running_loss_d, running_loss_g


def validation_one_epoch(
        val_dataloader=val_dataloader,
        latent_dim=2
    ):

    v_running_loss_d = 0.
    v_running_loss_g = 0.
    counter = 1 
    for i, v_data in enumerate(val_dataloader):
        counter += 1
        v_ims, _ = v_data.to(device)
        v_batch_size = v_ims.shape[0]

        with torch.no_grad():
            latents = torch.randn((v_batch_size, latent_dim)).to(device)
            fake_ims = generator(latents).to(device)

            combined_ims = torch.vstack((v_ims, fake_ims)).to(device)
            combined_labels = torch.hstack(
                (
                    torch.ones(v_batch_size).to(device),
                    torch.zeros(v_batch_size).to(device)
                )
            ).reshape((-1, 1)).to(device)

            v_d_loss = loss_fn(discriminator(combined_ims), combined_labels)
            v_running_loss_d += v_d_loss.item()

            latents = torch.randn((batch_size, latent_dim)).to(device)
            fake_ims = generator(latents).to(device)
            fake_labels = torch.ones((batch_size, 1)).to(device)

            v_g_loss = loss_fn(discriminator(fake_ims), fake_labels)
            v_running_loss_g += v_g_loss.item()

    avg_v_loss_d = v_running_loss_d / counter
    avg_v_loss_g = v_running_loss_g / counter

    return avg_v_loss_d, avg_v_loss_g

#|%%--%%| <9BG1XYDcSq|J9kMZL0mb0>

EPOCHS = 30
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch}')
    _ = discriminator.train()
    _ = generator.train()
    avg_d_loss, avg_g_loss = train_one_epoch()

    _ = discriminator.eval()
    _ = generator.eval()
    avg_v_loss_d, avg_v_loss_g = validation_one_epoch()


