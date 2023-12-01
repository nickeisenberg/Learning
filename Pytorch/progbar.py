import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128 * 16 * 16, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class MyDataset(Dataset):
    def __init__(self):
        self.data = [
            torch.randn(3, 256, 256) for i in range(1000)
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], idx

dataset = MyDataset()
dataloader = DataLoader(dataset, 8)

device = "cpu"

model = NeuralNet()
model = model.to(device)

def train_loop(loader, model):
    prog_bar = tqdm(loader, leave=True)
    losses = []
    for x, y in prog_bar:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        _loss = pred[..., 0] - y
        losses.append(_loss.sum().item())
        mean_loss = sum(losses) / len(losses)
        prog_bar.set_postfix(loss=mean_loss)

for i in range(10):
    train_loop(dataloader, model)
