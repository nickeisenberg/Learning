from _typeshed import NoneType
from abc import ABC
from collections.abc import Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset
from tqdm import tqdm

class Logger:
    def __init__(self, loss_keys: list):
        self.loss_keys = []
        self.history = {}

    def update(self, epoch, loss_history: dict):
        """
        To be run after each batch. Update self.history
        """

    def finalize(self):
        """
        Run at the end of the Epoch. 
        Save self.history to a file. Maybe a csv or a json.
        """

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_fn: Callable):

        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)
    
    def train_pass(self, inputs, targets):
        # make the prediction
        predicitons = self.model(inputs)
        
        # Calculate the batch loss
        loss, loss_history = self.loss_fn(predicitons, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_history

class Dataset(_Dataset):
    def __getitem__(self, idx):
        return None
    def __len__(self):
        return None


class Experiment:
    def __init__(self, network: Network, logger: Logger, dataloader: DataLoader,
                 num_epochs: int, device:str):

        self.network = network
        self.logger = logger
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device

    def _train(self):
        pass

    def _train_one_epoch(self, epoch):

        pbar = tqdm(self.dataloader, leave=True)
        for batch_num, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_history = self.network.train_pass(inputs, targets)
            self.logger.update(epoch, batch_history)
            pbar.set_postfix(loss=0)

        self.logger.finalize()
