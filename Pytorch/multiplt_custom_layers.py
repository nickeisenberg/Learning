import torch
import torch.nn as nn
import matplotlib.pyplot as plt

clusts1 = .25 * torch.randn([400, 2])
clusts2 = (.25 * torch.randn([400, 2])) + torch.tensor([1.3, 1.3])

clusts = torch.vstack([clusts1, clusts2])
labels = torch.hstack([torch.zeros(400), torch.ones(400)]).reshape((-1, 1))

# one model
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 3)
        self.lin2 = nn.Linear(3, 1)

    def forward(self, inputs):
        x = self.lin1(inputs)
        x = nn.ReLU()(x)
        x = self.lin2(x)
        x = nn.Sigmoid()(x)
        return x

model = Model()

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=.01)

def train_step(inputs, targets):

    total_loss = 0

    optimizer.zero_grad()
    guess = model(inputs)
    loss = loss_fn(guess, targets)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss

for i in range(10000):
    _ = model.train()
    loss = train_step(clusts, labels)
    if i%50 == 0:
        print(loss)

guesses = model(clusts).detach().numpy().reshape(-1)

plt.scatter(clusts[:, 0], clusts[:, 1], c=guesses)
plt.show()

# same thing but with "custom" layers

class Layer1(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 3)

    def forward(self, inputs):
        x = self.lin1(inputs)
        x = nn.ReLU()(x)
        return x

class Layer2(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin2 = nn.Linear(3, 1)

    def forward(self, inputs):
        x = self.lin2(inputs)
        x = nn.Sigmoid()(x)
        return x

class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = Layer1()
        self.lin2 = Layer2()

    def forward(self, inputs):
        x = self.lin1(inputs)
        x = self.lin2(x)
        return x

custom_model = CustomModel()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(custom_model.parameters(), lr=.01)

guesses = custom_model(clusts).detach().numpy().reshape(-1)
plt.scatter(clusts[:, 0], clusts[:, 1], c=guesses)
plt.show()

def train_step(inputs, targets):
    optimizer.zero_grad()
    guess = custom_model(inputs)
    loss = loss_fn(guess, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(10000):
    _ = custom_model.train()
    loss = train_step(clusts, labels)
    if i % 50 == 0:
        print(loss)

guesses = custom_model(clusts).detach().numpy().reshape(-1)
plt.scatter(clusts[:, 0], clusts[:, 1], c=guesses)
plt.show()
