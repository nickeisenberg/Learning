# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the training data from open data sets
training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor())

test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor())

type(training_data)
type(test_data)

batch_size = 64
train_dataloader = DataLoader(
        training_data, batch_size=batch_size)
test_dataloader = DataLoader(
        test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print(x.shape)
    print(type(x))
    print(y.shape)
    print(type(y))
    break

# Create the model

# Get cpu or gpu device for training. Way to check if I have a gpu or not
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using {device} device")  # I dont

# Define the neurnal net
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
                )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # picked up from nn.Module
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>.1f}%, AVG Loss: \
            {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n------------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("DONE")

# save the model 
path = '/Users/nickeisenberg/GitRepos/Python_Notebook/'
path += 'Pytorch/PytorchDocsTutorials/Models/'

torch.save(model.state_dict(), path + 'MNIST_model.pth')

# We can load the model
loaded_model = NeuralNetwork()

loaded_model.load_state_dict(torch.load(path + 'MNIST_model.pth'))

# We can use this model to make predictions

classes = np.linspace(0, 9, 10)

loaded_model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    print(f"prediction: {classes[pred[0].argmax(0)]}, Actual: {classes[y]}")
