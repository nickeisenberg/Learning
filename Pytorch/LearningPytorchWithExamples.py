# Quick tutorial from
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
import math
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device('cpu')

x=torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = (y_pred - y).pow(2).sum().item()

    if t % 100 == 99:
        print(t, loss)

    grad_y_pred = 2 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

plt.plot(y)
plt.plot(y_pred, linestyle='--')
plt.show()
#--------------------------------------------------

# Do the same but with the nn module of torch

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# The model has the form (b, c, d) * (x, x ** 2, x ** 3)^T + a. Torch has a
# layer nn.Linear which handles this.

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
        )

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        print(t, loss)

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Get the weights which are stored in the first layer
a = model[0].bias.item()

b, c, d = model[0].weight.detach().numpy().reshape(-1)

plt.plot(a + b * x + c * x ** 2 + d * x ** 3)
plt.plot(y)
plt.show()
