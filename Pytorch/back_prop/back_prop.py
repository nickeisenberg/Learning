import torch

w = torch.tensor([0.8], requires_grad=True)
b = torch.tensor([0.2], requires_grad=True)

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y_true = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)

learning_rate = .001

y_pred = torch.matmul(x, w) + b

for i in range(5000):
    y_pred = torch.matmul(x, w) + b
    loss = torch.mean((y_pred - y_true) ** 2)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        _ = w.grad.zero_()
        _ = b.grad.zero_()
