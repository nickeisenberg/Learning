'''
Speed comparision of pytorch and tesnorflow with a simply back property 
Pytorch is much faster
'''


import torch

w = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y_true = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)

def forward(w, b):
    return torch.matmul(x, w) + b

def loss(w, b):
    return torch.mean((forward(w, b) - y_true) ** 2)

forward(w, b)
loss(w, b)

learning_rate = .001

for i in range(5000):
    _loss = loss(w, b)
    _loss.backward()
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        _ = w.grad.zero_()
        _ = b.grad.zero_()


forward(w, b)
loss(w, b)

#--------------------------------------------------
import tensorflow as tf

w = tf.Variable(initial_value=[[.5]])
b = tf.Variable(initial_value=[.5])

x = tf.Variable([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.Variable([[1], [2], [3], [4]], dtype=tf.float32)


def forward(w, b):
    return tf.matmul(x, w) + b

def loss(w, b):
    return tf.reduce_mean((forward(w, b) - y_true) ** 2)

forward(w, b)
loss(w, b)

learning_rate = .001

for i in range(5000):
    with tf.GradientTape() as tape:
        _loss = loss(w, b)
    grads = tape.gradient(_loss, [w, b])
    _ = w.assign_sub(grads[0] * learning_rate)
    _ = b.assign_sub(grads[1] * learning_rate)

forward(w, b)
loss(w, b)

#--------------------------------------------------

class Linear_1D:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = torch.tensor(
            data=torch.randn(input_dim, output_dim), requires_grad=True
        )
        self.b = torch.tensor(
            data=torch.randn(output_dim), requires_grad=True
        )
        self.learning_rate = .001

    def forward(self, data):
        return torch.matmul(data, self.w) + self.b

    def loss(self, data, y_true):
        return torch.mean((self.forward(data) - y_true)**2)

    def train(self, data, y_true):
        _loss = self.loss(data, y_true)
        _loss.backward()
        with torch.no_grad():
            self.w -= self.w.grad * self.learning_rate
            self.b -= self.b.grad * self.learning_rate
            _ = self.w.grad.zero_()
            _ = self.b.grad.zero_()

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y_true = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

model = Linear_1D(4, 4)

for i in range(5000):
    _ = model.train(x, y_true)

model.forward(x)
















