import torch 

x = torch.ones(5)  # input
y = torch.zeros(3)  # output

# model
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient for z: {z.grad_fn}")
print(f"Gradient for loss: {loss.grad_fn}")

# To calculate the change in the loss with respect to the weights b and w,
# we need to call the backward() method of our loss function.

loss.backward()

w.grad
b.grad

# By default, the torch variables with requires_grad set true will always
# track history and compute the gradients after a forward pass. But there are
# time where we will not want to do this, like when wanting to run a test
# item through the model. When this is the case we will use torch.no_grad.

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
    print(z.requires_grad)

# We can also use the deteach() property of z to do this

z = torch.matmul(x, w) + b
z = z.detach()
print(z.requires_grad)

