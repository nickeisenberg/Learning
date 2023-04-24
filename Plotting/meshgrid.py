import matplotlib.pyplot as plt
import numpy as np

def func(r, std=1):
    return (1 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-(r ** 2) / (2 * std))

xaxis = np.linspace(-4, 4, 1000)
yaxis = np.linspace(-4, 4, 1000)

X, Y = np.meshgrid(xaxis, yaxis)
R = np.sqrt(X ** 2 + Y ** 2)
surf = func(R)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(R)
ax[1].imshow(func(R))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, surf, cmap='plasma')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.pcolormesh(X, Y, surf, cmap='plasma')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(R, cmap='plasma')
plt.show()

