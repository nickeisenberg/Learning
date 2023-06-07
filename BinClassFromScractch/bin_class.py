from os import close
import numpy as np
import matplotlib.pyplot as plt

blobs_x = np.random.normal(0, 1, 1000)
blobs_y = np.random.normal(0, 1, 1000)

blob_0 = np.vstack(
    (np.random.normal(0, .5, 1000),
    np.random.normal(4, .5, 1000))
).T

blob_1 = np.vstack(
    (np.random.normal(3, .5, 1000),
    np.random.normal(9, .5, 1000))
).T

plt.scatter(blob_0[:,0], blob_0[:,1])
plt.scatter(blob_1[:,0], blob_1[:,1])
plt.show()



a = np.random.normal(0, 1, 1)
b = np.random.normal(0, 1, 1)

def line(x):
    return a * x + b

def closest_point_on_line(point, a, b):
    x0 = point[0] + (point[1] * a) - (a * b)
    x0 /= 1 + a ** 2
    y0 = a * x0 + b
    return (x0, y0)

j, k = (1, -1)

plt.plot(np.linspace(-1, 1, 100), line(np.linspace(-1, 1, 100)))
plt.scatter(j, k)

x0, y0 = closest_point_on_line((j, k), a, b)
plt.scatter(x0, y0)

plt.show()


