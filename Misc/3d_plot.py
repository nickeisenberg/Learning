import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(-4, 4, 1000)
y = np.linspace(-4, 4, 1000)

xx, yy = np.meshgrid(x, y)

fun = lambda x, y : x ** 2 + y ** 2

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.plot_surface(xx, yy, fun(xx,yy))

plt.show()
