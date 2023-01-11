import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(0, 4, 5)

y = np.linspace(0, 4, 5)

xx, yy = np.meshgrid(x, y)

print(xx)
print(yy)

fun = lambda x, y : x ** 2 + y ** 2

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.plot_surface(xx, yy, fun(xx,yy))

# plt.show()
