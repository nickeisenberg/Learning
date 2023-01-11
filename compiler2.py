import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 1000)
data = lambda x : np.sin(2 * np.pi * 5 * x) + np.cos(2 * np.pi * 2 * x)

plt.plot(time, data(time))
plt.show()


