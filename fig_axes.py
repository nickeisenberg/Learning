import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0,  1, 100)
data = lambda x : x ** 2
data = data(time)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.plot(time, data)
ax2 = fig.add_subplot(2,2,2)
ax2.plot(time, data)
ax3 = fig.add_subplot(2,2,3)
ax3.plot(time, data)
ax4 = fig.add_subplot(2,2,4)
ax4.plot(time, data)
plt.show()

fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)
count = 0
for ax in axs:
    ax.plot(time, data)
plt.show()

fig, axs = plt.subplots(2, 2)
for ax in axs:
    ax[0].plot(time, data)
    ax[1].plot(time, data)
plt.show()
