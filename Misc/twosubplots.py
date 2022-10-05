import numpy as np
import matplotlib.pyplot as plt

# First create some toy data:
x1 = np.linspace(0, 2*np.pi, 400)
y1 = np.sin(x1**2)

x2 = np.linspace(0, 2*np.pi, 400)
y2 = np.cos(x2**2)


## Create two subolots side by side
#f, (ax1, ax2) = plt.subplots(2, sharey=True)
#ax1.plot(x, y)
#ax1.set_title('ax_1 title')
#ax2.scatter(x, y)
#ax2.set_title('ax_2 title')

# Another method 
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(x1 ,y1)
ax2.scatter(x2,y2)

## Plotting multiple functions on the same graph.
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(x1 ,y1)
#ax1.scatter(x2,y2)

## Good example illustrationg how to use fig.add_subplot()
#fig = plt.figure()
#fig.add_subplot(221)   #top left
#fig.add_subplot(222)   #top right
#fig.add_subplot(223)   #bottom left
#fig.add_subplot(224)   #bottom right
#plt.show()

## Create two subplots stacked vertically
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.plot(x, y)
#ax1.set_title('ax_1 title')
#ax2.scatter(x, y)
#ax2.set_title('ax_2 title')

plt.show()
