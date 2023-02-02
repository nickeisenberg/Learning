import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.cm as cm
import numpy as np

line = lambda x, b: x + b

lines = np.array(
        [line(np.linspace(0, 1, 100), b)
         for b in range(10)]
        )

cmap = cm.autumn
norm = mplc.Normalize(vmin=0, vmax=1)

for i, l in enumerate(lines):
    i = i / lines.shape[0]
    plt.plot(l, color=cmap(norm(i)))
plt.title('color by number')
plt.show()
