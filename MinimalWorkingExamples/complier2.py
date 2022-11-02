import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

x = np.random.randn(100, 20000)
print(np.mean(x))

xm = deepcopy(x)
med_ct = 0

for i in range(1, 200):
    xm += np.roll(x, i, axis=1)
    xm += np.roll(x, -i, axis=1)
    med_ct += 2

xm /= med_ct

print(np.mean(x - xm))

