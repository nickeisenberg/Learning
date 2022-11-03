import numpy as np

x = np.array([1,2,3,77,66,4,5])

val = []
for i in x:
    if i > 10:
        continue
    if len(val) == 3:
        break
    val.append(i)

print(val)
