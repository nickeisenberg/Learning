import numpy as np
import matplotlib.pyplot as plt

b = np.random.normal(0, .5, (400,4))

b1x = b[:, 0] + 3
b1y = b[:, 1]
b1 = np.vstack([b1x,b1y]).T

b2x = b[:, 2]
b2y = b[:, 3] + 2
b2 = np.vstack([b2x,b2y]).T

blobs = np.vstack([b1, b2])
targets = np.hstack([np.ones(len(b1)), np.zeros(len(b2))]).reshape((-1,1))

plt.scatter(blobs[:, 0], blobs[:, 1], c=targets[:, 0])

W = np.random.rand(2)
b = np.random.rand(1)[0]

def line_sep(inp):
    pred = np.dot(W, inp) + b
    return(pred)

guesses = np.array([line_sep(blob) for blob in blobs])

dom = np.linspace(-1, 4, 1000)
line = lambda x : -W[0] / W[1] * x + (.5 - b / W[1])
vals = line(dom)

plt.plot(dom, vals)
plt.show()
