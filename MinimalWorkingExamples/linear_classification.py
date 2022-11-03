import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

b = np.random.normal(0, .5, (400,4))

b1x = b[:, 0] + 3
b1y = b[:, 1]
b1 = np.vstack([b1x,b1y]).T

b2x = b[:, 2]
b2y = b[:, 3] + 2
b2 = np.vstack([b2x,b2y]).T

blobs = np.vstack([b1, b2])
targets = np.hstack([np.ones(len(b1)), np.zeros(len(b2))]).reshape((-1,1))

def model(inputs, params):
    return (np.sum(np.multiply(params[:2], inputs), axis=1) + params[2]).reshape((-1,1))

def loss(params, targets, inputs):
    predictions = model(inputs, params)
    residual = np.mean(np.square(targets - predictions))
    return residual

guess_params = np.random.randn(3)
opt_params = minimize(loss, guess_params, args=(targets, blobs)).x

dom = np.linspace(-1, 4, 1000)

line_guess = lambda x : (-guess_params[0] / guess_params[1] * x) + ((.5 - guess_params[2])/ guess_params[1])
vals_guess = line_guess(dom)

line = lambda x : (-opt_params[0] / opt_params[1] * x) + ((.5 - opt_params[2])/ opt_params[1])
vals = line(dom)
predicted_targets = model(blobs, opt_params)
for i, pred in enumerate(predicted_targets):
    if pred[0] > 1/2:
        predicted_targets[i] = [1]
    elif pred[0] <= 1/2:
        predicted_targets[i] = [0]

plt.subplot(121)
plt.scatter(blobs[:, 0], blobs[:, 1])
plt.title('unclustered blobs')

plt.subplot(122)
plt.scatter(blobs[:, 0], blobs[:, 1], c=predicted_targets[:,0])
plt.plot(dom, vals_guess, label='initial guess')
plt.plot(dom, vals, label='learned classifer')
plt.legend(loc='upper left')
plt.title('clustered blobs')

plt.suptitle('Linear Classification')
plt.show()
