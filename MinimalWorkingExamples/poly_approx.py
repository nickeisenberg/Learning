import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def poly_approx(fun, dom_l=0, dom_h=1, part_size=1000, order=1, noise=False):

    domain = np.linspace(dom_l, dom_h, part_size)
    data = fun(domain)

    if noise == True:
        noise = np.sqrt((dom_h - dom_l) / part_size) * np.random.randn(part_size)
        data += noise

    def poly(coef, partition, data):

        approx_data = []
        for t in partition:
            summands = []
            for a in range(len(coef)):
                summands.append(coef[a] * t ** a)
            approx_data.append(np.sum(np.array(summands)))

        return np.sqrt(np.sum(np.abs(approx_data - data) ** 2))
#         return np.max(np.abs(approx_data - data))

    guess = np.random.randn(order) # defines to order of the polynomial
    coefficients = minimize(poly, guess, args=(domain, data)).x

    return coefficients, data

#########################Some Examples#########################

splusc = lambda t : np.sin(t) + np.cos(t)
def func_approx(func, dom_l=0, dom_h=3*np.pi, order=5, part_size=1000, noise=False):
    coef = poly_approx(func, dom_l=dom_l, dom_h=dom_h, part_size=part_size, order=order, noise=noise)[0]
    domain = np.linspace(dom_l, dom_h, part_size)
    values = []
    for t in domain:
        summands = []
        for a in range(len(coef)):
            summands.append(coef[a] * t ** a)
        summands = np.array(summands)
        values.append(np.sum(summands))
    return values

domain = np.linspace(0, 3*np.pi, 1000)
values_approx = func_approx(func=np.sin)


# plt.plot(domain, np.sin(domain) + np.sqrt(2*np.pi/5000)*np.random.randn(5000))
data = poly_approx(np.sin, dom_l=0, dom_h=3*np.pi, part_size=1000, order=5, noise=False)[1]
plt.plot(domain, data)
# plt.plot(domain, np.sin(domain))
plt.plot(domain, values_approx)
plt.show()
