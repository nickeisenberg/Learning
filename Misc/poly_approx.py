import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def poly_approx(fun, dom_l=0, dom_h=1, part_size=100, order=1, noise=False):

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

    guess = np.random.randn(order)
    coefficients = minimize(poly, guess, args=(domain, data)).x

    return coefficients

#########################Some Examples#########################

'''
def approx_sin(dom_l=0, dom_h=3*np.pi, order=5, part_size=5000, noise=False):
    coef = poly_approx(np.sin, dom_l=dom_l, dom_h=dom_h, order=order)
    domain = np.linspace(dom_l, dom_h, part_size)
    values = []
    for t in domain:
        summands = []
        for a in range(len(coef)):
            summands.append(coef[a] * t ** a)
        summands = np.array(summands)
        values.append(np.sum(summands))
    return values

domain = np.linspace(0, 3*np.pi, 5000)
values_sin = approx_sin()

# plt.plot(domain, np.sin(domain) + np.sqrt(2*np.pi/5000)*np.random.randn(5000))
plt.plot(domain, np.sin(domain))
plt.plot(domain, values_sin)
plt.show()
'''

'''
def approx_poly(dom_l=-4, dom_h=4, order=5, part_size=5000, noise=False):
    fun = lambda x : x ** 4 - 3 * x ** 2 + x + 1
    coef = poly_approx(fun, dom_l=dom_l, dom_h=dom_h, order=order, noise=noise)
    domain = np.linspace(dom_l, dom_h, part_size)
    values = []
    for t in domain:
        summands = []
        for a in range(len(coef)):
            summands.append(coef[a] * t ** a)
        summands = np.array(summands)
        values.append(np.sum(summands))
    return values

domain = np.linspace(-4, 4, 5000)
values = approx_poly()

fun_ = lambda x : x ** 4 - 3 * x ** 2 + x + 1
plt.plot(domain, fun_(domain), label='$x^4 - 3 x^2 + x + 1$')
plt.plot(domain, values, ':', label='Order 4 approximation')
plt.legend(loc='upper center')
plt.show()
'''
