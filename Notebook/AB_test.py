import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.stats import norm

original = np.random.normal(0, 1, 1000)
mu = np.mean(original)
std = np.std(original)

outcome_1 = deepcopy(original)
outcome_2 = np.random.normal(.05, 1.5, 1000)
outcome_3 = np.random.normal(0, 3, 1000)

def z_test(data, sig, mu):
    z = (np.mean(data) - mu) / (sig / np.sqrt(len(data)))
    return z

z_1 = z_test(data = outcome_1, sig=sig, mu=mu)
z_2 = z_test(data = outcome_2, sig=sig, mu=mu)
z_3 = z_test(data = outcome_3, sig=sig, mu=mu)

def two_tail(x):
    return norm.cdf(-1 * np.abs(x)) * 2

print(two_tail(z_1))  # do not reject
print(two_tail(z_2))  # do not reject
print(two_tail(z_3))  # reject 
