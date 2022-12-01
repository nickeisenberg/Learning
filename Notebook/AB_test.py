import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.stats import norm

def z_test(data, sig, mu):
    z = (np.mean(data) - mu) / (sig / np.sqrt(len(data)))
    return z

def simulator(data, sig, mu, amt):
    sims = [np.random.choice(data, size=125, replace=False) for i in range(amt)]
    return np.array(sims)

def two_tail(x):
    return norm.cdf(-1 * np.abs(x)) * 2

original = np.random.normal(0, 1, 1000)
mu = np.mean(original)
std = np.std(original)

outcome_1 = deepcopy(original)
outcome_2 = np.random.normal(.05, 1.5, 1000)
outcome_3 = np.random.normal(0, 3, 1000)
sims_2 = simulator(outcome_2, sig=std, mu=mu, amt=100)
sims_3 = simulator(outcome_3, sig=std, mu=mu, amt=100)

z_1 = z_test(data=outcome_1, sig=std, mu=mu)
z_2 = z_test(data=outcome_2, sig=std, mu=mu)
z_3 = z_test(data=outcome_3, sig=std, mu=mu)
z_sims_2 = [z_test(data=sim, sig=std, mu=mu) for sim in sims_2]
z_sims_3 = [z_test(data=sim, sig=std, mu=mu) for sim in sims_3]
p_sims_2 = two_tail(np.array(z_sims_2))
p_sims_3 = two_tail(np.array(z_sims_3))


count = 0
for p in p_sims_2:
    if p < .05:
        count += 1
print(count / len(p_sims_2))

count = 0
for p in p_sims_3:
    if p < .05:
        count += 1
print(count / len(p_sims_3))

print(two_tail(z_1))  # do not reject
print(two_tail(z_2))  # do not reject
print(two_tail(z_3))  # reject 
