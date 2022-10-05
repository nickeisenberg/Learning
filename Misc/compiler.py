import pandas as pd
import numpy as np
from scipy.optimize import minimize

def fun(param):
    x, y = param
    return x * 2 - (y + x ) ** 2

initial_guess = [2,2]
param = minimize(fun, initial_guess)
print(param)
