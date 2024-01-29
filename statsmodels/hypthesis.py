import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from utils import make_blobs, t_statistic

#--------------------------------------------------
# Data
#--------------------------------------------------
blob0, blob1 = make_blobs(
    ((0, 0), (.2, .2), 100),
    ((.3, .3), (.2, .2), 100),
)
#--------------------------------------------------

#--------------------------------------------------
# Direct implementation
#--------------------------------------------------
T = t_statistic(blob0[:, 1], blob1[:, 1])
print(T)
#--------------------------------------------------

#--------------------------------------------------
# statsmodels
#--------------------------------------------------
t_test = sm.stats.ttest_ind(blob0[:, 1], blob1[:, 1])

# Outputting the results
print(f"T-statistic: {t_test[0]}, P-value: {t_test[1]}")
#--------------------------------------------------
