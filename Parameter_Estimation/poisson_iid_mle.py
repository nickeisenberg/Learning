import numpy as np
from scipy.stats import poisson
import plotly.graph_objs  as go
import pandas as pd

mu = 4
rv = poisson(mu)
pmf = rv.pmf
cdf = rv.cdf

x = np.arange(rv.ppf(.001), rv.ppf(.99))
prob = pmf(x)
cdf = np.cumsum(prob)

fig = go.Figure(
        [go.Scatter(x=x, y=pmf(x), name='pmf'),
         go.Scatter(x=x, y=cdf(x), name='cdf')]
        )
_ = fig.update_layout(title={'text': f'Possion Randon Variable with mean {mu}',
                             'x': .5})
fig.show()

random_sample = np.random.poisson(lam=4, size=50)

fig = go.Figure()
_ = fig.add_trace(go.Histogram(x=random_sample))
fig.show()

def likelihood(sample, mu):
    return np.product(poisson(mu).pmf(sample))

mus = np.arange(1, 11)
ll_ins = [] 
for mu in mus:
    ll_ins.append(likelihood(random_sample, mu))
ll_ins = np.array(ll_ins)

fig = go.Figure(
        [go.Scatter(x=mus, y=ll_ins)]
        )
fig.show()

print(f' MLE of the sample is {mus[np.argmax(ll_ins)]}')
