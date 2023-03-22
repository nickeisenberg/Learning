import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

data = np.random.normal(size=(1000, 5))
cols = [f'col{i}' for i in range(1, 6)]
df = pd.DataFrame(data=data, columns=cols)

pairs = [['col1', f'col{i}'] for i in [2, 3]]

traces = [
        px.scatter(df, x=pair[0], y=pair[1],
                   color=np.arange(0, 1000, 1),
                   color_continuous_scale=px.colors.sequential.Viridis,
                   )['data'][0] for pair in pairs
]

fig = make_subplots(rows=1, cols=2)

for i in [1, 2]:
    fig.add_trace(traces[i - 1],
                  row=1, col=i)

fig.show()
