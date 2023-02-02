import plotly.express as px
import pandas as pd
import numpy as np

bm = np.random.normal(size=(1000, 10))
bm *= np.sqrt(1 / 1000)
bm = np.cumsum(bm, axis=0)

time = np.linspace(0, 1, 1000)

bm_df = pd.DataFrame(
        data=bm,
        index=time)

fig = px.line(bm_df,
              x=bm_df.index,
              y=bm_df.columns,
              color=bm_df.columns)
fig.show()

print(bm_df.columns)
