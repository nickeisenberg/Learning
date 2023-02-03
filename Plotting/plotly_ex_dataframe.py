import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

no_sigs = 100
bm = np.random.normal(size=(1000, no_sigs))
bm *= np.sqrt(1 / 1000)
bm = np.cumsum(bm, axis=0)
time = np.linspace(0, 1, 1000)

sig_names = np.array([*range(1, no_sigs + 1)])
bm_df = pd.DataFrame(
        data=bm,
        index=time,
        columns=sig_names)

vid = px.colors.sequential.Viridis
vid0 = px.colors.hex_to_rgb(vid[0])
vid1 = px.colors.hex_to_rgb(vid[-1])
colors = px.colors.n_colors(vid0, vid1, no_sigs + 1)

fig = go.Figure()
for c in bm_df.columns:
    _ = fig.add_trace(
            go.Scatter(
                y=bm_df[c].values,
                line={'color': f'rgb{colors[c]}'})
            )
fig.show()
