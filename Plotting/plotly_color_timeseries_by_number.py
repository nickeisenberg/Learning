import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import numpy as np

no_sigs = 100

# Brownian motions
bm = np.random.normal(size=(1000, no_sigs))
bm *= np.sqrt(1 / 1000)
bm = np.cumsum(bm, axis=0)
time = np.linspace(0, 1, 1000)
sig_names = np.array([*range(1, no_sigs + 1)])
bm_df = pd.DataFrame(
        data=bm,
        index=time,
        columns=sig_names)
#-------------------------------------------------- 

# lines
time = np.linspace(0, 1, 100)
line = lambda x, b: x + b
lines = np.array(
        [line(time, b) for b in np.linspace(0, 1, 100)]
        )
line_names = np.array([*range(1, no_sigs + 1)])
lines_df = pd.DataFrame(
        data=lines.T,
        index=time,
        columns=line_names)
#--------------------------------------------------

# rgba colors and plot
cmap = plt.get_cmap('viridis', no_sigs)
norm = mplc.Normalize(vmin=line_names[0], vmax=line_names[-1])

# Initialize the color bar
c_bar_data = np.vstack([lines_df.columns.values,
                        np.empty((2, int(no_sigs)))]).T
colorbar_df = pd.DataFrame(
        data=c_bar_data,
        columns=['sig_num', 'empt1', 'empt2'])
fig = px.scatter(colorbar_df,
                 x='empt1',
                 y='empt2',
                 color='sig_num',
                 color_continuous_scale=px.colors.sequential.Viridis)
#--------------------------------------------------

# Plot the lines
for c in lines_df.columns:
    _ = fig.add_trace(
            go.Scatter(
                y=lines_df[c].values,
                line={'color': f'rgba{cmap(norm(c))}'},
                showlegend=False)
            )
_ = fig.update_xaxes(
        title={'text': 'x-axis'})
_ = fig.update_yaxes(
        title={'text': 'y-axis'})
fig.show()
#--------------------------------------------------
