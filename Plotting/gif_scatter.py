import numpy as np
import plotly.express as px
import pandas as pd

particle_dfs = {}
no_particles = 10
no_time_steps = 100
for i in range(no_particles):
    motion = np.random.normal(size=(2, no_time_steps))
    motion = np.round(np.cumsum(motion, axis=1), 2)
    weight = np.round(np.ones(no_time_steps) * np.random.normal(2, 1, 1), 2)
    time = np.arange(0, no_time_steps, 1)
    particle_dfs[i] = pd.DataFrame(
            data=np.vstack((motion, weight, time)).T,
            columns=['x_pos', 'y_pos', 'weight', 'time'])
    particle_dfs[i]['name'] = np.repeat([f'paricle_{i}'], no_time_steps)

combined_df = pd.concat((d for d in particle_dfs.values()))

fig = px.scatter(
        combined_df,
        x="x_pos", y="y_pos",
        animation_frame="time",
        animation_group="name",
        size="weight", color="name", hover_name="name",
        size_max=55, range_x=[-25, 25], range_y=[-25, 25]
        )
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 50
# fig.layout.updatemenus[0].buttons[0].args[1]['frame']['transition'] = 500
fig.show()
