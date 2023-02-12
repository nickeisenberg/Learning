import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

df1 = pd.DataFrame(
        data=np.random.normal(size=(1000, 2)),
        columns=['col1', 'col2']
        )

df2 = pd.DataFrame(
        data=np.array([3, 6, 11, 5]).reshape((-1, 1)),
        columns=['col1'],
        index=['ind1', 'ind2', 'ind3', 'ind4']
        )


# using plotly express
fig = px.histogram(
        df1,
        x=df1.col1)
fig.show()

fig = px.histogram(
        df2,
        x=df2.index,
        y=df2.col1)
fig.show()

# using plotly.graph_objects
fig = go.Figure()
_ = fig.add_trace(
        go.Histogram(
            x=df1.col1)
        )
fig.show()

fig = go.Figure()
_ = fig.add_trace(
        go.Bar(
            x=df2.index,
            y=df2.col1)
        )
fig.show()

