import panel as pn
from param import Parameterized, Parameter
import param
import pyarrow as pa
import pyarrow.parquet as pq
import hvplot.pandas
import os

# preloaded data
dir = './data/'
files = [f for f in os.listdir(dir) if f.endswith('.parquet')]
paths = [os.path.join(dir, file) for file in files]

df = pq.read_table(paths[0]).to_pandas()
#--------------------------------------------------

class params(Parameterized):
    kind = param.Selector()
    x = param.Selector()
    y = param.Selector()
    y_multi = param.ListSelector(default=[], label='y')
    by = param.ListSelector(default=[])
    groupby = param.ListSelector(default=[])
    x_lim = param.Range()
    logy = param.Boolean()
     
def ts_view_tabs():
    tabs = pn.Tabs(
        tabs_location='left', width=400
    )

    fields = pn.Param(
         params.param, parameters=['kind', 'x', 'y', 'by', 'groupby'],
         sizing_mode='stretch_width', max_width=300, show_name=False,
     )

    axes = pn.Param(
         params.param, parameters=['x_lim', 'logy'],
         sizing_mode='stretch_width', max_width=300, show_name=False,
     )

    tabs[:] = [('Main', fields), ('Axes', axes)]

    return tabs

def ts_view_plot(df=df, col='col_0'):
    return df.hvplot.line(y=col)

def ts_view():
    return pn.Row(
        ts_view_tabs(),
        ts_view_plot()
    )

app = ts_view()
app.show()

