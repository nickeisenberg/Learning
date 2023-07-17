import panel as pn
import hvplot
import hvplot.pandas
import pandas as pd
import numpy as np

pn.extension(design='material')

csv_file = ("https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv")
data = pd.read_csv(csv_file, parse_dates=["date"], index_col=['date'])

def view_hvplot(avg, highlight):
    plt0 = avg.hvplot(height=300, width=400, legend=False)
    plt1 = highlight.hvplot.scatter(
        color="orange", padding=0.1, legend=False
    )
    return plt0 * plt1

def find_outliers(
    data=data,
    variable="Temperature",
    window=30,
    sigma=10,
    view_fn=view_hvplot
    ):
    avg = data[variable].rolling(window=window).mean()
    residual = data[variable] - avg
    std = residual.rolling(window=window).std()
    outliers = np.abs(residual) > std * sigma
    return view_fn(avg, avg[outliers])

plot = find_outliers()
hvplot.show(plot)

# app
variable_widget = pn.widgets.Select(
    name="variable", value="Temperature", options=list(data.columns)
)
window_widget = pn.widgets.IntSlider(name="window", value=30, start=1, end=60)
sigma_widget = pn.widgets.IntSlider(name="sigma", value=10, start=0, end=20)

bound_plot = pn.bind(
    find_outliers, variable=variable_widget,
    window=window_widget, sigma=sigma_widget
)

widgets = pn.Column(variable_widget, window_widget, sigma_widget)
first_app = pn.Row(
    widgets, bound_plot
)

first_app.show()
