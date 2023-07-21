import panel as pn
from panel.viewable import Viewer
import param
import pyarrow.parquet as pq
import numpy as np
from hvplot.ui import Controls
import hvplot.pandas
from make_data import SENSOR_SHOTS, AXIS


class df_exp(Viewer):

    default_axis='C1'
    default_sensor = 'Sensor_1'
    default_shot = 4002
    
    axis = param.Selector(AXIS, default=default_axis)
    sensor = param.Selector(
        [*SENSOR_SHOTS['C1'].keys()], default=default_sensor
    )
    shot = param.Selector(
        [*SENSOR_SHOTS['C1'][default_sensor]]
    )

    text = param.String(default='A place to add widgets')

    xlim = param.Range()
    ylim = param.Range()

    def __panel__(self):
        return self._layout

    def __init__(self, **params):

        super().__init__(**params)

        # First tab controls
        self._main_controls = pn.Param(
            self.param, parameters=['axis', 'sensor', 'shot'],
            sizing_mode='stretch_width', max_width=300, show_name=False,
        )

        self._axes_controls = pn.Param(
            self.param, parameters=['xlim', 'ylim'],
            sizing_mode='stretch_width', max_width=300, show_name=False,
        )

        self._nt_controls = pn.Param(
            self.param, parameters=['text'],
            sizing_mode='stretch_width', max_width=300, show_name=False,
        )

        self._tabs = pn.Tabs(
            tabs_location='left', width=400
        )
        self._alert = pn.pane.Alert(
            alert_type='danger', visible=False, sizing_mode='stretch_width'
        )
        self._layout = pn.Column(
            pn.Row(
                self._tabs,
                pn.layout.HSpacer(),
                sizing_mode='stretch_width'
            ),
            pn.layout.HSpacer(),
            sizing_mode='stretch_both'
        )
        
        self._toggle_controls()
        self._retrieve_parquet()
        self._plot()

    @param.depends('axis', 'sensor', watch=True)
    def _retrieve_parquet(self):
        self._parquet = pq.read_table(
            f'./data/{self.axis}/{self.sensor}.parquet'
        )
        self._parq_df = self._parquet.to_pandas()

    @param.depends('axis', 'sensor', watch=True)
    def _update_shots(self):
        shots = SENSOR_SHOTS[self.axis][self.sensor]
        self.param['shot'].objects = shots
        if self.shot not in shots:
            self.shot = shots[0]

    @param.depends('axis', 'sensor', 'shot', watch=True)
    def _toggle_controls(self):
        parameters = ['axis', 'sensor', 'shot']
        self._main_controls.parameters = parameters
        # Control other tabs
        tabs = [
            ('Fields', self._main_controls),
            ('Axes', self._axes_controls),
            ('nt', self._nt_controls)
        ]
        self._tabs[:] = tabs

    @param.depends('axis', 'sensor', 'shot', watch=True)
    def _plot(self):
        try:
            self._hvplot = self._parq_df.hvplot.line(y=str(self.shot))
            self._hvpane = pn.pane.HoloViews(
                self._hvplot,
                sizing_mode='stretch_width',
                margin=(0, 20, 0, 20)
            )
            self._layout[0][1] = self._hvpane
            self._alert.visible = False
        except:
            print(self.shot)
            print(self.shot)


inst = df_exp()

inst.show()
