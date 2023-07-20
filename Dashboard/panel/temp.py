import param
import panel as pn
from make_data import SENSOR_SHOTS, AXIS, SHOTS

class which_shots(param.Parameterized):

    default_axis='C1'
    default_sensor = [*SENSOR_SHOTS[default_axis].keys()][0]
    default_shot = SENSOR_SHOTS[default_axis][default_sensor][0]
    
    axis = param.Selector(AXIS, default='C1')
    sensor = param.Selector(
        [*SENSOR_SHOTS['C1'].keys()], default=default_sensor
    )
    shot = param.Selector(
        [*SENSOR_SHOTS['C1'][default_sensor]]
    )
    
    @param.depends('axis', 'sensor', watch=True)
    def _update_shots(self):
        shots = self._sensor_shots[self.axis][self.sensor]
        self.param['shot'].objects = shots

c = which_shots()

pn.Row(c).show()

SENSOR_SHOTS['C1']['Sensor_1']

shots = param.Selector(
    [*SENSOR_SHOTS['C1']['Sensor_1']], default=4004
)


