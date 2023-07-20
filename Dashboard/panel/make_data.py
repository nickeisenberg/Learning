import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import os

class make_data():

    def __init__(self, path='./'):
        self.path = path
        self.axis = ['C1', 'C2']
        self.shots = np.arange(4000, 4040)
        self.sensor_shots = {
            'C1': {
                'Sensor_1': np.random.choice(self.shots, 19, replace=False),
                'Sensor_2': np.random.choice(self.shots, 13, replace=False),
                'Sensor_3': np.random.choice(self.shots, 9, replace=False),
                'Sensor_4': np.random.choice(self.shots, 18, replace=False),
            },
            'C2': {
                'Sensor_1': np.random.choice(self.shots, 13, replace=False),
                'Sensor_2': np.random.choice(self.shots, 14, replace=False),
                'Sensor_3': np.random.choice(self.shots, 18, replace=False),
                'Sensor_4': np.random.choice(self.shots, 2, replace=False),
            }
        }
        self._generate()
        self._generate_parquet()
    
    @classmethod
    def b_paths(cls, size):
        data = np.random.normal(0, 1, (100, size))
        data[0] *= 0
        data = data.cumsum(axis=0)
        return data
    
    def _generate(self):
        self.sensor_data = {}
        for k in self.sensor_shots.keys():
            self.sensor_data[k] = {}
            for sen, shot in self.sensor_shots[k].items():
                self.sensor_data[k][sen] = self.b_paths(shot.size)

    def _generate_parquet(self):
        for k in self.sensor_data.keys():
            to_path = os.path.join(self.path, k)
            os.makedirs(to_path)
            for sen, d in self.sensor_data[k].items():
                parq = pa.Table.from_pandas(
                    pd.DataFrame(d)
                )
                pq.write_table(parq, os.path.join(to_path, f'{sen}.parquet'))

inst = make_data(path='./data')


