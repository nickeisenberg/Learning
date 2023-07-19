import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import panel as pn

data = np.random.normal(0, 1, (100, 5))
data[0] *= 0
data = data.cumsum(axis=0)

cols = [f'col_{i}' for i in range(5)]
df = pd.DataFrame(
    data,
    columns=cols
)
parq = pa.Table.from_pandas(df)

pq.write_table(parq, 'data/brownian_motion.parquet')


