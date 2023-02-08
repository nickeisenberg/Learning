import pandas as pd
import numpy as np

index = np.linspace(0, 10, 11).astype(int)

clusters = np.ones(11)

df = pd.DataFrame(data=clusters,
                  index=index)

df.values
