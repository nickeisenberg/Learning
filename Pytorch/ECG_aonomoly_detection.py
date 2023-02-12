import pandas as pd
import arff as a2p
from scipy.io.arff import loadarff
import plotly.graph_objects as go
import plotly.express as px

path = '/Users/nickeisenberg/GitRepos/OFFLINE_DIRECTORY/Datasets/ECG5000/'

train = loadarff(path + 'ECG5000_TRAIN.arff')
test = loadarff(path + 'ECG5000_TEST.arff')

# combine the test and training data into one data frame and then shuffle
# the data.
df_train = pd.DataFrame(train[0])
df_test = pd.DataFrame(test[0])

df = pd.concat((df_train, df_test)).sample(frac=1.0)

# Each row in the dataframe is a single heartbeat
fig = go.Figure()
for i in range(10):
    _ = fig.add_trace(
            go.Scatter(
                y=df.iloc[i].values.astype(float),
                )
            )
_ = fig.update_xaxes(
        title={
            'text': 'text',
            })
_ = fig.update_yaxes(
        title={
            'text': 'text',
            })
_ = fig.update_layout(
        title={
            'text': 'text',
            })
fig.show()

# Name the possible classes

# the target column specifies which class the heartbeat falls into.
class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']
normal = 1

# We can see how many of each target we have
val_counts = df['target'].value_counts()
val_counts.index = class_names

val_counts_df = pd.DataFrame(val_counts)

fig = go.Figure()
_ = fig.add_trace(
        go.Bar(
            x=val_counts_df.index,
            y=val_counts_df['target']
            )
        )
_ = fig.update_xaxes(
        title={
            'text': 'Class of hearbeat',
            })
_ = fig.update_yaxes(
        title={
            'text': 'amount from each class',
            })
fig.show()
