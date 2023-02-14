import pandas as pd
import numpy as np
import arff as a2p
from scipy.io.arff import loadarff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from copy import deepcopy

path = '/Users/nickeisenberg/GitRepos/OFFLINE_DIRECTORY/Datasets/ECG5000/'

train = loadarff(path + 'ECG5000_TRAIN.arff')
test = loadarff(path + 'ECG5000_TEST.arff')

# combine the test and training data into one data frame and then shuffle
# the data.
df_train = pd.DataFrame(train[0])
df_test = pd.DataFrame(test[0])

df = pd.concat((df_train, df_test)).sample(frac=1.0)

df_heart = deepcopy(df[df.columns.values[: -1]])
df_target = deepcopy(pd.DataFrame(df['target'].astype(int)))

df_heart.head()
df_target.head()

# Each row in the dataframe is a single heartbeat
fig = go.Figure()
for i in range(10):
    _ = fig.add_trace(
            go.Scatter(
                y=df_heart.iloc[i].values.astype(float),
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
val_counts = df_target['target'].value_counts()
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

# Lets view a smoothed average of each class

class_avgs = {}
for i in range(1, 6):
    df_target_i_inds = df_target.loc[df_target.target == i].index
    df_heart_i = deepcopy(df_heart.loc[df_target_i_inds].values)
    avg = df_heart_i.mean(axis=0)
    smooth_avg = pd.Series(avg).rolling(window=10).mean().iloc[10 - 1:]
    class_avgs[i] = smooth_avg.values

nrow, ncol = 2,3 
rcs = []  # row and column subplot pairs
for i in range(1, nrow + 1):
    for j in range(1, ncol + 1):
        rcs.append((i, j))
fig = make_subplots(rows=nrow, cols=ncol,
                    subplot_titles=class_names)
for i, rc in enumerate(rcs[: 5]):
    _ = fig.add_trace(
            go.Scatter(
                y=class_avgs[i + 1],
                ),
            row=rc[0], col=rc[1]
            )
    _ = fig.update_xaxes(
            title={
                'text':'',
                },
            row=rc[0], col=rc[1])
    _ = fig.update_yaxes(
            title={
                'text':'',
                },
            row=rc[0], col=rc[1])
_ = fig.update_layout(
        title={
            'text':'',
            })
fig.show()

# Now we can start with getting the training set for the autoencoder
# We will train the autoecoder on the normal heartbeats.

df_normal_heart = df_heart.loc[df_target.target == 1]

# We can combine all the other anomalies into one 'anomalies' dataframe'
df_heart_anamoly = df_heart.loc[df_target.target != 1]

# Now we can split the normal heart rates into train, val and testing groups.
def train_val_test(df, tr_v_split=(.7, .15), seed=1):
    inds = df.index.values
    np.random.shuffle(inds)
    train_upper = int(len(inds) * tr_v_split[0])
    train_inds = inds[: train_upper]
    val_upper = train_upper + int(len(inds) * tr_v_split[1])
    val_inds = inds[train_upper: val_upper]
    test_inds = inds[val_upper: ]
    train_df = df.loc[train_inds]
    val_df = df.loc[val_inds]
    test_df = df.loc[test_inds]
    return train_df, val_df, test_df

train_df, val_df, test_df = train_val_test(df_normal_heart)


