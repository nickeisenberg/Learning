from tensorflow import keras
import tensorflow as tf 

data = np.linspace(0, 15, 16)
seq_len = 3
seq_stride = 1

targets = keras.utils.timeseries_dataset_from_array(
        data=data[seq_len:],
        targets=None,
        sequence_length=seq_len,
        batch_size=None)


targets = np.array([np.array(tar) for tar in targets])
print(targets)

for inp in targets:
    print(inp)
    

dataset = keras.utils.timeseries_dataset_from_array(
        data=data[: -seq_len],
        targets=targets,
        sequence_length=seq_len,
        sequence_stride=seq_stride,
        batch_size=None)


