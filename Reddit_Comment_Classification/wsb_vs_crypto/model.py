import numpy as np
import os
import string
import keras
from keras import layers, Model, Input, Sequential
import matplotlib.pyplot as plt


path = '/Users/nickeisenberg/GitRepos/Python_Notebook/'
path += 'Reddit_Comment_Classification/subreddit_texts/'

posts_and_comments = {}
for fn in os.listdir(path):
    if not fn.endswith('.txt'):
        continue
    name = fn.split('.')[0]
    posts_and_comments[name] = open(path + fn, 'r')

punc = string.punctuation
punc += "‘’'“”"
trans = str.maketrans('', '', punc)

raw_comments = {k: [] for k in posts_and_comments.keys()}
for key in posts_and_comments.keys():
    for line in posts_and_comments[key]:
        if line[:12] == '!!comment!!:':
            line = line[13:].strip().lower().translate(trans)
            raw_comments[key].append(line)

word_counts = {}
for key in raw_comments.keys():
    for line in raw_comments[key]:
        for word in line.split(' '):
            if word not in word_counts.keys():
                word_counts[word] = 1
            else:
                word_counts[word] += 1

arr = np.array([v for v in word_counts.values()])
arr = arr[np.argsort(arr)][::-1]
n_words = 10000
cutoff_val = arr[n_words] + 1
cutoff_ind = np.where(arr == cutoff_val)[0][-1]

remaining = arr[cutoff_ind + 1: n_words].size
vocabulary = {'[UNK]': 0}
for key in raw_comments.keys():
    for line in raw_comments[key]:
        for word in line.split(' '):
            if word_counts[word] >= cutoff:
                if word not in vocabulary.keys():
                    vocabulary[word] = len(vocabulary) + 1
            else:
                if remaining > 0:
                    if word not in vocabulary.keys():
                        vocabulary[word] = len(vocabulary) + 1
                        remaining -= 1

class text_processing:

    def __init__(self, vocab):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}

    def one_hot_encoding(self, line):
        one_hot = np.zeros(len(self.encoder))
        for word in line.split(' '):
            if word in self.encoder.keys():
                one_hot[self.encoder[word] - 1] = 1
            else:
                one_hot[0] = 1
        return one_hot

    def vectorize_encoding(self, line):
        vec = []
        for word in line.split(' '):
            if word in self.encoder.keys():
                vec.append(self.encoder[word])
            else:
                vec.append(0)
        vec = np.array(vec)
        return vec

    def vectorize_decoding(self, vec):
        line = []
        for v in vec:
            line.append(self.decoder[v])
        return line

text_processer = text_processing(vocabulary)
one_hot_comments = {k: [] for k in raw_comments.keys()}
for i, k in enumerate(raw_comments.keys()):
    for line in raw_comments[k]:
        one_hot_comments[k].append(text_processer.one_hot_encoding(line))
one_hot_comments = {
    k: np.array(one_hot_comments[k]) for k in one_hot_comments.keys()
}

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []
for i, k in enumerate(one_hot_comments.keys()):
    one_hots = one_hot_comments[k]
    amt = one_hots.shape[0]
    inds = np.arange(0, amt, 1)
    tr, vl = int(amt * .65), int(amt * .15)
    np.random.seed(41)
    np.random.shuffle(inds)
    x_train.append(one_hots[: tr])
    x_val.append(one_hots[tr: tr + vl])
    x_test.append(one_hots[tr + vl:])
    y_train.append(np.repeat(i, tr))
    y_val.append(np.repeat(i, tr + vl))
    y_test.append(np.repeat(i, amt - tr - vl))
x_train, x_val, x_test = np.vstack(x_train), np.vstack(x_val), np.vstack(x_test)
y_train, y_val, y_test = np.hstack(y_train), np.hstack(y_val), np.hstack(y_test)

train_inds = np.arange(0, x_train.shape[0], 1)
np.random.seed(31)
np.random.shuffle(train_inds)
val_inds = np.arange(0, x_val.shape[0], 1)
np.random.seed(31)
np.random.shuffle(val_inds)
test_inds = np.arange(0, x_test.shape[0], 1)
np.random.seed(31)
np.random.shuffle(test_inds)

x_train = x_train[train_inds]
y_train = y_train[train_inds]
x_val = x_val[val_inds]
y_val = y_val[val_inds]
x_test = x_test[test_inds]
y_test = y_test[test_inds]

inputs = Input((x_train.shape[1],))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath=f'Models/dense_model.keras',
    monitor='val_loss',
    save_best_only=True)
        ]

history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val), 
                    batch_size=512,
                    callbacks=callbacks,
                    epochs=50)

hist_dic = history.history

hist_dic.keys()

plt.plot(hist_dic['accuracy'])
plt.plot(hist_dic['val_accuracy'])
plt.show()
