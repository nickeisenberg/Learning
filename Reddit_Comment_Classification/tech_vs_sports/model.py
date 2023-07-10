import reddit_funcs as rfunc
import numpy as np
import keras
from keras import layers, Model, Input, Sequential

path = '/Users/nickeisenberg/GitRepos/'
path += 'DataSets_local/subreddit_texts/tech_vs_sport/'

# create the vocabulary
processeser = rfunc.TextProcessing()
processeser.get_words(
    f'{path}tech_coms.txt', ignore=['!!body!!'], from_file=True
)
processeser.get_words(
    f'{path}sport_coms.txt', ignore=['!!body!!'], from_file=True
)

no_words=100
processeser.get_vocab(no_words=no_words)

# encode the comments and create the datasets
dataset = rfunc.TextDataset(no_words + 1, 1)

text_encoder = rfunc.TextEncoding(vocab=processeser.vocab).one_hot_encoding

dataset.from_txt_file(
    encoder=text_encoder,
    path_to_txt_file=f'{path}tech_coms.txt',
    labels = 0
)

dataset.from_txt_file(
    encoder=text_encoder,
    path_to_txt_file=f'{path}sport_coms.txt',
    labels = 1
)

dataset.clean_up_nan()

# define the model
inputs = Input((dataset.x_train.shape[1],))
x = layers.Dense(1024, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
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

history = model.fit(
    dataset.x_train, dataset.y_train,
    validation_data=(dataset.x_val, dataset.y_val), 
    batch_size=512,
    callbacks=callbacks,
    epochs=10
)

preds = model.predict(dataset.x_test)
preds = np.round(preds).astype(int)

1 - np.mean(np.abs(preds - dataset.y_train))




