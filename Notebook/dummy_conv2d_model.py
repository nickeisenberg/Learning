from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(200, 200, 3))
x = layers.Conv2D(64, 3, padding='same')(inputs)
x = layers.Conv2D(128, 3, padding='same')(x)
x = layers.Conv2D(64, 3, padding='same')(x)
outputs = layers.Dense(3)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

train_inps = np.random.normal(size=(100, 200, 200, 3))
train_outs = np.random.normal(size=(100, 200, 200, 3))

model.fit(train_inps, train_outs,
          epochs=1)

test_inp = np.random.normal(size=(1, 200, 200, 3))
test_out = model.predict(test_inp)

plt.subplot(121)
plt.imshow(test_inp[0])
plt.subplot(122)
plt.imshow(test_out[0])
plt.show()


