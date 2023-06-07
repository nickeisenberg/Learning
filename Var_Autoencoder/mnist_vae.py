from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.io import renderers
import pandas as pd
renderers.default = 'browser'

#|%%--%%| <68q1He6YUn|IEJqCEYwDM>

(x_train, x_train_lab), (x_test, x_test_lab) = mnist.load_data()

mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

mnist_labels = np.hstack((x_train_lab, x_test_lab))

#|%%--%%| <IEJqCEYwDM|TE2jFfLVeI>

latent_dim = 2

enc_inps = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(
    32, 3, padding='same', strides=2)(enc_inps)
x = keras.layers.Conv2D(
    64, 3, activation='relu', padding='same', strides=2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(16, activation='relu')(x)

z_mu = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)

encoder = keras.models.Model(enc_inps, [z_mu, z_log_var], name="encoder")

encoder.summary()

#|%%--%%| <TE2jFfLVeI|IAm7DZfbgF>

class Sampler(keras.layers.Layer):
    def call(self, z_mu, z_log_var):
        batch_size = tf.shape(z_mu)[0]
        size = tf.shape(z_mu)[1]
        epsilon = tf.random.normal(shape=(batch_size, size))
        return z_mu + tf.exp(0.5 * z_log_var) * epsilon

#|%%--%%| <IAm7DZfbgF|65Y13TDVOh>

latent_inputs = keras.layers.Input(shape=(latent_dim,))

x = keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(
    64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(
    32, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.models.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

#|%%--%%| <65Y13TDVOh|otFriU1w3X>

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name='reconstruction_loss'
        )
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_weights)
            )
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                'total_loss': self.total_loss_tracker.result(),
                'reconstruction_loss': self.reconstruction_loss_tracker.result(),
                'kl_loss': self.kl_loss_tracker.result(),
            }

    def call(self, inp):
        mu, std = self.encoder(inp)
        z = self.sampler(mu, std)
        inp_recon = self.decoder(z)


#|%%--%%| <otFriU1w3X|uWQq4Biex2>

vae_model = VAE(encoder, decoder)

# vae_model.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
# vae_model.fit(train_imgs, epochs=30, batch_size=128)

#|%%--%%| <uWQq4Biex2|pKGWLDCna6>

vae_model.build(input_shape=(None, 28, 28, 1))

vae_model.summary()

#|%%--%%| <pKGWLDCna6|c59oCk4RK9>

vae_model.load_weights("/mnt/c/Users/EISENBNT/Projects/VAE_MNIST/model_weights.keras")

#|%%--%%| <c59oCk4RK9|7bo6CZ9Br4>

inp = mnist_digits[5]
inp = np.expand_dims(inp, 0)

inp_lab = mnist_labels[5]
print(inp_lab)

inp_recon = vae_model.decoder.predict(
    vae_model.sampler(*vae_model.encoder.predict(inp))
)

fig = px.imshow(inp_recon[0, :, :, 0])
fig.show()

#|%%--%%| <7bo6CZ9Br4|G8Okprc572>

fig = make_subplots(25, 25)
for i, xx in enumerate(np.linspace(-1, 1, 25)):
    for j, yy in enumerate(np.linspace(-1, 1, 25)):
        recon = vae_model.decoder.predict([[xx, yy]])
        _ = fig.add_trace(
            px.imshow(recon[0, :, :, 0]).data[0],
            row=i+1, col=j+1
        )
        _ = fig.update_xaxes(visible=False)
        _ = fig.update_yaxes(visible=False)

fig.show()

#|%%--%%| <G8Okprc572|qOEEZtQmOC>

latents = vae_model.sampler(*vae_model.encoder.predict(mnist_digits))

df_data = np.hstack((latents, mnist_labels.reshape((-1, 1))))

latents = pd.DataFrame(
    data=df_data,
    columns=['x', 'y', 'label']
)

fig = px.scatter(
    latents, x='x', y='y', color='label'
)

fig.show()
