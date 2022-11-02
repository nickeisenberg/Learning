import tensorflow as tf

x = tf.constant(2.0)
with tf.GradientTape() as gg:
    gg.watch(x)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x ** 5
    dy_dx = g.gradient(y, x)
print(f'dy/dx = {dy_dx}')
print(f'd^2y/dx^2 = {gg.gradient(dy_dx, x)}')

