import tensorflow as tf
import time
import numpy as np

from ltc import LTCCell


num_units = 32
num_batches = 5
num_timesteps = 10
num_features = 4
time_major = False
TF2 = True

np.random.seed(1)
x_np = np.random.randn(num_batches, num_timesteps, num_features)
# x_np = np.ones((num_batches, num_timesteps, num_features))
t_np = np.random.randn(num_batches, num_timesteps, 1)
inputs_np = np.concatenate((x_np, t_np), axis=-1)
if time_major:
    inputs_np = np.transpose(inputs_np, [1,0,2])

if TF2:
    tf.random.set_seed(1)
    cell = LTCCell(num_units)
    layer = tf.keras.layers.RNN(cell, return_sequences=True, time_major=time_major)

    output_np = layer(inputs_np)
    print("===============================================================")
    print(f"Got output of shape: {output_np.shape}")
    print(f"Expected shape:      {(num_batches, num_timesteps, num_units)}")
    print("===============================================================")

else:
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None, None, num_features+1])
    tf.random.set_seed(1)
    cell = LTCCell(num_units)
    output, _ = tf.compat.v1.nn.dynamic_rnn(cell, x, dtype=tf.float32, time_major=time_major)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    output_np = sess.run(output, feed_dict={x: inputs_np})

if time_major:
    output_np = np.transpose(output_np, [1,0,2])
print(output_np[0,9,7])
