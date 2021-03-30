import numpy as np
import tensorflow as tf

from ltc_tf1 import layer


# Create the model
x = tf.placeholder(tf.float32, [None, 1], name='input')
y = tf.placeholder(tf.float32, [None, 1], name='output')

hidden = layer.dense(x, 10)
output = layer.dense(x, 1)

# define the optimizer
loss = tf.reduce_mean((output - y)**2)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

# Create the dataset
x_data = np.linspace(0, 1, 20).reshape(-1,1)
y_data = x_data * 2

# start training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, loss_val = sess.run([train_step, loss], feed_dict={x: x_data, y: y_data})
    if i % 100 == 0:
        print loss_val
