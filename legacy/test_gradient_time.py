from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time
from ltc import LTC


x = np.random.randn(8214, 32)
x = tf.convert_to_tensor(x, dtype=tf.float32)

y = np.random.randn(8214, 100)
y = tf.convert_to_tensor(y, dtype=tf.float32)

# create single layer model with 1000 hidden neurons
model = tf.keras.Sequential([
    LTC(100)
])


with tf.GradientTape() as tape:
    y_ = model(x)
    loss = tf.reduce_mean(y-y_)**2
grad = tape.gradient(loss, model.variables)


time_history = []
for i in tqdm(range(20)):
    with tf.GradientTape() as tape:
        # record the forward pass on the tape
        tic = time.time()
        y_ = model(x)
        t_forward = time.time()-tic

        # compute the loss
        tic = time.time()
        loss = tf.reduce_mean(y-y_)**2
        t_loss = time.time()-tic

    # compute the gradient of loss with respect to x
    tic = time.time()
    grad = tape.gradient(loss, model.variables)
    t_grad = time.time()-tic

    # record the results
    time_history.append([t_forward, t_loss, t_grad])

# take the average and print
time_history = np.array(time_history)
print(time_history.mean(axis=0), time_history.var(axis=0))
