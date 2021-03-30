from tqdm import tqdm
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import time
from ltc import LTC


#x = (np.linspace(0,1,100).reshape(-1, 1))
#x = tf.convert_to_tensor(x, dtype=tf.float32)


#y = tf.sin(np.pi*2*x) + tf.cos(4*x)
#y = tf.convert_to_tensor(y, dtype=tf.float32)

x =np.loadtxt('ltc/sensory.csv',delimiter = ';')
x = np.float32(x[0:6000,1:])
x_train = x
x_size = np.shape(x)[0]
x = tf.convert_to_tensor(x, dtype=tf.float32)


y = np.loadtxt('ltc/output.csv', delimiter = ';')
y = np.float32(y[0:6000,1:])
y = tf.convert_to_tensor(y, dtype=tf.float32)


# create single layer model with 1000 hidden neurons
model = tf.keras.Sequential([
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(100, activation='relu'),
    LTC(10),
    #LTC(10),
    #LTC(10),
    #LTC(10),
    LTC(1),
])


#input = tf.keras.layers.Input(shape=[1])
#feed1 = LTC(3)(input)
#feed2 = model2.layers[0](input)
#feed3 = tf.keras.layers.Concatenate()([feed1,input])
#feed2 = tf.keras.layers.concatenate([input,feed1])
#feed4 = LTC(4)(feed1)
#feed5 = tf.keras.layers.Concatenate([feed3,feed4])
#output = LTC(1)(feed5)


#model = tf.keras.Model(input=input,output=output)


optimizer = tf.train.AdamOptimizer(learning_rate=0.5e-2) # define our optimizer

for i in range(5000):
    with tf.GradientTape() as tape:
        # record the forward pass on the tape
        y_ = model(x)
        loss = tf.reduce_mean((y-y_)**2)


    # compute the gradient of loss with respect to x
    grad = tape.gradient(loss, model.variables)

    # if np.sum([tf.reduce_sum(tf.cast(tf.is_nan(g), tf.float32)).numpy() for g in grad]) > 0:
        # break
    optimizer.apply_gradients(zip(grad, model.variables),
        global_step=tf.train.get_or_create_global_step())
    if i%100 ==0:
        print(loss)
print(int(np.sum([np.prod(v.shape) for v in model.variables])))

import matplotlib.pyplot as plt


x = np.linspace(1,x_size,x_size)

plt.plot(x,y,'k')
xx = x
plt.plot(xx, model(x_train), 'r')
plt.show()


x_in_test =np.loadtxt('ltc/sensory.csv',delimiter = ';')
x_in_test = np.float32(x_in_test[6001:,1:])
x_size = np.shape(x_in_test)[0]
x = np.linspace(1,x_size,x_size)

y_test = np.loadtxt('ltc/output.csv', delimiter = ';')
y_test = np.float32(y[6001:,1:])


plt.plot(x,y_test,'k')
xx = x
plt.plot(xx, model(x_in_test), 'r')
plt.show()

#x_test = (np.linspace(0, 10, int((100./1.)*10.)).reshape(-1, 1).astype(np.float32))
#y_test = tf.sin(np.pi*2*x_test) + tf.cos(4*x_test)
#plt.scatter(x_test,y_test)
#plt.plot(x_test, model(x_test), 'r')
#plt.show()
#import pdb; pdb.set_trace()
