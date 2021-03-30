# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

class LTCSolution(tf.keras.layers.Layer):

    def __init__(self,
                state_size,
                sigma_w=0.0001,
                sigma_b=0.0001,
                sigma_x=0.1,
                activation="relu",
                **kwargs):

        self.state_size = state_size
        self.output_size = state_size
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.sigma_x = sigma_x
        self.x_0 = None

        self.f = tf.keras.activations.get(activation)

        super(LTCSolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Save some convience variables

        # batch_size, self.num_features_with_time = input_shape
        # self.num_features = self.num_features_with_time - 1
        k = self.state_size
        # m = self.num_features

        m = input_shape[-1]
        if (isinstance(input_shape[0], tuple)):
            m = input_shape[0][-1]

        # Define the initializers
        W_init = initializers.RandomNormal(mean=0.0, stddev=self.sigma_w)
        b_init = initializers.RandomNormal(mean=0.0, stddev=self.sigma_b)


        # Add the weights
        self.W = self.add_weight("W", shape=(m, k),
            initializer=W_init)
        self.W_r = self.add_weight("W_r", shape=(k, k),
            initializer=W_init)
        self.A = self.add_weight("A", shape=(1, k),
            initializer=W_init)
        self.B = self.add_weight("B", shape=(1, k),
            initializer=W_init)
        self.W_l = self.add_weight("W_l", shape=(1, k),
             initializer=W_init)

        # Add the biases
        self.b = self.add_weight("b", shape=(1, k),
            initializer=b_init)
        self.tau = self.add_weight("omega", shape=(1, k),
            initializer=b_init)

        self.built = True


    # def get_initial_state(self, inputs, batch_size, dtype=None):
    #     if self.x_0 is None:
    #         self.x_0 = tf.random.normal((batch_size, self.state_size),
    #                                     mean=0.0, stddev=self.sigma_x, seed=1)
    #     return self.x_0

    def reset_states(self):
        """ Must be called on the LTC cell specifically since not stateful."""
        self.x_0 = None


    def call(self, inputs, states):
        """ Main forward pass function of the cell.
            Inputs contain both the signal (I) and time (t) concatenated
            together. States are fed in from the previous iteration.

            Return the output and next state of the cell.
        """
        t = 1.0
        if ((isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1):
            t = inputs[1]
            I = inputs[0]
        # inputs = (I and t) = (batch_size, m+1)
        #I = inputs[:, :-1]
        # t = tf.expand_dims(inputs[:, -1], -1)

        # states = x = (batch_size, k)
        x_prev, = states

        # Main equations defining the LTC
        h = tf.matmul(I, self.W) + tf.matmul(x_prev, self.W_r) + self.b
        z = tf.abs(self.tau) + tf.abs(self.W_l * self.f(h))
        #x = (self.x_0 - self.A) * tf.exp(-z*t) * (self.f(-h)) + self.A
        x = (self.B - self.A) * tf.exp(-z * t) * (self.f(-h)) + self.A

        # # Debug print for checking states
        # print(f"x0: {self.x_0[0,0]:.4f} \
        #     prev x: {x_prev[0,0]:.4f} \
        #     new x: {x[0,0]:.4f}")

        return x, [x]



if __name__ == "__main__":
    # Use the LTC cell in a RNN layer:
    num_units = 32
    num_batches = 5
    num_timesteps = 10
    num_features = 4

    cell = LTCSolution(num_units)
    layer = layers.RNN(cell, return_sequences=True)

    x = tf.zeros((num_batches, num_timesteps, num_features))
    t = tf.random.normal((num_batches, num_timesteps, 1))
    inputs = tf.concat((x,t), axis=-1)
    y = layer(inputs)

    print("===============================================================")
    print(f"Got output of shape: {y.shape}")
    print(f"Expected shape:      {(num_batches, num_timesteps, num_units)}")
    print("===============================================================")

    import pdb; pdb.set_trace()
