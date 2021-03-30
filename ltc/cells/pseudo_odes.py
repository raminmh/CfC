import tensorflow as tf
import numpy as np


# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


class PseudoODECell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(PseudoODECell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.base_ff = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(self.units, tf.nn.silu),  # Swish activation
                tf.keras.layers.Dense(self.units, tf.nn.silu),  # Swish activation
            ]
        )
        self.ff1 = tf.keras.layers.Dense(self.units, lecun_tanh)
        self.ff2 = tf.keras.layers.Dense(self.units, lecun_tanh)
        self.time_a = tf.keras.layers.Dense(self.units)
        self.time_b = tf.keras.layers.Dense(self.units)
        self.built = True

    def call(self, inputs, states, **kwargs):
        hidden_state = states[0]
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        x = tf.keras.layers.Concatenate()([inputs, hidden_state])
        x = self.base_ff(x)
        ff1 = self.ff1(x)
        ff2 = self.ff2(x)
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = tf.nn.sigmoid(t_a * tf.reshape(elapsed, [-1, 1]) + t_b)
        new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, [new_hidden]
