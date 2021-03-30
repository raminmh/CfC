import tensorflow as tf
import numpy  as np
import os


class CTGRU(tf.compat.v1.nn.rnn_cell.RNNCell):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, num_units,M=8,cell_clip=-1):
        self._num_units = num_units
        self.M = M
        self.cell_clip = cell_clip
        self.ln_tau_table = np.empty(self.M)
        tau = 1
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            tau = tau * (10.0**0.5)

    @property
    def state_size(self):
        return self._num_units*self.M

    @property
    def output_size(self):
        return self._num_units

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.compat.v1.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.compat.v1.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.compat.v1.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        return y

    def __call__(self, inputs, state, scope=None):
        self._input_size = int(inputs.shape[1])

        # CT-GRU input is actually a matrix and not a vector
        h_hat = tf.reshape(state,[-1,self._num_units,self.M])
        h = tf.reduce_sum(input_tensor=h_hat,axis=2)
        state = None # Set state to None, to avoid misuses (bugs) in the code below

        with tf.compat.v1.variable_scope(scope or type(self).__name__):
            with tf.compat.v1.variable_scope("Gates"):  # Reset gate and update gate.
                fused_input = tf.concat([inputs,h],axis=-1)
                ln_tau_r = tf.compat.v1.layers.Dense(self._num_units*self.M,activation=None,name="tau_r")(fused_input)
                ln_tau_r = tf.reshape(ln_tau_r,shape=[-1,self._num_units,self.M])
                sf_input_r = -tf.square(ln_tau_r-self.ln_tau_table)
                rki = tf.nn.softmax(logits=sf_input_r,axis=2)

                q_input = tf.reduce_sum(input_tensor=rki*h_hat,axis=2)
                reset_value = tf.concat([inputs,q_input],axis=1)
                qk = self._dense(units=self._num_units,inputs=reset_value,activation=tf.nn.tanh,name="detect_signal")

                qk = tf.reshape(qk,[-1,self._num_units,1]) # in order to broadcast

                ln_tau_s = tf.compat.v1.layers.Dense(self._num_units*self.M,activation=None,name="tau_s")(fused_input)
                ln_tau_s = tf.reshape(ln_tau_s,shape=[-1,self._num_units,self.M])
                sf_input_s = -tf.square(ln_tau_s-self.ln_tau_table)
                ski = tf.nn.softmax(logits=sf_input_s,axis=2)

                h_hat_next = ((1-ski)*h_hat + ski*qk)*np.exp(-1.0/self.ln_tau_table)

                if(self.cell_clip > 0):
                    h_hat_next = tf.clip_by_value(h_hat_next,-self.cell_clip,self.cell_clip)
                # Compute new state
                h_next = tf.reduce_sum(input_tensor=h_hat_next,axis=2)
                h_hat_next_flat = tf.reshape(h_hat_next,shape=[-1,self._num_units*self.M])

        return h_next, h_hat_next_flat
