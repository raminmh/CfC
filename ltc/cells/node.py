import tensorflow as tf
import numpy  as np
import os


class NODE(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1):
        self._num_units = num_units
        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.1

        self.cell_clip = cell_clip


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def export_weights(self,dirname,sess,output_weights=None):
        os.makedirs(dirname,exist_ok=True)
        w,b = sess.run([self.W,self.b])

        tau = np.ones(1)
        if(not self.fix_tau):
            sp_op = tf.nn.softplus(self._tau_var)
            tau = sess.run(sp_op)
        if(not output_weights is None):
            output_w,output_b = sess.run(output_weights)
            np.savetxt(os.path.join(dirname,"output_w.csv"),output_w)
            np.savetxt(os.path.join(dirname,"output_b.csv"),output_b)
        np.savetxt(os.path.join(dirname,"w.csv"),w)
        np.savetxt(os.path.join(dirname,"b.csv"),b)
        np.savetxt(os.path.join(dirname,"tau.csv"),b)

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass


    def _ode_step_runge_kutta(self,inputs,state):

        for i in range(self._unfolds):
            k1 = self._delta_t*self._f_prime(inputs,state)
            k2 = self._delta_t*self._f_prime(inputs,state+k1*0.5)
            k3 = self._delta_t*self._f_prime(inputs,state+k2*0.5)
            k4 = self._delta_t*self._f_prime(inputs,state+k3)

            state = state + (k1+2*k2+2*k3+k4)/6.0

            # Optional clipping of the RNN cell to enforce stability (not needed)
            if(self.cell_clip > 0):
                state = tf.clip_by_value(state,-self.cell_clip,self.cell_clip)

        return state

    def _f_prime(self,inputs,state):
        fused_input = tf.concat([inputs,state],axis=-1)
        input_f_prime = self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.tanh,name="step")
        return input_f_prime

    def _dense(self,units,inputs,activation,name,bias_initializer=tf.compat.v1.constant_initializer(0.0)):
        input_size = int(inputs.shape[-1])
        W = tf.compat.v1.get_variable('W_{}'.format(name), [input_size, units])
        b = tf.compat.v1.get_variable("bias_{}".format(name), [units],initializer=bias_initializer)

        y = tf.matmul(inputs,W) + b
        if(not activation is None):
            y = activation(y)

        self.W = W
        self.b = b

        return y

    def __call__(self, inputs, state, scope=None):
        # CTRNN ODE is: df/dt = NN(x) - f
        # where x is the input, and NN is a MLP.
        # Input could be: 1: just the input of the RNN cell
        # or 2: input of the RNN cell merged with the current state

        self._input_size = int(inputs.shape[-1])
        with tf.compat.v1.variable_scope(scope or type(self).__name__):
            with tf.compat.v1.variable_scope("RNN",reuse=tf.compat.v1.AUTO_REUSE):  # Reset gate and update gate.

                state = self._ode_step_runge_kutta(inputs,state)

        return state,state
