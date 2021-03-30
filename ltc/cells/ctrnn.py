import tensorflow as tf
import numpy  as np
import os

class CTRNN(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, num_units,cell_clip=-1,global_feedback=False,fix_tau=True):
        self._num_units = num_units
        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.1

        self.global_feedback = global_feedback

        # Time-constant of the cell
        self.fix_tau = fix_tau
        self.tau = 1
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
                if(not self.fix_tau):
                    tau = tf.compat.v1.get_variable('tau', [],initializer=tf.compat.v1.constant_initializer(self.tau))
                    self._tau_var = tau
                    tau = tf.nn.softplus(tau) # Make sure tau is positive
                else:
                    tau = self.tau

                # Input Option 1: RNNCell input
                if(not self.global_feedback):
                    input_f_prime = self._dense(units=self._num_units,inputs=inputs,activation=tf.nn.tanh,name="step")
                for i in range(self._unfolds):
                    # Input Option 2: RNNCell input AND RNN state
                    if(self.global_feedback):
                        fused_input = tf.concat([inputs,state],axis=-1)
                        input_f_prime = self._dense(units=self._num_units,inputs=fused_input,activation=tf.nn.tanh,name="step")

                    # df/dt
                    f_prime = -state/self.tau + input_f_prime

                    # If we solve this ODE with explicit euler we get
                    # f(t+deltaT) = f(t) + deltaT * df/dt
                    state = state + self._delta_t * f_prime

                    # Optional clipping of the RNN cell to enforce stability (not needed)
                    if(self.cell_clip > 0):
                        state = tf.clip_by_value(state,-self.cell_clip,self.cell_clip)

        return state,state
