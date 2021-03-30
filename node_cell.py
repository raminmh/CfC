import tensorflow as tf
import numpy as np

class CTRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, num_unfolds,method,tau=1, **kwargs):
        self.methods = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
            }
        self.units = units
        self.state_size = units
        self.num_unfolds = num_unfolds
        self.method = method
        self.tau = tau
        super(CTRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',
            name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name='bias')
        self.scale = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Constant(1.0),
            name='scale')

        self.built = True

    def call(self, inputs, states):
        hidden_state = states[0]
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        delta_t = elapsed/self.num_unfolds
        method = self.methods[self.method]
        for i in range(self.num_unfolds):
            hidden_state = method(inputs,hidden_state,delta_t)
        return hidden_state, [hidden_state]

    def dfdt(self,inputs, hidden_state):
        h_in = tf.matmul(inputs,self.kernel)
        h_rec = tf.matmul(hidden_state,self.recurrent_kernel)
        dh_in = self.scale * tf.nn.tanh(h_in + h_rec + self.bias)
        if(self.tau > 0):
            dh = dh_in - hidden_state*self.tau
        else:
            dh = dh_in
        return dh

    def euler(self,inputs, hidden_state, delta_t):
        dy = self.dfdt(inputs, hidden_state)
        return hidden_state+delta_t*dy

    def heun(self,inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state+delta_t*k1)
        return hidden_state + delta_t*0.5*(k1 + k2)

    def rk4(self,inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state+k1*delta_t*0.5)
        k3 = self.dfdt(inputs, hidden_state+k2*delta_t*0.5)
        k4 = self.dfdt(inputs, hidden_state+k3*delta_t)

        return hidden_state + delta_t *( k1 + 2*k2 + 2*k3 + k4)/6.0


class LSTMCell(tf.keras.layers.Layer):

    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = (units,units)
        self.initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        super(LSTMCell, self).__init__(**kwargs)

    def get_initial_state(self,inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size,self.units],dtype=tf.float32),
            tf.zeros([batch_size,self.units],dtype=tf.float32)
        )

    def build(self, input_shape):
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_shape = (input_shape[0][-1]+input_shape[1][-1],)

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], 4*self.units),
            initializer=self.initializer,
            name='input_kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4*self.units),
            initializer=self.recurrent_initializer,
            name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(4*self.units),
            initializer=tf.keras.initializers.Zeros(),
            name='bias')

        self.built = True

    def call(self, inputs, states):
        cell_state,output_state = states
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            inputs = tf.concat([inputs[0],inputs[1]],axis=-1)

        z = tf.matmul(inputs,self.input_kernel)+tf.matmul(output_state,self.recurrent_kernel)+self.bias
        i,ig,fg,og = tf.split(z,4,axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg+1.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state*forget_gate + input_activation*input_gate
        output_state = tf.nn.tanh(new_cell)*output_gate

        return output_state, [new_cell,output_state]



class CTLSTM(tf.keras.layers.Layer):

    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = (units,units)
        self.initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        self.ctrnn = CTRNNCell(self.units,num_unfolds=4,method="euler")
        super(CTLSTM, self).__init__(**kwargs)

    def get_initial_state(self,inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size,self.units],dtype=tf.float32),
            tf.zeros([batch_size,self.units],dtype=tf.float32)
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.ctrnn.build([self.units])
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4*self.units),
            initializer=self.initializer,
            name='input_kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4*self.units),
            initializer=self.recurrent_initializer,
            name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(4*self.units),
            initializer=tf.keras.initializers.Zeros(),
            name='bias')

        self.built = True

    def call(self, inputs, states):
        cell_state,ode_state = states
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        z = tf.matmul(inputs,self.input_kernel)+tf.matmul(ode_state,self.recurrent_kernel)+self.bias
        i,ig,fg,og = tf.split(z,4,axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg+3.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state*forget_gate + input_activation*input_gate
        ode_input = tf.nn.tanh(new_cell)*output_gate

        ode_output,new_ode_state = self.ctrnn.call([ode_input,elapsed],[ode_state])
        # ode_output = ode_input
        # new_ode_state = [ode_input]

        return ode_output, [new_cell,new_ode_state[0]]


class CTGRU(tf.keras.layers.Layer):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, units,M=8,**kwargs):
        self.units = units
        self.M = M
        self.state_size = units*self.M

        # Pre-computed tau table (as recommended in paper)
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0**0.5)

        super(CTGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.retrieval_layer = tf.keras.layers.Dense(self.units*self.M,activation=None)
        self.detect_layer = tf.keras.layers.Dense(self.units,activation="tanh")
        self.update_layer = tf.keras.layers.Dense(self.units*self.M,activation=None)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        batch_dim = tf.shape(inputs)[0]

        # States is actually 2D
        h_hat = tf.reshape(states[0],[batch_dim,self.units,self.M])
        h = tf.reduce_sum(h_hat,axis=2)
        states = None # Set state to None, to avoid misuses (bugs) in the code below

        # Retrieval
        fused_input = tf.concat([inputs,h],axis=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r,shape=[batch_dim,self.units,self.M])
        sf_input_r = -tf.square(ln_tau_r-self.ln_tau_table)
        rki = tf.nn.softmax(logits=sf_input_r,axis=2)

        q_input = tf.reduce_sum(rki*h_hat,axis=2)
        reset_value = tf.concat([inputs,q_input],axis=1)
        qk =  self.detect_layer(reset_value)
        qk = tf.reshape(qk,[batch_dim,self.units,1]) # in order to broadcast

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s,shape=[batch_dim,self.units,self.M])
        sf_input_s = -tf.square(ln_tau_s-self.ln_tau_table)
        ski = tf.nn.softmax(logits=sf_input_s,axis=2)

        # Now the elapsed time enters the state update
        base_term = (1-ski)*h_hat + ski*qk
        exp_term = tf.exp(-elapsed/self.tau_table)
        # Clipping at 1 added to avoid exploding gradient
        # exp_term = tf.clip_by_value(exp_term,tf.constant(-1.0,dtype=tf.float32),tf.constant(1.0,dtype=tf.float32))
        exp_term = tf.reshape(exp_term,[batch_dim,1,self.M])
        h_hat_next = base_term*exp_term

        # Compute new state
        # h_hat_next = tf.clip_by_value(h_hat_next,-5,5)
        h_next = tf.reduce_sum(h_hat_next,axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next,shape=[batch_dim,self.units*self.M])
        return h_next, [h_hat_next_flat]

class VanillaRNN(tf.keras.layers.Layer):
    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = units
        
        super(VanillaRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._layer = tf.keras.layers.Dense(self.units,activation="tanh")
        self._out_layer = tf.keras.layers.Dense(self.units,activation=None)
        self._tau = self.add_weight("tau",shape=(self.units),dtype=tf.float32,initializer=tf.keras.initializers.Constant(0.1))
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        fused_input = tf.concat([inputs,states[0]],axis=-1)
        new_states = self._out_layer(self._layer(fused_input)) - elapsed*self._tau

        return new_states, [new_states]
        

class BidirectionalRNN(tf.keras.layers.Layer):
    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = (units,units,units)

        self.ctrnn = CTRNNCell(self.units,num_unfolds=4,method="euler")
        self.lstm = LSTMCell(units=self.units)

        super(BidirectionalRNN, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._out_layer = tf.keras.layers.Dense(self.units,activation=None)
        fused_dim = ((input_dim+self.units,),(1,))
        self.lstm.build(fused_dim)
        self.ctrnn.build(fused_dim)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        lstm_state = [states[0],states[1]]
        lstm_input = [tf.concat([inputs,states[2]],axis=-1),elapsed]
        ctrnn_state = [states[2]]
        ctrnn_input = [tf.concat([inputs,states[1]],axis=-1),elapsed]
        
        lstm_out,new_lstm_states = self.lstm.call(lstm_input,lstm_state)
        ctrnn_out, new_ctrnn_state = self.ctrnn.call(ctrnn_input,ctrnn_state)

        fused_output = lstm_out + ctrnn_out
        return fused_output, [new_lstm_states[0],new_lstm_states[1],new_ctrnn_state[0]]
        


class GRUD(tf.keras.layers.Layer):
    # Implemented according to 
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = units
        super(GRUD, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]
        initializer = 'glorot_uniform'

        self._reset_gate = tf.keras.layers.Dense(self.units,activation="sigmoid",kernel_initializer=initializer)
        self._detect_signal = tf.keras.layers.Dense(self.units,activation="tanh",kernel_initializer=initializer)
        self._update_gate = tf.keras.layers.Dense(self.units,activation="sigmoid",kernel_initializer=initializer)
        self._d_gate = tf.keras.layers.Dense(self.units,activation="relu",kernel_initializer=initializer)

        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]
        
        dt = self._d_gate(elapsed)
        gamma = tf.exp(-dt)
        h_hat = states[0]*gamma

        fused_input = tf.concat([inputs,h_hat],axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs,rt*h_hat],axis=-1)
        h_tilde = self._detect_signal(reset_value)

        # Compute new state
        ht = zt*h_hat + (1.0-zt) * h_tilde

        return ht, [ht]



class PhasedLSTM(tf.keras.layers.Layer):
    # Implemented according to 
    # https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

    def __init__(self, units,**kwargs):
        self.units = units
        self.state_size = (units,units)
        self.initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        super(PhasedLSTM, self).__init__(**kwargs)

    def get_initial_state(self,inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size,self.units],dtype=tf.float32),
            tf.zeros([batch_size,self.units],dtype=tf.float32)
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.input_kernel = self.add_weight(
            shape=(input_dim, 4*self.units),
            initializer=self.initializer,
            name='input_kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4*self.units),
            initializer=self.recurrent_initializer,
            name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(4*self.units),
            initializer=tf.keras.initializers.Zeros(),
            name='bias')
        self.tau = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),name="tau")
        self.ron = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),name="ron")
        self.s = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),name="s")

        self.built = True

    def call(self, inputs, states):
        cell_state,hidden_state = states
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]

        # Leaky constant taken fromt he paper
        alpha = 0.001
        # Make sure these values are positive
        tau = tf.nn.softplus(self.tau)
        s = tf.nn.softplus(self.s)
        ron = tf.nn.softplus(self.ron)

        phit = tf.math.mod(elapsed-s,tau)/tau
        kt = tf.where(tf.less(phit,0.5*ron),2*phit*ron,tf.where(tf.less(phit,ron),2.0-2*phit/ron,alpha*phit))

        z = tf.matmul(inputs,self.input_kernel)+tf.matmul(hidden_state,self.recurrent_kernel)+self.bias
        i,ig,fg,og = tf.split(z,4,axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg+1.0)
        output_gate = tf.nn.sigmoid(og)

        c_tilde = cell_state*forget_gate + input_activation*input_gate
        c = kt *c_tilde + (1.0-kt)*cell_state

        h_tilde = tf.nn.tanh(c_tilde)*output_gate
        h = kt*h_tilde + (1.0-kt)*hidden_state

        return h, [c,h]


class GRUODE(tf.keras.layers.Layer):
    # Implemented according to 
    # https://arxiv.org/pdf/1905.12374.pdf
    # without the Bayesian stuff

    def __init__(self, units,num_unfolds=4,**kwargs):
        self.units = units
        self.num_unfolds = num_unfolds
        self.state_size = units
        super(GRUODE, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[-1]
        if(isinstance(input_shape[0],tuple)):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._reset_gate = tf.keras.layers.Dense(self.units,activation="sigmoid",bias_initializer=tf.constant_initializer(1))
        self._detect_signal = tf.keras.layers.Dense(self.units,activation="tanh")
        self._update_gate = tf.keras.layers.Dense(self.units,activation="sigmoid")

        self.built = True

    def _dh_dt(self,inputs,states):
        fused_input = tf.concat([inputs,states],axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs,rt*states],axis=-1)
        gt = self._detect_signal(reset_value)

        # Compute new state
        dhdt = (1.0-zt)*(gt-states)
        return dhdt

    def euler(self,inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state+delta_t*dy

    def call(self, inputs, states):
        elapsed = 1.0
        if((isinstance(inputs,tuple) or isinstance(inputs,list)) and len(inputs)>1):
            elapsed = inputs[1]
            inputs = inputs[0]
        
        delta_t = elapsed/self.num_unfolds
        hidden_state = states[0]
        for i in range(self.num_unfolds):
            hidden_state = self.euler(inputs,hidden_state,delta_t)
        return hidden_state, [hidden_state]


        return ht, [ht]
