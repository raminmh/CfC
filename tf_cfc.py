import tensorflow as tf
import numpy as np

# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


class CfcCell(tf.keras.layers.Layer):
    def __init__(self, units, hparams, **kwargs):
        super(CfcCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.hparams = hparams
        self._no_gate = False

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        if self.hparams.get("backbone_activation") == "silu":
            backbone_activation = tf.nn.silu
        elif self.hparams.get("backbone_activation") == "relu":
            backbone_activation = tf.nn.relu
        elif self.hparams.get("backbone_activation") == "tanh":
            backbone_activation = tf.nn.tanh
        elif self.hparams.get("backbone_activation") == "gelu":
            backbone_activation = tf.nn.gelu
        elif self.hparams.get("backbone_activation") == "lecun":
            backbone_activation = lecun_tanh
        elif self.hparams.get("backbone_activation") == "softplus":
            backbone_activation = tf.nn.softplus
        else:
            raise ValueError("Unknown backbone activation")

        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        self.backbone = []
        for i in range(self.hparams["backbone_layers"]):

            self.backbone.append(
                tf.keras.layers.Dense(
                    self.hparams["backbone_units"],
                    backbone_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(
                        self.hparams["weight_decay"]
                    ),
                )
            )
            self.backbone.append(tf.keras.layers.Dropout(self.hparams["backbone_dr"]))

        self.backbone = tf.keras.models.Sequential(self.backbone)

        if self._minimal:
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.w_tau = self.add_weight(
                shape=(1, self.units), initializer=tf.keras.initializers.Zeros()
            )
            self.A = self.add_weight(
                shape=(1, self.units), initializer=tf.keras.initializers.Ones()
            )
        else:
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.ff2 = tf.keras.layers.Dense(
                self.units,
                lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.time_a = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.time_b = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
        self.built = True

    def call(self, inputs, states, **kwargs):
        hidden_state = states[0]
        t = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            t = tf.reshape(elapsed, [-1, 1])
            inputs = inputs[0]

        x = tf.keras.layers.Concatenate()([inputs, hidden_state])
        x = self.backbone(x)
        ff1 = self.ff1(x)
        if self._minimal:
            # Solution
            new_hidden = (
                -self.A
                * tf.math.exp(-t * (tf.math.abs(self.w_tau) + tf.math.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            ff2 = self.ff2(x)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = tf.nn.sigmoid(-t_a * t + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]


class MixedCfcCell(tf.keras.layers.Layer):
    def __init__(self, units, hparams, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        self.forget_gate_bias = 1
        if "forget_bias" in hparams.keys():
            self.forget_gate_bias = hparams["forget_bias"]
        self.cfc = CfcCell(self.units, hparams)
        super(MixedCfcCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.cfc.build(input_shape)
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states, **kwargs):
        cell_state, ode_state = states
        elapsed = tf.zeros((1,), dtype=tf.float32)
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(ode_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + self.forget_gate_bias)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = tf.nn.tanh(new_cell) * output_gate  # LSTM output = ODE input

        # Implementation choice on how to parametrize ODE component
        ode_output, new_ode_state = self.cfc([ode_input, elapsed], [ode_state])
        # ode_output, new_ode_state = self.ctrnn.call([ode_input, elapsed], [ode_input])

        return ode_output, [new_cell, new_ode_state[0]]


class LTCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, ode_unfolds=3, epsilon=1e-8, **kwargs):

        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }

        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        super(LTCCell, self).__init__(name="ltc_cell")

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.input_dim

    def _get_initializer(self, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return tf.keras.initializers.Constant(minval)
        else:
            return tf.keras.initializers.RandomUniform(minval, maxval)

    def _erev_initializer(self, shape=None, dtype=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def build(self, input_shape):

        # Check if input_shape is nested tuple/list
        if isinstance(input_shape[0], (tuple, list)):
            input_shape = input_shape[0]

        self.input_dim = input_shape[-1]

        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("gleak"),
        )
        self._params["vleak"] = self.add_weight(
            name="vleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=self._get_initializer("vleak"),
        )
        self._params["cm"] = self.add_weight(
            name="cm",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("cm"),
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sigma"),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._erev_initializer,
        )

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_mu"),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("sensory_w"),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._erev_initializer,
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            shape=(self.sensory_size,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            shape=(self.sensory_size,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0),
        )

        self._params["output_w"] = self.add_weight(
            name="output_w",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0),
        )
        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.expand_dims(v_pre, axis=-1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant
        cm_t = self._params["cm"] / tf.cast(
            (elapsed_time + 1e-3) / self._ode_unfolds, dtype=tf.float32
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self._params["gleak"] * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self._params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            elapsed_time = 1.0
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states[0], elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, [next_state]
