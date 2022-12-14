import copy

import numpy as np
import tensorflow as tf

import cfc_model.configuration
import cfc_model.data_types as data_types
from cfc_model.tf_cfc import CfcCell, MixedCfcCell, LTCCell


class Args:
    """Params expected for the keras api."""

    def __init__(self):
        self.model = 'cfc'
        self.backbone_activation = 'gelu'
        self.backbone_layers = 1
        self.backbone_units = 128
        self.backbone_dr = 0.0
        self.forget_bias = 3
        self.weight_decay = 0
        self.size = 64
        self.epochs = 200
        self.lr = 0.0005
        self.batch_size = 64
        self.use_lstm = False
        self.no_gate = False
        self.minimal = False


def convert_xy_data_predict(x):
    """
    Converts an x,y format into the cfc expected data structure for the SequentialModel predict method.
    Expects:
        x (list, np.ndarray):   An n x m matrix where n is a fixed and expected size of
                                sequential data and m is the number of samples.
    Returns:
        data (cfc_model.data_types.GenericData): A sequential structure of data expected in the
                                cfc model.
    """

    assert isinstance(x, (list, np.ndarray)), f'Expected X to be type <np.ndarray>, got {type(x)}.'

    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x])

    assert len(x.shape), f'Expected X.shape to be size 2, got {x.shape}.'

    data = data_types.GenericData()
    data.pad_size = x.shape[1]
    for i, x in enumerate(x):
        data.test_events.append(x[i])
        data.test_mask.append([True for _ in range(data.pad_size)])
        test_elapsed = [ii / data.pad_size for ii in range(data.pad_size)]
        data.test_elapsed.append(test_elapsed)

    data.test_events = np.array(data.test_events)
    data.test_y = np.array(data.test_y)
    data.test_mask = np.array(data.test_mask)
    data.test_elapsed = np.array(data.test_elapsed)

    return data


def convert_xy_data_fit(x, y, train_size=0.7):
    """
    Converts an x,y format into the cfc expected data structure. Assumes no
    shuffling will be done for sequential data.
    Expects:
        x (list, np.ndarray):   An n x m matrix where n is a fixed and expected size of
                                sequential data and m is the number of samples.
        y (list, np.ndarray):   An 1D array containing the discrete labels associated with
                                the series.
    Returns:
        data (cfc_model.data_types.GenericData): A sequential structure of data expected in the
                                cfc model.
    """

    assert isinstance(x, np.ndarray), f'Expected X to be type <np.ndarray>, got {type(x)}.'
    assert isinstance(y, (list, np.ndarray)), f'Expected X to be type <np.ndarray> or <list>, got {type(y)}.'
    assert len(x.shape), f'Expected X.shape to be size 2, got {x.shape}.'

    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x, list):
        x = np.array(x)

    data = data_types.GenericData()
    data.pad_size = x.shape[1]
    train_pop_size = int(x.shape[0] * train_size)
    for i, x in enumerate(x):
        if i < train_pop_size:
            data.train_events.append(x[i])
            data.train_y.append(y[i])
            data.train_mask.append([True for _ in range(data.pad_size)])
            train_elapsed = [ii / data.pad_size for ii in range(data.pad_size)]
            data.train_elapsed.append(train_elapsed)
        else:
            data.test_events.append(x[i])
            data.test_y.append(y[i])
            data.test_mask.append([True for _ in range(data.pad_size)])
            test_elapsed = [ii / data.pad_size for ii in range(data.pad_size)]
            data.test_elapsed.append(test_elapsed)

    # Cast as numpy arrays from list
    data.train_events = np.array(data.train_events)
    data.train_y = np.array(data.train_y)
    data.train_mask = np.array(data.train_mask)
    data.train_elapsed = np.array(data.train_elapsed)
    data.test_events = np.array(data.test_events)
    data.test_y = np.array(data.test_y)
    data.test_mask = np.array(data.test_mask)
    data.test_elapsed = np.array(data.test_elapsed)

    return data


class SequentialModel:

    def __init__(self):
        self.model = None
        self.fitted = False

    def predict(self, x, data=None):
        """
        Predict a sample label.
        """

        assert self.fitted, f'fit() must be called prior to predict().'

        # Convert X, y data into standard data structure with fixed time interval between samples.
        if isinstance(data, type(None)) and isinstance(x, (list, np.ndarray)):
            data = convert_xy_data_predict(x)
        if isinstance(data, type(None)):
            raise data_types.MissingDataError

        res = self.model.predict(
            x=(data.test_events, data.test_elapsed, data.test_mask),
            verbose=False
        )

        return np.argmax(res)

    def fit(self, x=None, y=None, data=None, config=None):
        """Train the cfc model"""

        assert isinstance(x, np.ndarray) and isinstance(y, (list, np.ndarray)) or isinstance(data,
                                                                                             data_types.GenericData), \
            f'Expected X and y or data.'

        if isinstance(config, type(None)):
            config = copy.copy(cfc_model.configuration.tf['default'])

        args = Args()

        # Defaults are only inserted if they are not found in the provided config.
        for key in copy.copy(args.__dict__):
            if key not in config:
                config[key] = args.__dict__[key]

        if config.get("use_ltc"):
            cell = LTCCell(units=config["size"], ode_unfolds=6)
        elif config.get("use_mixed", None) and config.get("use_mixed", None) is not None:

            cell = MixedCfcCell(units=config["size"], hparams=config)
        else:
            cell = CfcCell(units=config["size"], hparams=config)

        # Convert X, y data into standard data structure with fixed time interval between samples.
        if isinstance(data, type(None)) and isinstance(x, np.ndarray) and isinstance(y, (list, np.ndarray)):
            data = convert_xy_data_fit(x, y)
        if isinstance(data, type(None)):
            raise data_types.MissingDataError

        pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="pixel")
        time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
        mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

        rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
        dense_layer = tf.keras.layers.Dense(data.train_y.max() + 1)

        output_states = rnn((pixel_input, time_input), mask=mask_input)
        y = dense_layer(output_states)

        self.model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        self.model.summary()

        # Fit and evaluate
        self.model.fit(
            x=(data.train_events, data.train_elapsed, data.train_mask),
            y=data.train_y,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
        )

        self.fitted = True
        return self
