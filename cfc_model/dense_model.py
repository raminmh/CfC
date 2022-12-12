import os
import sys
import time

import tensorflow as tf
import argparse
from cfc_model.tf_cfc import CfcCell, MixedCfcCell
import cfc_model.data_types as data_types
import cfc_model.configuration
import copy

import numpy as np


class Args:
    """Params expected for the keras api."""
    def __init__(self):
        self.model = 'cfc'
        self.size = 64
        self.epochs = 200
        self.lr = 0.0005

def convert_xy_data(X, y, train_size=0.7):
    """
    Converts an x,y format into the cfc expected data structure. Assumes no
    shuffling will be done for sequential data.
    Expects:
        X (np.ndarray):   An n x m matrix where n is a fixed and expected size of
                                sequential data and m is the number of samples.
        y (list, np.ndarray):   An 1D array containing the discrete labels associated with
                                the series.
    Returns:
        data (cfc_model.data_types.GenericData): A sequential structure of data expected in the
                                cfc model.
    """

    assert isinstance(X, np.ndarray), f'Expected X to be type <np.ndarray>, got {type(X)}.'
    assert isinstance(y, (list, np.ndarray)), f'Expected X to be type <np.ndarray> or <list>, got {type(y)}.'
    assert len(X.shape), f'Expected X.shape to be size 2, got {X.shape}.'

    if isinstance(y, list):
        y = np.array(y)

    data = data_types.GenericData()
    data.pad_size = X.shape[1]
    train_pop_size = int(X.shape[0]*train_size)
    for i, x in enumerate(X):
        if i < train_pop_size:
            data.train_events.append(X[i])
            data.train_y.append(y[i])
            data.train_mask.append([True for ii in range(data.pad_size)])
            train_elapsed = [ii/data.pad_size for ii in range(data.pad_size)]
            data.train_elapsed.append(train_elapsed)
        else:
            data.test_events.append(X[i])
            data.test_y.append(y[i])
            data.test_mask.append([True for ii in range(data.pad_size)])
            test_elapsed = [ii/data.pad_size for ii in range(data.pad_size)]
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

def fit(X=None, y=None, data=None, config=None):
    """Create and fit cfc_model model."""

    assert isinstance(X, np.ndarray) and isinstance(y, (list, np.ndarray)) or isinstance(data, data_types.GenericData),\
        f'Expected X and y or data.'

    if isinstance(config, type(None)):
        config = copy.copy(cfc_model.configuration.tf['default'])

    args = Args()
    cell = CfcCell(units=args.size, hparams=config)

    # Convert X, y data into standard data structure with fixed time interval between samples.
    if isinstance(data, type(None)) and isinstance(X, np.ndarray) and isinstance(y,(list, np.ndarray)):
        data = convert_xy_data(X, y)
    if isinstance(data, type(None)):
        raise data_types.MissingDataError
    data

    pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="pixel")
    time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
    mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(10)

    output_states = rnn((pixel_input, time_input), mask=mask_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    # Fit and evaluate
    model.fit(
        x=(data.train_events, data.train_elapsed, data.train_mask),
        y=data.train_y,
        batch_size=128,
        epochs=args.epochs,
    )
    _, best_test_acc = model.evaluate(
        x=(data.test_events, data.test_elapsed, data.test_mask), y=data.test_y
    )
