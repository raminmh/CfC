import os
import sys
import time

import tensorflow as tf
import argparse
from CFC.tf_cfc import CfcCell, MixedCfcCell
import numpy as np

class Args:
    """Params expected for the keras api."""
    def __init__(self):
        self.model = 'cfc'
        self.size = 64
        self.epochs = 200
        self.lr = 0.0005

def fit(data, config=None):
    """Create and fit CFC model."""
    args = Args()
    cell = CfcCell(units=args.size, hparams=config)

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
