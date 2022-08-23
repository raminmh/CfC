import os
import sys
import time

import tensorflow as tf
import argparse
from irregular_sampled_datasets import ETSMnistData
from tf_cfc import CfcCell, MixedCfcCell
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cfc")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
args = parser.parse_args()

data = ETSMnistData(time_major=False)

CFC_CONFIG = {
    "backbone_activation": "gelu",
    "backbone_dr": 0.0,
    "forget_bias": 3.0,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 0,
    "use_lstm": False,
    "no_gate": False,
    "minimal": False,
}
if args.model == "cfc":
    cell = CfcCell(units=args.size, hparams=CFC_CONFIG)
elif args.model == "no_gate":
    CFC_CONFIG["no_gate"] = True
    cell = CfcCell(units=args.size, hparams=CFC_CONFIG)
elif args.model == "minimal":
    CFC_CONFIG["minimal"] = True
    cell = CfcCell(units=args.size, hparams=CFC_CONFIG)
elif args.model == "mixed":
    cell = MixedCfcCell(units=args.size, hparams=CFC_CONFIG)
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

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
hist = model.fit(
    x=(data.train_events, data.train_elapsed, data.train_mask),
    y=data.train_y,
    batch_size=128,
    epochs=args.epochs,
)

_, best_test_acc = model.evaluate(
    x=(data.test_events, data.test_elapsed, data.test_mask), y=data.test_y
)
