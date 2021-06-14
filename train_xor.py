import os
import subprocess

from irregular_sampled_datasets import XORData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import time
import sys



def eval(config, index_arg, verbose=0):
    data = XORData(time_major=False, event_based=True, pad_size=32)

    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"], ode_unfolds=6)
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="input")
    time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
    mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(1)

    output_states = rnn((pixel_input, time_input), mask=mask_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    # end_lr = config["end_lr"]
    train_steps = data.train_events.shape[0] // config["batch_size"]
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )
    opt = (
        tf.keras.optimizers.Adam
        if config["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate_fn, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
    )
    # model.summary()

    # Fit model
    hist = model.fit(
        x=(data.train_events, data.train_elapsed, data.train_mask),
        y=data.train_y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=0,
    )
    # Evaluate model after training
    _, best_test_acc = model.evaluate(
        x=(data.test_events, data.test_elapsed, data.test_mask),
        y=data.test_y,
        verbose=2,
    )
    return best_test_acc

# Accuracy: 99.72 +- 0.08
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 64,
    "epochs": 200,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "forget_bias": 0.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 2e-06,
    "use_mixed": True,
}
# Accuracy: 99.42% +- 0.42
# DENSE: 97.34\% $\pm$ 1.85
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 192,
    "epochs": 200,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "forget_bias": 1.2,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 3e-06,
    "use_mixed": False,
}

# 96.29% +- 1.61
BEST_NO_GATE = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 128,
    "epochs": 200,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.3,
    "forget_bias": 4.7,
    "backbone_units": 192,
    "backbone_layers": 1,
    "weight_decay": 5e-06,
    "use_mixed": False,
    "no_gate": True,
}

# 85.42\% $\pm$ 2.84
BEST_MINIMAL = {
    "clipnorm": 5,
    "optimizer": "adam",
    "batch_size": 256,
    "size": 64,
    "epochs": 200,
    "base_lr": 0.005,
    "decay_lr": 0.9,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 1.2,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}
#  49.11\% $\pm$ 0.00
LTC_TEST = {
    "clipnorm": 5,
    "optimizer": "adam",
    "batch_size": 256,
    "size": 64,
    "epochs": 200,
    "base_lr": 0.005,
    "decay_lr": 0.9,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 1.2,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}


def score(config):
    acc = []
    for i in range(3):
        acc.append(100 * eval(config, i))
        print(
            f"Accuracy [n={len(acc)}]: {np.mean(acc):0.2f}\\% $\\pm$ {np.std(acc):0.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--use_ltc", action="store_true")
    args = parser.parse_args()

    if args.minimal:
        score(BEST_MINIMAL)
    elif args.no_gate:
        score(BEST_NO_GATE)
    elif args.use_ltc:
        score(LTC_TEST)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)
