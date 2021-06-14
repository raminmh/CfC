# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell

import sys

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review


def load_imdb():

    """
    ## Download and prepare dataset
    """
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
    return (x_train, y_train), (x_val, y_val)



def eval(config, index_arg, verbose=0):
    (train_x, train_y), (test_x, test_y) = load_imdb()
    if config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    # pixel_input = tf.keras.Input(shape=(28 * 28, 1), name="pixel")
    inputs = tf.keras.layers.Input(shape=(maxlen,))
    token_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=config["embed_dim"]
    )
    cell_input = token_emb(inputs)
    cell_input = tf.keras.layers.Dropout(config["embed_dr"])(cell_input)

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(10)

    output_states = rnn(cell_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs, y)

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    # end_lr = config["end_lr"]
    train_steps = train_x.shape[0] // config["batch_size"]
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit and evaluate
    hist = model.fit(
        x=train_x,
        y=train_y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(test_x, test_y) if verbose else None,
        verbose=verbose,
    )
    # test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    # return np.max(test_accuracies)
    _, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
    return test_accuracy



BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 64,
    "embed_dim": 32,
    "embed_dr": 0.3,
    "epochs": 20,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00029,
    "use_mixed": True,
}

# 87.04% (MAX)
#  85.91% $\pm$ 0.99
BEST_DEFAULT = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 192,
    "embed_dim": 192,
    "embed_dr": 0.0,
    "epochs": 47,
    "base_lr": 0.0005,
    "decay_lr": 0.7,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 2,
    "weight_decay": 3.6e-05,
    "use_mixed": False,
    "no_gate": False,
}
# 87.52\% $\pm$ 0.09
BEST_NO_GATE = {
    "clipnorm": 5,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 224,
    "embed_dim": 192,
    "embed_dr": 0.2,
    "epochs": 37,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 2.7e-05,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
}
# 81.72\% $\pm$ 0.50
BEST_MINIMAL = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 320,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 27,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00048,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}
# 61.76\% $\pm$ 6.14
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 50,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}




def score(config):
    acc = []
    for i in range(5):
        acc.append(100 * eval(config, i))
        print(
            f"IMDB test accuracy [{len(acc)}/5]: {np.mean(acc):0.2f}\\% $\\pm$ {np.std(acc):0.2f}"
        )
    print(f"IMDB test accuracy: {np.mean(acc):0.2f}\\% $\\pm$ {np.std(acc):0.2f}")


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
