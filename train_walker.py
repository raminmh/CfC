import os
import subprocess

from irregular_sampled_datasets import Walker2dImitationData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import sys



class BackupCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(BackupCallback, self).__init__()
        self.saved_weights = None
        self.model = model
        self.best_loss = np.PINF

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best_loss:
            self.best_loss = logs["val_loss"]
            # print(f" new best -> {logs['val_loss']:0.3f}")
            self.saved_weights = self.model.get_weights()

    def restore(self):
        if self.best_loss is not None:
            self.model.set_weights(self.saved_weights)


def eval(config, index_arg, verbose=0):
    data = Walker2dImitationData(seq_len=64)

    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"])
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    signal_input = tf.keras.Input(shape=(data.seq_len, data.input_size), name="robot")
    time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)

    output_states = rnn((signal_input, time_input))
    y = tf.keras.layers.Dense(data.input_size)(output_states)

    model = tf.keras.Model(inputs=[signal_input, time_input], outputs=[y])

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    train_steps = data.train_x.shape[0] // config["batch_size"]
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
        loss=tf.keras.losses.MeanSquaredError(),
    )
    # model.summary()

    # Fit model
    hist = model.fit(
        x=(data.train_x, data.train_times),
        y=data.train_y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((data.valid_x, data.valid_times), data.valid_y),
        callbacks=[BackupCallback(model)],
        verbose=0,
    )
    # Evaluate model after training
    test_loss = model.evaluate(
        x=(data.test_x, data.test_times), y=data.test_y, verbose=2
    )
    return test_loss



# 0.64038 +- 0.00574
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 256,
    "size": 64,
    "epochs": 50,
    "base_lr": 0.02,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
}

# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.2,
    "forget_bias": 2.1,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 6e-06,
    "use_mixed": True,
    "no_gate": False,
}

# 0.65040 $\pm$ 0.00814
BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.008,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.1,
    "forget_bias": 2.8,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": True,
}
# 0.94844 $\pm$ 0.00988
BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.006,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 5.0,
    "backbone_units": 192,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}

# 0.66225 $\pm$ 0.01330
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
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
    acc = [eval(config, i) for i in range(5)]
    print(f"MSE: {np.mean(acc):0.5f} $\\pm$ {np.std(acc):0.5f}")


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
        score(BEST_LTC)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)
