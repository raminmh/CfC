import argparse
import os
import subprocess
import time
from unittest.loader import VALID_MODULE_NAME

import torch
import torch.nn as nn
import pytorch_lightning as pl
from duv_person_activity import get_person_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from torch_cfc import Cfc

import numpy as np
import sys


class PersonActivityLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self._hparams = hparams

    def _prepare_batch(self, batch):
        _, t, x, mask, y = batch

        t_elapsed = t[:, 1:] - t[:, :-1]
        t_fill = torch.zeros(t.size(0), 1, device=x.device)
        t = torch.cat((t_fill, t_elapsed), dim=1)

        t = t * self._hparams["tau"]
        # return new_x, t, new_mask, y
        return x, t, mask, y

    def training_step(self, batch, batch_idx):
        x, t, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, t, mask=mask)

        enable_signal = torch.sum(y, -1) > 0.0

        y_hat = y_hat[enable_signal]
        y = y[enable_signal]


        y = torch.argmax(y.detach(), dim=-1)
        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat.detach(), dim=-1)  # labels are given as one-hot
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, t, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, t, mask=mask)

        enable_signal = torch.sum(y, -1) > 0.0

        y_hat = y_hat[enable_signal]
        y = y[enable_signal]

        y = torch.argmax(y, dim=-1)  # labels are given as one-hot

        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss, acc

    def validation_epoch_end(self, validation_step_outputs):
        val_acc = torch.stack([l[1] for l in validation_step_outputs])

        val_acc = torch.mean(val_acc)
        print(f"\nval_acc: {val_acc.item():0.3f}\n")

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._hparams["base_lr"],
            weight_decay=self._hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: self._hparams["decay_lr"] ** epoch
        )
        return [optimizer], [scheduler]


class FakeArg:
    batch_size = 32
    classif = True
    extrap = False
    sample_tp = None
    cut_tp = None
    n = 10000



def eval(hparams):

    model = Cfc(
        in_features=12 * 2,
        hidden_size=hparams["hidden_size"],
        out_feature=11,
        hparams=hparams,
        return_sequences=True,
        use_mixed=hparams["use_mixed"],
    )
    learner = PersonActivityLearner(model, hparams)
    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    data_obj = get_person_dataset(fake_arg)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        gradient_clip_val=hparams["clipnorm"],
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(
        learner, train_loader
    )
    results = trainer.test(learner, test_loader)[0]
    return float(results["val_acc"])



CFC = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 448,
    "base_lr": 0.002,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_layers": 1,
    "backbone_dr": 0.0,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 64,
    "optim": "adamw",
    "init": 0.84,
    "use_mixed": False,
}
CFC_MIXED = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.0005,
    "decay_lr": 0.99,
    "backbone_activation": "gelu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.5,
    "weight_decay": 4e-05,
    "tau": 10,
    "batch_size": 64,
    "optim": "adamw",
    "init": 1.35,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
}
CFC_NOGATE = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.005,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 192,
    "backbone_layers": 2,
    "backbone_dr": 0.2,
    "weight_decay": 0.0002,
    "tau": 0.5,
    "batch_size": 64,
    "optim": "adamw",
    "init": 0.78,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
}
CFC_MINIMAL = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "gelu",
    "backbone_units": 256,
    "backbone_layers": 3,
    "backbone_dr": 0.4,
    "weight_decay": 3e-05,
    "tau": 0.1,
    "batch_size": 64,
    "optim": "adamw",
    "init": 0.67,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}

model_zoo = {"cfc":CFC,"minimal":CFC_MINIMAL,"no_gate":CFC_NOGATE,"mixed":CFC_MIXED}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cfc")
    args = parser.parse_args()

    if args.model not in model_zoo.keys():
        raise ValueError(f"Unknown model '{args.model}', available: {list(model_zoo.keys())}")

    eval(model_zoo[args.model])
