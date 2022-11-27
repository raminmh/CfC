import argparse
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from torch_cfc import Cfc
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional
from sklearn.metrics import roc_auc_score
import sys
from pytorch_lightning.loggers import CSVLogger
from duv_physionet import get_physio
import numpy as np
import time

from pytorch_lightning.callbacks import Callback


class SpeedCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # The reproducing the mTAN times and calibrating to our GPU shows that my GPU is 1.33 times faster
        print(f"Took {1.34*(time.time()-self._start)/60:0.3f} minutes")


class PhysionetLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.Tensor((1.0, hparams["class_weight"]))
        )
        self._hparams = hparams
        self._all_rocs = []

    def _prepare_batch(self, batch):
        x, tt, mask, y = batch
        t_elapsed = tt[:, 1:] - tt[:, :-1]
        t_fill = torch.zeros(tt.size(0), 1, device=x.device)
        t = torch.cat((t_fill, t_elapsed), dim=1)
        return x, t, mask, y

    def training_step(self, batch, batch_idx):
        x, tt, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = accuracy(preds, y)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, tt, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1).long()

        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        softmax = torch.nn.functional.softmax(y_hat, dim=1)[:, 1]
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return [softmax, y]

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.cat([l[0] for l in validation_step_outputs])
        all_labels = torch.cat([l[1] for l in validation_step_outputs])

        auc = auroc(all_preds, all_labels, pos_label=1)
        self._all_rocs.append(auc)
        self.log("val_rocauc", auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optim = "rmsprop"
        if "optim" in self._hparams.keys():
            optim = self._hparams["optim"]
        optimizer = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop,
        }[optim]
        # optimizer = torch.optim.Adam(
        optimizer = optimizer(
            self.model.parameters(),
            lr=self._hparams["base_lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        def lamb_f(epoch):
            lr = self._hparams["decay_lr"] ** epoch
            # print(f"LEARNING RATE = {lr:0.4g} (epoch={epoch})")
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lamb_f)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lambda epoch: self._hparams["decay_lr"] ** epoch
        # )
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.optimizer.step(closure=closure)
        # Apply weight constraints
        if self._hparams["use_ltc"]:
            self.model.rnn_cell.apply_weight_constraints()


def eval(hparams, speed=False):
    # torch.set_num_threads(4)
    model = Cfc(
        in_features=41 * 2,
        hidden_size=hparams["hidden_size"],
        out_feature=2,
        hparams=hparams,
        use_mixed=hparams["use_mixed"],
        use_ltc=hparams["use_ltc"],
    )
    learner = PhysionetLearner(model, hparams)

    class FakeArg:
        batch_size = 32
        classif = True
        n = 8000
        extrap = False
        sample_tp = None
        cut_tp = None

    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    device = "cpu"
    data_obj = get_physio(fake_arg, device)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]

    gpu_name = "cpu"
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_name = str(os.environ["CUDA_VISIBLE_DEVICES"])

    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        gradient_clip_val=hparams["clipnorm"],
        gpus=1,
        callbacks=[SpeedCallback()] if speed else None,
    )
    trainer.fit(
        learner,
        train_loader,
    )
    results = trainer.test(learner, test_loader)[0]
    return float(results["val_rocauc"])



# AUC: 83.90 % +-0.22
BEST_DEFAULT = {
    "epochs": 57,
    "class_weight": 11.69,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.002,
    "decay_lr": 0.9,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_dr": 0.2,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}

# 0.8397588133811951
BEST_MIXED = {
    "epochs": 65,
    "class_weight": 5.91,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.001,
    "decay_lr": 0.9,
    "backbone_activation": "lecun",
    "backbone_units": 64,
    "backbone_dr": 0.3,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.6,
    "batch_size": 128,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}

# 0.8395 $\pm$ 0.0033
BEST_NO_GATE = {
    "epochs": 58,
    "class_weight": 7.73,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.73,
    "backbone_activation": "relu",
    "backbone_units": 192,
    "backbone_dr": 0.0,
    "backbone_layers": 2,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.55,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
    "use_ltc": False,
}
# test AUC 0.6431 $\pm$ 0.0180
BEST_MINIMAL = {
    "epochs": 116,
    "class_weight": 18.25,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.72,
    "backbone_activation": "tanh",
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
    "use_ltc": False,
}
# 0.6577
BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 64,
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}


def score(config, n=5):

    means = []
    for i in range(n):
        means.append(eval(config, speed=True))
    print(f"Test AUC: {np.mean(means):0.4f} $\\pm$ {np.std(means):0.4f} ")




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
