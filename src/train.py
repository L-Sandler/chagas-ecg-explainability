"""
Training entry point for ECGResNet on CODE-15%.

Usage:
    python src/train.py \
        --hdf5 data/code15/exams_part0.hdf5 \
        --labels data/code15/code15_chagas_labels.csv \
        --epochs 20 --batch-size 64

Smoke test (CPU):
    python src/train.py \
        --hdf5 data/code15/exams_part0.hdf5 \
        --labels data/code15/code15_chagas_labels.csv \
        --fast --epochs 5 --batch-size 32

Optional:
    --wandb    enable W&B logging
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from sklearn.metrics import roc_auc_score
import numpy as np

from model import ECGResNet
from dataset import Code15Dataset


class ECGLightningModule(pl.LightningModule):
    def __init__(self, pos_weight: float = 32.0, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGResNet()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self._val_preds = []
        self._val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        self._val_preds.append(probs.detach().cpu())
        self._val_labels.append(y.detach().cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        probs = torch.cat(self._val_preds).numpy()
        labels = torch.cat(self._val_labels).numpy()
        self._val_preds.clear()
        self._val_labels.clear()

        if labels.sum() == 0:
            return

        auroc = roc_auc_score(labels, probs)
        tpr_at_5 = _tpr_at_fpr(labels, probs, fpr_target=0.05)
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/tpr_at_5pct", tpr_at_5, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]


def _tpr_at_fpr(labels: np.ndarray, probs: np.ndarray, fpr_target: float) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--fast", action="store_true", help="200 train / 50 val samples (CPU smoke test)")
    args = parser.parse_args()

    max_train = 200 if args.fast else None
    max_val = 50 if args.fast else None

    train_ds = Code15Dataset(args.hdf5, args.labels, split="train", max_samples=max_train)
    val_ds = Code15Dataset(args.hdf5, args.labels, split="val", max_samples=max_val)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Pos weight (from train split): {train_ds.pos_weight():.1f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0,  # 0 keeps HDF5 file handle in main process; safe on all platforms
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    module = ECGLightningModule(pos_weight=train_ds.pos_weight(), lr=args.lr)

    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        ModelCheckpoint(
            monitor="val/auroc", mode="max", save_top_k=1,
            filename="best-{epoch}-{val/auroc:.3f}",
        ),
        EarlyStopping(monitor="val/auroc", mode="max", patience=5),
    ]

    loggers = []
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(project="chagas-ecg", log_model=False))

    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
