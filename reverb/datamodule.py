"""Lightning DataModule utilities.

The default training setup in this repo uses a single impulse input and a single
(target) RIR. Historically the code expanded this pair to create many identical
batches. This module provides a minimal Lightning-native alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


@dataclass(frozen=True)
class ImpulseRIRDataConfig:
    num_copies: int = 1
    batch_size: int = 1
    num_workers: int = 0


class ImpulseRIRDataModule(pl.LightningDataModule):
    """DataModule that repeats a single (input, target) pair."""

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        config: ImpulseRIRDataConfig,
        pin_memory: bool = False,
    ):
        super().__init__()
        if inputs.ndim != 3 or targets.ndim != 3:
            raise ValueError("inputs and targets must be shaped [B, T, C]")
        if inputs.shape != targets.shape:
            raise ValueError(f"inputs shape {inputs.shape} must match targets shape {targets.shape}")
        self.inputs = inputs.detach().cpu()
        self.targets = targets.detach().cpu()
        self.config = config
        self.pin_memory = bool(pin_memory)

        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        x = self.inputs.repeat(int(self.config.num_copies), 1, 1)
        y = self.targets.repeat(int(self.config.num_copies), 1, 1)
        self._train_ds = TensorDataset(x, y)
        self._val_ds = TensorDataset(self.inputs, self.targets)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(self.config.num_workers),
            pin_memory=self.pin_memory,
            drop_last=False,
        )


__all__ = ["ImpulseRIRDataConfig", "ImpulseRIRDataModule"]
