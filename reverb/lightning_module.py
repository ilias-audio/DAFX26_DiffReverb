"""PyTorch Lightning wrappers for training flamo `system.Shell` models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import pytorch_lightning as pl


@dataclass(frozen=True)
class WeightedCriterion:
    alpha: float
    criterion: Callable
    requires_model: bool = False
    name: Optional[str] = None

    def criterion_name(self) -> str:
        if self.name is not None:
            return self.name
        return getattr(self.criterion, "__class__", type(self.criterion)).__name__


class ShellLightningModule(pl.LightningModule):
    """LightningModule for a flamo `system.Shell` model.

    Notes:
    - Batch is expected to be `(inputs, targets)`.
    - Criteria can optionally depend on the model (e.g. parameter regularization).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criteria: Iterable[WeightedCriterion],
        lr: float = 1e-3,
        step_size: int = 100,
        step_gamma: float = 0.5,
        grad_clip_norm: float = 1.0,
        skip_on_nonfinite_grad: bool = True,
        sample_rate: int = 48000,
        log_hist_every_n_epochs: int = 10,
        log_audio_every_n_epochs: int = 25,
        log_stft_every_n_epochs: int = 25,
        delay_lr_mult: float = 1.0,
        matrix_lr_mult: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.criteria = list(criteria)
        self.lr = lr
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.grad_clip_norm = grad_clip_norm
        self.skip_on_nonfinite_grad = skip_on_nonfinite_grad
        self.sample_rate = int(sample_rate)
        self.log_hist_every_n_epochs = int(log_hist_every_n_epochs)
        self.log_audio_every_n_epochs = int(log_audio_every_n_epochs)
        self.log_stft_every_n_epochs = int(log_stft_every_n_epochs)
        self.delay_lr_mult = float(delay_lr_mult)
        self.matrix_lr_mult = float(matrix_lr_mult)

        self._last_valid_estimations: Optional[torch.Tensor] = None
        self._last_valid_targets: Optional[torch.Tensor] = None

        # Per-step and per-epoch loss history for post-training analysis.
        self._step_train_losses: list[float] = []
        self._step_valid_losses: list[float] = []
        self._step_train_components: dict[str, list[float]] = {}
        self._epoch_train_losses: list[float] = []
        self._epoch_valid_losses: list[float] = []
        self._epoch_accum: dict[str, float] = {
            "train_sum": 0.0, "train_n": 0,
            "valid_sum": 0.0, "valid_n": 0,
        }

        # Lightning: avoid dumping the whole model config into checkpoints.
        self.save_hyperparameters(ignore=["model", "criteria"])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        inputs, targets = batch
        return inputs.to(device), targets.to(device)

    def _compute_loss(self, estimations: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Estimation-side feature cache is per-step: clear before computing
        # so each loss's first lookup of est-STFT / est-rfft populates it and
        # subsequent losses reuse it.
        from .feature_cache import reset_step_cache
        reset_step_cache()

        loss = torch.zeros((), device=estimations.device)
        logs: dict[str, torch.Tensor] = {}
        for wc in self.criteria:
            if wc.requires_model:
                term = wc.criterion(estimations, targets, self.model)
            else:
                term = wc.criterion(estimations, targets)

            # Guard against non-finite criterion outputs (NaN/Inf). We still
            # return a tensor so Lightning can continue; the caller can decide
            # how to handle the batch.
            if not torch.isfinite(term):
                logs[f"nonfinite/{wc.criterion_name()}"] = torch.ones((), device=estimations.device)
            else:
                logs[f"nonfinite/{wc.criterion_name()}"] = torch.zeros((), device=estimations.device)
            loss = loss + float(wc.alpha) * term
            logs[wc.criterion_name()] = term.detach()
        return loss, logs

    def _repeat_last_history(self):
        """Append the previous step's values for non-finite steps."""
        self._step_train_losses.append(
            self._step_train_losses[-1] if self._step_train_losses else 0.0
        )
        for vals in self._step_train_components.values():
            vals.append(vals[-1] if vals else 0.0)

    def _record_step(self, loss: torch.Tensor, logs: dict[str, torch.Tensor]):
        loss_val = loss.item()
        self._step_train_losses.append(loss_val)
        self._epoch_accum["train_sum"] += loss_val
        self._epoch_accum["train_n"] += 1
        for wc in self.criteria:
            name = wc.criterion_name()
            if name in logs:
                self._step_train_components.setdefault(name, []).append(
                    float(wc.alpha) * float(logs[name])
                )

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        estimations = self(inputs)

        if not torch.isfinite(estimations).all():
            self.log("train/nonfinite_output", torch.ones((), device=self.device), prog_bar=True, on_step=True, on_epoch=False)
            self._repeat_last_history()
            return estimations.sum() * 0.0

        loss, logs = self._compute_loss(estimations, targets)

        if not torch.isfinite(loss):
            self.log("train/nonfinite_loss", torch.ones((), device=self.device), prog_bar=True, on_step=True, on_epoch=False)
            for k, v in logs.items():
                if k.startswith("nonfinite/"):
                    self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=False)
            self._repeat_last_history()
            return estimations.sum() * 0.0

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)
        self._record_step(loss, logs)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        estimations = self(inputs)

        if not torch.isfinite(estimations).all():
            self.log("valid/nonfinite_output", torch.ones((), device=self.device), prog_bar=True, on_step=False, on_epoch=True)
            self._step_valid_losses.append(
                self._step_valid_losses[-1] if self._step_valid_losses else 0.0
            )
            return None

        loss, logs = self._compute_loss(estimations, targets)

        if not torch.isfinite(loss):
            self.log("valid/nonfinite_loss", torch.ones((), device=self.device), prog_bar=True, on_step=False, on_epoch=True)
            for k, v in logs.items():
                if k.startswith("nonfinite/"):
                    self.log(f"valid/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
            self._step_valid_losses.append(
                self._step_valid_losses[-1] if self._step_valid_losses else 0.0
            )
            return None

        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in logs.items():
            self.log(f"valid/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        loss_val = loss.item()
        self._step_valid_losses.append(loss_val)
        self._epoch_accum["valid_sum"] += loss_val
        self._epoch_accum["valid_n"] += 1

        # Save the last validation batch for TensorBoard media logging (on CPU
        # to avoid holding GPU memory during training).
        self._last_valid_estimations = estimations.detach().cpu()
        self._last_valid_targets = targets.detach().cpu()
        return loss

    def on_train_epoch_end(self):
        n = self._epoch_accum["train_n"]
        if n > 0:
            self._epoch_train_losses.append(self._epoch_accum["train_sum"] / n)
        self._epoch_accum["train_sum"] = 0.0
        self._epoch_accum["train_n"] = 0

    def on_validation_epoch_end(self) -> None:
        n = self._epoch_accum["valid_n"]
        if n > 0:
            self._epoch_valid_losses.append(self._epoch_accum["valid_sum"] / n)
        self._epoch_accum["valid_sum"] = 0.0
        self._epoch_accum["valid_n"] = 0

        exp = getattr(getattr(self.logger, "experiment", None), "add_histogram", None)
        if exp is None:
            return

        epoch = int(self.current_epoch)

        # ---------------- Histograms ----------------
        if self.log_hist_every_n_epochs > 0 and epoch % self.log_hist_every_n_epochs == 0:
            tb = self.logger.experiment
            for name, param in self.model.named_parameters():
                if param is None or param.numel() == 0:
                    continue
                tb.add_histogram(f"params/{name}", param.detach().float().cpu(), global_step=epoch)

        # ---------------- Audio + STFT image ----------------
        if self._last_valid_estimations is None or self._last_valid_targets is None:
            return

        tb = self.logger.experiment

        if self.log_audio_every_n_epochs > 0 and epoch % self.log_audio_every_n_epochs == 0:
            est = self._last_valid_estimations[0, :, 0].detach().float().cpu().clamp(-1, 1)
            tgt = self._last_valid_targets[0, :, 0].detach().float().cpu().clamp(-1, 1)
            tb.add_audio("audio/estim", est, global_step=epoch, sample_rate=self.sample_rate)
            tb.add_audio("audio/target", tgt, global_step=epoch, sample_rate=self.sample_rate)

        if self.log_stft_every_n_epochs > 0 and epoch % self.log_stft_every_n_epochs == 0:
            def _stft_db(x: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
                w = torch.hann_window(n_fft)
                X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=w, return_complex=True)
                mag = torch.abs(X).clamp_min(1e-12)
                return 20.0 * torch.log10(mag)

            est = self._last_valid_estimations[0, :, 0].detach().float().cpu()
            tgt = self._last_valid_targets[0, :, 0].detach().float().cpu()
            est_db = _stft_db(est)
            tgt_db = _stft_db(tgt)
            diff_db = (tgt_db - est_db).abs()

            def _norm_img(img: torch.Tensor) -> torch.Tensor:
                lo = torch.quantile(img, 0.05)
                hi = torch.quantile(img, 0.95)
                img = (img - lo) / (hi - lo + 1e-8)
                return img.clamp(0, 1)

            # Stack target/estim/diff as RGB for quick visual comparison.
            rgb = torch.stack([_norm_img(tgt_db), _norm_img(est_db), _norm_img(diff_db)], dim=0)
            tb.add_image("stft/target_estim_diff", rgb, global_step=epoch)

    def on_before_optimizer_step(self, optimizer):
        # Hard guard against NaN/Inf gradients: skip step before Adam moments are corrupted.
        if self.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))
            self.log("train/grad_norm", grad_norm, prog_bar=False, on_step=True, on_epoch=False)

            if self.skip_on_nonfinite_grad and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                # Zero grads and mark step skipped.
                optimizer.zero_grad(set_to_none=True)
                self.log("train/skip_step", torch.ones((), device=self.device), prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        delay_params, matrix_params, default_params = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            n = name.lower()
            # The FDN's `self.delays` is consumed by Recursion, which renames
            # the child to `feedforward`. The literal substring "delays" no
            # longer appears in the parameter path, so match on "feedforward"
            # (the only feedforward inside the recursion is the delay block).
            if "feedforward" in n or "delays" in n:
                delay_params.append(p)
            elif "mixing_matrix" in n:
                matrix_params.append(p)
            else:
                default_params.append(p)

        base_lr = float(self.lr)
        groups = []
        if default_params:
            groups.append({"params": default_params, "lr": base_lr})
        if delay_params:
            groups.append({"params": delay_params,
                           "lr": base_lr * self.delay_lr_mult})
        if matrix_params:
            groups.append({"params": matrix_params,
                           "lr": base_lr * self.matrix_lr_mult})

        opt = torch.optim.Adam(groups if groups else [{"params": []}], lr=base_lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(self.step_size), gamma=float(self.step_gamma))
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
            },
        }


__all__ = ["WeightedCriterion", "ShellLightningModule"]
