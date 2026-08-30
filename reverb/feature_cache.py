"""Cross-loss feature cache for FDN training.

During a single forward/backward pass, multiple losses recompute the same
target-side and estimation-side features (STFTs at identical params, full
rffts, Gaussian smoothing). Caching them once per step saves >50% of the
loss-side compute.

Two caches:
- ``TGT_CACHE`` persistent across steps and keyed by ``id(target)``. The
  target IR tensor is the same Python object every step (the DataModule
  reuses it), so this caches once and reuses forever.
- ``EST_CACHE`` cleared at the start of every loss evaluation (the est
  tensor is freshly produced by the forward pass each step).
"""

from __future__ import annotations

from typing import Optional

import torch


class _Cache:
    def __init__(self):
        self._store: dict = {}
        self._owner_id: Optional[int] = None

    def clear(self):
        self._store.clear()
        self._owner_id = None

    def bind(self, owner: torch.Tensor) -> None:
        """Bind cache to a specific tensor; if it changes, flush."""
        oid = id(owner)
        if self._owner_id is not None and self._owner_id != oid:
            self._store.clear()
        self._owner_id = oid

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


# Module-level singletons. Modules don't share state across processes, so a
# global is fine for our single-process training script.
TGT_CACHE = _Cache()
EST_CACHE = _Cache()


def stft_cached(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor,
                is_target: bool) -> torch.Tensor:
    cache = TGT_CACHE if is_target else EST_CACHE
    cache.bind(x)
    key = ("stft", n_fft, hop)
    cached = cache.get(key)
    if cached is not None:
        return cached
    out = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window,
                     return_complex=True)
    cache.set(key, out)
    return out


def rfft_cached(x: torch.Tensor, is_target: bool) -> torch.Tensor:
    """Cache full-length rfft(x). Length is inferred from x.shape[-1]."""
    cache = TGT_CACHE if is_target else EST_CACHE
    cache.bind(x)
    key = ("rfft", int(x.shape[-1]))
    cached = cache.get(key)
    if cached is not None:
        return cached
    out = torch.fft.rfft(x)
    cache.set(key, out)
    return out


def reset_step_cache() -> None:
    """Call this once per training/validation step before computing losses."""
    EST_CACHE.clear()


def reset_all_caches() -> None:
    """Drop both caches (e.g. when target tensor is replaced)."""
    EST_CACHE.clear()
    TGT_CACHE.clear()


__all__ = [
    "stft_cached",
    "rfft_cached",
    "reset_step_cache",
    "reset_all_caches",
    "TGT_CACHE",
    "EST_CACHE",
]
