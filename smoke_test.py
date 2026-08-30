#!/usr/bin/env python
"""Quick install check.

    python smoke_test.py

Builds the model at a small FFT size, runs a few Adam steps against a synthetic
target on CPU, and checks that imports work, gradients flow, the loss drops and
the rendered IR is finite and non-silent. Needs only torch and the vendored
flamo. Takes about 30 s.

This is not a numeric check of the paper results: for that, fit a real RIR and
compare metrics/metrics_comparison.csv against EXPECTED.md.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import model as M
from reverb.losses import SpectralEDCLoss, T30Loss


def main():
    torch.manual_seed(0)
    fs, nfft = 48000, 8192
    device = "cpu"

    # Synthetic onset-aligned target: direct + a couple of early taps + a decaying tail.
    t = np.zeros(nfft, dtype=np.float32)
    t[0], t[60], t[140] = 1.0, 0.4, -0.25
    tail = np.arange(nfft - 200)
    t[200:] += (0.05 * np.random.randn(nfft - 200) * np.exp(-tail / 1500.0)).astype(np.float32)
    target_ir = torch.tensor(t, device=device).unsqueeze(0).unsqueeze(-1)
    target = target_ir[:, :, 0]

    cfg = M.ReverbConfig.proposed()
    model, fdn = M.build_fdn(target_ir, nfft, fs, cfg, device=device, seed=999)

    impulse = M.signal_gallery(1, n_samples=nfft, n=1, signal_type="impulse", fs=fs, device=device)
    assert torch.isfinite(model(impulse)).all(), "init IR is non-finite"

    # A small slice of the paper's loss stack (registry-free → no Lightning needed).
    losses = [(2.0, SpectralEDCLoss(fs=fs, device=device)),
              (5.0, T30Loss(fs=fs, device=device))]

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=0.01)

    def total_loss():
        est = model(impulse)[:, :, 0]
        return sum(w * fn(est, target) for w, fn in losses)

    first = float(total_loss().item())
    for _ in range(5):
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        opt.step()
    last = float(total_loss().item())

    ir = model(impulse).detach()[0, :, 0]
    finite = bool(torch.isfinite(ir).all())
    peak = float(ir.abs().max())

    print(f"  trainable params : {n_params}")
    print(f"  loss             : {first:.4f} -> {last:.4f}")
    print(f"  IR finite/peak   : {finite} / {peak:.4f}")

    assert finite, "rendered IR is non-finite"
    assert peak > 1e-3, "rendered IR is silent"
    assert last < first, "loss did not decrease"
    print("\nSMOKE TEST PASSED\n")


if __name__ == "__main__":
    main()
