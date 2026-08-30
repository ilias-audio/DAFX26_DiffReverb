#!/usr/bin/env python
"""Fit the differentiable FDN to a target room impulse response.

    python fit.py --target room.wav                 # trainable delays
    python fit.py --target room.wav --fixed-delays  # fixed delays, train the rest

Writes to --out: ir_target.wav, ir_init.wav, ir_optim.wav, loss_history.csv,
comparison.png and metrics/metrics_comparison.csv. About two minutes per RIR on
one GPU. The model and its losses are in model.py and reverb/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Make the package importable and prefer the vendored flamo (model.py also does
# this on import, but do it here too so `import model` resolves when run from
# anywhere).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_FLAMO = os.path.join(_HERE, "flamo")
if os.path.isdir(_FLAMO) and _FLAMO not in sys.path:
    sys.path.insert(0, _FLAMO)

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import model as M
from reverb import utils as utilities


# --------------------------------------------------------------------------- #
#  Target T30 -> per-band RT initialization.
# --------------------------------------------------------------------------- #
def _measure_t30(ir_np, fs):
    """Per-octave-band T30 of an IR via the ReverbAnalyzer (pyrato regression)."""
    import tempfile
    from reverb import metrics as RA
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, ir_np, fs)
        analyzer = RA.ReverbAnalyzer(tmp)
        analyzer.compute_octave_bands()
        analyzer.compute_energy_decay_curves()
        analyzer.measure_reverberation_time()
        rt_table = analyzer.rt_table
        result = {}
        if "T30" in rt_table.index:
            for freq in rt_table.columns:
                val = float(np.squeeze(rt_table.at["T30", freq]))
                if np.isfinite(val):
                    result[float(freq)] = val
        return result
    finally:
        os.unlink(tmp)


def _init_rt_from_target(fdn, target_ir, fs, bands, device):
    """Set each PEQ band's RT to the target's measured T30 at the nearest band."""
    target_np = target_ir[0, :, 0].cpu().numpy()
    target_t30 = _measure_t30(target_np, fs)
    if not target_t30:
        print("  T30 init: measurement empty, keeping default RT=3.0 s")
        return
    with torch.no_grad():
        mapped = fdn.attenuation.map_eq(fdn.attenuation.param)
        band_freqs_hz = (mapped[0, :, 0] / (2 * np.pi) * fs).cpu().numpy()
        t30_freqs = sorted(target_t30.keys())
        t30_vals = [target_t30[f] for f in t30_freqs]
        for b_idx in range(bands):
            nearest = int(np.argmin(np.abs(np.array(t30_freqs) - band_freqs_hz[b_idx])))
            band_rt = max(0.1, min(t30_vals[nearest], 20.0))
            rt_norm = (band_rt - utilities.MIN_RT) / (utilities.MAX_RT - utilities.MIN_RT)
            rt_norm = max(1e-4, min(1 - 1e-4, rt_norm))
            fdn.attenuation.param[b_idx, 2, 0] = float(np.log(rt_norm / (1 - rt_norm)))
    median_rt = float(np.median(list(target_t30.values())))
    print(f"  T30 init: per-band RT from target (median={median_rt:.2f} s, {len(t30_freqs)} bands)")


# --------------------------------------------------------------------------- #
#  Colored-noise-floor injection during training.
# --------------------------------------------------------------------------- #
def _noise_floor(target_trimmed, nfft, fs, device):
    """Estimate the target's noise-floor RMS + spectral shape from its tail."""
    with torch.no_grad():
        tail = target_trimmed[-int(0.1 * fs):]
        noise_rms = torch.sqrt((tail ** 2).mean()).item()
        tail_padded = F.pad(tail, (0, nfft - len(tail)))
        noise_spectrum = torch.abs(torch.fft.rfft(tail_padded))
        noise_spectrum = (noise_spectrum / (noise_spectrum.max() + 1e-10)).to(device)
    return noise_rms, noise_spectrum


class _NoisyModel(torch.nn.Module):
    """Adds colored noise (shaped like the target's floor) during training only,
    so the objective sees a realistic noise floor. Validation output is clean."""

    def __init__(self, shell, spec, rms):
        super().__init__()
        self.shell = shell
        self.rms = rms
        self.register_buffer("spec", spec)

    def forward(self, x):
        y = self.shell(x)
        if self.training:
            B, T, _C = y.shape
            white = torch.randn(B, T, device=y.device)
            colored = torch.fft.irfft(torch.fft.rfft(white) * self.spec, n=T)
            colored = colored / (colored.std(dim=-1, keepdim=True) + 1e-10) * self.rms
            y = y.clone()
            y[:, :, 0] = y[:, :, 0] + colored
        return y


# --------------------------------------------------------------------------- #
#  Comparison figure.
# --------------------------------------------------------------------------- #
def _comparison_figure(target_np, optim_np, fs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _mag_db(x):
        X = np.abs(np.fft.rfft(x)) + 1e-12
        return 20 * np.log10(X / X.max())

    def _edc_db(x):
        e = np.cumsum(x[::-1] ** 2)[::-1]
        return 10 * np.log10(e / (e[0] + 1e-30) + 1e-30)

    freqs = np.fft.rfftfreq(len(target_np), 1.0 / fs)
    t = np.arange(len(target_np)) / fs
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(freqs, _mag_db(target_np), label="target", lw=1)
    ax[0].semilogx(freqs, _mag_db(optim_np), label="FDN", lw=1, alpha=0.8)
    ax[0].set(xlim=(20, fs / 2), ylim=(-80, 5), xlabel="Hz", ylabel="dB",
              title="Magnitude spectrum")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
    ax[1].plot(t, _edc_db(target_np), label="target", lw=1)
    ax[1].plot(t, _edc_db(optim_np), label="FDN", lw=1, alpha=0.8)
    ax[1].set(ylim=(-70, 2), xlabel="s", ylabel="dB", title="Energy decay curve")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main.
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_target = os.path.join("docs", "ir_wav", "01_studio_genesis6", "reference.wav")
    p.add_argument("--target", default=default_target, help="target RIR (wav)")
    p.add_argument("--out", default=None, help="output dir (default: output/<timestamp>_fit)")
    p.add_argument("--fixed-delays", action="store_true",
                   help="recommended fixed-delay / train-mix variant (default: proposed all-diff)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--nfft", type=int, default=None, help="FFT size (default: auto from target RT)")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--delay-lr-mult", type=float, default=5.0)
    p.add_argument("--matrix-lr-mult", type=float, default=1.0)
    p.add_argument("--loss-alpha-cap", type=float, default=20.0,
                   help="cap each loss term's init-normalized weight (paper: 20)")
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (val checks)")
    p.add_argument("--es-min-delta", type=float, default=3e-2)
    p.add_argument("--val-every-n-epochs", type=int, default=5)
    p.add_argument("--no-t30-init", action="store_true", help="disable target-T30 per-band RT init")
    p.add_argument("--no-noise", action="store_true", help="disable colored-noise-floor injection")
    p.add_argument("--smooth-stft", type=float, default=0.0,
                   help="optional Savgol multi-res STFT loss weight (default 0 = paper front-door)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fs = args.samplerate
    cfg = M.ReverbConfig.fixed() if args.fixed_delays else M.ReverbConfig.proposed()
    variant = "fixed-delay / train-mix" if args.fixed_delays else "proposed (all-diff)"

    out_dir = args.out or os.path.join("output", time.strftime("%Y%m%d-%H%M%S") + "_fit")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(out_dir, "command.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    # ---- Target ---------------------------------------------------------- #
    target_trimmed, fs = M.load_target(args.target, fs=fs)
    nfft = args.nfft if args.nfft is not None else M.auto_nfft(target_trimmed, fs)
    target_ir = M.prepare_target_ir(target_trimmed, nfft, device)
    sf.write(os.path.join(out_dir, "ir_target.wav"), target_ir[0, :, 0].cpu().numpy(), fs)

    print(f"\n{'=' * 72}\n  Fitting FDN — {variant}")
    print(f"  target : {args.target}")
    print(f"  N={cfg.n_delays}, bands={cfg.bands}, nfft={nfft} ({nfft / fs:.1f}s), device={device}")
    print(f"  output : {out_dir}\n{'=' * 72}")

    # ---- Build model ----------------------------------------------------- #
    model, fdn = M.build_fdn(target_ir, nfft, fs, cfg, device=device, seed=args.seed)
    if not args.no_t30_init:
        _init_rt_from_target(fdn, target_ir, fs, cfg.bands, device)

    init_ir = M.render_ir(model, nfft, fs, device=device)
    init_np = init_ir[0, :, 0].cpu().numpy()
    sf.write(os.path.join(out_dir, "ir_init.wav"),
             init_np / max(float(np.abs(init_np).max()), 1e-12), fs, subtype="FLOAT")

    # ---- Loss (fdn_best preset, init-normalized) ------------------------- #
    from reverb.loss_registry import resolve_weights, build_criteria
    overrides = {"smooth_stft": args.smooth_stft} if args.smooth_stft > 0 else None
    spec = resolve_weights("fdn_best", overrides)
    impulse = M.signal_gallery(1, n_samples=nfft, n=1, signal_type="impulse", fs=fs, device=device)
    with torch.no_grad():
        init_hat = model(impulse)[:, :, 0]
    target_squeezed = target_ir[:, :, 0]
    criteria = build_criteria(spec, fs=fs, device=device,
                              init_ir=init_hat, target=target_squeezed, normalize=True,
                              alpha_cap=args.loss_alpha_cap)

    # ---- Colored-noise floor (training only) ----------------------------- #
    if args.no_noise:
        train_model = model
    else:
        noise_rms, noise_spectrum = _noise_floor(target_trimmed, nfft, fs, device)
        if noise_rms > 0:
            print(f"  Noise floor: RMS={noise_rms:.2e} "
                  f"({20 * np.log10(noise_rms + 1e-12):.1f} dB), injected during training")
            train_model = _NoisyModel(model, noise_spectrum, noise_rms)
        else:
            train_model = model

    # ---- Train (Lightning) ----------------------------------------------- #
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from reverb.lightning_module import ShellLightningModule
    from reverb.datamodule import ImpulseRIRDataModule, ImpulseRIRDataConfig

    lit = ShellLightningModule(
        model=train_model, criteria=criteria, lr=args.lr,
        step_size=200, step_gamma=0.5, sample_rate=fs,
        delay_lr_mult=args.delay_lr_mult, matrix_lr_mult=args.matrix_lr_mult,
    )
    data = ImpulseRIRDataModule(
        inputs=impulse, targets=target_ir,
        config=ImpulseRIRDataConfig(num_copies=1, batch_size=1),
    )
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(out_dir, "checkpoints"),
                              filename="best", monitor="valid/loss", mode="min", save_top_k=1)
    early_stop_cb = EarlyStopping(monitor="valid/loss", patience=args.patience,
                                  min_delta=args.es_min_delta, mode="min", verbose=True)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu", devices=1,
        callbacks=[ckpt_cb, early_stop_cb],
        logger=False, enable_checkpointing=True, enable_progress_bar=True,
        log_every_n_steps=1, check_val_every_n_epoch=args.val_every_n_epochs,
    )
    t0 = time.perf_counter()
    trainer.fit(lit, datamodule=data)
    train_s = time.perf_counter() - t0

    if ckpt_cb.best_model_path:
        state = torch.load(ckpt_cb.best_model_path, map_location=device, weights_only=False)
        lit.load_state_dict(state["state_dict"])
    lit.to(device)
    print(f"  Trained {len(lit._step_train_losses)} steps in {train_s:.1f}s")

    # ---- Render + write outputs ------------------------------------------ #
    optim_ir = M.render_ir(model, nfft, fs, device=device)
    optim_np = optim_ir[0, :, 0].cpu().numpy()
    fade_n = min(int(0.02 * fs), len(optim_np) // 4)   # 20 ms anti-click fade
    if fade_n > 0:
        optim_np[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    optim_np = optim_np / max(float(np.abs(optim_np).max()), 1e-12)
    sf.write(os.path.join(out_dir, "ir_optim.wav"), optim_np, fs, subtype="FLOAT")

    with open(os.path.join(out_dir, "loss_history.csv"), "w", newline="") as f:
        names = [wc.criterion_name() for wc in lit.criteria]
        w = csv.writer(f)
        w.writerow(["step", "total"] + names)
        comp = lit._step_train_components
        for i, total in enumerate(lit._step_train_losses):
            w.writerow([i, f"{total:.6f}"] +
                       [f"{comp.get(n, [0.0] * (i + 1))[i]:.6f}" if i < len(comp.get(n, [])) else "0.0"
                        for n in names])

    _comparison_figure(target_ir[0, :, 0].cpu().numpy(), optim_np, fs,
                        os.path.join(out_dir, "comparison.png"))

    # ---- Metrics --------------------------------------------------------- #
    from reverb.metrics import compare_rir_metrics
    try:
        compare_rir_metrics(os.path.join(out_dir, "ir_target.wav"),
                            os.path.join(out_dir, "ir_optim.wav"),
                            os.path.join(out_dir, "metrics"))
        print(f"  Metrics → {os.path.join(out_dir, 'metrics', 'metrics_comparison.csv')}")
    except Exception as e:
        print(f"  Metrics computation failed: {e}")

    print(f"\n  Done. Results in {out_dir}\n")


if __name__ == "__main__":
    main()
