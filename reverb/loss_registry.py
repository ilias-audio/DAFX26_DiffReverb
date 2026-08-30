"""Loss registry — the single place that maps short names to loss functions.

This is the backbone of the "mix any losses with any approach" workflow.  Instead
of a wall of ``--w_*`` flags and a ``LOSS_PRESET`` env-var cascade, a run picks a
weighted **set** of losses by name, either as a dict (YAML) or a string (CLI):

    "spectral_edc=2, band_t30=5, early_energy=3"   ==   {"spectral_edc": 2.0, ...}

``build_criteria`` turns that spec into the ``WeightedCriterion`` list the trainer
consumes, applying the same init-normalisation (``alpha = weight / loss(init)``)
that the training scripts used inline.

To understand one loss: find its row in ``LOSSES`` below — name -> the class it
builds.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

import torch

from . import losses as _losses


# --------------------------------------------------------------------------- #
# Registry: short name -> factory(fs, device, **kwargs) -> nn.Module loss.
# Defaults below reproduce exactly how the training scripts instantiated each
# loss, so swapping to the registry does not change results.
# --------------------------------------------------------------------------- #
LOSSES: Dict[str, Callable[..., torch.nn.Module]] = {
    # --- Energy-decay-curve family ---------------------------------------- #
    "spectral_edc": lambda fs, device, **kw: _losses.SpectralEDCLoss(
        fs=fs, device=device, db_lo=kw.pop("db_lo", -5), db_hi=kw.pop("db_hi", -35),
        use_freq_weighting=kw.pop("use_freq_weighting", True), **kw),
    "band_t30": lambda fs, device, **kw: _losses._BandEDCBase(
        fs=fs, device=device, db_lo=kw.pop("db_lo", -5), db_hi=kw.pop("db_hi", -35),
        emphasis_bands=kw.pop("emphasis_bands", None), **kw),
    "edt": lambda fs, device, **kw: _losses._BandEDCBase(
        fs=fs, device=device, db_lo=kw.pop("db_lo", 0), db_hi=kw.pop("db_hi", -10),
        emphasis_bands=kw.pop("emphasis_bands", None), **kw),
    "t30": lambda fs, device, **kw: _losses.T30Loss(fs=fs, device=device, **kw),
    "linear_edc": lambda fs, device, **kw: _losses.LinearEDCLoss(fs=fs, device=device, **kw),
    "diff_t30": lambda fs, device, **kw: _losses.DifferentiableT30Loss(fs=fs, device=device, **kw),

    # --- Spectral family --------------------------------------------------- #
    "mrstft": lambda fs, device, **kw: _losses.MultiResoSTFT(device=device, **kw),
    "smooth_stft": lambda fs, device, **kw: _losses.SmoothSTFTLoss(
        savgol_window=kw.pop("savgol_window", 11), savgol_poly=kw.pop("savgol_poly", 3),
        skip_samples=kw.pop("skip_samples", 0), device=device, **kw),
    "power_spec": lambda fs, device, **kw: _losses.PowerSpectrumLoss(
        fs=fs, device=device, use_freq_weighting=kw.pop("use_freq_weighting", True), **kw),
    "band_energy": lambda fs, device, **kw: _losses.BandEnergyLoss(fs=fs, device=device, **kw),

    # --- Descriptor / time-domain family ----------------------------------- #
    "early_energy": lambda fs, device, **kw: _losses.EarlyEnergyLoss(
        fs=fs, t_boundaries_ms=kw.pop("t_boundaries_ms", [5, 50, 80]), **kw),
    "drr": lambda fs, device, **kw: _losses.DRRLoss(fs=fs, **kw),
    "echo_density": lambda fs, device, **kw: _losses.EchoDensityLoss(
        fs=fs, window_ms=kw.pop("window_ms", 10.0), **kw),
    "gauss_time": lambda fs, device, **kw: _losses.GaussianSmoothedTimeLoss(
        fs=fs, sigma_ms=kw.pop("sigma_ms", 3.0), **kw),

    # --- Baseline-only (Mezza DAFx 2024); fixed paper losses --------------- #
    "broadband_edc": lambda fs, device, **kw: _losses.BroadbandEDCLoss(**kw),
    "mel_edr": lambda fs, device, **kw: _losses.MelEDRLoss(
        fs=fs, device=device, **{**_mel_edr_defaults(fs), **kw}),
    "soft_edp": lambda fs, device, **kw: _losses.SoftEDPLoss(**kw),
}


def _mel_edr_defaults(fs: int) -> dict:
    """Scale Mezza's 16 kHz MelEDR STFT params to ``fs``, preserving the paper's
    time-frequency tiling (n_fft 32 ms, win 20 ms, hop 10 ms). At 16 kHz this is
    the paper-exact 512/320/160; at 48 kHz it becomes 1536/960/480. Explicit
    kwargs in the loss spec still override these."""
    f = max(1, round(fs / 16000))
    return {"n_fft": 512 * f, "win_length": 320 * f, "hop_length": 160 * f}

# Human-readable labels kept identical to the old loss_history.csv column names,
# so downstream analysis/figures see the same names.
DISPLAY_NAMES: Dict[str, str] = {
    "spectral_edc": "SpectralEDC", "band_t30": "BandT30", "edt": "EDT", "t30": "T30",
    "linear_edc": "LinearEDC", "diff_t30": "DiffT30", "mrstft": "MRSTFT",
    "smooth_stft": "SmoothSTFT", "power_spec": "PowSpec", "band_energy": "BandEnergy",
    "early_energy": "EarlyEnergy", "drr": "DRR", "echo_density": "EchoDensity",
    "gauss_time": "GaussTime", "broadband_edc": "BroadbandEDC", "mel_edr": "MelEDR",
    "soft_edp": "SoftEDP",
}

# --------------------------------------------------------------------------- #
# Named presets: a preset is just a {loss_name: weight} dict, defined once here
# instead of scattered across scripts / env-vars.  `--losses` overrides on top.
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, Dict[str, float]] = {
    "default": {"spectral_edc": 2.0, "band_t30": 5.0, "linear_edc": 3.0},
    "v3": {"spectral_edc": 2.0, "band_energy": 3.0, "power_spec": 3.0,
           "band_t30": 5.0, "linear_edc": 3.0},
    # The arch-study FDN baseline loss set (run_arch_study v2 defaults).
    "arch_fdn": {"spectral_edc": 2.0, "band_t30": 5.0, "linear_edc": 3.0,
                 "drr": 2.0, "echo_density": 0.5, "early_energy": 3.0},
    # Paper config.
    "fdn_best": {"spectral_edc": 2.0, "band_t30": 5.0, "linear_edc": 3.0,
                 "band_energy": 2.0, "power_spec": 2.0, "early_energy": 3.0,
                 "drr": 2.0, "echo_density": 2.0, "gauss_time": 1.0},
    # Mezza DAFx 2024 baseline (used only by --method mezza).
    "mezza": {"broadband_edc": 0.5, "mel_edr": 1.0, "soft_edp": 0.1},
}


# --------------------------------------------------------------------------- #
# Spec parsing / resolution
# --------------------------------------------------------------------------- #
def parse_loss_spec(spec: Union[str, dict, None]) -> Dict[str, float]:
    """Parse a loss spec into a ``{name: weight}`` dict.

    Accepts:
      - ``"spectral_edc=2, band_t30=5"``  (CLI string)
      - ``{"spectral_edc": 2.0, ...}``    (already a dict, e.g. from YAML)
      - ``None`` / ``""`` -> ``{}``
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    out: Dict[str, float] = {}
    for tok in str(spec).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(
                f"Bad loss term {tok!r}; expected 'name=weight' (e.g. 'band_t30=5')")
        name, weight = tok.split("=", 1)
        out[name.strip()] = float(weight)
    return out


def resolve_weights(preset: Union[str, None],
                    overrides: Union[str, dict, None]) -> Dict[str, float]:
    """Resolve final ``{name: weight}`` from an optional preset + overrides.

    A ``weight <= 0`` in the overrides removes that loss (lets you drop a term
    from a preset, e.g. ``--preset fdn_best --losses "smooth_stft=0"``).
    """
    weights: Dict[str, float] = {}
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}. Known: {sorted(PRESETS)}")
        weights.update(PRESETS[preset])
    weights.update(parse_loss_spec(overrides))
    return {k: v for k, v in weights.items() if v > 0}


def build_criteria(spec: Union[str, dict], fs: int, device: str,
                   init_ir: torch.Tensor, target: torch.Tensor,
                   normalize: bool = True, verbose: bool = True,
                   alpha_cap: float = 0.0,
                   scale_overrides: dict = None):
    """Build the ``WeightedCriterion`` list from a loss spec.

    Args:
        spec:             ``{name: weight}`` dict or ``"name=weight,..."`` string.
        fs:               sample rate.
        device:           torch device string.
        init_ir:          model output at init, shape ``[B, T]`` (for normalisation).
        target:           target IR, shape ``[B, T]``.
        normalize:        apply ``alpha = weight / loss(init_ir, target)``.
        alpha_cap:        cap per-term alpha (0 = disabled).
        scale_overrides:  map display_name → init_val to use instead of the per-RIR
                          value; used for dataset-level normalisation where the mean
                          init value across all dataset RIRs is pre-computed and passed
                          in. If a loss name is in this dict its override value is used
                          in place of the live per-RIR measurement.

    Returns:
        ``list[WeightedCriterion]`` ready for ``ShellLightningModule``.
    """
    from .lightning_module import WeightedCriterion  # lazy: avoids pl import on inspect

    spec = parse_loss_spec(spec) if not isinstance(spec, dict) else spec
    scale_overrides = scale_overrides or {}
    criteria = []
    if verbose:
        print("\n  Loss spec (init normalization):")
    for name, val in spec.items():
        if name not in LOSSES:
            raise KeyError(f"Unknown loss {name!r}. Known: {sorted(LOSSES)}")
        # Allow {"weight": w, **kwargs} or a bare numeric weight.
        if isinstance(val, dict):
            kwargs = dict(val)
            weight = float(kwargs.pop("weight"))
        else:
            kwargs, weight = {}, float(val)
        if weight <= 0:
            continue
        loss_fn = LOSSES[name](fs, device, **kwargs).to(device)
        alpha = weight
        if normalize:
            display_name = DISPLAY_NAMES.get(name, name)
            with torch.no_grad():
                per_rir_val = float(loss_fn(init_ir, target).item())
            # Use dataset-level override if provided, else per-RIR value
            init_val = scale_overrides.get(display_name, per_rir_val)
            if init_val > 1e-10:
                alpha = weight / init_val
            if alpha_cap > 0:
                alpha = min(alpha, alpha_cap)
            if verbose:
                src = " (dataset)" if display_name in scale_overrides else ""
                print(f"    {display_name:>15}: per_rir={per_rir_val:.4f}, "
                      f"init={init_val:.4f}{src}, weight={weight}, alpha={alpha:.4f}")
        criteria.append(WeightedCriterion(
            alpha=alpha, criterion=loss_fn, name=DISPLAY_NAMES.get(name, name)))
    return criteria


def compute_init_vals(spec: Union[str, dict], fs: int, device: str,
                      init_ir: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute raw (unweighted) init loss for each term in *spec*.

    Returns ``{display_name: scalar}`` — the value each loss would produce on
    the given (init_ir, target) pair with no alpha scaling.  Useful for
    dataset-level normalisation: call this for every RIR, average the dicts,
    then pass the result as ``scale_overrides`` to :func:`build_criteria`.

    Args:
        spec:     ``{name: weight}`` dict or ``"name=weight,..."`` string.
        fs, device, init_ir, target: same as :func:`build_criteria`.

    Returns:
        ``{display_name: float}``
    """
    spec_dict = parse_loss_spec(spec) if not isinstance(spec, dict) else spec
    out: Dict[str, float] = {}
    for name, val in spec_dict.items():
        if name not in LOSSES:
            raise KeyError(f"Unknown loss {name!r}. Known: {sorted(LOSSES)}")
        kwargs = dict(val) if isinstance(val, dict) else {}
        if isinstance(val, dict):
            kwargs.pop("weight", None)
        loss_fn = LOSSES[name](fs, device, **kwargs).to(device)
        with torch.no_grad():
            v = float(loss_fn(init_ir, target).item())
        out[DISPLAY_NAMES.get(name, name)] = v
    return out


def compute_dataset_scales(spec: Union[str, dict], fs: int, device: str,
                           init_target_pairs: list) -> Dict[str, float]:
    """Average :func:`compute_init_vals` over a list of (init_ir, target) pairs.

    Args:
        spec:               ``{name: weight}`` dict or string.
        fs, device:         sample rate and torch device string.
        init_target_pairs:  ``[(init_ir, target), ...]``, each shape ``[1, T]``.

    Returns:
        ``{display_name: mean_float}`` — suitable for passing as
        ``scale_overrides`` to :func:`build_criteria`.
    """
    if not init_target_pairs:
        raise ValueError("init_target_pairs must be non-empty")
    accum: Dict[str, list] = {}
    for init_ir, target in init_target_pairs:
        vals = compute_init_vals(spec, fs, device, init_ir, target)
        for k, v in vals.items():
            accum.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in accum.items()}


__all__ = ["LOSSES", "PRESETS", "DISPLAY_NAMES",
           "parse_loss_spec", "resolve_weights", "build_criteria",
           "compute_init_vals", "compute_dataset_scales"]
