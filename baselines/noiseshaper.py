import argparse
import csv
import os
import sys
import time

# Release layout: baselines/<this>.py → package root is one level up; prefer the
# vendored, patched flamo submodule.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAMO_ROOT = os.path.join(REPO_ROOT, "flamo")
for _p in (REPO_ROOT, FLAMO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
import dasp_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from dasp_pytorch import NoiseShapedReverb, noise_shaped_reverberation

from flamo.functional import find_onset, signal_gallery
from flamo.utils import save_audio
from reverb.metrics import compare_rir_metrics

from scipy.signal import fftconvolve

from reverb.losses import (
    BandEnergyEnvelopeLoss,
    SpectralEDCLoss,
    BandEnergyLoss,
    PowerSpectrumLoss,
    DRRLoss,
    MultiResoSTFT,
)


def analyze_target_bands(target_np, sr, num_bandpass_taps, max_band_decay=10.0,
                          max_band_gain=5.0):
    """Analyze target RIR per octave band to derive initial gain/decay norms.

    Gain calibration uses a model-aware approach:
    1. Run dasp NoiseShapedReverb with reference gain=0.5 (all bands, flat decay).
    2. Measure per-band RMS of model output vs target RIR (50-500ms window).
    3. Compute optimal_gain = target_rms / (model_rms_at_ref / ref_gain).
    4. Clip to [0, max_band_gain] and normalize to [0, 1] for dasp.

    Decay estimation uses T30 (-5 to -35 dB) with T20 fallback (-5 to -25 dB).
    Bands where EDC doesn't have enough dynamic range (e.g. 31.5 Hz, 16 kHz)
    are filled by interpolating from neighboring band estimates instead of
    defaulting to t60=duration, which avoids spurious slow-decay initialization.

    This correctly accounts for dasp's filterbank response (filter energy doubles
    per octave) so the initial model energy per band actually matches the target.
    """
    filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, sr)
    filters_np = filters.detach().cpu().numpy().squeeze(1)
    duration = len(target_np) / sr

    # Use a wider late-reverb window (50–500ms) for gain calibration.
    # Wider window improves RMS estimates for low-energy extreme-frequency bands
    # (31.5 Hz, 16 kHz) that may have very little energy in a narrow window.
    win_start = max(0, int(0.05 * sr))
    win_end = min(len(target_np), int(0.50 * sr))

    # --- Decay estimation ---
    band_decay_norm = []
    target_band_rms = []
    band_t60 = []
    for i in range(12):
        filtered = fftconvolve(target_np, filters_np[i], mode="same")
        rms = np.sqrt(np.mean(filtered[win_start:win_end] ** 2))
        target_band_rms.append(rms)

        # Estimate T60 from Schroeder EDC.
        # Try T30 range (-5 to -35 dB) first; fall back to T20 (-5 to -25 dB);
        # if neither is reachable, store None and interpolate from neighbors later.
        energy = filtered**2
        edc = np.flip(np.cumsum(np.flip(energy)))
        edc_db = 10 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
        idx_5 = np.searchsorted(-edc_db, 5)

        t60 = None
        for lo_db, hi_db, extrap in [(5, 35, 2.0), (5, 25, 3.0)]:
            idx_hi = np.searchsorted(-edc_db, hi_db)
            if idx_hi > idx_5 + 10 and idx_hi < len(edc_db):
                slope = (edc_db[idx_hi] - edc_db[idx_5]) / ((idx_hi - idx_5) / sr)
                if slope < 0:
                    t60 = -60.0 / slope  # extrapolate to T60
                    break
        band_t60.append(t60)  # None means "needs neighbor interpolation"

    # Fill bands where EDC was insufficient using nearest valid neighbor.
    # This avoids t60=duration for silent extreme-frequency bands.
    valid_t60 = [t for t in band_t60 if t is not None]
    fallback = float(np.mean(valid_t60)) if valid_t60 else duration
    for i in range(12):
        if band_t60[i] is None:
            # Find nearest neighbor with a valid estimate
            left = next((band_t60[j] for j in range(i - 1, -1, -1) if band_t60[j] is not None), None)
            right = next((band_t60[j] for j in range(i + 1, 12) if band_t60[j] is not None), None)
            if left is not None and right is not None:
                band_t60[i] = (left + right) / 2
            elif left is not None:
                band_t60[i] = left
            elif right is not None:
                band_t60[i] = right
            else:
                band_t60[i] = fallback

    for i in range(12):
        t60 = np.clip(band_t60[i], 0.05, duration * 1.5)

        # Map T60 → decay_norm through dasp's internal mapping
        actual_decay = np.log(1000) * duration / t60
        denorm = max((actual_decay - 1) / 10, 0)
        norm = np.clip(denorm / max_band_decay, 1e-4, 1 - 1e-4)
        band_decay_norm.append(norm)

    target_band_rms = np.array(target_band_rms)

    # --- Gain calibration via model calibration run ---
    # Run model with gain_norm=0.5 for all bands to measure model response per band.
    # This accounts for dasp's per-band filter energy (doubles per octave).
    nfft_ir = len(target_np)
    ref_gain_norm = 0.5
    ref_decay_norm = 0.25  # T60 ≈ 1s; keeps model energy above noise floor in 80-250ms window
    processor = dasp_pytorch.NoiseShapedReverb(
        sample_rate=sr,
        min_band_gain=0.0,
        max_band_gain=max_band_gain,
        min_band_decay=0.0,
        max_band_decay=max_band_decay,
        min_mix=1.0,
        max_mix=1.0,
    )
    x_imp = torch.zeros(1, 1, nfft_ir)
    x_imp[0, 0, 0] = 1.0
    gains_ref = torch.full((1, 12), ref_gain_norm)
    decays_ref = torch.full((1, 12), ref_decay_norm)
    mix_ref = torch.ones(1, 1)
    params_all = torch.cat([gains_ref, decays_ref, mix_ref], dim=1)
    torch.manual_seed(42)
    with torch.no_grad():
        param_dict = processor.extract_param_dict(params_all)
        denorm_dict = processor.denormalize_param_dict(param_dict)
        from dasp_pytorch import noise_shaped_reverberation
        y_ref = noise_shaped_reverberation(
            x_imp, sample_rate=sr, **denorm_dict,
            num_samples=nfft_ir, num_bandpass_taps=num_bandpass_taps,
        )
    y_ref_mono = y_ref.mean(dim=1).squeeze().numpy()

    # Measure per-band RMS of model reference output in the same 50-500ms window
    model_ref_rms = []
    for i in range(12):
        m_filtered = fftconvolve(y_ref_mono, filters_np[i], mode="same")
        m_rms = np.sqrt(np.mean(m_filtered[win_start:win_end] ** 2))
        model_ref_rms.append(max(m_rms, 1e-12))
    model_ref_rms = np.array(model_ref_rms)

    # model_output_rms ∝ gain_norm (linear, since dasp actual_gain = gain_norm * max_band_gain).
    # model_rms_per_unit_gain_norm = model_ref_rms / ref_gain_norm
    # We want: gain_norm * model_rms_per_unit_gain_norm = target_band_rms
    # → gain_norm = target_band_rms / model_rms_per_unit_gain_norm
    #             = target_band_rms * ref_gain_norm / model_ref_rms
    model_rms_per_unit_gain_norm = model_ref_rms / ref_gain_norm
    band_gain_norm = target_band_rms / model_rms_per_unit_gain_norm
    # Clip to valid sigmoid range and dasp's [0, 1] norm range
    band_gain_norm = np.clip(band_gain_norm, 0.01, 0.95)

    return band_gain_norm, np.array(band_decay_norm)


def plot_ir(ir_optim, ir_target, args, end_t=0.25, start_t=-0.01):
    ir_optim_np = ir_optim.detach().cpu().numpy().squeeze()
    ir_target_np = ir_target.detach().cpu().numpy().squeeze()
    fs = args.samplerate

    start_shift_samples = int(np.floor(-start_t * fs))
    if start_shift_samples < 0:
        start_shift_samples = 0
        start_t = 0.0

    end_idx = min(ir_optim_np.shape[0], int(np.ceil(end_t * fs)))
    t_start = start_t
    t_end = (end_idx - 1) / fs
    num_points = start_shift_samples + end_idx
    t = np.linspace(t_start, t_end, num_points, endpoint=True)

    padding = np.zeros(start_shift_samples)
    ir_optim_padded = np.concatenate((padding, ir_optim_np[:end_idx]))
    ir_target_padded = np.concatenate((padding, ir_target_np[:end_idx]))

    plt.figure(figsize=(8, 12))
    for idx, y in enumerate([ir_optim_padded, ir_target_padded], start=1):
        plt.subplot(2, 1, idx)
        plt.plot(t, y, linewidth=1)
        plt.title(f"IR ({start_t:.3f} s - {end_t} s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(start_t, end_t)
        plt.ylim(-1.0, 1.0)
        plt.grid(True, which="both", ls="--", alpha=0.4)

    save_path = os.path.join(args.train_dir, f"ir_0-{int(end_t * 1000)}ms_shifted.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_history(train_loss, valid_loss, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss", color="blue", linewidth=2)
    if valid_loss and len(valid_loss) > 0:
        plt.plot(
            valid_loss,
            label="Validation Loss",
            color="orange",
            linestyle="--",
            linewidth=2,
        )
    plt.title("Loss Evolution per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    save_path = os.path.join(save_dir, "loss_history.png")
    plt.savefig(save_path)
    plt.close()


class NoiseShapingReverbModel(nn.Module):
    """
    Trainable wrapper around dasp-pytorch noise-shaped reverb.

    Optimized parameters:
    - 12 band gains
    - 12 band decays

    Mix is fixed at 1.0 (fully wet) to match IR synthesis.
    """

    def __init__(
        self,
        sample_rate: int,
        ir_num_samples: int,
        num_bandpass_taps: int = 1023,
        init_gain_norm=0.5,
        init_decay_norm=0.5,
        max_band_decay: float = 10.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.ir_num_samples = ir_num_samples
        self.num_bandpass_taps = num_bandpass_taps
        self.max_band_decay = max_band_decay

        # max_band_decay controls the denormalized range [0, max_band_decay].
        # Inside dasp, actual_decay = band_decay * 10 + 1, and envelope = exp(-actual * t)
        # where t ∈ [0,1] spans ir_num_samples/sample_rate seconds.
        # T60_min = duration * log(1000) / (max_band_decay * 10 + 1).
        # max_band_decay is scaled with duration so T60_min stays ~0.3s (see main()).
        #
        # max_band_gain=5.0: calibration shows dasp needs actual gains of 1-4x to match
        # typical room IR levels (target band RMS exceeds model output by up to 4x).
        # Old max_band_gain=1.0 forced gain_norm ≡ 0.99 (clipped) for most bands
        # → optimizer had zero headroom → gains stuck at upper wall → loss plateaus.
        self.processor = NoiseShapedReverb(
            sample_rate=sample_rate,
            min_band_gain=0.0,
            max_band_gain=5.0,
            min_band_decay=0.0,
            max_band_decay=max_band_decay,
            min_mix=1.0,
            max_mix=1.0,
        )

        # Fixed HP/LP: 4th-order butterworth-style in frequency domain.
        # Removes DC drift (<30 Hz) and limits HF aliasing (>16 kHz).
        freqs = torch.fft.rfftfreq(ir_num_samples, 1.0 / sample_rate)
        f_lo, f_hi, order = 30.0, 16000.0, 4
        hp = 1.0 / torch.sqrt(1 + (f_lo / freqs.clamp(min=0.1)) ** (2 * order))
        lp = 1.0 / torch.sqrt(1 + (freqs / f_hi) ** (2 * order))
        self.register_buffer("bp_mask", (hp * lp).float())

        # Accept per-band arrays or scalar init values
        gain_arr = np.atleast_1d(np.asarray(init_gain_norm, dtype=np.float64))
        decay_arr = np.atleast_1d(np.asarray(init_decay_norm, dtype=np.float64))
        if gain_arr.size == 1:
            gain_arr = np.full(12, gain_arr.item())
        if decay_arr.size == 1:
            decay_arr = np.full(12, decay_arr.item())

        gain_arr = np.clip(gain_arr, 1e-4, 1 - 1e-4)
        decay_arr = np.clip(decay_arr, 1e-4, 1 - 1e-4)

        gain_logits = np.log(gain_arr / (1.0 - gain_arr))
        decay_logits = np.log(decay_arr / (1.0 - decay_arr))

        init_raw = torch.from_numpy(
            np.concatenate([gain_logits, decay_logits])[np.newaxis, :].astype(np.float32)
        )
        self.raw_params = nn.Parameter(init_raw)

    @staticmethod
    def _stable_sigmoid(logits: torch.Tensor) -> torch.Tensor:
        # Keep parameters in a trainable range and avoid hard saturation at 0/1.
        return torch.sigmoid(torch.clamp(logits, min=-6.0, max=6.0))

    def get_normalized_band_params(self):
        p = self._stable_sigmoid(self.raw_params)
        return p[:, :12], p[:, 12:24]

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError("Expected input shape [B, T, 1] or [B, T].")

        # Fix noise seed always (train and val) so the same parameters always produce
        # the same output → deterministic loss surface, consistent gradients.
        # dasp's noise_shaped_reverberation calls torch.randn() internally.
        # Must be set in both modes — without this, validation sees a random noise
        # draw unrelated to what was optimized, causing train/val divergence.
        torch.manual_seed(42)

        x_ch_first = x.permute(0, 2, 1)
        batch_size = x_ch_first.shape[0]

        params_norm = self._stable_sigmoid(self.raw_params).expand(batch_size, -1)
        mix = torch.ones(batch_size, 1, device=x.device)
        params_all = torch.cat([params_norm, mix], dim=1)

        param_dict = self.processor.extract_param_dict(params_all)
        denorm_param_dict = self.processor.denormalize_param_dict(param_dict)

        y = noise_shaped_reverberation(
            x_ch_first,
            sample_rate=self.sample_rate,
            **denorm_param_dict,
            num_samples=self.ir_num_samples,
            num_bandpass_taps=self.num_bandpass_taps,
        )

        y_mono = y.mean(dim=1)  # [B, T]
        # Apply fixed HP/LP in frequency domain (differentiable)
        Y = torch.fft.rfft(y_mono)
        Y = Y * self.bp_mask.to(Y.device)
        y_filt = torch.fft.irfft(Y, n=y_mono.shape[-1])
        return y_filt.unsqueeze(-1)  # [B, T, 1]


def _custom_bandpass_filterbank(num_bands: int, num_taps: int, sample_rate: float):
    """FIR filterbank with log-spaced centers, supporting arbitrary band counts.

    Returns (filters, centers):
      filters  – [num_bands, 1, num_taps] float32 tensor (flipped for conv1d)
      centers  – [num_bands] float64 ndarray of nominal center frequencies (Hz)
    """
    f_lo_global = 20.0
    f_hi_global = min(18000.0, sample_rate * 0.45)
    centers = np.geomspace(f_lo_global, f_hi_global, num_bands)
    # half-bandwidth in octaves between adjacent centres
    if num_bands > 1:
        bw_oct = np.log2(centers[1] / centers[0]) / 2.0
    else:
        bw_oct = 1.0

    filts = []
    nyq = sample_rate / 2.0
    for i, fc in enumerate(centers):
        if i == 0:
            f_cut = min(fc * 2 ** bw_oct, nyq * 0.999)
            filt = scipy.signal.firwin(num_taps, f_cut, fs=sample_rate)
        elif i == num_bands - 1:
            f_cut = max(fc / 2 ** bw_oct, 1.0)
            filt = scipy.signal.firwin(num_taps, f_cut, fs=sample_rate, pass_zero=False)
        else:
            f_min = max(fc / 2 ** bw_oct, 1.0)
            f_max = min(fc * 2 ** bw_oct, nyq * 0.999)
            filt = scipy.signal.firwin(num_taps, [f_min, f_max], fs=sample_rate, pass_zero=False)
        t = torch.from_numpy(filt.astype("float32"))
        filts.append(torch.flip(t, dims=[0]))

    filters = torch.stack(filts, dim=0).unsqueeze(1)  # [num_bands, 1, num_taps]
    return filters, centers


class CustomBandNoiseShaper(nn.Module):
    """Noise-shaped reverb with an arbitrary number of log-spaced bands.

    Implements the same gain + exponential-decay forward pass as dasp's
    noise_shaped_reverberation, but with a user-defined filterbank so band
    count is not limited to dasp's hardcoded 12.
    """

    MAX_BAND_GAIN = 5.0  # matches NoiseShapingReverbModel

    def __init__(
        self,
        num_bands: int,
        sample_rate: int,
        ir_num_samples: int,
        num_bandpass_taps: int = 1023,
        init_gain_norm=0.5,
        init_decay_norm=0.5,
        max_band_decay: float = 10.0,
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        self.sample_rate = int(sample_rate)
        self.ir_num_samples = int(ir_num_samples)
        self.num_bandpass_taps = int(num_bandpass_taps)
        self.max_band_decay = float(max_band_decay)

        filters, centers = _custom_bandpass_filterbank(num_bands, num_bandpass_taps, sample_rate)
        self.register_buffer("filterbank", filters)
        self.register_buffer("band_centers", torch.from_numpy(centers.astype("float32")))

        freqs = torch.fft.rfftfreq(ir_num_samples, 1.0 / sample_rate)
        f_lo, f_hi, order = 30.0, 16000.0, 4
        hp = 1.0 / torch.sqrt(1 + (f_lo / freqs.clamp(min=0.1)) ** (2 * order))
        lp = 1.0 / torch.sqrt(1 + (freqs / f_hi) ** (2 * order))
        self.register_buffer("bp_mask", (hp * lp).float())

        _g = np.asarray(init_gain_norm, dtype=float).ravel()
        _d = np.asarray(init_decay_norm, dtype=float).ravel()
        gain_arr = np.clip(np.broadcast_to(_g if _g.size == num_bands else np.full(num_bands, _g.item() if _g.size == 1 else _g[0]), (num_bands,)).copy(), 1e-4, 1 - 1e-4)
        decay_arr = np.clip(np.broadcast_to(_d if _d.size == num_bands else np.full(num_bands, _d.item() if _d.size == 1 else _d[0]), (num_bands,)).copy(), 1e-4, 1 - 1e-4)
        gain_logits = np.log(gain_arr / (1 - gain_arr))
        decay_logits = np.log(decay_arr / (1 - decay_arr))
        init_raw = torch.from_numpy(
            np.concatenate([gain_logits, decay_logits])[np.newaxis, :].astype("float32")
        )
        self.raw_params = nn.Parameter(init_raw)

    @staticmethod
    def _stable_sigmoid(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(logits, -6.0, 6.0))

    def get_normalized_band_params(self):
        p = self._stable_sigmoid(self.raw_params)
        return p[:, : self.num_bands], p[:, self.num_bands :]

    def forward(self, x):
        torch.manual_seed(42)
        bs = x.shape[0]
        gains_norm, decays_norm = self.get_normalized_band_params()

        gains = gains_norm * self.MAX_BAND_GAIN  # [1, num_bands]
        actual_decay = (decays_norm * self.max_band_decay) * 10.0 + 1.0  # [1, num_bands]

        num_samples = self.ir_num_samples
        pad_size = self.num_bandpass_taps - 1
        wn = torch.randn(
            bs * 2, self.num_bands, num_samples + pad_size,
            device=x.device, dtype=x.dtype,
        )
        wn_filt = F.conv1d(wn, self.filterbank.to(x.device, x.dtype), groups=self.num_bands)
        wn_filt = wn_filt.view(bs, 2, self.num_bands, num_samples)

        t = torch.linspace(0, 1, steps=num_samples, device=x.device, dtype=x.dtype)
        env = torch.exp(-actual_decay.view(1, 1, self.num_bands, 1) * t.view(1, 1, 1, -1))
        wn_filt = wn_filt * gains.view(1, 1, self.num_bands, 1) * env

        ir_mono = wn_filt.sum(dim=2).mean(dim=1)  # [bs, num_samples]

        IR = torch.fft.rfft(ir_mono)
        IR = IR * self.bp_mask.to(IR.device)
        ir_filt = torch.fft.irfft(IR, n=num_samples)
        return ir_filt.unsqueeze(-1)  # [B, T, 1]


def visualize_noise_shaping_parameters(model, samplerate, num_bandpass_taps, save_dir):
    gains, decays = model.get_normalized_band_params()
    gains = gains.detach().cpu().numpy().squeeze()
    decays = decays.detach().cpu().numpy().squeeze()

    if isinstance(model, CustomBandNoiseShaper):
        filters_np = model.filterbank.detach().cpu().numpy().squeeze(1)
        center_freqs = model.band_centers.detach().cpu().numpy().astype(np.float64)
    else:
        filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, samplerate)
        filters_np = filters.detach().cpu().numpy().squeeze(1)
        nominal_centers = np.array(
            [12.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 18000.0],
            dtype=np.float64,
        )
        center_freqs = np.clip(nominal_centers, 1.0, 0.999 * samplerate / 2.0)

    fft_len = max(8192, 2 ** int(np.ceil(np.log2(num_bandpass_taps * 4))))
    response = np.fft.rfft(filters_np, n=fft_len, axis=-1)
    response_mag = np.abs(response)
    response_db = 20.0 * np.log10(response_mag + 1e-8)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / samplerate)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True)

    for band_idx in range(response_db.shape[0]):
        axes[0].plot(
            freqs,
            response_db[band_idx],
            linewidth=1.2,
            alpha=0.85,
            label=f"Band {band_idx + 1}",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlim(20, samplerate / 2)
    axes[0].set_title("Noise-Shaping Bandpass Filter Responses")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, which="both", alpha=0.25)

    sort_idx = np.argsort(center_freqs)
    sorted_freqs = center_freqs[sort_idx]
    sorted_gains = gains[sort_idx]
    sorted_energy = sorted_gains**2
    sorted_decays = decays[sort_idx]

    ir_duration_s = model.ir_num_samples / float(samplerate)

    # Denormalize decay: both model types store decay in [0,1] norm space where
    # actual_decay = decay_norm * max_band_decay * 10 + 1 (dasp convention).
    if isinstance(model, CustomBandNoiseShaper):
        max_bd = model.max_band_decay
    else:
        decay_min, decay_max = model.processor.param_ranges.get("band0_decay", (0.0, 1.0))
        max_bd = decay_max - decay_min
    actual_decay = sorted_decays * max_bd * 10.0 + 1.0

    t60_s = (np.log(1000.0) * ir_duration_s) / np.maximum(actual_decay, 1e-8)

    axes[1].bar(sorted_freqs, sorted_gains, width=0.15 * sorted_freqs, alpha=0.8, label="Learned gain")
    ax_energy = axes[1].twinx()
    ax_energy.plot(sorted_freqs, sorted_energy, color="black", linewidth=2, marker="o", label="Initial energy proxy (gain²)")
    ax_energy.plot(
        sorted_freqs,
        t60_s,
        color="tab:green",
        linewidth=2,
        marker="s",
        linestyle="--",
        label="Estimated T60",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlim(20, samplerate / 2)
    left_max = float(np.max(sorted_gains)) if sorted_gains.size > 0 else 1.0
    axes[1].set_ylim(0, 1.1 * left_max if left_max > 0 else 1.0)
    right_max = max(float(np.max(sorted_energy)), float(np.max(t60_s)))
    ax_energy.set_ylim(0, 1.1 * right_max if right_max > 0 else 1.0)
    axes[1].set_title("Learned Initial Energy and Decay Rate per Band")
    axes[1].set_xlabel("Band center frequency (Hz)")
    axes[1].set_ylabel("Gain")
    ax_energy.set_ylabel("Energy proxy / T60 (s)")
    axes[1].grid(True, which="both", alpha=0.25)

    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax_energy.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper right")

    fig.savefig(os.path.join(save_dir, "bandpass_response_and_energy.png"))
    plt.close(fig)

    np.savetxt(
        os.path.join(save_dir, "noise_shaping_params.csv"),
        np.stack([sorted_freqs, sorted_gains, t60_s], axis=1),
        delimiter=",",
        header="center_freq_hz,band_gain,t60_seconds",
        comments="",
    )

def plot_band_energy_envelopes(ir_optim, target_rir, samplerate, save_dir,
                               nfft=4096, hop=512):
    """
    Save the per-band temporal energy envelopes (dB) that BandEnergyEnvelopeLoss
    actually compares, plus the per-band difference.  Useful for diagnosing why
    the loss is not converging.
    """
    from reverb.losses import BandEnergyEnvelopeLoss

    loss_fn = BandEnergyEnvelopeLoss(fs=samplerate, nfft=nfft, hop=hop, device="cpu")

    # Ensure [B, T] shape — keep everything on CPU for diagnostics
    est = ir_optim.detach().cpu()
    tgt = target_rir.detach().cpu()
    if est.dim() == 3 and est.shape[-1] == 1:
        est = est.squeeze(-1)
    if tgt.dim() == 3 and tgt.shape[-1] == 1:
        tgt = tgt.squeeze(-1)
    if est.dim() == 1:
        est = est.unsqueeze(0)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    # Same normalisation as the loss: divide both by target RMS in signal region
    skip = loss_fn.skip_frames
    ref_rms = tgt[:, skip * hop:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    tgt_n = tgt / ref_rms
    est_n = est / ref_rms

    window = loss_fn.window
    spec_est = torch.stft(est_n, n_fft=nfft, hop_length=hop, window=window, return_complex=True)
    spec_tgt = torch.stft(tgt_n, n_fft=nfft, hop_length=hop, window=window, return_complex=True)

    eps = 1e-8
    smooth_k = loss_fn.smooth_frames
    band_labels = [f"{fc}" for fc in BandEnergyEnvelopeLoss.BAND_CENTERS
                   if (min(nfft // 2, int((fc * 2**0.5) / (samplerate / nfft))) >
                       max(1, int((fc / 2**0.5) / (samplerate / nfft))))]
    n_bands = len(loss_fn.band_ranges)
    t_frames = spec_est.shape[-1]
    t_axis = np.arange(t_frames) * hop / samplerate

    db_est_all = np.zeros((n_bands, t_frames))
    db_tgt_all = np.zeros((n_bands, t_frames))
    for i, (k_lo, k_hi) in enumerate(loss_fn.band_ranges):
        n_bins = k_hi - k_lo
        # Per-bin power (consistent with BandEnergyEnvelopeLoss)
        e_est = torch.sum(torch.abs(spec_est[0, k_lo:k_hi, :]) ** 2, dim=0) / n_bins
        e_tgt = torch.sum(torch.abs(spec_tgt[0, k_lo:k_hi, :]) ** 2, dim=0) / n_bins
        if smooth_k > 1:
            k = smooth_k | 1  # force odd
            e_est = torch.nn.functional.avg_pool1d(
                e_est.unsqueeze(0).unsqueeze(0), k, stride=1,
                padding=k // 2, count_include_pad=False).squeeze()
            e_tgt = torch.nn.functional.avg_pool1d(
                e_tgt.unsqueeze(0).unsqueeze(0), k, stride=1,
                padding=k // 2, count_include_pad=False).squeeze()
        db_est_all[i] = 10 * np.log10(e_est.cpu().numpy() + eps)
        db_tgt_all[i] = 10 * np.log10(e_tgt.cpu().numpy() + eps)

    diff = db_est_all - db_tgt_all

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    for i in range(n_bands):
        label = band_labels[i] if i < len(band_labels) else f"band{i}"
        axes[0].plot(t_axis, db_est_all[i], linewidth=1, label=label)
    axes[0].set_title("Model — per-band energy envelope (dB)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("dB")
    axes[0].legend(ncol=4, fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i in range(n_bands):
        label = band_labels[i] if i < len(band_labels) else f"band{i}"
        axes[1].plot(t_axis, db_tgt_all[i], linewidth=1, label=label)
    axes[1].set_title("Target — per-band energy envelope (dB)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("dB")
    axes[1].legend(ncol=4, fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Weighted difference: show what the loss actually penalises
    time_decay = 3.0
    w = np.exp(-time_decay * np.linspace(0, 1, t_frames))
    w /= w.sum()
    weighted_diff = diff * w[np.newaxis, :]  # [bands, T]

    im = axes[2].imshow(
        weighted_diff,
        aspect="auto",
        origin="lower",
        extent=[t_axis[0], t_axis[-1], 0, n_bands],
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )
    axes[2].set_title("Weighted difference (model − target) × time_weight — red=too loud, blue=too quiet")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Band index")
    axes[2].set_yticks(np.arange(n_bands) + 0.5)
    axes[2].set_yticklabels(band_labels[:n_bands], fontsize=7)
    plt.colorbar(im, ax=axes[2], label="weighted dB difference")

    fig.savefig(os.path.join(save_dir, "band_energy_envelopes.png"), dpi=120)
    plt.close(fig)

    # Also save raw arrays for further analysis
    np.savez(
        os.path.join(save_dir, "band_energy_envelopes.npz"),
        db_est=db_est_all,
        db_tgt=db_tgt_all,
        diff=diff,
        t_axis=t_axis,
        band_labels=np.array(band_labels[:n_bands]),
    )


def optimize_noise_shaping_baseline(args):
    import math
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=torch.float32)
    if target_rir.dim() > 1:
        target_rir = target_rir.mean(dim=-1)

    rir_onset = find_onset(target_rir)
    target_trimmed = target_rir[rir_onset:]

    # Auto-size nfft from broadband RT60.
    if args.nfft is None:
        try:
            raw_np = target_trimmed.numpy()
            edc_bb = np.cumsum(raw_np[::-1] ** 2)[::-1]
            edc_bb_db = 10 * np.log10(edc_bb / (edc_bb[0] + 1e-30) + 1e-30)
            mask_bb = (edc_bb_db >= -35) & (edc_bb_db <= -5)
            if mask_bb.sum() > 10:
                t_bb = np.arange(len(edc_bb_db))[mask_bb] / args.samplerate
                slope_bb, _ = np.polyfit(t_bb, edc_bb_db[mask_bb], 1)
                max_rt = -60.0 / slope_bb
            else:
                max_rt = len(raw_np) / args.samplerate
            max_rt = min(max_rt, 8.0)
            ir_duration = max(max_rt * 1.2, 2.0)
        except Exception:
            ir_duration = max(len(target_trimmed) / args.samplerate, 2.0)
        half_seconds = math.ceil(ir_duration / 0.5)
        args.nfft = int(half_seconds * 0.5 * args.samplerate)
        print(f"  Auto nfft: {args.nfft} ({args.nfft / args.samplerate:.1f}s) from estimated RT")

    target_rir = target_trimmed[: args.nfft]
    if target_rir.shape[0] < args.nfft:
        target_rir = torch.nn.functional.pad(target_rir, (0, args.nfft - target_rir.shape[0]))

    target_rir = target_rir.view(1, -1, 1)
    target_rir = target_rir / torch.max(torch.abs(target_rir))

    save_audio(
        os.path.join(args.train_dir, "ir_target.wav"),
        target_rir.squeeze(),
        fs=args.samplerate,
    )

    # Fixed initialization: T60=3s for all bands, max gain just below clipping.
    # Same starting point regardless of target RIR — matches FDN default init (RT=3s).
    # decay_norm is computed from IR duration because dasp's decay envelope spans [0,1]
    # relative to IR duration: actual_decay = log(1000) * duration / t60.
    #
    # MAX_BAND_DECAY is scaled with duration to keep T60_min ≈ 0.3s regardless of IR length.
    # Formula: T60_min = duration * log(1000) / (max_band_decay * 10 + 1)
    # → max_band_decay = (duration * log(1000) / T60_min - 1) / 10
    # Floored at 10.0 so short RIRs keep the original (well-conditioned) range.
    T60_MIN_ACHIEVABLE = 0.1  # seconds — ensure high-freq bands are reachable
    MAX_BAND_GAIN = 5.0
    duration = args.nfft / args.samplerate
    MAX_BAND_DECAY = max(
        (duration * float(np.log(1000)) / T60_MIN_ACHIEVABLE - 1.0) / 10.0,
        10.0,
    )
    print(f"  max_band_decay={MAX_BAND_DECAY:.1f} (duration={duration:.1f}s, T60_min≈{duration * np.log(1000) / (MAX_BAND_DECAY * 10 + 1):.2f}s)")
    init_t60 = 3.0
    actual_decay = np.log(1000) * duration / init_t60
    denorm = max((actual_decay - 1.0) / 10.0, 0.0)
    decay_norm_3s = float(np.clip(denorm / MAX_BAND_DECAY, 1e-4, 1 - 1e-4))

    noise_bands = int(getattr(args, "noise_bands", 12))

    # Gain calibration: find the max gain_norm that keeps peak output < 0.90.
    _test_gain = 0.90
    DESIRED_PEAK = 0.90
    if noise_bands == 12:
        # Use dasp's noise_shaped_reverberation for calibration (matches the 12-band model).
        _proc_cal = NoiseShapedReverb(
            sample_rate=args.samplerate, min_band_gain=0.0, max_band_gain=MAX_BAND_GAIN,
            min_band_decay=0.0, max_band_decay=float(MAX_BAND_DECAY), min_mix=1.0, max_mix=1.0,
        )
        _x_imp = torch.zeros(1, 1, args.nfft); _x_imp[0, 0, 0] = 1.0
        _params_cal = torch.cat([
            torch.full((1, 12), _test_gain),
            torch.full((1, 12), decay_norm_3s),
            torch.ones(1, 1),
        ], dim=1)
        torch.manual_seed(42)
        with torch.no_grad():
            _pd = _proc_cal.extract_param_dict(_params_cal)
            _dd = _proc_cal.denormalize_param_dict(_pd)
            _y_cal = noise_shaped_reverberation(
                _x_imp, sample_rate=args.samplerate, **_dd,
                num_samples=args.nfft, num_bandpass_taps=args.num_bandpass_taps,
            )
        _peak = float(_y_cal.mean(dim=1).squeeze().abs().max())
    else:
        # Calibrate via a trial CustomBandNoiseShaper forward pass.
        _x_imp = torch.zeros(1, 1, args.nfft); _x_imp[0, 0, 0] = 1.0
        _cal_model = CustomBandNoiseShaper(
            num_bands=noise_bands,
            sample_rate=args.samplerate,
            ir_num_samples=args.nfft,
            num_bandpass_taps=args.num_bandpass_taps,
            init_gain_norm=_test_gain,
            init_decay_norm=decay_norm_3s,
            max_band_decay=float(MAX_BAND_DECAY),
        )
        with torch.no_grad():
            _y_cal = _cal_model(_x_imp)
        _peak = float(_y_cal.squeeze().abs().max())
        del _cal_model

    calibrated_gain = float(np.clip(_test_gain * DESIRED_PEAK / max(_peak, 1e-6), 0.01, 0.95))
    print(f"  Gain calibration ({noise_bands} bands): trial peak={_peak:.4f} @ gain_norm={_test_gain} → calibrated_gain={calibrated_gain:.4f}")

    init_gains = np.full(noise_bands, calibrated_gain)
    init_decays = np.full(noise_bands, decay_norm_3s)
    print(f"  Fixed init: T60={init_t60}s → decay_norm={decay_norm_3s:.4f}, gain_norm={calibrated_gain:.4f} (all bands)")

    if noise_bands == 12:
        model = NoiseShapingReverbModel(
            sample_rate=args.samplerate,
            ir_num_samples=args.nfft,
            num_bandpass_taps=args.num_bandpass_taps,
            init_gain_norm=init_gains,
            init_decay_norm=init_decays,
            max_band_decay=float(MAX_BAND_DECAY),
        )
    else:
        model = CustomBandNoiseShaper(
            num_bands=noise_bands,
            sample_rate=args.samplerate,
            ir_num_samples=args.nfft,
            num_bandpass_taps=args.num_bandpass_taps,
            init_gain_norm=init_gains,
            init_decay_norm=init_decays,
            max_band_decay=float(MAX_BAND_DECAY),
        )

    input_impulse = signal_gallery(
        1,
        n_samples=args.nfft,
        n=1,
        signal_type="impulse",
        fs=args.samplerate,
        device=args.device,
    )

    with torch.no_grad():
        _init_gains, _init_decays = model.get_normalized_band_params()
        init_decays = _init_decays.detach().cpu().clone()
        init_gains = _init_gains.detach().cpu().clone()

    # ---- Build loss stack ----
    from reverb.lightning_module import ShellLightningModule, WeightedCriterion
    from reverb.datamodule import ImpulseRIRDataModule, ImpulseRIRDataConfig

    if getattr(args, "steinmetz_loss", False):
        # Original Steinmetz et al. (2021) formulation: a single MRSTFT loss
        # (multi-resolution log-magnitude STFT) on the output, with the first
        # 200 ms skipped to account for the noise-shaper's group delay.
        # No SpectralEDC / BandEnergy / PowSpec / DRR terms — this is the
        # baseline's own loss, so the comparison vs our FDN is fair under
        # each model's intended objective.
        skip = int(0.2 * args.samplerate)
        raw_losses = [
            ("MRSTFT", 1.0, MultiResoSTFT(skip_samples=skip, level_invariant=True)),
        ]
    else:
        w_sedc, w_be, w_ps = [float(x) for x in args.loss_weights.split(",")]
        raw_losses = [
            ("SpectralEDC", w_sedc, SpectralEDCLoss(fs=args.samplerate, device=args.device, db_lo=-5, db_hi=-35, use_freq_weighting=True)),
            ("BandEnergy",  w_be,   BandEnergyLoss(fs=args.samplerate, device=args.device)),
            ("PowSpec",     w_ps,   PowerSpectrumLoss(fs=args.samplerate, device=args.device, use_freq_weighting=True)),
        ]
        if args.w_drr > 0:
            raw_losses.append(("DRR", args.w_drr, DRRLoss(fs=args.samplerate)))

    # Normalize alpha by init loss value — same as FDN (alpha = weight / init_val)
    target_sq = target_rir[:, :, 0].to(args.device)
    model.to(args.device)
    torch.manual_seed(42)
    with torch.no_grad():
        init_hat = model(input_impulse.to(args.device))
        init_hat_sq = init_hat[:, :, 0]
    print(f"\n  Loss normalization (init values):")
    criteria = []
    for name, weight, loss_fn in raw_losses:
        loss_fn.to(args.device)
        with torch.no_grad():
            init_val = loss_fn(init_hat_sq, target_sq).item()
        alpha = weight / init_val if init_val > 1e-10 else weight
        print(f"    {name:>15}: init={init_val:.4f}, weight={weight}, alpha={alpha:.4f}")
        criteria.append(WeightedCriterion(alpha=alpha, criterion=loss_fn))

    # ---- Colored noise floor injection (same as FDN training) ----
    # Real RIRs have a non-decaying noise floor at the tail. Without this,
    # losses penalize the clean model tail vs the noisy target tail.
    # Extract noise from the last 100ms of the (peak-normalized) target RIR.
    import torch.nn.functional as _F
    _tail_samples = int(0.1 * args.samplerate)
    _target_np_tail = target_rir[0, -_tail_samples:, 0]  # [T]
    _noise_rms = float(torch.sqrt((_target_np_tail ** 2).mean()))
    _noise_spectrum = None
    if _noise_rms > 1e-7:
        _tail_padded = _F.pad(_target_np_tail, (0, args.nfft - _tail_samples))
        _noise_spectrum = torch.abs(torch.fft.rfft(_tail_padded))
        _noise_spectrum = _noise_spectrum / (_noise_spectrum.max() + 1e-10)
        print(f"  Noise floor RMS={_noise_rms:.6f} ({20*np.log10(_noise_rms+1e-10):.1f} dB) — injecting during training")

    if _noise_spectrum is not None:
        class _NoisyModel(torch.nn.Module):
            def __init__(self_, inner, spec, rms):
                super().__init__()
                self_.inner = inner
                self_.rms = rms
                self_.register_buffer("spec", spec)
            def forward(self_, x):
                y = self_.inner(x)
                if self_.training:
                    B, T, _C = y.shape
                    white = torch.randn(B, T, device=y.device)
                    colored = torch.fft.irfft(torch.fft.rfft(white) * self_.spec.to(y.device), n=T)
                    colored = colored / (colored.std(dim=-1, keepdim=True) + 1e-10) * self_.rms
                    y = y.clone()
                    y[:, :, 0] = y[:, :, 0] + colored
                return y
        train_model = _NoisyModel(model, _noise_spectrum, _noise_rms)
    else:
        train_model = model

    # ---- Lightning training ----
    import pytorch_lightning as _pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    lit = ShellLightningModule(
        model=train_model, criteria=criteria,
        lr=args.lr, step_size=args.lr_step_size, step_gamma=0.5,
        sample_rate=args.samplerate,
    )
    data = ImpulseRIRDataModule(
        inputs=input_impulse, targets=target_rir,
        config=ImpulseRIRDataConfig(num_copies=max(1, args.num), batch_size=args.batch_size),
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.train_dir, "checkpoints"),
        filename="best", monitor="valid/loss", mode="min", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="valid/loss", patience=args.patience, min_delta=args.es_min_delta,
        mode="min", verbose=True,
    )
    pl_trainer = _pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_stop_cb],
        logger=False,  # release: no TensorBoard dependency
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    _train_start = time.perf_counter()
    pl_trainer.fit(lit, datamodule=data)
    _train_wall_s = time.perf_counter() - _train_start

    # Restore best checkpoint
    if ckpt_cb.best_model_path:
        state = torch.load(ckpt_cb.best_model_path, map_location=args.device, weights_only=False)
        lit.load_state_dict(state["state_dict"])
    lit.to(args.device)

    plot_loss_history(lit._epoch_train_losses, lit._epoch_valid_losses, args.train_dir)

    model.eval()
    with torch.no_grad():
        # Fix the noise seed during inference so the saved WAV uses the same
        # noise realization as training — otherwise the output is a random
        # draw that looks nothing like what was actually optimized.
        torch.manual_seed(42)
        ir_optim = model(input_impulse).squeeze()
        ir_optim = ir_optim.clamp(-1.0, 1.0)

    save_audio(
        os.path.join(args.train_dir, "ir_optim.wav"),
        ir_optim,
        fs=args.samplerate,
    )

    plot_ir(ir_optim, target_rir, args)

    gains, decays = model.get_normalized_band_params()
    decays_cpu = decays.detach().cpu()
    gains_cpu = gains.detach().cpu()
    decay_delta = torch.abs(decays_cpu - init_decays)
    gain_delta = torch.abs(gains_cpu - init_gains)

    np.save(os.path.join(args.train_dir, "initial_band_gains.npy"), init_gains.numpy())
    np.save(os.path.join(args.train_dir, "initial_band_decays.npy"), init_decays.numpy())
    np.save(os.path.join(args.train_dir, "optimized_band_gains.npy"), gains.detach().cpu().numpy())
    np.save(os.path.join(args.train_dir, "optimized_band_decays.npy"), decays.detach().cpu().numpy())

    # ---- Save parameter CSV ----
    try:
        gains_np  = gains.detach().cpu().numpy().flatten()
        decays_np = decays.detach().cpu().numpy().flatten()
        ig_np = init_gains.numpy().flatten()
        id_np = init_decays.numpy().flatten()
        # dasp uses 12 octave bands; center frequencies ~31.5 Hz to 16 kHz
        import dasp_pytorch
        _filters = dasp_pytorch.signal.octave_band_filterbank(args.num_bandpass_taps, args.samplerate)
        # Approximate center frequencies from filter count
        _n = len(gains_np)
        _fc = [31.5 * (2 ** i) for i in range(_n)]
        with open(os.path.join(args.train_dir, "noiseshaper_parameters.csv"), "w", newline="") as _pf:
            _pw = csv.DictWriter(_pf, fieldnames=["band", "approx_fc_hz",
                                                    "gain_norm_init", "gain_norm_optim",
                                                    "decay_norm_init", "decay_norm_optim"])
            _pw.writeheader()
            for i in range(_n):
                _pw.writerow({"band": i, "approx_fc_hz": f"{_fc[i]:.1f}",
                               "gain_norm_init":  f"{ig_np[i]:.4f}",
                               "gain_norm_optim": f"{gains_np[i]:.4f}",
                               "decay_norm_init": f"{id_np[i]:.4f}",
                               "decay_norm_optim": f"{decays_np[i]:.4f}"})
        print(f"  Saved: noiseshaper_parameters.csv")
    except Exception as _pe:
        print(f"  [noiseshaper_parameters.csv skipped: {_pe}]")

    visualize_noise_shaping_parameters(
        model=model,
        samplerate=args.samplerate,
        num_bandpass_taps=args.num_bandpass_taps,
        save_dir=args.train_dir,
    )

    # ---- analysis_overview.png + analysis_data.npz ----
    try:
        from scipy.signal import hilbert
        fs = args.samplerate
        nfft = args.nfft
        out_dir = args.train_dir

        target_np = target_rir.squeeze().cpu().numpy()
        optim_np  = ir_optim.detach().cpu().numpy()
        ml = min(len(target_np), len(optim_np))
        target_np = target_np[:ml]
        optim_np  = optim_np[:ml]

        # Power spectrum (log-magnitude, 1/3-oct smoothed)
        freq_axis = np.fft.rfftfreq(ml, d=1.0 / fs)
        S_t = np.abs(np.fft.rfft(target_np)) + 1e-10
        S_o = np.abs(np.fft.rfft(optim_np))  + 1e-10
        spec_target = 20 * np.log10(S_t)
        spec_optim  = 20 * np.log10(S_o)

        def _smooth(spec, freq_axis, frac=0.333):
            out = spec.copy()
            for i, f in enumerate(freq_axis):
                if f < 20:
                    continue
                f_lo, f_hi = f * 2 ** (-frac / 2), f * 2 ** (frac / 2)
                mask = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                if mask.sum() > 0:
                    out[i] = spec[mask].mean()
            return out

        spec_target_s = _smooth(spec_target, freq_axis)
        spec_optim_s  = _smooth(spec_optim,  freq_axis)

        # Per-octave-band energy
        octave_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        def _band_energy_db(sig, fc, fs):
            f_lo, f_hi = fc / np.sqrt(2), fc * np.sqrt(2)
            from scipy.signal import butter, sosfilt
            sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
            filtered = sosfilt(sos, sig)
            rms = np.sqrt(np.mean(filtered ** 2) + 1e-30)
            return 20 * np.log10(rms)

        energy_target = [_band_energy_db(target_np, fc, fs) for fc in octave_centers]
        energy_optim  = [_band_energy_db(optim_np,  fc, fs) for fc in octave_centers]
        energy_init   = energy_target  # NoiseShaper has no separate init IR; skip

        # Energy decay curves
        t_axis = np.arange(ml) / fs
        edc_t = np.cumsum(target_np[::-1] ** 2)[::-1]
        edc_o = np.cumsum(optim_np[::-1]  ** 2)[::-1]
        _ref = edc_t[0] + 1e-30
        edc_target_db = 10 * np.log10(edc_t / _ref + 1e-30)
        edc_optim_db  = 10 * np.log10(edc_o / _ref + 1e-30)

        loss_history = list(lit._epoch_train_losses)

        # ---- Figure ----
        fig, axes_ov = plt.subplots(3, 2, figsize=(14, 15))

        # (0,0) Loss convergence
        ax = axes_ov[0, 0]
        epochs = np.arange(len(loss_history))
        ax.plot(epochs, loss_history, "k-", label="Train loss", linewidth=2)
        if lit._epoch_valid_losses:
            ax.plot(np.arange(len(lit._epoch_valid_losses)), lit._epoch_valid_losses,
                    "r--", label="Valid loss", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Convergence")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # (0,1) Power spectrum
        ax = axes_ov[0, 1]
        ax.semilogx(freq_axis, spec_target_s, label="Target", color="tab:blue", linewidth=2)
        ax.semilogx(freq_axis, spec_optim_s,  label="Optimized", color="tab:red", linewidth=1.5)
        ax.set_xlim(20, fs / 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Power Spectrum (1/3-oct smoothed)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        # (1,0) Per-band energy
        ax = axes_ov[1, 0]
        x_pos = np.arange(len(octave_centers))
        w = 0.35
        ax.bar(x_pos - w / 2, energy_target, w, label="Target",    color="tab:blue", alpha=0.8)
        ax.bar(x_pos + w / 2, energy_optim,  w, label="Optimized", color="tab:red",  alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(f) for f in octave_centers], rotation=45, fontsize=7)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Energy (dB)")
        ax.set_title("Per-Band Energy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,1) Energy decay envelope
        ax = axes_ov[1, 1]
        env_target = 20 * np.log10(np.abs(hilbert(target_np)) + 1e-10)
        env_optim  = 20 * np.log10(np.abs(hilbert(optim_np))  + 1e-10)
        ds = max(1, ml // 5000)
        ax.plot(t_axis[::ds], env_target[::ds], label="Target",    color="tab:blue", alpha=0.6, linewidth=0.8)
        ax.plot(t_axis[::ds], env_optim[::ds],  label="Optimized", color="tab:red",  alpha=0.6, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Envelope (dB)")
        ax.set_title("Energy Decay Envelope")
        ax.set_ylim(-80, 5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (2,0) First 50ms time domain
        ax = axes_ov[2, 0]
        n_50ms = int(0.05 * fs)
        ax.plot(t_axis[:n_50ms] * 1000, target_np[:n_50ms], label="Target",    color="tab:blue", alpha=0.7)
        ax.plot(t_axis[:n_50ms] * 1000, optim_np[:n_50ms],  label="Optimized", color="tab:red",  alpha=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title("First 50ms (Time Domain)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (2,1) T30 comparison — target vs optimized
        ax = axes_ov[2, 1]
        try:
            from reverb.metrics import measure_rt60_continuous
            from scipy.signal import median_filter
            f_t, rt60_t = measure_rt60_continuous(target_np, fs, nfft=nfft)
            f_o, rt60_o = measure_rt60_continuous(optim_np, fs, nfft=nfft)
            valid_t = np.isfinite(rt60_t); rt60_t = np.where(valid_t, median_filter(np.where(valid_t, rt60_t, 0.0), size=15), np.nan)
            valid_o = np.isfinite(rt60_o); rt60_o = np.where(valid_o, median_filter(np.where(valid_o, rt60_o, 0.0), size=15), np.nan)
            ax.plot(f_t, rt60_t, color="tab:blue", linewidth=1.5, label="Target")
            ax.plot(f_o, rt60_o, color="tab:red", linewidth=1.5, label="Optimized", alpha=0.8)
        except Exception as _te:
            ax.text(0.5, 0.5, f"T30 failed:\n{_te}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("T30 (s)")
        ax.set_title("T30 — Target vs Optimized")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        fig.suptitle(
            f"NoiseShaper — {args.max_epochs} epochs\nTarget: {os.path.basename(args.target_rir)}",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "analysis_overview.png"), dpi=150)
        plt.close(fig)
        print("  Saved: analysis_overview.png")

        # ---- analysis_data.npz ----
        np.savez_compressed(
            os.path.join(out_dir, "analysis_data.npz"),
            fs=np.array([fs]),
            nfft=np.array([nfft]),
            freq_axis=freq_axis,
            t_axis=t_axis,
            spec_target=spec_target,
            spec_optim=spec_optim,
            spec_target_s=spec_target_s,
            spec_optim_s=spec_optim_s,
            octave_centers=np.array(octave_centers, dtype=float),
            energy_target=np.array(energy_target),
            energy_optim=np.array(energy_optim),
            edc_target_db=edc_target_db,
            edc_optim_db=edc_optim_db,
            loss_history=np.array(loss_history),
        )
        print("  Saved: analysis_data.npz")
    except Exception as _e:
        print(f"  [analysis_overview/npz skipped: {_e}]")

    plot_band_energy_envelopes(
        ir_optim=ir_optim,
        target_rir=target_rir.squeeze(),
        samplerate=args.samplerate,
        save_dir=args.train_dir,
    )

    # Run standardized ReverbAnalyzer metrics for compatibility with analysis scripts
    try:
        metrics_dir = os.path.join(args.train_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        compare_rir_metrics(
            os.path.join(args.train_dir, "ir_target.wav"),
            os.path.join(args.train_dir, "ir_optim.wav"),
            metrics_dir
        )
    except Exception as e:
        print(f"Metrics comparison failed: {e}")

    # ---- Save training time ----
    try:
        with open(os.path.join(args.train_dir, "train_time.txt"), "w") as _tf:
            _tf.write(f"{_train_wall_s:.1f}\n")
        print(f"  Training time: {_train_wall_s/60:.1f} min")
    except Exception as _te:
        print(f"  [train_time.txt skipped: {_te}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=None, help="FFT size (default: auto from RIR RT60)")
    parser.add_argument("--samplerate", type=int, default=48000, help="Sampling rate")
    parser.add_argument("--num", type=int, default=2, help="Dataset expansion size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dataset_split",
        type=float,
        default=0.5,
        help="Train/validation split ratio",
    )
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping: stop if valid/loss doesn't improve for this many epochs.")
    parser.add_argument("--es_min_delta", type=float, default=3e-2,
                        help="Early stopping: minimum absolute improvement in valid/loss to count as progress.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--lr_step_size", type=int, default=50, help="LR scheduler step size")
    parser.add_argument("--train_dir", type=str, help="Directory to save training results")
    parser.add_argument(
        "--target_rir",
        type=str,
        default="flamo/rirs/arni_35_3541_4_2.wav",
        help="Path to target RIR",
    )

    parser.add_argument(
        "--num_bandpass_taps",
        type=int,
        default=1023*4 + 1,
        help="Odd number of taps for noise-shaping octave filterbank",
    )
    parser.add_argument(
        "--noise_bands",
        type=int,
        default=12,
        help="Number of bands for noise-shaping filterbank. 12 uses dasp's octave filterbank; "
             "other values build a log-spaced custom filterbank (e.g. 24 for half-octave, 32 for ~third-octave).",
    )
    parser.add_argument("--loss_weights", type=str, default="10,5,5",
                        help="Loss weights as 'SpectralEDC,BandEnergy,PowSpec' (default: 10,5,5)")
    parser.add_argument("--w_drr", type=float, default=2.0, help="Weight for DRR loss (0=disabled)")
    parser.add_argument("--steinmetz_loss", action="store_true",
                        help="Use the original Steinmetz et al. (2021) MRSTFT-only loss "
                             "instead of our composite loss. Overrides --loss_weights and --w_drr.")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Using device: {args.device}")

    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S") + "_noise_shaping")
        os.makedirs(args.train_dir)

    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )

    optimize_noise_shaping_baseline(args)
