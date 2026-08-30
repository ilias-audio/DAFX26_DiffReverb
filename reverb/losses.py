import torch
import torch.nn as nn

from . import feature_cache as _fc


def _peak_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize each batch element to peak=1.0 (only scales down, never up)."""
    if x.dim() == 2:
        peak = torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1.0)
    elif x.dim() == 3:
        peak = torch.amax(torch.abs(x), dim=(-2, -1), keepdim=True).clamp(min=1.0)
    else:
        peak = torch.max(torch.abs(x)).clamp(min=1.0)
    return x / peak


class _BandEDCBase(nn.Module):
    """
    Base class for per-band EDC losses. Subclasses define which dB range
    and which frequency bands to emphasize.
    """

    def __init__(
        self,
        fs=48000,
        nfft=4096,
        hop=2048,
        n_bands=8,
        device="cuda",
        db_lo=-5,
        db_hi=-35,
        emphasis_bands=None,
        emphasis_factor=3.0,
    ):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.db_lo = db_lo
        self.db_hi = db_hi
        self.emphasis_factor = emphasis_factor
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)

        band_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        self.band_ranges = []
        self.band_weights = []
        bin_hz = fs / nfft
        emphasis_set = set(emphasis_bands or [])
        for fc in band_centers[:n_bands]:
            f_lo = fc / 2**0.5
            f_hi = fc * 2**0.5
            if f_hi > fs / 2:
                break
            k_lo = max(1, int(f_lo / bin_hz))
            k_hi = min(nfft // 2, int(f_hi / bin_hz))
            if k_hi > k_lo:
                self.band_ranges.append((k_lo, k_hi, fc))
                self.band_weights.append(emphasis_factor if fc in emphasis_set else 1.0)
        # Persistent per-target cache.
        self._tgt_id: int | None = None
        self._tgt_band_edc_db: list[torch.Tensor] | None = None
        self._tgt_band_mask: list[torch.Tensor] | None = None

    def _target_features(self, target):
        if id(target) == self._tgt_id and self._tgt_band_edc_db is not None:
            return self._tgt_band_edc_db, self._tgt_band_mask
        with torch.no_grad():
            spec_tgt = _fc.stft_cached(target, self.nfft, self.hop, self.window,
                                       is_target=True)
            edc_db_list, mask_list = [], []
            eps = 1e-10
            for k_lo, k_hi, _band_fc in self.band_ranges:
                band_tgt = torch.sum(torch.abs(spec_tgt[:, k_lo:k_hi, :]) ** 2, dim=1)
                edc_tgt = torch.flip(torch.cumsum(torch.flip(band_tgt, [-1]), dim=-1), [-1])
                edc_tgt_db = 10 * torch.log10(edc_tgt / (edc_tgt[:, :1] + eps) + eps)
                mask = torch.sigmoid(5 * (edc_tgt_db - self.db_hi)) * torch.sigmoid(
                    5 * (self.db_lo - edc_tgt_db)
                )
                edc_db_list.append(edc_tgt_db)
                mask_list.append(mask)
        self._tgt_id = id(target)
        self._tgt_band_edc_db = edc_db_list
        self._tgt_band_mask = mask_list
        return edc_db_list, mask_list

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        estimation = _peak_normalize(estimation)

        spec_est = _fc.stft_cached(estimation, self.nfft, self.hop, self.window,
                                   is_target=False)
        edc_db_list, mask_list = self._target_features(target)

        loss = 0.0
        total_weight = 0.0
        eps = 1e-10
        for (k_lo, k_hi, _band_fc), bw, edc_tgt_db, mask in zip(
            self.band_ranges, self.band_weights, edc_db_list, mask_list
        ):
            band_est = torch.sum(torch.abs(spec_est[:, k_lo:k_hi, :]) ** 2, dim=1)
            edc_est = torch.flip(torch.cumsum(torch.flip(band_est, [-1]), dim=-1), [-1])
            edc_est_db = 10 * torch.log10(edc_est / (edc_est[:, :1] + eps) + eps)

            diff = (edc_est_db - edc_tgt_db) ** 2
            band_loss = (diff * mask).sum() / (mask.sum() + 1)
            loss += bw * band_loss
            total_weight += bw

        return loss / (total_weight + 1e-10) / 900


class LinearEDCLoss(nn.Module):
    """Per-band EDC loss in linear scale (energy domain).

    Unlike T30Loss/EDTLoss which operate in dB, this computes the Schroeder
    Energy Decay Curve in linear scale and uses L2 loss directly.  This avoids
    gradient pathologies near the noise floor where dB = 10*log10(x) has
    d/dx → ∞ as x → 0.

    Dal Santo et al. (2025) show that linear EDC loss (L_EDC,lin) achieves
    5.15% MAE vs 105% for multi-resolution STFT loss when training recursive
    attenuation filters.
    """

    def __init__(
        self,
        fs=48000,
        nfft=4096,
        hop=2048,
        n_bands=8,
        device="cuda",
    ):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)

        band_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        self.band_ranges = []
        bin_hz = fs / nfft
        for fc in band_centers[:n_bands]:
            f_lo = fc / 2**0.5
            f_hi = fc * 2**0.5
            if f_hi > fs / 2:
                break
            k_lo = max(1, int(f_lo / bin_hz))
            k_hi = min(nfft // 2, int(f_hi / bin_hz))
            if k_hi > k_lo:
                self.band_ranges.append((k_lo, k_hi))
        # Persistent per-target cache.
        self._tgt_id: int | None = None
        self._tgt_band_edc: list[torch.Tensor] | None = None

    def _target_features(self, target):
        if id(target) == self._tgt_id and self._tgt_band_edc is not None:
            return self._tgt_band_edc
        with torch.no_grad():
            spec_tgt = _fc.stft_cached(target, self.nfft, self.hop, self.window,
                                       is_target=True)
            edc_list = []
            for k_lo, k_hi in self.band_ranges:
                band_tgt = torch.sum(torch.abs(spec_tgt[:, k_lo:k_hi, :]) ** 2, dim=1)
                edc_tgt = torch.flip(torch.cumsum(torch.flip(band_tgt, [-1]), dim=-1), [-1])
                edc_tgt = edc_tgt / (edc_tgt[:, :1] + 1e-10)
                edc_list.append(edc_tgt)
        self._tgt_id = id(target)
        self._tgt_band_edc = edc_list
        return edc_list

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        spec_est = _fc.stft_cached(estimation, self.nfft, self.hop, self.window,
                                   is_target=False)
        edc_tgt_list = self._target_features(target)

        loss = 0.0
        for (k_lo, k_hi), edc_tgt in zip(self.band_ranges, edc_tgt_list):
            band_est = torch.sum(torch.abs(spec_est[:, k_lo:k_hi, :]) ** 2, dim=1)
            edc_est = torch.flip(torch.cumsum(torch.flip(band_est, [-1]), dim=-1), [-1])
            edc_est = edc_est / (edc_est[:, :1] + 1e-10)
            loss += torch.mean((edc_est - edc_tgt) ** 2)

        return loss / len(self.band_ranges)


class T30Loss(_BandEDCBase):
    def __init__(self, fs=48000, device="cuda"):
        super().__init__(
            fs=fs,
            device=device,
            db_lo=-5,
            db_hi=-35,
            emphasis_bands=[125, 250, 500],
            emphasis_factor=3.0,
        )


class EDTLoss(_BandEDCBase):
    def __init__(self, fs=48000, device="cuda"):
        super().__init__(
            fs=fs,
            device=device,
            db_lo=0,
            db_hi=-10,
            emphasis_bands=[63, 125, 250],
            emphasis_factor=3.0,
        )


class SpectralEDCLoss(nn.Module):
    """Per-frequency-bin EDC loss.

    Instead of grouping STFT bins into octave bands, this computes the
    Schroeder Energy Decay Curve at *every* STFT frequency bin and
    matches the decay shape between estimation and target.

    This gives the PEQ continuous gradient signal across the full
    spectrum — every filter band receives direct feedback from the
    frequency bins it influences, regardless of its center frequency.

    ``db_lo`` / ``db_hi`` select the dB range of the target EDC to
    match (same sigmoid-mask approach as T30Loss / EDTLoss).
    Use ``db_lo=-5, db_hi=-35`` for T30-like late-decay matching and
    ``db_lo=0, db_hi=-10`` for EDT-like early-decay matching.
    """

    def __init__(
        self,
        fs=48000,
        nfft=4096,
        hop=2048,
        db_lo=-5,
        db_hi=-35,
        f_lo=20.0,
        f_hi=20000.0,
        device="cuda",
        use_freq_weighting=True,
    ):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.db_lo = db_lo
        self.db_hi = db_hi
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)

        # Restrict to bins within [f_lo, f_hi] to avoid noise-floor bins.
        bin_hz = fs / nfft
        self.k_lo = max(1, int(f_lo / bin_hz))
        self.k_hi = min(nfft // 2, int(f_hi / bin_hz))

        # Optional 1/f weighting so each octave contributes equally to the loss.
        bin_freqs = torch.arange(self.k_lo, self.k_hi, device=device, dtype=torch.float32) * bin_hz
        if use_freq_weighting:
            freq_weights = 1.0 / bin_freqs.clamp(min=1.0)
        else:
            freq_weights = torch.ones_like(bin_freqs)
        freq_weights = freq_weights / freq_weights.sum()
        self.register_buffer("freq_weights", freq_weights)
        # Per-target-tensor cache for the heavy work (STFT, EDC, mask).
        self._tgt_id: int | None = None
        self._tgt_edc_db: torch.Tensor | None = None
        self._tgt_mask: torch.Tensor | None = None

    def _target_features(self, target):
        if id(target) == self._tgt_id and self._tgt_edc_db is not None:
            return self._tgt_edc_db, self._tgt_mask
        with torch.no_grad():
            spec_tgt = _fc.stft_cached(target, self.nfft, self.hop, self.window,
                                       is_target=True)
            power_tgt = torch.abs(spec_tgt[:, self.k_lo : self.k_hi, :]) ** 2
            edc_tgt = torch.flip(torch.cumsum(torch.flip(power_tgt, [-1]), dim=-1), [-1])
            eps = 1e-10
            edc_tgt_db = 10 * torch.log10(edc_tgt / (edc_tgt[:, :, :1] + eps) + eps)
            mask = torch.sigmoid(5 * (edc_tgt_db - self.db_hi)) * torch.sigmoid(
                5 * (self.db_lo - edc_tgt_db)
            )
        self._tgt_id = id(target)
        self._tgt_edc_db = edc_tgt_db
        self._tgt_mask = mask
        return edc_tgt_db, mask

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        estimation = _peak_normalize(estimation)

        spec_est = _fc.stft_cached(estimation, self.nfft, self.hop, self.window,
                                   is_target=False)
        edc_tgt_db, mask = self._target_features(target)

        # Per-bin power in the selected frequency range: [B, K, T]
        power_est = torch.abs(spec_est[:, self.k_lo : self.k_hi, :]) ** 2
        edc_est = torch.flip(torch.cumsum(torch.flip(power_est, [-1]), dim=-1), [-1])

        eps = 1e-10
        edc_est_db = 10 * torch.log10(edc_est / (edc_est[:, :, :1] + eps) + eps)

        diff = (edc_est_db - edc_tgt_db) ** 2

        # --- Per-bin normalization then frequency weighting ---
        # Normalize each bin's loss by its own mask sum.  Without this,
        # bins with long decays (many masked frames) dominate the global
        # denominator and dilute the contribution of fast-decaying HF bins,
        # causing systematic HF over-attenuation for long-RT rooms.
        per_bin_loss = (diff * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1)  # [B, K]

        # Weight by 1/f so each octave contributes equally.
        w = self.freq_weights.unsqueeze(0)  # [1, K]
        loss = (per_bin_loss * w).sum(dim=-1).mean()  # scalar

        return loss / 900


class DifferentiableT30Loss(nn.Module):
    """Per-band differentiable T30 loss in seconds.

    Computes Schroeder EDC per octave band, fits a linear slope over the
    [-5, -35] dB range via differentiable masked linear regression, converts
    the slope to T30 = -60 / slope (seconds), and penalizes L1 deviation
    from the target T30.

    Unlike T30Loss / EDTLoss which match EDC *curves*, this directly
    penalizes the scalar reverberation time per band — giving an
    interpretable loss magnitude (seconds of T30 error).
    """

    def __init__(self, fs=48000, nfft=4096, hop=2048, n_bands=8,
                 device="cuda", t30_clamp=(0.05, 30.0)):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.t30_min, self.t30_max = t30_clamp
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)

        band_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        self.band_ranges = []
        bin_hz = fs / nfft
        for fc in band_centers[:n_bands]:
            f_lo = fc / 2**0.5
            f_hi = fc * 2**0.5
            if f_hi > fs / 2:
                break
            k_lo = max(1, int(f_lo / bin_hz))
            k_hi = min(nfft // 2, int(f_hi / bin_hz))
            if k_hi > k_lo:
                self.band_ranges.append((k_lo, k_hi))

        # Cached target T30 (computed on first forward call)
        self._target_t30 = None

    def _fit_t30(self, edc_db, time_sec, mask):
        """Differentiable weighted linear regression → T30 in seconds.

        Args:
            edc_db: [B, T] EDC in dB (normalized to 0 dB at t=0).
            time_sec: [T] time axis in seconds.
            mask: [B, T] sigmoid weights selecting the [-5, -35] dB range.

        Returns:
            t30: [B] T30 values in seconds, clamped to [t30_min, t30_max].
        """
        # Weighted mean of time
        t = time_sec.unsqueeze(0)  # [1, T]
        w_sum = mask.sum(dim=-1, keepdim=True) + 1e-10  # [B, 1]
        t_mean = (mask * t).sum(dim=-1, keepdim=True) / w_sum  # [B, 1]
        t_c = t - t_mean  # centered time [B, T]

        # Weighted linear regression: slope = Σ(w * t_c * y) / Σ(w * t_c²)
        numerator = (mask * t_c * edc_db).sum(dim=-1)  # [B]
        denominator = (mask * t_c ** 2).sum(dim=-1) + 1e-10  # [B]
        slope = numerator / denominator  # dB/s

        # T30 = -60 / slope; clamp to avoid explosion near slope≈0
        t30 = (-60.0 / slope).clamp(self.t30_min, self.t30_max)
        return t30

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        estimation = _peak_normalize(estimation)

        spec_est = torch.stft(
            estimation, n_fft=self.nfft, hop_length=self.hop,
            window=self.window, return_complex=True,
        )
        spec_tgt = torch.stft(
            target, n_fft=self.nfft, hop_length=self.hop,
            window=self.window, return_complex=True,
        )

        # Time axis in seconds for the STFT frames
        n_frames = spec_est.shape[-1]
        time_sec = torch.arange(n_frames, device=estimation.device,
                                dtype=estimation.dtype) * (self.hop / self.fs)

        eps = 1e-10
        loss = 0.0
        n_valid = 0

        for k_lo, k_hi in self.band_ranges:
            band_est = torch.sum(torch.abs(spec_est[:, k_lo:k_hi, :]) ** 2, dim=1)
            band_tgt = torch.sum(torch.abs(spec_tgt[:, k_lo:k_hi, :]) ** 2, dim=1)

            # Schroeder backward integration
            edc_est = torch.flip(torch.cumsum(torch.flip(band_est, [-1]), dim=-1), [-1])
            edc_tgt = torch.flip(torch.cumsum(torch.flip(band_tgt, [-1]), dim=-1), [-1])

            # Normalize to 0 dB at onset, convert to dB
            edc_est_db = 10 * torch.log10(edc_est / (edc_est[:, :1] + eps) + eps)
            edc_tgt_db = 10 * torch.log10(edc_tgt / (edc_tgt[:, :1] + eps) + eps)

            # Sigmoid mask selecting the [-5, -35] dB range of the TARGET EDC
            mask = (torch.sigmoid(5 * (edc_tgt_db - (-35.0)))
                    * torch.sigmoid(5 * ((-5.0) - edc_tgt_db)))

            if mask.sum() < 2:
                continue

            # Fit T30 for estimation (gradient flows) and target (no gradient)
            t30_est = self._fit_t30(edc_est_db, time_sec, mask)
            with torch.no_grad():
                t30_tgt = self._fit_t30(edc_tgt_db, time_sec, mask)

            loss += torch.mean(torch.abs(t30_est - t30_tgt))
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=estimation.device,
                                requires_grad=True)
        return loss / n_valid


class EarlyEnergyLoss(nn.Module):
    def __init__(self, fs=48000, t_boundaries_ms=[5, 50, 80]):
        super().__init__()
        self.boundaries = [int(t * fs / 1000) for t in t_boundaries_ms]

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        estimation = _peak_normalize(estimation)

        loss = 0.0
        eps = 1e-10
        for boundary in self.boundaries:
            e_early_est = torch.sum(estimation[:, :boundary] ** 2, dim=-1)
            e_early_tgt = torch.sum(target[:, :boundary] ** 2, dim=-1)
            e_late_est = torch.sum(estimation[:, boundary:] ** 2, dim=-1)
            e_late_tgt = torch.sum(target[:, boundary:] ** 2, dim=-1)

            ratio_est = 10 * torch.log10(e_early_est / (e_late_est + eps) + eps)
            ratio_tgt = 10 * torch.log10(e_early_tgt / (e_late_tgt + eps) + eps)

            loss += torch.mean((ratio_est - ratio_tgt) ** 2)

        return loss / len(self.boundaries)


class MultiResoSTFT(nn.Module):
    """
    Multi-Resolution STFT loss.
    By default this computes plain (unweighted) log-magnitude MSE across
    time-frequency bins. Optional time/frequency weighting can be enabled.
    """

    def __init__(
        self,
        fft_sizes=[256, 1024, 4096],
        win_lengths=[256, 1024, 4096],
        hop_sizes=[64, 256, 1024],
        device="cpu",
        use_time_weighting=False,
        use_freq_weighting=False,
        time_decay_factor=2.0,
        time_boost=9.0,
        freq_exponent=0.3,
        level_invariant=False,
        skip_samples=0,
        truncate_samples=None,
        target_mask_db=None,
        normalize=True,
    ):
        super().__init__()

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = 1e-3
        self.use_time_weighting = use_time_weighting
        self.use_freq_weighting = use_freq_weighting
        self.time_decay_factor = time_decay_factor
        self.time_boost = time_boost
        self.freq_exponent = freq_exponent

        self.level_invariant = level_invariant
        self.skip_samples = skip_samples
        self.truncate_samples = truncate_samples
        self.target_mask_db = target_mask_db
        self.normalize = normalize

        self.windows = nn.ParameterList()
        for win_len in win_lengths:
            window = torch.hann_window(win_len, device=device)
            self.windows.append(nn.Parameter(window, requires_grad=False))

    def forward(self, rir1, rir2):
        if rir1.dim() == 3 and rir1.shape[-1] == 1:
            rir1 = rir1.squeeze(-1)
        if rir2.dim() == 3 and rir2.shape[-1] == 1:
            rir2 = rir2.squeeze(-1)

        # Match lengths: truncate longer to shorter so STFT frames are identical.
        min_len = min(rir1.shape[-1], rir2.shape[-1])
        rir1 = rir1[..., :min_len]
        rir2 = rir2[..., :min_len]

        if self.skip_samples > 0:
            rir1 = rir1[:, self.skip_samples:]
            rir2 = rir2[:, self.skip_samples:]

        if self.truncate_samples is not None:
            rir1 = rir1[:, :self.truncate_samples]
            rir2 = rir2[:, :self.truncate_samples]

        if self.normalize:
            if self.level_invariant:
                # Normalize both to unit RMS so the loss compares spectral shape,
                # not absolute level.  Noise-shaped reverb (colored noise) has much
                # lower crest factor than an impulsive RIR, creating a systematic
                # ~10 dB RMS offset even when both peak at 1.0.
                rms1 = torch.sqrt(torch.mean(rir1**2, dim=-1, keepdim=True) + 1e-10)
                rms2 = torch.sqrt(torch.mean(rir2**2, dim=-1, keepdim=True) + 1e-10)
                rir1 = rir1 / rms1
                rir2 = rir2 / rms2
            else:
                # Normalize both signals to their own peak so the loss compares
                # spectral shape and temporal decay, not absolute level.
                # Using a tiny eps floor instead of clamp(min=1.0) so signals
                # quieter than 1.0 are also brought to unit peak.
                peak1 = torch.amax(torch.abs(rir1), dim=-1, keepdim=True).clamp(min=1e-8)
                peak2 = torch.amax(torch.abs(rir2), dim=-1, keepdim=True).clamp(min=1e-8)
                rir1 = rir1 / peak1
                rir2 = rir2 / peak2

        total_loss = 0.0
        for i, (n_fft, hop, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = self.windows[i]
            if window.device != rir1.device:
                window = window.to(rir1.device)

            spec_x = torch.stft(
                rir1,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_len,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            spec_y = torch.stft(
                rir2,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_len,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )

            mag_x = torch.abs(spec_x) + self.eps
            mag_y = torch.abs(spec_y) + self.eps

            db_x = 20.0 * torch.log10(mag_x)
            db_y = 20.0 * torch.log10(mag_y)

            if self.use_time_weighting:
                time_indices = db_x.shape[-1]
                t_ramp = (
                    torch.arange(time_indices, device=rir1.device).float()
                    / max(time_indices, 1)
                )
                time_weight = (
                    1 + self.time_boost * torch.exp(-self.time_decay_factor * t_ramp)
                ).view(1, 1, -1)
            else:
                time_weight = 1.0

            if self.use_freq_weighting:
                freq_indices = db_x.shape[-2]
                f_ramp = torch.arange(1, freq_indices + 1, device=rir1.device).float()
                freq_weight = (1.0 / f_ramp.pow(self.freq_exponent)).view(1, -1, 1)
            else:
                freq_weight = 1.0

            diff = (db_y - db_x) ** 2
            
            if self.target_mask_db is not None:
                # Mask out time-frequency bins where the target (rir2) is below the threshold
                # Allows the network to "ring out" without being falsely penalized by the target's noise floor.
                mask = (db_y > self.target_mask_db).float()
                diff = diff * mask

            weighted_error = diff * freq_weight * time_weight
            
            # Use sum / non-masked elements if masked, else mean
            if self.target_mask_db is not None:
                active_bins = mask.sum().clamp(min=1.0)
                total_loss += (weighted_error.sum() / active_bins)
            else:
                total_loss += weighted_error.mean()

        return total_loss / len(self.fft_sizes)


class FilterbankEnvelopeLoss(nn.Module):
    """
    Per-octave-band energy envelope loss using dasp's exact filterbank.

    Uses the same octave bandpass filters as NoiseShapedReverb to measure per-band
    energy envelopes.  This eliminates STFT spectral leakage between adjacent octave
    bands — a key source of gradient confusion when optimizing per-band gains.

    Per-band power is estimated via short-time squared amplitude:
      e[t] = moving average of signal^2 (window = smooth_samples)

    Normalization: both signals are divided by the target's own RMS in the
    post-skip region.  This keeps the reference fixed (no gradient through norm).
    """

    def __init__(self, fs=48000, num_bandpass_taps=1023, skip_samples=0,
                 smooth_ms=200.0, time_decay=0.5, device="cuda"):
        super().__init__()
        import dasp_pytorch
        self.fs = fs
        self.skip_samples = skip_samples
        self.time_decay = time_decay
        # Smooth window in samples for short-time power estimate
        self.smooth_samples = max(1, int(smooth_ms * 1e-3 * fs))

        filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, fs)
        # filters: [12, 1, num_bandpass_taps]
        self.register_buffer("filters", filters)
        self.num_bands = filters.shape[0]

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        # estimation, target: [B, T]
        B, T = target.shape

        # Normalise by target's post-skip RMS (fixed reference, no grad through it)
        skip = self.skip_samples
        ref_rms = target[:, skip:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        est_n = estimation / ref_rms
        tgt_n = target / ref_rms

        # Apply filterbank: conv each band filter over each signal.
        # filters [num_bands, 1, K]; signals [B, 1, T].
        # Use groups=1 broadcasting via unfold approach: iterate over bands.
        # For gradient flow, use F.conv1d per band.
        import torch.nn.functional as F
        filters = self.filters.to(estimation.device)  # [num_bands, 1, K]
        K = filters.shape[-1]
        pad = K // 2

        eps = 1e-8
        smooth_k = self.smooth_samples | 1  # force odd for avg_pool1d padding
        loss = 0.0

        for b_idx in range(self.num_bands):
            filt = filters[b_idx:b_idx+1]  # [1, 1, K]
            # Apply to all batch elements: broadcast by repeating filter
            filt_b = filt.expand(B, 1, -1)  # [B, 1, K]
            # F.conv1d with groups=B: each example gets its own filter (which are all the same)
            # Alternative: reshape for batched conv
            est_band = F.conv1d(est_n.unsqueeze(1), filt, padding=pad).squeeze(1)  # [B, T]
            tgt_band = F.conv1d(tgt_n.unsqueeze(1), filt, padding=pad).squeeze(1)

            # Short-time power via avg_pool1d on squared signal
            e_est = F.avg_pool1d(
                est_band.pow(2).unsqueeze(1), smooth_k, stride=1,
                padding=smooth_k // 2, count_include_pad=False,
            ).squeeze(1)  # [B, T]
            e_tgt = F.avg_pool1d(
                tgt_band.pow(2).unsqueeze(1), smooth_k, stride=1,
                padding=smooth_k // 2, count_include_pad=False,
            ).squeeze(1)

            # Skip the early-reflection region
            if skip > 0:
                e_est = e_est[:, skip:]
                e_tgt = e_tgt[:, skip:]

            db_est = 10 * torch.log10(e_est + eps)
            db_tgt = 10 * torch.log10(e_tgt + eps)

            # Time weighting: mild exponential down-weights the noisy tail while
            # keeping late reverb frames in the gradient.
            n_t = db_tgt.shape[-1]
            t = torch.linspace(0.0, 1.0, n_t, device=db_tgt.device)
            w = torch.exp(-self.time_decay * t)
            w = w / w.sum()

            band_loss = torch.sum((db_est - db_tgt) ** 2 * w, dim=-1)
            loss += band_loss.mean()

        return loss / self.num_bands


class FilterbankSlopeLoss(nn.Module):
    """
    Per-band decay slope loss using dasp's exact filterbank (same as FilterbankEnvelopeLoss).
    Compares decay slope (dB/s) over the first half of the post-skip signal,
    independent of absolute level — pure gradient signal for decay parameters.
    """

    def __init__(self, fs=48000, num_bandpass_taps=1023, skip_samples=0,
                 smooth_ms=200.0, device="cuda"):
        super().__init__()
        import dasp_pytorch
        self.fs = fs
        self.skip_samples = skip_samples
        self.smooth_samples = max(1, int(smooth_ms * 1e-3 * fs))

        filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, fs)
        self.register_buffer("filters", filters)
        self.num_bands = filters.shape[0]

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        B, T = target.shape

        skip = self.skip_samples
        ref_rms = target[:, skip:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        est_n = estimation / ref_rms
        tgt_n = target / ref_rms

        import torch.nn.functional as F
        filters = self.filters.to(estimation.device)
        K = filters.shape[-1]
        pad = K // 2
        eps = 1e-8
        smooth_k = self.smooth_samples | 1
        loss = 0.0

        for b_idx in range(self.num_bands):
            filt = filters[b_idx:b_idx+1]
            est_band = F.conv1d(est_n.unsqueeze(1), filt, padding=pad).squeeze(1)
            tgt_band = F.conv1d(tgt_n.unsqueeze(1), filt, padding=pad).squeeze(1)

            e_est = F.avg_pool1d(
                est_band.pow(2).unsqueeze(1), smooth_k, stride=1,
                padding=smooth_k // 2, count_include_pad=False,
            ).squeeze(1)
            e_tgt = F.avg_pool1d(
                tgt_band.pow(2).unsqueeze(1), smooth_k, stride=1,
                padding=smooth_k // 2, count_include_pad=False,
            ).squeeze(1)

            if skip > 0:
                e_est = e_est[:, skip:]
                e_tgt = e_tgt[:, skip:]

            db_est = 10 * torch.log10(e_est + eps)
            db_tgt = 10 * torch.log10(e_tgt + eps)

            # Linear regression slope over first half (avoids noise floor divergence).
            # Scale by T_half to get total dB drop over T_half (rather than dB/sample).
            # dB/sample values are ~1e-3 and their squared differences are ~1e-6,
            # which underflows to 0.0 in float32.  Scaling by T_half gives differences
            # in dB (range 5-50 dB) with squared values 25-2500 — well in float32 range.
            T_half = max(4, db_est.shape[-1] // 2)
            t = torch.arange(T_half, dtype=torch.float32, device=db_est.device)
            t_c = t - t.mean()
            t_c2 = (t_c * t_c).sum()

            # total_dB_drop = slope_dB_per_sample * T_half
            slope_est = T_half * (db_est[:, :T_half] * t_c).sum(dim=-1) / t_c2
            slope_tgt = T_half * (db_tgt[:, :T_half] * t_c).sum(dim=-1) / t_c2
            loss += torch.mean((slope_est - slope_tgt) ** 2)

        return loss / self.num_bands


class BandEnergyEnvelopeLoss(nn.Module):
    """
    Per-octave-band energy envelope loss for noise-shaping models.
    Compares temporal energy decay per band without RMS-normalizing model output,
    giving nonzero gradients for both gain (level shift) and decay (slope).
    """

    BAND_CENTERS = [12, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 18000]

    def __init__(self, fs=48000, nfft=4096, hop=512, skip_frames=0,
                 smooth_frames=21, time_decay=0.5, device="cuda"):
        super().__init__()
        self.fs = fs
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)
        self.nfft = nfft
        self.hop = hop
        self.skip_frames = skip_frames
        # smooth_frames: moving-average window applied to per-band power before
        # taking log.  Averages out the ±10-20 dB stochastic fluctuations of
        # the noise model so the loss sees the mean energy envelope, not noise.
        # ~20 frames ≈ 200 ms at hop=512/fs=48k — long enough to suppress noise,
        # short enough to resolve the decay slope.
        self.smooth_frames = smooth_frames
        # time_decay=0.5: mild exponential downweighting of late frames.
        # Old value 3.0 gave exp(-3)=0.05 weight at the end, essentially ignoring
        # the late reverb tail where T60 information lives.  With 0.5, late frames
        # get exp(-0.5)=0.6 weight — enough gradient to drive decay parameters.
        self.time_decay = time_decay
        bin_hz = fs / nfft
        self.band_ranges = []
        for fc in self.BAND_CENTERS:
            k_lo = max(1, int((fc / 2**0.5) / bin_hz))
            k_hi = min(nfft // 2, int((fc * 2**0.5) / bin_hz))
            if k_hi > k_lo:
                self.band_ranges.append((k_lo, k_hi))

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # Normalise both by the target's RMS in the signal region (after skip).
        # Using the target RMS as the reference is a fixed constant w.r.t. the
        # model → gain gradients survive.  Peak normalization is wrong here:
        # the model peak is a random noise spike, not a physically meaningful
        # reference, so it would create an unstable, misleading level comparison.
        skip = self.skip_frames
        ref_rms = target[:, skip * self.hop:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        estimation = estimation / ref_rms
        target = target / ref_rms

        spec_est = torch.stft(
            estimation,
            n_fft=self.nfft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )
        spec_tgt = torch.stft(
            target,
            n_fft=self.nfft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )

        # Drop the initial double-slope region.
        if skip > 0:
            spec_est = spec_est[:, :, skip:]
            spec_tgt = spec_tgt[:, :, skip:]

        # Exponential time weight: down-weights noisy tail.
        n_frames = spec_tgt.shape[-1]
        t = torch.linspace(0.0, 1.0, n_frames, device=spec_tgt.device)
        w = torch.exp(-self.time_decay * t)
        w = w / w.sum()

        eps = 1e-8
        loss = 0.0
        for k_lo, k_hi in self.band_ranges:
            n_bins = k_hi - k_lo
            # Divide by bin count to get mean per-bin power.
            # dasp's filterbank has filter_energy ∝ bandwidth ∝ n_bins, so
            # per-bin power is approximately proportional to gain^2 alone,
            # independent of band centre frequency.  Without this division, the
            # summed energy grows with n_bins (doubles per octave), creating a
            # systematic 14 dB bias between low and high bands.
            e_est = torch.sum(torch.abs(spec_est[:, k_lo:k_hi, :]) ** 2, dim=1) / n_bins  # [B, T]
            e_tgt = torch.sum(torch.abs(spec_tgt[:, k_lo:k_hi, :]) ** 2, dim=1) / n_bins

            # Smooth power over time before taking log: removes ±10-20 dB
            # stochastic frame-to-frame fluctuations of the noise model.
            # The model can only control the mean energy envelope, not individual
            # noise realisations — smoothing makes the loss match what the model
            # can actually achieve.
            if self.smooth_frames > 1:
                k = self.smooth_frames | 1  # force odd so output length == input length
                e_est = torch.nn.functional.avg_pool1d(
                    e_est.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False
                ).squeeze(1)
                e_tgt = torch.nn.functional.avg_pool1d(
                    e_tgt.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False
                ).squeeze(1)

            db_est = 10 * torch.log10(e_est + eps)
            db_tgt = 10 * torch.log10(e_tgt + eps)
            band_loss = torch.sum((db_est - db_tgt) ** 2 * w, dim=-1)
            loss += band_loss.mean()

        return loss / len(self.band_ranges)


class BandSlopeLoss(nn.Module):
    """
    Per-band decay SLOPE loss for noise-shaping models.

    Compares only the dB-per-frame slope of each band's energy envelope,
    decoupled from absolute level.  This gives a pure gradient to the decay
    parameters with no gain confounding — prevents the optimizer from
    stretching T60 to compensate for a wrong gain level.

    Slope is estimated via linear regression over the first half of the
    post-skip signal.  Using only the first half avoids the noise floor issue:
    - The target IR eventually hits the recording noise floor (not -inf)
    - The model hits the mathematical eps floor (-80 dB)
    - Comparing last-quarter means across these different floors creates
      enormous spurious slope errors (thousands of dB^2)
    Linear regression over the initial decay captures the actual T60 slope
    where both signals are well above their respective noise floors.
    """

    BAND_CENTERS = [12, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 18000]

    def __init__(self, fs=48000, nfft=4096, hop=512, skip_frames=0,
                 smooth_frames=21, device="cuda"):
        super().__init__()
        window = torch.hann_window(nfft, device=device)
        self.register_buffer("window", window)
        self.nfft = nfft
        self.hop = hop
        self.skip_frames = skip_frames
        self.smooth_frames = smooth_frames
        bin_hz = fs / nfft
        self.band_ranges = []
        for fc in self.BAND_CENTERS:
            k_lo = max(1, int((fc / 2**0.5) / bin_hz))
            k_hi = min(nfft // 2, int((fc * 2**0.5) / bin_hz))
            if k_hi > k_lo:
                self.band_ranges.append((k_lo, k_hi))

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # Same normalisation as BandEnergyEnvelopeLoss
        skip = self.skip_frames
        ref_rms = target[:, skip * self.hop:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        estimation = estimation / ref_rms
        target = target / ref_rms

        spec_est = torch.stft(estimation, n_fft=self.nfft, hop_length=self.hop,
                              window=self.window, return_complex=True)
        spec_tgt = torch.stft(target, n_fft=self.nfft, hop_length=self.hop,
                              window=self.window, return_complex=True)

        if skip > 0:
            spec_est = spec_est[:, :, skip:]
            spec_tgt = spec_tgt[:, :, skip:]

        eps = 1e-8
        loss = 0.0
        T_full = spec_est.shape[-1]
        # Use only first half: avoids noise-floor divergence between model (hits
        # eps≈-80 dB) and target (hits recording noise floor at ~+18 dB).
        T_half = max(4, T_full // 2)

        # Precompute linear regression basis (centred frame indices).
        # Slope estimator: b = sum(t_c * db) / sum(t_c^2), where t_c = t - mean(t).
        t = torch.arange(T_half, dtype=torch.float32, device=spec_est.device)
        t_c = t - t.mean()
        t_c2_sum = (t_c * t_c).sum()  # scalar denominator

        for k_lo, k_hi in self.band_ranges:
            n_bins = k_hi - k_lo
            # Per-bin power (consistent with BandEnergyEnvelopeLoss)
            e_est = torch.sum(torch.abs(spec_est[:, k_lo:k_hi, :T_half]) ** 2, dim=1) / n_bins
            e_tgt = torch.sum(torch.abs(spec_tgt[:, k_lo:k_hi, :T_half]) ** 2, dim=1) / n_bins

            if self.smooth_frames > 1:
                k = self.smooth_frames | 1
                e_est = torch.nn.functional.avg_pool1d(
                    e_est.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False
                ).squeeze(1)
                e_tgt = torch.nn.functional.avg_pool1d(
                    e_tgt.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False
                ).squeeze(1)

            db_est = 10 * torch.log10(e_est + eps)  # [B, T_half]
            db_tgt = 10 * torch.log10(e_tgt + eps)

            # Linear regression slope (dB/frame) — level-invariant
            slope_est = (db_est * t_c).sum(dim=-1) / t_c2_sum  # [B]
            slope_tgt = (db_tgt * t_c).sum(dim=-1) / t_c2_sum
            loss += torch.mean((slope_est - slope_tgt) ** 2)

        return loss / len(self.band_ranges)


class FilterbankEnergyAndSlopeLoss(nn.Module):
    """
    Combined per-band energy + Schroeder EDC slope loss in a single forward pass.

    Applies dasp's octave filterbank once and returns:
        energy_loss + slope_weight × slope_loss

    energy_loss  — 10·log10(mean(filtered²)) MSE per band, no normalisation.
                   One scalar per band → unambiguous gradient for gain params.
                   Mirrors the per-band energy bar chart in the visualiser.

    slope_loss   — Schroeder EDC slope MSE per band, level-invariant.
                   Mirrors the T60 estimate shown per band in the visualiser.
                   Gradient signal for decay params.

    Key design choices:
    • Single F.conv1d call for all 12 bands (filters [12,1,K] acts as a
      1→12 channel conv on the [B,1,T] input) — replaces 24 serial calls.
    • Target path is wrapped in torch.no_grad() — it never needs a gradient.
    • Per-band energy downsampled by ds_factor before Schroeder integration,
      reducing cumsum length from ~92 k to ~1800 frames and preventing:
        a) an enormous backward graph from torch.cumsum over the full sequence
        b) gradient explosion near the noise floor (EDC → ε → large dlog10/dx)
    • EDC clamped to −60 dB before regression: removes residual noise-floor
      gradients from bands that have already decayed below that threshold.
    """

    def __init__(self, fs=48000, num_bandpass_taps=1023, skip_samples=0,
                 ds_factor=50, slope_weight=2.0):
        super().__init__()
        import dasp_pytorch
        self.skip         = skip_samples
        self.ds_factor    = ds_factor
        self.slope_weight = slope_weight
        filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, fs)
        self.register_buffer("filters", filters)   # [12, 1, K]
        self.num_bands = filters.shape[0]

    def forward(self, estimation, target):
        import torch.nn.functional as F
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        # estimation, target: [B, T]
        B, T = estimation.shape[0], estimation.shape[-1]

        # Normalize by target's post-skip RMS (fixed reference)
        ref_rms = target[:, self.skip:].pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        estimation = estimation / ref_rms
        target = target / ref_rms

        K  = self.filters.shape[-1]
        DS = self.ds_factor

        # FFT-based convolution: O(T log T) vs O(T×K) for direct conv.
        # For T=96000, K=4093 this is ~180× fewer MACs.
        fft_len  = 1 << (T + K - 2).bit_length()   # next power-of-2 ≥ T+K-1
        start    = K // 2                            # 'same' trim offset
        # Filter FFTs: [12, fft_len//2+1] — recomputed if signal length changes.
        filt_fft = torch.fft.rfft(self.filters.squeeze(1).to(estimation.device), n=fft_len, dim=-1)  # [12, F]

        def _fft_apply(x):
            """Apply all 12 band filters to x [B,T] → [B,12,T] via FFT."""
            x_fft  = torch.fft.rfft(x, n=fft_len, dim=-1)              # [B, F]
            out    = torch.fft.irfft(
                x_fft.unsqueeze(1) * filt_fft.unsqueeze(0),            # [B,12,F]
                n=fft_len, dim=-1
            )                                                            # [B,12,fft_len]
            return out[:, :, start:start + T]                           # [B,12,T]

        est_all = _fft_apply(estimation)
        with torch.no_grad():
            tgt_all = _fft_apply(target)

        if self.skip > 0:
            est_all = est_all[:, :, self.skip:]
            tgt_all = tgt_all[:, :, self.skip:]

        # ── Energy loss ──────────────────────────────────────────────────────
        eps_e   = 1e-8
        db_est  = 10 * torch.log10(est_all.pow(2).mean(dim=-1) + eps_e)  # [B,12]
        with torch.no_grad():
            db_tgt = 10 * torch.log10(tgt_all.pow(2).mean(dim=-1) + eps_e)
            
        # Mean-center to only penalize spectral shape, not absolute broadband level
        db_est_c = db_est - db_est.mean(dim=-1, keepdim=True)
        db_tgt_c = db_tgt - db_tgt.mean(dim=-1, keepdim=True)
        energy_loss = ((db_est_c - db_tgt_c) ** 2).mean()

        # ── Schroeder EDC slope loss ─────────────────────────────────────────
        eps_s = 1e-10
        BN    = B * self.num_bands

        # Downsample per-band energy: avg_pool over DS samples → [B,12,T_ds]
        # The 1/DS normalisation cancels in the EDC ratio below.
        e_est = F.avg_pool1d(
            est_all.pow(2).reshape(BN, 1, -1), DS, stride=DS
        ).reshape(B, self.num_bands, -1)
        with torch.no_grad():
            e_tgt = F.avg_pool1d(
                tgt_all.pow(2).reshape(BN, 1, -1), DS, stride=DS
            ).reshape(B, self.num_bands, -1)

        # Schroeder backward integration on the short downsampled sequence
        edc_est = torch.flip(torch.cumsum(torch.flip(e_est, [-1]), dim=-1), [-1])
        with torch.no_grad():
            edc_tgt = torch.flip(torch.cumsum(torch.flip(e_tgt, [-1]), dim=-1), [-1])

        # Normalise to 0 dB at onset; clamp at −60 dB to kill noise-floor gradients
        edc_est_db = (10 * torch.log10(
            edc_est / edc_est[:, :, :1].clamp(min=eps_s) + eps_s
        )).clamp(min=-60.0)
        with torch.no_grad():
            edc_tgt_db = (10 * torch.log10(
                edc_tgt / edc_tgt[:, :, :1].clamp(min=eps_s) + eps_s
            )).clamp(min=-60.0)

        # Linear regression slope over first half of downsampled EDC.
        # Multiplying by T_half converts units to total-dB-drop (not dB/frame)
        # so squared differences stay in float32-friendly range (25–3600 dB²).
        T_half = max(4, edc_est_db.shape[-1] // 2)
        t      = torch.arange(T_half, dtype=torch.float32, device=edc_est_db.device)
        t_c    = t - t.mean()                          # [T_half], centred indices
        t_c2   = (t_c * t_c).sum()

        slope_est = T_half * (edc_est_db[:, :, :T_half] * t_c).sum(dim=-1) / t_c2
        with torch.no_grad():
            slope_tgt = T_half * (edc_tgt_db[:, :, :T_half] * t_c).sum(dim=-1) / t_c2

        slope_loss = ((slope_est - slope_tgt) ** 2).mean()

        return energy_loss + self.slope_weight * slope_loss


class NoiseShapingParameterRegularizationLoss(nn.Module):
    """
    L2 regularization for normalized noise-shaping parameters.
    """

    def __init__(self, model, weight_gain=1e-2, weight_decay=1e-2):
        super().__init__()
        self.model = model
        self.weight_gain = weight_gain
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        gains, decays = self.model.get_normalized_band_params()
        return self.weight_gain * torch.norm(gains, p=2) + self.weight_decay * torch.norm(
            decays, p=2
        )


class EchoDensityLoss(nn.Module):
    """
    Echo Density profile MAE loss.
    Computes the Normalized Echo Density (NED) profile in a fully differentiable
    manner using local statistical moments (kurtosis-based approximation),
    and minimizes the L1 difference (MAE) between estimation and target.
    """

    def __init__(self, fs=48000, window_ms=10.0, hop_ms=2.0):
        super().__init__()
        self.fs = fs
        self.win_samples = max(1, int(window_ms * fs / 1000))
        self.hop_samples = max(1, int(hop_ms * fs / 1000))

    def compute_ned(self, x):
        import torch.nn.functional as F
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]

        # Local moments via average pooling
        m1 = F.avg_pool1d(x, self.win_samples, stride=self.hop_samples)
        m2 = F.avg_pool1d(x**2, self.win_samples, stride=self.hop_samples)
        m3 = F.avg_pool1d(x**3, self.win_samples, stride=self.hop_samples)
        m4 = F.avg_pool1d(x**4, self.win_samples, stride=self.hop_samples)

        # Central moments
        v = m2 - m1**2
        mu4 = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*(m1**4)

        # Kurtosis (Pearson's, Fisher=False)
        # Add epsilon to variance to prevent division by zero in silence
        kurtosis = mu4 / (v**2 + 1e-8)

        # Normalized Echo Density: 3.0 / kurtosis
        # Valid kurtosis for signals is typically >= 1.0 (for Bernoulli coin toss).
        # A Gaussian has a kurtosis of 3.0 (which gives NED = 1.0).
        ned = 3.0 / (kurtosis + 1e-8)
        ned = torch.clamp(ned, min=0.0, max=1.0)
        return ned.squeeze(1)

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # Match lengths
        min_len = min(estimation.shape[-1], target.shape[-1])
        estimation = estimation[..., :min_len]
        target = target[..., :min_len]

        ned_est = self.compute_ned(estimation)
        with torch.no_grad():
            ned_tgt = self.compute_ned(target)

        return torch.nn.functional.l1_loss(ned_est, ned_tgt)

class DRRLoss(nn.Module):
    """
    Direct-to-Reverberant Ratio (DRR) Loss.
    Penalizes differences in the DRR (in dB) between estimation and target.
    DRR = 10 * log10( Energy(0 to t_direct) / Energy(t_direct to end) ).
    """
    def __init__(self, fs=48000, t_direct_ms=2.5):
        super().__init__()
        self.n_direct = int(t_direct_ms * fs / 1000)

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
            
        estimation = _peak_normalize(estimation)

        eps = 1e-10
        e_dir_est = torch.sum(estimation[:, :self.n_direct] ** 2, dim=-1)
        e_rev_est = torch.sum(estimation[:, self.n_direct:] ** 2, dim=-1)
        drr_est = 10 * torch.log10((e_dir_est + eps) / (e_rev_est + eps))

        e_dir_tgt = torch.sum(target[:, :self.n_direct] ** 2, dim=-1)
        e_rev_tgt = torch.sum(target[:, self.n_direct:] ** 2, dim=-1)
        drr_tgt = 10 * torch.log10((e_dir_tgt + eps) / (e_rev_tgt + eps))

        return torch.mean((drr_est - drr_tgt) ** 2)


class GaussianSmoothedTimeLoss(nn.Module):
    """L1/MSE between Gaussian-smoothed pred and target IRs.

    Convolving both signals with a Gaussian kernel turns each discrete
    echo into a continuous bump, so the loss becomes differentiable in
    delay-position space. Gives FDN delay parameters a usable gradient.

    Implementation: FFT-based circular convolution. Picks up
    ``rfft_cached`` from the global feature cache so that the full-length
    rffts are shared with ``BandEnergyLoss`` / ``PowerSpectrumLoss``.
    Target output is cached forever; est output is recomputed each step.
    """
    def __init__(self, fs=48000, sigma_ms=3.0, base_loss="l1",
                 level_invariant=True, skip_ms=0.0):
        super().__init__()
        self.fs = int(fs)
        self.base_loss = base_loss
        self.level_invariant = bool(level_invariant)
        self.skip_samples = int(round(skip_ms * fs / 1000.0))
        sigma_samples = max(1.0, float(sigma_ms) * fs / 1000.0)
        self.sigma_samples = sigma_samples
        K = int(8 * sigma_samples) | 1  # odd, ≥ 8σ
        x = torch.arange(K, dtype=torch.float32) - K // 2
        g = torch.exp(-(x ** 2) / (2 * sigma_samples ** 2))
        g = g / g.sum()
        # Stored as 1D for FFT path; the time-domain conv path is unused now
        # but kept around so old checkpoints still load.
        self.register_buffer("kernel", g.view(1, 1, -1))
        self.pad = K // 2
        # Filled lazily on first forward when the signal length is known.
        self._kernel_freq: torch.Tensor | None = None
        self._kernel_freq_n: int = -1

    def _build_kernel_freq(self, n: int, device, dtype):
        g = self.kernel.view(-1).to(device=device, dtype=dtype)
        K = g.shape[0]
        # Place the kernel so its center sits at index 0 (wrap-around) — this
        # makes the circular convolution shift-free.
        kern = torch.zeros(n, device=device, dtype=dtype)
        kern[: (K // 2) + 1] = g[K // 2:]
        kern[n - (K // 2):] = g[: K // 2]
        self._kernel_freq = torch.fft.rfft(kern)
        self._kernel_freq_n = n

    def _smooth(self, x: torch.Tensor, is_target: bool) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        n = x.shape[-1]
        if self._kernel_freq is None or self._kernel_freq_n != n:
            self._build_kernel_freq(n, x.device, x.dtype)
        # Cache hits when BandEnergy/PowSpec also rfft this tensor.
        X = _fc.rfft_cached(x, is_target=is_target)
        if X.dtype != self._kernel_freq.dtype:
            # Keep multiplication in matching complex dtype.
            kf = self._kernel_freq.to(X.dtype)
        else:
            kf = self._kernel_freq
        y = torch.fft.irfft(X * kf, n=n)
        return y

    def forward(self, estimation, target):
        import torch.nn.functional as F
        est = self._smooth(estimation, is_target=False)
        # Target smoothing is fully cached: id(target) is stable, so
        # rfft_cached returns the same tensor and irfft is the only cost.
        # Wrap target path in no_grad — target requires no gradient.
        with torch.no_grad():
            tgt = self._smooth(target, is_target=True)
        if self.skip_samples > 0:
            est = est[..., self.skip_samples:]
            tgt = tgt[..., self.skip_samples:]
        if self.level_invariant:
            est = est / est.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            tgt = tgt / tgt.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.base_loss == "mse":
            return F.mse_loss(est, tgt)
        return F.l1_loss(est, tgt)


class EnergyEnvelopeLoss(nn.Module):
    """Compare windowed RMS envelopes. Penalizes both level and decay shape."""
    def __init__(self, window_ms=5.0, fs=48000):
        super().__init__()
        self.window = int(window_ms * fs / 1000)

    def forward(self, estimation, target):
        win = self.window
        n = min(estimation.shape[-1], target.shape[-1])
        est = estimation[..., :n]
        tgt = target[..., :n]
        n_trim = (n // win) * win
        est = est[..., :n_trim].reshape(*est.shape[:-1], -1, win)
        tgt = tgt[..., :n_trim].reshape(*tgt.shape[:-1], -1, win)
        est_rms = 10 * torch.log10((est ** 2).mean(dim=-1) + 1e-10)
        tgt_rms = 10 * torch.log10((tgt ** 2).mean(dim=-1) + 1e-10)
        return ((est_rms - tgt_rms) ** 2).mean()


class BandEnergyLoss(nn.Module):
    """Per-octave-band total energy matching in dB."""
    CENTERS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    def __init__(self, fs=48000, device="cuda"):
        super().__init__()
        self.fs = fs
        # Persistent per-target cache.
        self._tgt_id: int | None = None
        self._tgt_band_db: dict[float, torch.Tensor] | None = None
        self._cached_masks: dict[tuple, torch.Tensor] = {}

    def _target_features(self, target, freqs):
        if id(target) == self._tgt_id and self._tgt_band_db is not None:
            return self._tgt_band_db
        with torch.no_grad():
            spec_tgt = _fc.rfft_cached(target, is_target=True)
            band_db: dict[float, torch.Tensor] = {}
            for fc in self.CENTERS:
                lo, hi = fc / 2**0.5, fc * 2**0.5
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() < 1:
                    continue
                band_db[fc] = torch.log10((spec_tgt[:, mask].abs() ** 2).sum(dim=-1) + 1e-10)
        self._tgt_id = id(target)
        self._tgt_band_db = band_db
        return band_db

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        n = estimation.shape[-1]
        spec_est = _fc.rfft_cached(estimation, is_target=False)
        freqs = torch.fft.rfftfreq(n, 1.0 / self.fs).to(estimation.device)
        tgt_band_db = self._target_features(target, freqs)
        loss = torch.tensor(0.0, device=estimation.device)
        n_bands = 0
        self._band_losses = {}
        for fc in self.CENTERS:
            if fc not in tgt_band_db:
                continue
            lo, hi = fc / 2**0.5, fc * 2**0.5
            mask = (freqs >= lo) & (freqs < hi)
            e_est = torch.log10((spec_est[:, mask].abs() ** 2).sum(dim=-1) + 1e-10)
            e_tgt = tgt_band_db[fc]
            band_loss = ((e_est - e_tgt) ** 2).mean()
            self._band_losses[fc] = band_loss.item()
            loss = loss + band_loss
            n_bands += 1
        return loss / max(n_bands, 1)


class PowerSpectrumLoss(nn.Module):
    """Smoothed power spectrum L2 in dB (1/f weighted)."""
    def __init__(self, fs=48000, smooth_bins=201, device="cuda", use_freq_weighting=True):
        super().__init__()
        self.fs = fs
        self.smooth_bins = smooth_bins
        self.use_freq_weighting = use_freq_weighting
        # Persistent per-target cache (smoothed dB spectrum + freq weights).
        self._tgt_id: int | None = None
        self._tgt_db: torch.Tensor | None = None
        self._weights: torch.Tensor | float | None = None

    def _target_features(self, target, n: int, device):
        if id(target) == self._tgt_id and self._tgt_db is not None:
            return self._tgt_db, self._weights
        with torch.no_grad():
            spec_tgt_rfft = _fc.rfft_cached(target, is_target=True)
            spec_tgt = torch.abs(spec_tgt_rfft) ** 2
            k = self.smooth_bins
            spec_tgt_s = torch.nn.functional.avg_pool1d(
                spec_tgt.unsqueeze(1), k, stride=1, padding=k // 2).squeeze(1)
            tgt_db = 10 * torch.log10(spec_tgt_s + 1e-10)
            if self.use_freq_weighting:
                freqs = torch.fft.rfftfreq(n, 1.0 / self.fs).to(device)
                w = 1.0 / freqs.clamp(min=20.0)
                w = w / w.sum()
            else:
                w = 1.0 / spec_tgt_s.shape[-1]
        self._tgt_id = id(target)
        self._tgt_db = tgt_db
        self._weights = w
        return tgt_db, w

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        n = estimation.shape[-1]
        spec_est_rfft = _fc.rfft_cached(estimation, is_target=False)
        spec_est = torch.abs(spec_est_rfft) ** 2
        k = self.smooth_bins
        spec_est_s = torch.nn.functional.avg_pool1d(
            spec_est.unsqueeze(1), k, stride=1, padding=k // 2).squeeze(1)
        est_db = 10 * torch.log10(spec_est_s + 1e-10)
        tgt_db, w = self._target_features(target, n, estimation.device)
        return (w * (est_db - tgt_db) ** 2).sum(dim=-1).mean()


def _savgol_kernel(window: int, poly: int = 3) -> torch.Tensor:
    """1-D Savitzky-Golay smoothing kernel for the 0-th derivative.

    Equivalent to scipy.signal.savgol_coeffs(window, poly, deriv=0). Computed
    analytically: kernel[i] = e_0^T (A^T A)^{-1} A^T e_i where A is the
    Vandermonde-style design matrix on x = -(M..M).
    """
    if window % 2 == 0:
        window += 1
    M = window // 2
    x = torch.arange(-M, M + 1, dtype=torch.float64)
    A = torch.stack([x ** k for k in range(poly + 1)], dim=1)  # [window, poly+1]
    AtA_inv = torch.linalg.inv(A.T @ A)
    coeffs = (AtA_inv @ A.T)[0]  # row 0 of pseudo-inverse
    return coeffs.to(torch.float32)


class SmoothSTFTLoss(nn.Module):
    """Multi-resolution log-magnitude STFT distance with Savitzky-Golay smoothing
    along the frequency axis.

    The intuition: PEQ attenuation produces smooth magnitude responses; the
    target RIR has fine spectral structure (modes). Smoothing both before
    comparison removes the un-matchable detail and gives the optimizer a
    cleaner gradient for the broad spectral envelope. Mirrors what
    GaussianSmoothedTimeLoss does in the time domain.

    Computed as L1 on dB-magnitude, peak-normalised per-signal.
    """

    def __init__(
        self,
        fft_sizes=(512, 2048, 8192),
        win_lengths=(512, 2048, 8192),
        hop_sizes=(128, 512, 2048),
        savgol_window=11,
        savgol_poly=3,
        skip_samples=0,
        device="cpu",
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.win_lengths = list(win_lengths)
        self.hop_sizes = list(hop_sizes)
        self.skip_samples = int(skip_samples)
        self.eps = 1e-8

        self.windows = nn.ParameterList()
        for win_len in self.win_lengths:
            w = torch.hann_window(win_len, device=device)
            self.windows.append(nn.Parameter(w, requires_grad=False))

        kern = _savgol_kernel(int(savgol_window), int(savgol_poly)).to(device)
        # Stored as [1, 1, K] for conv1d along the freq axis.
        self.register_buffer("savgol_kernel", kern.view(1, 1, -1))
        self.savgol_pad = int(savgol_window) // 2

    def _smooth_freq(self, mag_db: torch.Tensor) -> torch.Tensor:
        # mag_db: [B, F, T] — smooth along F (dim=1).
        B, F_, T = mag_db.shape
        x = mag_db.permute(0, 2, 1).reshape(B * T, 1, F_)
        kern = self.savgol_kernel.to(x.dtype)
        y = torch.nn.functional.conv1d(x, kern, padding=self.savgol_pad)
        return y.reshape(B, T, F_).permute(0, 2, 1)

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        min_len = min(estimation.shape[-1], target.shape[-1])
        estimation = estimation[..., :min_len]
        target = target[..., :min_len]
        if self.skip_samples > 0:
            estimation = estimation[..., self.skip_samples:]
            target = target[..., self.skip_samples:]
        # Peak-normalise so we compare shape, not absolute level.
        est_peak = estimation.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)
        tgt_peak = target.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)
        estimation = estimation / est_peak
        target = target / tgt_peak

        total = 0.0
        for i, (n_fft, hop, win_len) in enumerate(zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = self.windows[i].to(estimation.device)
            spec_e = torch.stft(estimation, n_fft=n_fft, hop_length=hop,
                                 win_length=win_len, window=window,
                                 return_complex=True, center=True)
            with torch.no_grad():
                spec_t = torch.stft(target, n_fft=n_fft, hop_length=hop,
                                     win_length=win_len, window=window,
                                     return_complex=True, center=True)
            mag_e = 20 * torch.log10(spec_e.abs() + 1e-7)
            mag_t = 20 * torch.log10(spec_t.abs() + 1e-7)
            mag_e_s = self._smooth_freq(mag_e)
            mag_t_s = self._smooth_freq(mag_t)
            total = total + torch.nn.functional.l1_loss(mag_e_s, mag_t_s)
        return total / len(self.fft_sizes)


# ---------------------------------------------------------------------------
# Mezza et al. DAFx 2024 loss functions
# ---------------------------------------------------------------------------

class BroadbandEDCLoss(nn.Module):
    """Broadband time-domain Energy Decay Curve loss (Mezza et al. L_EDC).

    Computes the Schroeder EDC via backward integration of the squared IR,
    normalises both EDCs to 1.0 at n=0, and returns the mean squared error.
    This is the L_EDC term in eq (5) of Mezza DAFx 2024 — broadband, linear
    scale, no STFT or per-band decomposition.
    """

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        min_len = min(estimation.shape[-1], target.shape[-1])
        est = estimation[..., :min_len]
        tgt = target[..., :min_len]

        # Schroeder backward integration: ε[n] = Σ_{τ=n}^{L-1} x[τ]²
        edc_est = torch.flip(torch.cumsum(torch.flip(est ** 2, [-1]), -1), [-1])
        edc_tgt = torch.flip(torch.cumsum(torch.flip(tgt ** 2, [-1]), -1), [-1])

        # Normalise to ε[0] = 1
        edc_est = edc_est / (edc_est[..., :1] + 1e-10)
        edc_tgt = edc_tgt / (edc_tgt[..., :1] + 1e-10)

        return torch.mean((edc_est - edc_tgt) ** 2)


class MelEDRLoss(nn.Module):
    """Mel-scale Energy Decay Relief loss (Mezza et al. DAFx 2024, L_EDR).

    This is the key novel contribution of the paper. It computes a mel-scale
    EDR via backward integration of the mel-frequency spectrogram and
    minimises the L1 distance in dB between target and predicted EDRs.

    Paper-exact parameters (for fs=16 kHz):
        STFT:  512-bin, 320-sample Hann window (20 ms), 160-sample hop (10 ms)
        Mel:   64 triangular mel filters over [0, fs/2]
        EDR:   backward frame integration of H_mel[k, m]
        Loss:  L1 in dB
    """

    def __init__(self, fs: int = 16000, n_fft: int = 512, win_length: int = 320,
                 hop_length: int = 160, n_mels: int = 64, device: str = "cpu"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length, device=device)
        self.register_buffer("window", window)

        try:
            import torchaudio
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=float(fs) / 2.0,
                n_mels=n_mels,
                sample_rate=fs,
                norm=None,
                mel_scale="htk",
            )  # (n_freqs, n_mels)
        except ImportError:
            # Manual triangular mel filterbank fallback
            mel_fb = self._build_mel_fb(n_fft // 2 + 1, fs, n_mels, device)
        self.register_buffer("mel_fb", mel_fb.to(device))

        # Target cache
        self._tgt_id: int | None = None
        self._tgt_edr_db: torch.Tensor | None = None

    @staticmethod
    def _build_mel_fb(n_freqs: int, fs: int, n_mels: int, device) -> torch.Tensor:
        """Simple triangular mel filterbank, shape (n_freqs, n_mels)."""
        import numpy as np
        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        mel_lo, mel_hi = hz_to_mel(0.0), hz_to_mel(fs / 2.0)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        bin_pts = np.floor((n_freqs * 2 - 1) * hz_pts / fs).astype(int)
        fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
        for m in range(n_mels):
            for k in range(bin_pts[m], bin_pts[m + 1]):
                if k < n_freqs:
                    fb[k, m] = (k - bin_pts[m]) / (bin_pts[m + 1] - bin_pts[m] + 1e-10)
            for k in range(bin_pts[m + 1], bin_pts[m + 2]):
                if k < n_freqs:
                    fb[k, m] = (bin_pts[m + 2] - k) / (bin_pts[m + 2] - bin_pts[m + 1] + 1e-10)
        return torch.tensor(fb, device=device)

    def _mel_edr_db(self, ir: torch.Tensor) -> torch.Tensor:
        """Compute mel-scale EDR in dB. ir: [B, T] → [B, n_mels, M]."""
        if ir.dim() == 1:
            ir = ir.unsqueeze(0)
        B = ir.shape[0]
        window = self.window.to(ir.device)
        # Pad so STFT covers full IR with center=False
        spec = torch.stft(
            ir.reshape(B, -1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=False,
            pad_mode="reflect",
        )  # (B, n_fft//2+1, M)
        mag = spec.abs()  # (B, F, M)
        # Apply mel filterbank: (B, F, M) @ (F, n_mels) → (B, n_mels, M)
        mel_fb = self.mel_fb.to(mag.dtype)
        H_mel = torch.einsum("bfm,fn->bnm", mag, mel_fb)  # (B, n_mels, M)
        # Backward integration of power (eq. 9): R[k,m] = Σ_{τ=m}^{M-1} |H_mel[k,τ]|²
        R = torch.flip(torch.cumsum(torch.flip(H_mel ** 2, [-1]), -1), [-1])  # (B, n_mels, M)
        R_db = 10.0 * torch.log10(R + 1e-8)
        return R_db

    def _target_features(self, target: torch.Tensor) -> torch.Tensor:
        if id(target) == self._tgt_id and self._tgt_edr_db is not None:
            return self._tgt_edr_db
        with torch.no_grad():
            edr = self._mel_edr_db(target)
        self._tgt_id = id(target)
        self._tgt_edr_db = edr
        return edr

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        edr_est = self._mel_edr_db(estimation)
        edr_tgt = self._target_features(target)

        # Match frame count (estimation may differ from target in length)
        min_frames = min(edr_est.shape[-1], edr_tgt.shape[-1])
        diff = (edr_est[..., :min_frames] - edr_tgt[..., :min_frames]).abs()
        # Eq. 10: normalize by Σ|R_target^dB| so loss is scale-invariant
        norm = edr_tgt[..., :min_frames].abs().sum() + 1e-8
        return diff.sum() / norm


class SoftEDPLoss(nn.Module):
    """Soft Echo Density Profile loss (Mezza et al. [18], used in DAFx 2024).

    Differentiable approximation of the Normalized Echo Density profile via
    the complementary error function and a κ-scaled sigmoid. κ increases
    linearly from κ_start=100 to κ_end=100000 indexed by sample position
    (not training step), giving a curriculum that sharpens the soft threshold
    toward the end of the IR.

    η_κ[n] = Σ_τ w[τ] · sigmoid(κ_n · (erfc(|h[n+τ]| / (σ_n·√2 + ε)) − 0.5))

    where w[τ] is a Hann-tapered window with Σ w[τ]=1, σ_n is the local
    standard deviation within that window, and κ_n interpolates from
    κ_start to κ_end over the IR.

    Loss: L1 MAE between estimated and target η_κ profiles.
    """

    def __init__(self, window_samples: int = 321, kappa_start: float = 100.0,
                 kappa_end: float = 100000.0):
        super().__init__()
        # Ensure odd window size so padding is symmetric and unfold gives exactly T outputs
        win_size = window_samples if window_samples % 2 == 1 else window_samples + 1
        self.win_size = win_size
        self.half_w = win_size // 2
        self.kappa_start = kappa_start
        self.kappa_end = kappa_end

        # Hann-tapered window, normalised to sum=1
        w = torch.hann_window(win_size)
        w = w / w.sum()
        self.register_buffer("win", w)

    def _soft_edp(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Soft EDP profile. x: [B, T] → [B, T]."""
        import torch.nn.functional as F
        B, T = x.shape
        hw = self.half_w

        # Pad signal for windowed computation (symmetric, hw each side → same-length output)
        x_pad = F.pad(x, (hw, hw))  # (B, T + 2*hw)

        # Unfold into windows: (B, T, win_size) — with symmetric hw padding gives exactly T windows
        windows = x_pad.unfold(-1, self.win_size, 1)  # (B, T, win_size)

        # Local std per window
        mu = windows.mean(dim=-1, keepdim=True)
        sigma = (windows.var(dim=-1, keepdim=True) + 1e-10).sqrt()

        # κ schedule: linearly from kappa_start to kappa_end over T samples
        kappa = torch.linspace(
            self.kappa_start, self.kappa_end, T, device=x.device, dtype=x.dtype
        ).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        # erfc(|h| / (σ√2))
        from torch.special import erfc
        z = windows.abs() / (sigma * (2.0 ** 0.5) + 1e-10)
        erfc_val = erfc(z)  # (B, T, win_size)

        # Soft thresholding via κ-scaled sigmoid
        soft = torch.sigmoid(kappa * (erfc_val - 0.5))  # (B, T, win_size)

        # Weighted sum over window
        win = self.win.to(x.device)
        eta = (soft * win.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, T)
        return eta

    def forward(self, estimation, target):
        if estimation.dim() == 3 and estimation.shape[-1] == 1:
            estimation = estimation.squeeze(-1)
        if target.dim() == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        min_len = min(estimation.shape[-1], target.shape[-1])
        est = estimation[..., :min_len]
        tgt = target[..., :min_len]

        eta_est = self._soft_edp(est)
        with torch.no_grad():
            eta_tgt = self._soft_edp(tgt)
        # Eq. 7: MSE (not L1)
        return torch.nn.functional.mse_loss(eta_est, eta_tgt)
