"""The differentiable FDN reverberator.

A 16-line FDN at 48 kHz, fitted to a measured room impulse response by gradient
descent:

    impulse -FFT-> [ Branch A: late reverb                    ]  -+
                     input_gain -> Recursion(delays,               |-> tone -> bandpass -iFFT-> IR
                       mixing o PEQ attenuation) -> output_gain    |
                   [ Branch B: early reflections               ]  -+
                     K trainable delay taps (position + gain)

There is no time-domain recursion: the impulse goes through a flamo
``system.Shell`` that does FFT -> FDN transfer function -> iFFT.

Trained jointly: input/output gains, delay lengths (trainable-delay config
only), the orthogonal mixing matrix (Hadamard warm start), the shared
proportional PEQ attenuation, the post-sum tone PEQ, the early-reflection taps
and the bandpass cutoffs. ``ReverbConfig.proposed()`` trains the delays,
``ReverbConfig.fixed()`` keeps them fixed and trains everything else.

Needs the vendored flamo submodule; normalization constants are in
``reverb/utils.py``.
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

# --- Prefer the vendored, patched flamo submodule over any pip-installed copy. -
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_FLAMO = os.path.join(_HERE, "flamo")
if os.path.isdir(_FLAMO) and _FLAMO not in sys.path:
    sys.path.insert(0, _FLAMO)

import numpy as np
import torch
import torch.nn as nn

from flamo.processor import dsp, system
from flamo.functional import signal_gallery, find_onset
from flamo.auxiliary.reverb import parallelFDNPEQ

from reverb import utils as utilities


# Attenuation filter: one shared proportional PEQ across all delay lines.
class parallelFDNScalablePEQ(parallelFDNPEQ):
    """FDN attenuation PEQ with one shared parameter set across all delay lines.

    Overrides ``parallelFDNPEQ.map_eq()`` so that:
    - a single set of raw params is shared across delays,
    - per-band gains are derived from target RT values and per-delay scaling
      (``convert_time_to_response``), so the PEQ controls decay directly,
    - shelf bands use a safe S-range to avoid NaNs in RBJ shelf design.

    Raw parameter layout: ``[n_bands, 3, 1]`` with (freq_raw, q_raw, rt_raw).
    """

    def __init__(
        self,
        n_bands: int = 10,
        fs: int = 48000,
        delays: Optional[torch.Tensor] = None,
        nfft: int = 2 ** 11,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: Optional[str] = None,
        bypass_cascade_solver: bool = False,
        min_broadband_atten_db: float = 0.0,
        train_frequencies: bool = False,
    ):
        self.bypass_cascade_solver = bypass_cascade_solver
        self.min_broadband_atten_db = min_broadband_atten_db
        self.train_frequencies = train_frequencies
        super().__init__(
            n_bands=n_bands,
            delays=delays,
            nfft=nfft,
            fs=fs,
            alias_decay_db=alias_decay_db,
            requires_grad=requires_grad,
            device=device,
        )
        self.overwrite_number_of_parameters()

    def map_eq(self, param: torch.Tensor) -> torch.Tensor:
        """Map raw params -> per-delay biquad params.

        Returns tensor shaped ``[3, n_bands, n_delays]`` with (omega0, Q/S, gain_db).
        All band frequencies (including shelves) are trainable.
        """
        freq_raw = param[:, 0, 0]
        if not self.train_frequencies:
            freq_raw = freq_raw.detach()
        freq_raw_param = freq_raw.unsqueeze(-1).repeat(1, len(self.delays))
        q_raw_param = param[:, 1, 0].unsqueeze(-1).repeat(1, len(self.delays))

        freq_hz = utilities.frequency_denormalize(torch.sigmoid(freq_raw_param))
        freq = 2 * torch.pi * freq_hz / self.fs

        # Peaking bands (index 1:-1): Q∈[0.2, 2.0] log-mapped.
        # Shelf bands (index 0=high shelf, -1=low shelf): S∈[0.1, 0.95] linear.
        q_peaking = utilities.q_denormalize(torch.sigmoid(q_raw_param[1:-1]))
        q_shelf = 0.1 + torch.sigmoid(q_raw_param[[0, -1]]) * 0.85
        q_factor = torch.cat([q_shelf[:1], q_peaking, q_shelf[1:]], dim=0)

        rt_raw_param = param[:, 2, 0]
        rt_denormalized = utilities.rt_denormalize(torch.sigmoid(rt_raw_param))
        desired_gain = utilities.convert_time_to_response(rt_denormalized, self.delays, self.fs)
        if self.bypass_cascade_solver:
            # Scale per-band gains by the overlap row sums so the cascade total
            # at each centre freq matches the target. Detach freq/Q from the
            # overlap so gradients only flow RT → gain.
            overlap = self._build_overlap_matrix(freq.detach(), q_factor.detach())
            row_sums = overlap.sum(dim=1, keepdim=True).clamp(min=1.0)
            gain = desired_gain / row_sums
        else:
            gain = self._solve_cascade_matched_gain(desired_gain, freq, q_factor)
        gain = torch.clamp(gain, min=-30.0, max=-1e-6)

        return torch.stack((freq, q_factor, gain), dim=0)

    def _section_type(self, index: int) -> str:
        if index == 0:
            return "highshelf"
        if index == self.n_bands - 1:
            return "lowshelf"
        return "peaking"

    def _evaluate_section_db_at_controls(
        self,
        omega0: torch.Tensor,
        q_factor: torch.Tensor,
        gain_db: torch.Tensor,
        section_type: str,
        control_omega: torch.Tensor,
    ) -> torch.Tensor:
        a, b = self.compute_biquad_coeff(
            f=omega0,
            R=q_factor,
            G=gain_db,
            type=section_type,
        )

        z1 = torch.complex(torch.cos(control_omega), -torch.sin(control_omega))
        z2 = torch.complex(torch.cos(2.0 * control_omega), -torch.sin(2.0 * control_omega))

        b0, b1, b2 = b[..., 0].unsqueeze(-1), b[..., 1].unsqueeze(-1), b[..., 2].unsqueeze(-1)
        a0, a1, a2 = a[..., 0].unsqueeze(-1), a[..., 1].unsqueeze(-1), a[..., 2].unsqueeze(-1)
        num = b0 + b1 * z1 + b2 * z2
        den = a0 + a1 * z1 + a2 * z2
        mag = torch.abs(num / (den + 1e-12)).squeeze(0 if num.ndim == 2 and num.shape[0] == 1 else -1)
        return 20.0 * torch.log10(mag.clamp(min=1e-12))

    def _build_overlap_matrix(self, freq: torch.Tensor, q_factor: torch.Tensor) -> torch.Tensor:
        control_omega = freq[:, 0]
        ref_gain = torch.tensor(-1.0, device=freq.device, dtype=freq.dtype)

        columns = []
        for idx in range(self.n_bands):
            response_db = self._evaluate_section_db_at_controls(
                omega0=freq[idx, 0],
                q_factor=q_factor[idx, 0],
                gain_db=ref_gain,
                section_type=self._section_type(idx),
                control_omega=control_omega,
            )
            columns.append(response_db / ref_gain)

        overlap = torch.stack(columns, dim=1)
        regularizer = 1e-3 * torch.eye(self.n_bands, device=freq.device, dtype=freq.dtype)
        return overlap + regularizer

    def _solve_cascade_matched_gain(
        self,
        desired_gain: torch.Tensor,
        freq: torch.Tensor,
        q_factor: torch.Tensor,
    ) -> torch.Tensor:
        overlap = self._build_overlap_matrix(freq, q_factor)
        return torch.linalg.pinv(overlap) @ desired_gain

    def get_poly_coeff(self, ext_param=None):
        """Apply a broadband minimum attenuation so the cascade loop gain stays
        below 1.0 at ALL frequencies (prevents matrix singularity in the
        closed-form recursion solve)."""
        H, B, A = super().get_poly_coeff(ext_param)
        if self.min_broadband_atten_db != 0.0:
            gain_linear = 10 ** (self.min_broadband_atten_db / 20.0)
            H = H * gain_linear
        return H, B, A

    def overwrite_number_of_parameters(self) -> None:
        """Initialize the shared parameter tensor with sensible defaults.

        High shelf inits at 16 kHz, low shelf at 50 Hz, peaking bands log-spaced
        100 Hz–10 kHz; all RTs init to 3.0 s (overwritten per-band from the
        target's measured T30 by the training driver).
        """
        if self.n_bands < 3:
            raise ValueError("n_bands must be at least 3")

        n_peak = self.n_bands - 2
        f_min = utilities.MIN_FREQ   # 20 Hz
        f_max = utilities.MAX_FREQ   # 20000 Hz

        k = torch.arange(0, n_peak, device=self.device, dtype=torch.float32)
        f_peak_targets = 100.0 * (10000.0 / 100.0) ** (k / max(n_peak - 1, 1))

        all_freqs = torch.cat([
            torch.tensor([16000.0], device=self.device),
            f_peak_targets,
            torch.tensor([50.0], device=self.device),
        ])
        x_all = (torch.log(all_freqs / f_min) / math.log(f_max / f_min)).clamp(1e-4, 1 - 1e-4)
        freq_tensor = torch.logit(x_all).unsqueeze(-1)  # [n_bands, 1]

        q_tensor = torch.zeros([self.n_bands, 1], device=self.device)

        init_rt = 3.0
        rt_norm = (init_rt - utilities.MIN_RT) / (utilities.MAX_RT - utilities.MIN_RT)
        rt_norm = float(torch.clamp(torch.tensor(rt_norm), 1e-4, 1 - 1e-4))
        rt_logit = float(torch.log(torch.tensor(rt_norm / (1 - rt_norm))))
        gain_tensor = rt_logit * torch.ones([self.n_bands, 1], device=self.device)

        init_param = torch.stack((freq_tensor, q_tensor, gain_tensor), dim=1)
        self.param = torch.nn.Parameter(init_param, requires_grad=self.requires_grad)


# Post-sum band-limiting filter (trainable HP + LP cutoffs).
class TrainableBandpass(torch.nn.Module):
    """Trainable HP/LP filter applied in the frequency domain. Only the cutoff
    frequencies are trainable (4th-order Butterworth-shaped magnitude masks)."""

    def __init__(self, nfft, fs, f_lo=20.0, f_hi=20000.0, order=4, device="cpu"):
        super().__init__()
        self.nfft = nfft
        self.fs = fs
        self.order = order
        self.f_lo_param = nn.Parameter(torch.tensor(np.log(f_lo), dtype=torch.float32, device=device))
        self.f_hi_param = nn.Parameter(torch.tensor(np.log(f_hi), dtype=torch.float32, device=device))

    def forward(self, x, **kwargs):
        freqs = torch.fft.rfftfreq(self.nfft, 1.0 / self.fs).to(x.device)
        f_lo = torch.exp(self.f_lo_param).clamp(min=1.0, max=self.fs / 2.0)
        f_hi = torch.exp(self.f_hi_param).clamp(min=1.0, max=self.fs / 2.0)
        hp = 1.0 / torch.sqrt(1 + (f_lo / freqs.clamp(min=0.1)) ** (2 * self.order))
        lp = 1.0 / torch.sqrt(1 + (freqs / f_hi) ** (2 * self.order))
        return x * (hp * lp).float().view(1, -1, 1)


def extract_early_peaks(ir, fs, n_peaks, er_ms=50.0, min_distance_ms=0.5):
    """Top-N absolute peaks from the early portion of an IR (onset-aligned).

    Returns ``(positions, amplitudes)`` — sample indices and signed amplitudes,
    used to warm-start the early-reflection taps from the target RIR.
    """
    from scipy.signal import find_peaks as _find_peaks
    ir_np = ir.cpu().numpy() if hasattr(ir, "cpu") else np.asarray(ir).squeeze()
    n_er = int(er_ms * fs / 1000)
    ir_early = ir_np[:n_er]

    min_dist = max(1, int(min_distance_ms * fs / 1000))
    abs_ir = np.abs(ir_early)

    peak_idx, _props = _find_peaks(abs_ir, distance=min_dist)
    if len(peak_idx) == 0:
        peak_idx = np.linspace(0, n_er - 1, min(n_peaks, n_er), dtype=int)

    sorted_idx = peak_idx[np.argsort(abs_ir[peak_idx])[::-1]]
    top_idx = np.sort(sorted_idx[:n_peaks])
    positions = top_idx.tolist()
    amplitudes = [float(ir_early[i]) for i in positions]
    return positions, amplitudes


def _real_orthogonal_log(H, tol=1e-8):
    """Real skew-symmetric matrix X with expm(X) = H, for H real orthogonal in
    SO(N). Uses the real Schur decomposition so it handles -1 eigenvalues."""
    import scipy.linalg as _sla
    import numpy as _np
    N = H.shape[0]
    T, Z = _sla.schur(H, output="real")
    L = _np.zeros_like(T)
    pending_neg = []
    i = 0
    while i < N:
        if i + 1 < N and abs(T[i + 1, i]) > tol:
            c = 0.5 * (T[i, i] + T[i + 1, i + 1])
            s = 0.5 * (T[i + 1, i] - T[i, i + 1])
            theta = _np.arctan2(s, c)
            L[i, i + 1] = -theta
            L[i + 1, i] = theta
            i += 2
        elif abs(T[i, i] + 1.0) < tol:
            pending_neg.append(i)
            i += 1
        else:
            i += 1  # +1 eigenvalue: log = 0
    if len(pending_neg) % 2 != 0:
        raise ValueError(f"Odd number of -1 eigenvalues ({len(pending_neg)}) — H not in SO(N)")
    for j in range(0, len(pending_neg), 2):
        a, b = pending_neg[j], pending_neg[j + 1]
        L[a, b] = -_np.pi
        L[b, a] = _np.pi
    X = Z @ L @ Z.T
    return 0.5 * (X - X.T)


def _warm_start_orthogonal_from_hadamard(matrix_module, N, device):
    """Set the raw param of a Matrix(matrix_type='orthogonal') so that
    expm(skew(param)) reproduces the Hadamard matrix H/sqrt(N). Falls back to a
    small-random init on failure."""
    import scipy.linalg as _sla
    import numpy as _np
    H = _np.array([[1.0]])
    while H.shape[0] < N:
        H = _np.kron(H, _np.array([[1.0, 1.0], [1.0, -1.0]])) / _np.sqrt(2.0)
    if _np.linalg.det(H) < 0:
        H[:, 0] = -H[:, 0]
    try:
        X = _real_orthogonal_log(H)
        H_recon = _sla.expm(X)
        residual = float(_np.linalg.norm(H_recon - H, ord="fro"))
        if residual > 1e-5:
            raise ValueError(f"roundtrip residual {residual:.2e} > 1e-5")
        with torch.no_grad():
            matrix_module.param.data = torch.tensor(
                X, dtype=matrix_module.param.dtype, device=device)
        print(f"  Mixing matrix: warm-started from Hadamard (roundtrip residual={residual:.2e})")
    except Exception as e:
        with torch.no_grad():
            init = 0.05 * torch.randn(N, N, dtype=matrix_module.param.dtype, device=device)
            matrix_module.param.data = init
        print(f"  Mixing matrix: Hadamard warm-start failed ({e}); using small-random init")


class TrainableTapsDelay(torch.nn.Module):
    """K trainable-position, trainable-gain delay taps as a frequency-domain FIR.

    Each tap contributes ``g_i * exp(-j * omega * tau_i)`` to the frequency
    response. ``tau_i`` is a sigmoid over ``[tau_min, tau_max]`` samples;
    ``g_i`` is tanh-clamped to (-1, 1). Both are trainable. Slots into the
    early-reflections branch in place of a dense FIR (K params vs thousands).
    """

    def __init__(self, n_taps, nfft, fs,
                 tau_min_samples=0.0, tau_max_samples=2400.0,
                 init_positions=None, init_gains=None, device=None,
                 gain_clamp=True):
        super().__init__()
        self.n_taps = int(n_taps)
        self.nfft = int(nfft)
        self.fs = int(fs)
        self.tau_min = float(tau_min_samples)
        self.tau_max = float(tau_max_samples)
        self.gain_clamp = bool(gain_clamp)
        dev = device or "cpu"

        if init_positions is None:
            init_positions = torch.linspace(
                self.tau_min + 1.0, max(self.tau_max - 1.0, self.tau_min + 2.0),
                self.n_taps, device=dev)
        else:
            init_positions = torch.as_tensor(init_positions, dtype=torch.float32, device=dev)
        if init_gains is None:
            init_gains = torch.full((self.n_taps,), 0.05, dtype=torch.float32, device=dev)
        else:
            init_gains = torch.as_tensor(init_gains, dtype=torch.float32, device=dev)

        tau_range = max(self.tau_max - self.tau_min, 1e-3)
        norm = ((init_positions - self.tau_min) / tau_range).clamp(1e-4, 1 - 1e-4)
        self.tau_logit = torch.nn.Parameter(torch.log(norm / (1.0 - norm)))

        if self.gain_clamp:
            safe = init_gains.clamp(-0.999, 0.999)
            self.gain = torch.nn.Parameter(torch.atanh(safe))
        else:
            self.gain = torch.nn.Parameter(init_gains)

        omega = 2.0 * float(torch.pi) * torch.fft.rfftfreq(self.nfft, d=1.0).to(dev)
        self.register_buffer("omega", omega)

        # flamo's system.Series/Parallel inspect these to wire I/O.
        self.input_channels = 1
        self.output_channels = 1
        self.alias_decay_db = 0.0

    def positions_samples(self):
        return torch.sigmoid(self.tau_logit) * (self.tau_max - self.tau_min) + self.tau_min

    def _gains(self):
        return torch.tanh(self.gain) if self.gain_clamp else self.gain

    def build_fir(self):
        """Reconstruct the time-domain FIR (for inspection / export)."""
        with torch.no_grad():
            tau = self.positions_samples()
            g = self._gains()
            length = int(self.tau_max) + 8
            fir = torch.zeros(length, device=tau.device)
            for i in range(self.n_taps):
                p = float(tau[i].item())
                lo, hi = int(p), int(p) + 1
                frac = p - lo
                if 0 <= lo < length:
                    fir[lo] += g[i] * (1.0 - frac)
                if 0 <= hi < length:
                    fir[hi] += g[i] * frac
            return fir

    def forward(self, x, ext_param=None):
        tau = self.positions_samples()
        phase = -self.omega[:, None] * tau[None, :]
        H_taps = torch.exp(1j * phase) * self._gains()[None, :]
        H = H_taps.sum(dim=-1)
        return x * H.view(1, -1, 1).to(x.dtype)


# The FDN: two-branch assembly (paper topology: tone + bandpass on the sum).
class FDN:
    """Assemble the full two-branch FDN core (``self.FDN`` is the flamo module).

    Branch A (late reverb): input_gain → Recursion(delays, mixing ∘ PEQ) → output_gain.
    Branch B (early reflections): K trainable delay taps warm-started from the target.
    The tone PEQ and the band-limiting filter are applied to the *sum* of both
    branches, so the feedback loop sees a flat input and the attenuation PEQ
    controls decay directly.
    """

    def __init__(self, alias_decay_db, delay_lengths, N, BANDS, args, er_init=None,
                 mixing_type="orthogonal", train_mixing=True):
        # Input and output gains -------------------------------------------- #
        self.input_gain = dsp.Gain(
            size=(N, 1), nfft=args.nfft, requires_grad=True,
            map=lambda x: x, alias_decay_db=alias_decay_db, device=args.device,
        )
        self.output_gain = dsp.Gain(
            size=(1, N), nfft=args.nfft, requires_grad=True,
            map=lambda x: x / (N ** 0.5),  # 1/sqrt(N) FDN energy scaling
            alias_decay_db=alias_decay_db, device=args.device,
        )

        # Delay lines. isint=True uses .round() (zero gradient); switch to
        # fractional delays whenever the lengths are trainable.
        train_delays = bool(getattr(args, "train_delays", False))
        self.delays = dsp.parallelDelay(
            size=(N,), max_len=delay_lengths.max(), nfft=args.nfft,
            isint=not train_delays, unit=1000, requires_grad=train_delays,
            alias_decay_db=alias_decay_db, device=args.device,
        )
        self.delays.assign_value(self.delays.sample2s(delay_lengths))

        # Feedback mixing matrix.
        #   orthogonal — N×N orthogonal via matrix-exp of a skew matrix,
        #                warm-started from Hadamard, trainable (paper choice).
        #   hadamard   — fixed unitary Hadamard, no params.
        self.mixing_type = mixing_type
        if mixing_type == "orthogonal":
            self.mixing_matrix = dsp.Matrix(
                size=(N, N), nfft=args.nfft, matrix_type="orthogonal",
                requires_grad=train_mixing, alias_decay_db=alias_decay_db,
                device=args.device,
            )
            _warm_start_orthogonal_from_hadamard(self.mixing_matrix, N, args.device)
        else:
            self.mixing_matrix = dsp.Matrix(
                size=(N, N), nfft=args.nfft, matrix_type="hadamard",
                requires_grad=False, alias_decay_db=alias_decay_db, device=args.device,
            )
            self.mixing_matrix.param.data = self.mixing_matrix.param.data / (N ** 0.5)

        # Shared proportional PEQ attenuation (controls per-band decay).
        self.attenuation = parallelFDNScalablePEQ(
            n_bands=BANDS, nfft=args.nfft, fs=args.samplerate, delays=delay_lengths,
            requires_grad=True, alias_decay_db=alias_decay_db, device=args.device,
            bypass_cascade_solver=True,
            min_broadband_atten_db=getattr(args, "min_broadband_atten_db", 0.0),
            train_frequencies=getattr(args, "train_frequencies", True),
        )

        self.feedback = system.Series(
            OrderedDict({"mixing_matrix": self.mixing_matrix, "attenuation": self.attenuation})
        )

        # Closed-loop recursion. solve_epsilon adds diagonal jitter before the
        # linear solve (a numerical-stability patch in the vendored flamo).
        solve_eps_raw = getattr(args, "solve_epsilon", 0.0)
        if solve_eps_raw is None or float(solve_eps_raw) == 0.0:
            solve_eps = None          # flamo's auto heuristic
        elif float(solve_eps_raw) < 0.0:
            solve_eps = 0.0           # explicitly disable
        else:
            solve_eps = float(solve_eps_raw)
        self.feedback_loop = system.Recursion(
            fF=self.delays, fB=self.feedback, solve_epsilon=solve_eps
        )

        # Post-sum tone PEQ: shelving + peaking. Placed after the loop so it
        # shapes colour without affecting loop stability; gain clamped so it
        # cannot compensate for decay.
        self.tone_filter = dsp.PEQ(
            size=(1, 1), n_bands=BANDS, nfft=args.nfft, fs=args.samplerate,
            requires_grad=True, alias_decay_db=alias_decay_db, device=args.device,
        )
        with torch.no_grad():
            new_bias = self.tone_filter.center_freq_bias.clone()
            new_bias[0] = 20.0       # lowshelf bias (flamo band 0 = lowest freq)
            new_bias[-1] = 10000.0   # highshelf bias (band -1 = highest freq)
            self.tone_filter.center_freq_bias.copy_(new_bias)

            two_pi_fs = 2.0 * math.pi / args.samplerate
            new_param = self.tone_filter.param.clone()
            diff_low = (100.0 - 20.0) * two_pi_fs          # lowshelf target 100 Hz
            new_param[0, 0, :] = math.log(max(diff_low, 1e-4) / max(1 - diff_low, 1e-4))
            diff_high = (12000.0 - 10000.0) * two_pi_fs    # highshelf target 12 kHz
            new_param[-1, 0, :] = math.log(max(diff_high, 1e-4) / max(1 - diff_high, 1e-4))
            self.tone_filter.param.copy_(new_param)

        tone_g_lo = float(getattr(args, "tone_clamp_lo", -30.0))
        tone_g_hi = float(getattr(args, "tone_clamp_hi", 18.0))
        original_tone_map = self.tone_filter.map_eq

        def constrained_tone_map(param):
            mapped = original_tone_map(param)
            f, R, G = mapped[0], mapped[1], mapped[2]
            G = torch.clamp(G, tone_g_lo, tone_g_hi)
            R = torch.clamp(R, min=0.1)  # Q/S must stay positive for stable biquads
            return torch.cat((f.unsqueeze(0), R.unsqueeze(0), G.unsqueeze(0)), dim=0)
        self.tone_filter.map_eq = constrained_tone_map

        # Branch B: K trainable early-reflection taps (frequency-domain FIR),
        # warm-started from the top-N peaks of the target's early portion.
        n_taps = int(getattr(args, "er_n_taps", 64))
        max_ms = float(getattr(args, "er_max_ms", 50.0))
        tau_max = max_ms * args.samplerate / 1000.0
        init_positions = init_gains = None
        if er_init is not None:
            try:
                pos, amps = extract_early_peaks(er_init, args.samplerate, n_peaks=n_taps, er_ms=max_ms)
                init_positions = torch.as_tensor(pos, dtype=torch.float32)
                init_gains = torch.as_tensor(amps, dtype=torch.float32)
            except Exception:
                init_positions = init_gains = None
        self.early_reflections = TrainableTapsDelay(
            n_taps=n_taps, nfft=args.nfft, fs=args.samplerate,
            tau_min_samples=0.0, tau_max_samples=tau_max,
            init_positions=init_positions, init_gains=init_gains,
            device=args.device, gain_clamp=bool(getattr(args, "er_gain_clamp", True)),
        )
        print(f"  ER: {n_taps} trainable taps over [0, {max_ms} ms] "
              f"({'peak-init' if init_positions is not None else 'linspace-init'})")

        # Post-sum band-limiting (20 Hz HP + 20 kHz LP; cutoffs trainable).
        self.bandpass = TrainableBandpass(args.nfft, args.samplerate, f_lo=20.0, f_hi=20000.0, device=args.device)

        # Assemble: (branchA + branchB) → tone → bandpass.
        self.branchA = system.Series(OrderedDict({
            "input_gain": self.input_gain,
            "feedback_loop": self.feedback_loop,
            "output_gain": self.output_gain,
        }))
        self.branchB = system.Series(OrderedDict({"early_reflections": self.early_reflections}))
        self.FDN_parallel = system.Parallel(brA=self.branchA, brB=self.branchB)
        self.FDN = system.Series(OrderedDict({
            "parallel": self.FDN_parallel,
            "tone_filter": self.tone_filter,
            "bandpass": self.bandpass,
        }))


# High-level API used by fit.py.
@dataclass
class ReverbConfig:
    """Fixed paper configuration for fitting the FDN to a target RIR.

    The two headline configs differ only in ``train_delays``. Defaults reproduce
    the paper's published listening-site methods (FDN_DiffER_PEQ10 /
    FDN_FixedDelayTrainMix_ER_PEQ10): a 16-line FDN with a shared 10-band PEQ.
    """
    n_delays: int = 16
    bands: int = 10
    er_n_taps: int = 64
    er_max_ms: float = 50.0
    mixing_type: str = "orthogonal"
    train_mixing: bool = True
    train_delays: bool = True
    delay_range_scale: float = 0.6
    solve_epsilon: float = 1e-3
    tone_clamp_lo: float = -30.0
    tone_clamp_hi: float = 18.0
    min_broadband_atten_db: float = 0.0
    train_frequencies: bool = True

    @classmethod
    def proposed(cls, **kw) -> "ReverbConfig":
        """The proposed all-differentiable FDN (delay lengths train)."""
        return cls(train_delays=True, **kw)

    @classmethod
    def fixed(cls, **kw) -> "ReverbConfig":
        """The recommended fixed-delay / train-mix variant (delays fixed)."""
        return cls(train_delays=False, **kw)


def _get_delays(n_delays, fs, range_scale=1.0):
    """Mutually-prime delay-line lengths."""
    s = float(range_scale)
    if n_delays == 8:
        return utilities.setup_fdn_delays(num_delays=8, min_ms=15.0 * s, max_ms=85.0 * s, sample_rate=fs)
    elif n_delays == 16:
        return utilities.setup_fdn_delays(num_delays=16, min_ms=10.0 * s, max_ms=70.0 * s, sample_rate=fs)
    else:
        return utilities.setup_fdn_delays(num_delays=n_delays, min_ms=20.0 * s, max_ms=60.0 * s, sample_rate=fs)


def load_target(path, fs=48000):
    """Load a RIR, resample to ``fs``, take channel 0, and onset-align.

    Returns ``(target_trimmed, fs)`` — a 1-D tensor starting at the direct sound.
    """
    import soundfile as sf
    target_raw, rir_fs = sf.read(path)
    if getattr(target_raw, "ndim", 1) > 1:
        target_raw = target_raw[:, 0]                     # first channel if multichannel
    if rir_fs != fs:
        import torchaudio
        t = torch.tensor(target_raw, dtype=torch.float32)
        target_raw = torchaudio.functional.resample(t, rir_fs, fs).numpy()
    target_tensor = torch.tensor(target_raw, dtype=torch.float32)
    onset = find_onset(target_tensor)
    return target_tensor[onset:], fs


def auto_nfft(target_trimmed, fs):
    """FFT size from the target's broadband RT60 (rounded up to the next 0.5 s),
    from the target's broadband RT60."""
    raw_np = target_trimmed.detach().cpu().numpy()
    try:
        edc_bb = np.cumsum(raw_np[::-1] ** 2)[::-1]
        edc_bb_db = 10 * np.log10(edc_bb / (edc_bb[0] + 1e-30) + 1e-30)
        mask_bb = (edc_bb_db >= -35) & (edc_bb_db <= -5)
        if mask_bb.sum() > 10:
            t_bb = np.arange(len(edc_bb_db))[mask_bb] / fs
            slope_bb, _ = np.polyfit(t_bb, edc_bb_db[mask_bb], 1)
            max_rt = -60.0 / slope_bb
        else:
            max_rt = len(raw_np) / fs
        file_duration = len(target_trimmed) / fs
        ir_duration = min(max(max_rt * 1.2, 2.0), file_duration)
    except Exception:
        ir_duration = max(len(target_trimmed) / fs, 2.0)
    half_seconds = math.ceil(ir_duration / 0.5)
    return int(half_seconds * 0.5 * fs)


def prepare_target_ir(target_trimmed, nfft, device):
    """Pad/truncate the onset-aligned target to ``nfft`` and peak-normalize.

    Returns ``target_ir`` shaped ``[1, nfft, 1]`` on ``device`` (the exact form
    the losses and metrics consume).
    """
    import torch.nn.functional as F
    if len(target_trimmed) < nfft:
        target_padded = F.pad(target_trimmed, (0, nfft - len(target_trimmed)))
    else:
        target_padded = target_trimmed[:nfft]
    target_padded = target_padded / target_padded.abs().max()
    return target_padded.unsqueeze(0).unsqueeze(-1).to(device)


def build_fdn(target_ir, nfft, fs, cfg: ReverbConfig, device="cpu", seed=999):
    """Build the FDN and wrap it in a flamo ``system.Shell`` (FFT → FDN → iFFT).

    Args:
        target_ir: ``[1, nfft, 1]`` onset-aligned, peak-normalized target (for ER init).
        nfft, fs:  FFT size and sample rate.
        cfg:       a :class:`ReverbConfig` (use ``.proposed()`` / ``.fixed()``).
        device:    "cpu" or "cuda".
        seed:      RNG seed set before student construction (paper uses 999).

    Returns ``(model, fdn)`` — the trainable Shell and the underlying FDN object.
    """
    from types import SimpleNamespace
    delay_lengths = _get_delays(cfg.n_delays, fs, range_scale=cfg.delay_range_scale).to(device)
    er_init = target_ir[0, :, 0].squeeze(-1)

    base_args = SimpleNamespace(
        nfft=nfft, samplerate=fs, device=device,
        solve_epsilon=cfg.solve_epsilon, train_delays=cfg.train_delays,
        min_broadband_atten_db=cfg.min_broadband_atten_db,
        train_frequencies=cfg.train_frequencies,
        tone_clamp_lo=cfg.tone_clamp_lo, tone_clamp_hi=cfg.tone_clamp_hi,
        er_n_taps=cfg.er_n_taps, er_max_ms=cfg.er_max_ms,
    )

    torch.manual_seed(seed)
    fdn = FDN(0, delay_lengths, cfg.n_delays, cfg.bands, base_args,
              er_init=er_init, mixing_type=cfg.mixing_type, train_mixing=cfg.train_mixing)

    # Freeze mixing / delays unless their flag is set.
    if hasattr(fdn.mixing_matrix, "param") and not cfg.train_mixing:
        fdn.mixing_matrix.param.requires_grad = False
    if hasattr(fdn.delays, "param") and not cfg.train_delays:
        fdn.delays.param.requires_grad = False

    model = system.Shell(
        core=fdn.FDN,
        input_layer=dsp.FFT(nfft),
        output_layer=dsp.iFFTAntiAlias(nfft=nfft, alias_decay_db=0, device=device),
    ).to(device)
    return model, fdn


def render_ir(model, nfft, fs, device="cpu"):
    """Render the model's impulse response (closed form: FFT → FDN → iFFT)."""
    impulse = signal_gallery(1, n_samples=nfft, n=1, signal_type="impulse", fs=fs, device=device)
    with torch.no_grad():
        ir = model(impulse).detach()
    return ir


__all__ = [
    "parallelFDNScalablePEQ", "TrainableBandpass", "TrainableTapsDelay", "FDN",
    "ReverbConfig", "load_target", "auto_nfft", "prepare_target_ir",
    "build_fdn", "render_ir", "extract_early_peaks",
]
