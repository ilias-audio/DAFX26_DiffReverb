"""Reimplementation of Mezza et al., DAFx 2024, used as a baseline.

Mezza, Giampiccolo, Bernardini, "Modeling the Frequency-Dependent Sound Energy
Decay of Acoustic Environments With Differentiable Feedback Delay Networks",
DAFx 2024.

    python baselines/dfdn_mezza.py --target_rir <wav> --train_dir <dir>

Modified FDN (paper Fig. 2): 6 delay lines, a 63-tap FIR per line for the
frequency-dependent attenuation, a learned orthogonal feedback matrix (matrix
exponential of a skew-symmetric W, no diagonal gain), and a 63-tap tone filter
on the FDN output only, with the direct path d*u bypassing it.

Loss L_FD = 0.5*L_EDC + 1.0*L_EDR + 0.1*L_EDP: broadband Schroeder EDC (L2,
linear), mel-scale energy decay relief (L1, dB) and soft echo density profile
(L1). Two Adam optimizers, lr 0.1 for W, b, c, m, d and 0.001 for the FIR taps,
650 steps, losses evaluated over ceil(T60*fs) samples.

The paper works at 16 kHz on the MIT corpus; targets are resampled to match.
Output layout is the same as fit.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

# Release layout: baselines/<this>.py → package root is one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAMO_ROOT = os.path.join(REPO_ROOT, "flamo")
for p in (REPO_ROOT, FLAMO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from flamo.functional import signal_gallery, find_onset
from flamo.processor import dsp, system


# ---------------------------------------------------------------------------
# DepthwiseFIRBank: N independent p-tap FIR filters (one per delay line).
# ---------------------------------------------------------------------------

class DepthwiseFIRBank(nn.Module):
    """N independent FIR attenuation filters, one per delay line.

    Implements the depthwise convolutional filterbank from Mezza et al.
    (Section 3.2) in the frequency domain: each delay line i has its own
    p-tap FIR h_i[n], and the frequency-domain response is computed as
    rFFT(h_i, n=nfft). Applied elementwise to the freq-domain signal.

    Initialization: h_i[0] = gamma_init (≈0.9), h_i[k>0] = 0 for all i,
    matching the paper's "scaled Kronecker delta γ_i^(0) · δ[n]".

    Compatible with flamo's frequency-domain pipeline: forward(x) takes
    x of shape (B, nfft//2+1, N) and returns the same shape.
    """

    def __init__(self, N: int, p: int, nfft: int, gamma_init: float = 0.9,
                 device: str = "cpu"):
        super().__init__()
        self.N = N
        self.p = p
        self.nfft = nfft
        self.input_channels = N
        self.output_channels = N
        self.register_buffer("alias_decay_db", torch.zeros(1, device=device))

        # (N, p) — each row is the FIR kernel for one delay line
        taps = torch.zeros(N, p, device=device)
        taps[:, 0] = gamma_init  # initialize to gamma_init · delta[n]
        self.taps = nn.Parameter(taps)

    def _freq_response(self) -> torch.Tensor:
        """Compute per-channel frequency response. Returns (nfft//2+1, N) complex."""
        pad_len = self.nfft - self.p
        taps_padded = F.pad(self.taps, (0, pad_len))  # (N, nfft)
        H = torch.fft.rfft(taps_padded, n=self.nfft, dim=1)  # (N, nfft//2+1)
        return H.T  # (nfft//2+1, N)

    def forward(self, x, ext_param=None):
        """x: (B, nfft//2+1, N) -> (B, nfft//2+1, N)"""
        H = self._freq_response()  # (M, N)
        if x.dim() == 3:
            return x * H.unsqueeze(0)  # (B, M, N)
        elif x.dim() == 4:
            return x * H.unsqueeze(0).unsqueeze(-1)
        raise ValueError(f"Unsupported x.dim()={x.dim()}")

    @torch.no_grad()
    def project_to_stable(self, eps: float = 1e-3) -> int:
        """Renormalize each line's FIR so max_f |H_i(f)| <= 1 - eps.

        The per-line FIR sits inside the feedback loop; a frequency band where
        |H_i(f)| >= 1 sustains energy indefinitely on every loop traversal and
        produces audible buzz. Paper init (gamma*delta) is stable, but training
        drift produces unstable bands. This projection leaves stable lines
        untouched and shrinks any line that has crossed the cap. Returns the
        number of lines that were rescaled.
        """
        pad_len = self.nfft - self.p
        taps_padded = F.pad(self.taps, (0, pad_len))
        H = torch.fft.rfft(taps_padded, n=self.nfft, dim=1)
        mag_max = H.abs().amax(dim=1)  # (N,)
        cap = 1.0 - eps
        scale = torch.clamp(cap / mag_max.clamp_min(1e-12), max=1.0)
        self.taps.data.mul_(scale.unsqueeze(1))
        return int((scale < 1.0).sum().item())


# ---------------------------------------------------------------------------
# ToneFIRFilter: single p-tap output tone correction filter T(z).
# ---------------------------------------------------------------------------

class ToneFIRFilter(nn.Module):
    """Single-channel output FIR tone correction filter T(z).

    Implements the tone correction filter from Mezza et al. (Section 3.2)
    in the frequency domain. Initialized to identity delta[n] so the initial
    FDN output is unchanged by the tone filter.

    Compatible with flamo's pipeline: forward(x) takes (B, nfft//2+1, 1).
    """

    def __init__(self, p: int, nfft: int, device: str = "cpu"):
        super().__init__()
        self.p = p
        self.nfft = nfft
        self.input_channels = 1
        self.output_channels = 1
        self.register_buffer("alias_decay_db", torch.zeros(1, device=device))

        # Initialize to identity (delta function at n=0)
        taps = torch.zeros(p, device=device)
        taps[0] = 1.0
        self.taps = nn.Parameter(taps)

    def _freq_response(self) -> torch.Tensor:
        """Returns (nfft//2+1,) complex frequency response."""
        pad_len = self.nfft - self.p
        taps_padded = F.pad(self.taps, (0, pad_len))
        return torch.fft.rfft(taps_padded, n=self.nfft)  # (nfft//2+1,)

    def forward(self, x, ext_param=None):
        """x: (B, nfft//2+1, 1) -> (B, nfft//2+1, 1)"""
        T = self._freq_response()  # (M,)
        if x.dim() == 3:
            return x * T.unsqueeze(0).unsqueeze(-1)  # (B, M, 1)
        elif x.dim() == 4:
            return x * T.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        raise ValueError(f"Unsupported x.dim()={x.dim()}")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _get_delays_beta(N: int, fs: int, psi: int = 1024, alpha: float = 1.1,
                     beta: float = 6.0, seed: int | None = None) -> torch.Tensor:
    """Sample N delay lengths from Beta(alpha, beta) * psi.

    Paper: psi=1024, alpha=1.1, beta=6. At fs=16 kHz this gives max delay
    1024 samples = 64 ms and mean ~= alpha/(alpha+beta) * psi ~= 160 samples ~= 10 ms.
    """
    gen = torch.Generator()
    gen.manual_seed(seed if seed is not None else 42)
    u = torch.rand(N, generator=gen)
    v = torch.rand(N, generator=gen)
    # Beta sample via two Gamma samples: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
    # For simplicity use torch.distributions which handles seeding via generator param
    dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
    m_tilde = dist.sample((N,))
    delays = (m_tilde * psi).clamp(min=1.0)  # keep as float for fractional delay training
    return delays


def _estimate_t60(ir: np.ndarray, fs: int) -> float | None:
    """Estimate broadband T60 from Schroeder backward integration."""
    try:
        sq = ir.astype(np.float64) ** 2
        edc = np.cumsum(sq[::-1])[::-1]
        edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
        mask = (edc_db >= -35.0) & (edc_db <= -5.0)
        if mask.sum() > 10:
            t = np.arange(len(edc_db))[mask] / fs
            slope, _ = np.polyfit(t, edc_db[mask], 1)
            return float(-60.0 / slope)
    except Exception:
        pass
    return None


def _measure_t30(ir: np.ndarray, fs: int) -> dict:
    """Per-octave-band T30 in seconds."""
    import tempfile
    from reverb import metrics as RA
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, ir, fs)
        analyzer = RA.ReverbAnalyzer(tmp)
        analyzer.compute_octave_bands()
        analyzer.compute_energy_decay_curves()
        analyzer.measure_reverberation_time()
        result = {}
        if "T30" in analyzer.rt_table.index:
            for freq in analyzer.rt_table.columns:
                val = float(np.squeeze(analyzer.rt_table.at["T30", freq]))
                if np.isfinite(val):
                    result[float(freq)] = val
        return result
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# MezzaFDN: the modified FDN architecture.
# ---------------------------------------------------------------------------

class MezzaFDN:
    """Modified FDN: independent per-line FIR + learned orthogonal matrix + tone FIR.

    All components are flamo-compatible. self.FDN is a system.Series that can
    be wrapped by system.Shell for inference and training.
    """

    def __init__(self, delay_lengths: torch.Tensor, N: int, p: int,
                 nfft: int, fs: int, device: str = "cpu"):
        # Input gains b ~ N(0, 1/N): shape (N, 1)
        self.input_gain = dsp.Gain(
            size=(N, 1), nfft=nfft, requires_grad=True,
            map=lambda x: x, device=device,
        )
        with torch.no_grad():
            self.input_gain.param.data.normal_(0.0, 1.0 / N ** 0.5)

        # Output gains c = 1/N * 1_N: shape (1, N)
        self.output_gain = dsp.Gain(
            size=(1, N), nfft=nfft, requires_grad=True,
            map=lambda x: x, device=device,
        )
        with torch.no_grad():
            self.output_gain.param.data.fill_(1.0 / N)

        # Delay lines: fractional delays, trained with structural optimizer (LR=0.1).
        # isint=False → freq response is exp(-j·ω·m) with continuous m → differentiable.
        # unit=fs makes the raw parameter live in *samples* (sample2s(m) = m/fs*fs = m),
        # so LR=0.1 gives ≈0.31 sample/step updates (gradient magnitude ≤ π). Using
        # unit=1 (seconds) would multiply the gradient by fs=16000 and blow up training.
        # When requires_grad=True, flamo sets map=softplus; for m>>1 softplus(m)≈m so
        # init raw_param ≈ delay_samples is fine (softplus_inv(160) ≈ 160).
        max_len = int(delay_lengths.max().item() * 4)
        self.delays = dsp.parallelDelay(
            size=(N,), max_len=max_len, nfft=nfft,
            isint=False, unit=fs, fs=fs,
            requires_grad=True,
            device=device,
        )
        # All delays ≥ 10 samples, so softplus_inv(m) ≈ m (softplus(m) ≈ m for m >> 0).
        self.delays.assign_value(delay_lengths.float())

        # Per-line FIR attenuation bank H_i(z): N independent p-tap filters
        self.attenuation = DepthwiseFIRBank(N=N, p=p, nfft=nfft, device=device)

        # Orthogonal feedback matrix U = exp(skew(W)), trainable.
        # Paper §3.5: W_ij ~ N(0, 1/N) — scalar-normal notation, variance=1/N, std=1/√N.
        self.mixing_matrix = dsp.Matrix(
            size=(N, N), nfft=nfft, matrix_type="orthogonal",
            requires_grad=True, device=device,
        )
        with torch.no_grad():
            self.mixing_matrix.param.data.normal_(0.0, 1.0 / N ** 0.5)

        # Direct path d: trainable scalar, |·| parameterization, init=1
        self.direct = dsp.Gain(
            size=(1, 1), nfft=nfft, requires_grad=True,
            map=lambda x: x.abs(),
            device=device,
        )
        with torch.no_grad():
            self.direct.param.data.fill_(1.0)

        # Output tone correction filter T(z): single p-tap FIR, init to delta
        self.tone_filter = ToneFIRFilter(p=p, nfft=nfft, device=device)

        # Feedback path: H_i(z) attenuation THEN U orthogonal mixing
        self.feedback = system.Series(OrderedDict({
            "attenuation": self.attenuation,
            "mixing_matrix": self.mixing_matrix,
        }))
        self.feedback_loop = system.Recursion(
            fF=self.delays, fB=self.feedback
        )

        # Branch A: b → recursion(delays → H_i → U) → c → T(z)
        # Paper Fig 2: tone filter T(z) sits at the FDN output, BEFORE the
        # direct path d·u is summed in. So y[n] = T(c·s[n]) + d·u[n].
        self.branchA = system.Series(OrderedDict({
            "input_gain": self.input_gain,
            "feedback_loop": self.feedback_loop,
            "output_gain": self.output_gain,
            "tone_filter": self.tone_filter,
        }))

        # Branch B: direct path d, bypassing T(z)
        self.branchB = system.Series(OrderedDict({
            "direct": self.direct,
        }))

        # Full system: Parallel(A, B) — sums T(c·s) and d·u into y[n]
        self.FDN = system.Parallel(brA=self.branchA, brB=self.branchB)


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def train(args):
    fs_target = args.samplerate  # 16 kHz (paper default)
    device = args.device
    N = args.n_delays
    p = args.fir_taps

    torch.manual_seed(args.seed)

    # ---- Load and resample target RIR ----
    target_raw, rir_fs = sf.read(args.target_rir)
    if target_raw.ndim > 1:
        target_raw = target_raw[:, 0]  # mono
    if rir_fs != fs_target:
        import torchaudio
        t = torch.tensor(target_raw, dtype=torch.float32)
        t = torchaudio.functional.resample(t, rir_fs, fs_target)
        target_raw = t.numpy()
    target_tensor = torch.tensor(target_raw, dtype=torch.float32)

    onset = find_onset(target_tensor)
    target_trimmed = target_tensor[onset:]

    # Estimate broadband T60 for nfft sizing and loss window
    t60_est = _estimate_t60(target_trimmed.numpy(), fs_target)
    if t60_est is not None and 0.1 < t60_est < 30.0:
        print(f"  Estimated T60: {t60_est:.3f} s")
    else:
        t60_est = max(len(target_trimmed) / fs_target, 0.5)
        print(f"  T60 estimation failed; using {t60_est:.2f} s fallback")

    if args.nfft is not None:
        nfft = args.nfft
    else:
        ir_duration = max(t60_est * 1.2, 0.5)
        half_seconds = math.ceil(ir_duration / 0.5)
        nfft = int(half_seconds * 0.5 * fs_target)
    print(f"  nfft: {nfft} ({nfft / fs_target:.2f} s)")

    # L_h: losses evaluated only up to T60 (paper Section 3.3 note)
    L_h = min(int(math.ceil(t60_est * fs_target)), nfft)
    print(f"  L_h (training trim to T60): {L_h} ({L_h / fs_target:.2f} s)")

    if len(target_trimmed) < nfft:
        target_padded = F.pad(target_trimmed, (0, nfft - len(target_trimmed)))
    else:
        target_padded = target_trimmed[:nfft]
    target_padded = target_padded / target_padded.abs().max().clamp(min=1e-10)

    target_ir = target_padded.unsqueeze(0).unsqueeze(-1).to(device)
    target_loss = target_padded[:L_h].unsqueeze(0).to(device)  # [1, L_h]

    out_dir = args.train_dir or os.path.join(
        "output", time.strftime("%Y%m%d-%H%M%S") + "_mezza"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "command.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    print(f"\n{'='*80}\n  Mezza DAFx 2024 — Modified FDN\n"
          f"  Target: {args.target_rir}\n"
          f"  N={N}, p={p}, nfft={nfft} ({nfft/fs_target:.2f}s), fs={fs_target} Hz\n"
          f"  output: {out_dir}\n{'='*80}")

    sf.write(os.path.join(out_dir, "ir_target.wav"),
             target_ir[0, :, 0].cpu().numpy(), fs_target)

    # ---- Build model ----
    delay_lengths = _get_delays_beta(N=N, fs=fs_target, psi=args.psi,
                                     alpha=args.delay_alpha, beta=args.delay_beta,
                                     seed=args.seed)
    print(f"  Delay lengths init (samples): {[f'{x:.1f}' for x in delay_lengths.tolist()]}")
    print(f"  Delay lengths init (ms):      "
          f"{[f'{d/fs_target*1000:.1f}' for d in delay_lengths.tolist()]}")

    mezza = MezzaFDN(delay_lengths=delay_lengths, N=N, p=p,
                     nfft=nfft, fs=fs_target, device=device)

    impulse = signal_gallery(
        1, n_samples=nfft, n=1, signal_type="impulse", fs=fs_target, device=device
    )
    model = system.Shell(
        core=mezza.FDN,
        input_layer=dsp.FFT(nfft),
        output_layer=dsp.iFFTAntiAlias(nfft=nfft, alias_decay_db=0, device=device),
    ).to(device)

    with torch.no_grad():
        init_ir = model(impulse).detach()
    init_np = init_ir[0, :, 0].cpu().numpy()
    init_np = init_np / max(float(np.abs(init_np).max()), 1e-12)
    sf.write(os.path.join(out_dir, "ir_init.wav"), init_np, fs_target, subtype="FLOAT")

    # ---- Capture init parameter snapshot ----
    # .clone() is essential: .numpy() on a CPU tensor shares memory, so without it
    # snap_init would silently reflect the trained values after the optimizer steps.
    def _snap(t): return t.clone().detach().cpu().numpy()
    def _delay_samples(fdn): return fdn.delays.s2sample(fdn.delays.map(fdn.delays.param)).detach().cpu().numpy().copy()
    snap_init = {
        "delay_lengths_samples": _delay_samples(mezza),
        "attenuation_taps": _snap(mezza.attenuation.taps),  # (N, p)
        "tone_taps":        _snap(mezza.tone_filter.taps),  # (p,)
        "input_gain_b":     _snap(mezza.input_gain.param),  # (N, 1)
        "output_gain_c":    _snap(mezza.output_gain.param), # (1, N)
        "direct_d":         _snap(mezza.direct.param),      # (1, 1)
        "mixing_W":         _snap(mezza.mixing_matrix.param),
    }

    # ---- Loss setup (paper eq. 11: L_FD = λ1·L_EDC + λ2·L_EDR + λ3·L_EDP) ----
    from reverb.losses import BroadbandEDCLoss, MelEDRLoss, SoftEDPLoss

    loss_edc = BroadbandEDCLoss().to(device)
    loss_edr = MelEDRLoss(
        fs=fs_target,
        n_fft=args.mel_nfft,
        win_length=args.mel_win,
        hop_length=args.mel_hop,
        n_mels=args.mel_bands,
        device=device,
    ).to(device)
    loss_edp = SoftEDPLoss(
        window_samples=args.edp_win,
        kappa_start=args.kappa_start,
        kappa_end=args.kappa_end,
    ).to(device)

    w_edc, w_edr, w_edp = args.w_edc, args.w_edr, args.w_edp

    def compute_loss(ir_hat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Trim to T60 window, compute L_FD."""
        est = ir_hat[:, :L_h, :]  # [B, L_h, 1]
        if est.dim() == 3 and est.shape[-1] == 1:
            est = est.squeeze(-1)  # [B, L_h]
        l_edc = loss_edc(est, target_loss)
        l_edr = loss_edr(est, target_loss)
        l_edp = loss_edp(est, target_loss)
        total = w_edc * l_edc + w_edr * l_edr + w_edp * l_edp
        return total, {
            "EDC": l_edc.item(),
            "EDR": l_edr.item(),
            "EDP": l_edp.item(),
            "total": total.item(),
        }

    with torch.no_grad():
        _ir0 = model(impulse)
        _, _info0 = compute_loss(_ir0)
    print(f"\n  Initial losses: {_info0}")

    # ---- Two Adam optimizers (paper Section 3.4) ----
    fir_param_ids = {
        id(q) for q in list(mezza.attenuation.parameters()) +
                        list(mezza.tone_filter.parameters())
    }
    structural_params = [q for q in model.parameters()
                         if q.requires_grad and id(q) not in fir_param_ids]
    fir_params = [q for q in model.parameters()
                  if q.requires_grad and id(q) in fir_param_ids]

    print(f"\n  Structural params: {sum(q.numel() for q in structural_params)}")
    print(f"  FIR params:        {sum(q.numel() for q in fir_params)}")

    opt_struct = torch.optim.Adam(structural_params, lr=args.lr_struct,
                                  betas=(0.9, 0.999), weight_decay=0.0)
    opt_fir = torch.optim.Adam(fir_params, lr=args.lr_fir,
                               betas=(0.9, 0.999), weight_decay=0.0)

    # ---- Training loop: 650 gradient steps ----
    print(f"\n  Training for {args.max_iters} gradient steps...\n")
    loss_history = []
    best_loss = float("inf")
    best_state = None
    train_start = time.perf_counter()

    for step in range(args.max_iters):
        opt_struct.zero_grad()
        opt_fir.zero_grad()

        ir_hat = model(impulse)
        total_loss, info = compute_loss(ir_hat)

        if not torch.isfinite(total_loss):
            print(f"  step {step}: non-finite loss, skipping")
            continue

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(structural_params, args.grad_clip)
            torch.nn.utils.clip_grad_norm_(fir_params, args.grad_clip)

        opt_struct.step()
        opt_fir.step()

        if args.enforce_stability:
            n_projected = mezza.attenuation.project_to_stable(eps=args.stability_eps)
            info["n_projected"] = n_projected

        loss_history.append({"step": step, **info})

        if info["total"] < best_loss:
            best_loss = info["total"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.perf_counter() - train_start
            print(f"  step {step+1:4d}/{args.max_iters} | "
                  f"total={info['total']:.4f}  "
                  f"EDC={info['EDC']:.4f}  EDR={info['EDR']:.4f}  EDP={info['EDP']:.4f}"
                  f" | {elapsed:.1f}s")

    train_wall_s = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ---- Capture final parameter snapshot and save analysis_data.npz ----
    snap_final = {
        "delay_lengths_samples": _delay_samples(mezza),
        "attenuation_taps": _snap(mezza.attenuation.taps),
        "tone_taps":        _snap(mezza.tone_filter.taps),
        "input_gain_b":     _snap(mezza.input_gain.param),
        "output_gain_c":    _snap(mezza.output_gain.param),
        "direct_d":         _snap(mezza.direct.param),
        "mixing_W":         _snap(mezza.mixing_matrix.param),
    }
    np.savez(
        os.path.join(out_dir, "analysis_data.npz"),
        fs=np.array(fs_target),
        **{f"init_{k}": v for k, v in snap_init.items()},
        **{f"final_{k}": v for k, v in snap_final.items()},
    )

    # ---- Save optimised IR ----
    with torch.no_grad():
        optim_ir = model(impulse).detach()
    optim_np = optim_ir[0, :, 0].cpu().numpy()
    target_np = target_ir[0, :, 0].cpu().numpy()
    init_np_final = init_ir[0, :, 0].cpu().numpy()

    fade_samples = min(int(0.02 * fs_target), len(optim_np) // 4)
    if fade_samples > 0:
        optim_np[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    optim_np = optim_np / max(float(np.abs(optim_np).max()), 1e-12)
    sf.write(os.path.join(out_dir, "ir_optim.wav"), optim_np, fs_target, subtype="FLOAT")

    # ---- Loss history CSV ----
    extra_keys = sorted({k for row in loss_history for k in row}
                        - {"step", "total", "EDC", "EDR", "EDP"})
    fieldnames = ["step", "total", "EDC", "EDR", "EDP", *extra_keys]
    with open(os.path.join(out_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(loss_history)

    # ---- T30 comparison CSV (consumed by run_arch_study) ----
    try:
        t30_target = _measure_t30(target_np, fs_target)
        t30_optim = _measure_t30(optim_np, fs_target)
        t30_init = _measure_t30(init_np_final, fs_target)
    except Exception as e:
        t30_target, t30_optim, t30_init = {}, {}, {}
        print(f"  T30 measurement failed: {e}")

    octave_centers = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    with open(os.path.join(out_dir, "t30_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "target_t30", "init_t30", "optim_t30", "error_pct"])
        for cf in octave_centers:
            tt = t30_target.get(cf, float("nan"))
            ti = t30_init.get(cf, float("nan"))
            to_ = t30_optim.get(cf, float("nan"))
            err = (abs(to_ - tt) / tt * 100
                   if (np.isfinite(tt) and np.isfinite(to_) and tt > 0)
                   else float("nan"))
            w.writerow([cf, f"{tt:.4f}", f"{ti:.4f}", f"{to_:.4f}", f"{err:.2f}"])

    # ---- Training summary CSV ----
    total_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    with open(os.path.join(out_dir, "training_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["best_loss", "train_wall_s", "steps_run", "trainable_params"])
        w.writerow([f"{best_loss:.6f}", f"{train_wall_s:.1f}",
                    args.max_iters, total_params])

    # ---- Standard metrics CSV (consumed by run_arch_study) ----
    try:
        from reverb.metrics import compare_rir_metrics
        metrics_dir = os.path.join(out_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        compare_rir_metrics(
            target_path=os.path.join(out_dir, "ir_target.wav"),
            optimized_path=os.path.join(out_dir, "ir_optim.wav"),
            output_dir=metrics_dir,
        )
    except Exception as e:
        print(f"  compare_rir_metrics failed: {e}")

    print(f"\n  Done. best_loss={best_loss:.4f}, wall={train_wall_s:.1f}s, "
          f"out={out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_rir", required=True, type=str,
                   help="Path to target RIR (wav). Resampled to --samplerate if needed.")
    p.add_argument("--train_dir", type=str, default=None,
                   help="Output directory. Defaults to output/<timestamp>_mezza.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu.")
    p.add_argument("--samplerate", type=int, default=16000,
                   help="Internal sample rate (paper: 16 kHz). Input is resampled if needed.")
    p.add_argument("--nfft", type=int, default=None,
                   help="Override auto nfft (in samples at --samplerate).")

    # FDN architecture (paper: N=6, p=63)
    p.add_argument("--n_delays", type=int, default=6,
                   help="Number of delay lines (paper: 6).")
    p.add_argument("--fir_taps", type=int, default=63,
                   help="FIR taps per delay line and for the tone filter (paper: 63).")

    # Delay initialization (paper: Beta(1.1, 6) * psi=1024)
    p.add_argument("--psi", type=int, default=1024,
                   help="Delay scaling psi. At 16 kHz: max delay = psi samples = 64 ms.")
    p.add_argument("--delay_alpha", type=float, default=1.1,
                   help="Beta distribution alpha for delay init (paper: 1.1).")
    p.add_argument("--delay_beta", type=float, default=6.0,
                   help="Beta distribution beta for delay init (paper: 6.0).")

    # Training (paper: 650 gradient steps, LR=0.1/0.001)
    p.add_argument("--max_iters", type=int, default=650,
                   help="Gradient steps (paper: 650).")
    p.add_argument("--lr_struct", type=float, default=0.1,
                   help="Adam LR for structural params W,b,c,d (paper: 0.1).")
    p.add_argument("--lr_fir", type=float, default=0.001,
                   help="Adam LR for FIR tap params H_i,T (paper: 0.001).")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clip norm (<=0 disables).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enforce_stability", action="store_true", default=True,
                   help="After each FIR optimizer step, project the depthwise FIR "
                        "bank so max_f |H_i(f)| <= 1-eps per delay line. Eliminates "
                        "the buzz caused by feedback-loop bands with gain >= 1. "
                        "Default ON.")
    p.add_argument("--no_enforce_stability", dest="enforce_stability",
                   action="store_false",
                   help="Disable the stability projection (paper-faithful behavior).")
    p.add_argument("--stability_eps", type=float, default=1e-3,
                   help="Per-line magnitude headroom (default 1e-3, i.e. cap at "
                        "0.999 ≈ -0.0087 dB).")

    # Loss weights (paper: lambda1=0.5, lambda2=1.0, lambda3=0.1)
    p.add_argument("--w_edc", type=float, default=0.5,
                   help="L_EDC weight (paper: 0.5).")
    p.add_argument("--w_edr", type=float, default=1.0,
                   help="L_EDR (mel-EDR) weight (paper: 1.0).")
    p.add_argument("--w_edp", type=float, default=0.1,
                   help="L_EDP (soft echo density) weight (paper: 0.1).")

    # Mel-EDR STFT params (paper at 16 kHz: n_fft=512, win=320 [20ms], hop=160 [10ms], 64 mels)
    p.add_argument("--mel_nfft", type=int, default=512)
    p.add_argument("--mel_win", type=int, default=320)
    p.add_argument("--mel_hop", type=int, default=160)
    p.add_argument("--mel_bands", type=int, default=64)

    # Soft EDP params (paper: window ~20 ms at 16 kHz = 320 samples, kappa: 10^2 -> 10^5)
    p.add_argument("--edp_win", type=int, default=321,
                   help="Soft EDP window size in samples (must be odd; paper: ~20ms at 16kHz).")
    p.add_argument("--kappa_start", type=float, default=100.0,
                   help="kappa at start of IR (paper: 10^2).")
    p.add_argument("--kappa_end", type=float, default=100000.0,
                   help="kappa at end of IR (paper: 10^5).")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)
