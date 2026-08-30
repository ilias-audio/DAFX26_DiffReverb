import torch 
import os
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pyfar as pf
import pyrato as ra
import pandas as pd
import argparse
from scipy.stats import kurtosis as scipy_kurtosis



class ReverbAnalyzer(torch.nn.Module):
    def __init__(self, RIR_path: str):
        super().__init__()
        self.RIR_path = RIR_path
        self.result_folder = "analysis_results"
        self.eps = self.dB2lin(-140)
        # NOTE (release): the upstream copy eagerly created ./analysis_results in
        # the cwd here. compare_rir_metrics() overrides result_folder to the run's
        # metrics dir (created by the caller), so we defer directory creation and
        # avoid littering the working directory.
        self.load_RIR(RIR_path)

    def lin2dB(self, lin):
        return 20 * torch.log10(lin + self.eps)
    
    def dB2lin(self, dB):
        return 10 ** (dB / 20)

    def load_RIR(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"RIR file not found at {path}")
        rir_waveform, self.sample_rate = torchaudio.load(path)
        rir_waveform = rir_waveform[0] / torch.max(torch.abs(rir_waveform[0]))
        signal = pf.Signal(rir_waveform.numpy(), self.sample_rate)
        initial_delay = pf.dsp.find_impulse_response_start(signal)
        self.rir = pf.dsp.time_shift(signal, -initial_delay, 'linear')


    def plot_ir(self, signal,  title: str = "Impulse Response", filename: str = "impulse_response", log_scale: bool = True):
        if log_scale:
            plt.figure(figsize=(10, 4))
            rir_magnitude = torch.abs(signal)
            rir_db = self.lin2dB(rir_magnitude)
            plt.plot(rir_db.numpy())
            plt.title(f"{title} in dB")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (dB)")
            plt.grid()
            plt.savefig(os.path.join(self.result_folder, f"{filename}_dB.png"))
            plt.close()
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(signal)
            plt.title(title)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.savefig(os.path.join(self.result_folder, f"{filename}.png"))
            plt.close()
    
    def compute_octave_bands(self, band_per_octave = 1):
        self.rir_filterbank = pf.dsp.filter.fractional_octave_bands(self.rir, num_fractions=band_per_octave)


    def compute_energy_decay_curves(self, band_per_octave = 1):
        self.EDC = np.empty((0, 0), dtype=float)
        self.EDC_center_frequencies = np.array([], dtype=float)
        self.EDC_band_indices = np.array([], dtype=int)

        edc_list = []
        freq_list = []
        band_idx_list = []
        for band in range(self.rir_filterbank.cshape[0]):
            center_frequency = pf.dsp.filter.fractional_octave_frequencies(num_fractions=band_per_octave)[0][band]
            edc = None
            # Try truncation method first, then Chu, then raw Schroeder
            for _, method_fn in [
                ("truncation", lambda sig: ra.energy_decay_curve_truncation(sig)),
                ("chu", lambda sig: ra.energy_decay_curve_chu(sig)),
            ]:
                try:
                    edc = method_fn(self.rir_filterbank[band])
                    break
                except Exception:
                    continue
            if edc is None:
                # Last resort: raw backward integration (no noise compensation)
                try:
                    sig = np.squeeze(self.rir_filterbank[band].time)
                    raw_edc = np.cumsum(sig[::-1] ** 2)[::-1]
                    raw_edc = raw_edc / (raw_edc[0] + 1e-30)
                    edc = pf.TimeData(raw_edc[np.newaxis, :], [1.0 / self.sample_rate * i for i in range(len(raw_edc))])
                    print(f"  Band {center_frequency:.0f} Hz: using raw Schroeder fallback")
                except Exception as e:
                    print(f"Error processing band[{band}] {center_frequency} Hz: {e}")
                    continue

            edc_list.append(np.asarray(edc))
            freq_list.append(float(center_frequency))
            band_idx_list.append(int(band))

        if len(edc_list) > 0:
            # (n_bands, n_samples)
            self.EDC = np.stack(edc_list, axis=0)
            self.EDC_center_frequencies = np.asarray(freq_list, dtype=float)
            self.EDC_band_indices = np.asarray(band_idx_list, dtype=int)

        print(f"EDC computed for {len(edc_list)}/{self.rir_filterbank.cshape[0]} bands.")
    
    def measure_reverberation_time(self):
        measurements_type = ['T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'EDT', 'LDT']
        if self.EDC.size == 0 or self.EDC.shape[0] == 0:
            print("No valid octave bands for EDC/RT measurement (likely low SNR).")
            self.rt_table = pd.DataFrame()
            return
        self.rt_table = pd.DataFrame()
        self.rt_table = pd.DataFrame(columns= self.EDC_center_frequencies.tolist())
        for type in measurements_type:
            for band_i in range(self.EDC.shape[0]):
                try:
                    measurement_time = ra.reverberation_time_linear_regression(
                        self.EDC[band_i], T=type).round(3)
                except Exception:
                    # pyrato raises "All-NaN slice" when the EDC doesn't span
                    # the dB range required for this T value (common on short
                    # rooms where T60 needs more decay than the IR contains).
                    # Record NaN and keep going so downstream metrics still work.
                    measurement_time = float("nan")
                self.rt_table.at[type, self.EDC_center_frequencies[band_i]] = measurement_time


        clarity = {'E05': 0.005, 'E10': 0.01, 'E15': 0.015, 'E20': 0.02, 'E30': 0.03, 'E50': 0.05, 'E80': 0.08, 'E2K': 2.0}
        for time in clarity.keys():
            for band_i, band in enumerate(self.EDC_band_indices.tolist()):
                self.rt_table.at[time, self.EDC_center_frequencies[band_i]] = self.compute_energy(
                    self.rir_filterbank[band],
                    time_in_s=clarity[time],
                ).round(3)

        for band_i, band in enumerate(self.EDC_band_indices.tolist()):
            c80, d50, ts = self.compute_iso3382_metrics(self.rir_filterbank[band])
            self.rt_table.at['C80', self.EDC_center_frequencies[band_i]] = round(c80, 2)
            self.rt_table.at['D50', self.EDC_center_frequencies[band_i]] = round(d50 * 100, 2) # in %
            self.rt_table.at['Ts', self.EDC_center_frequencies[band_i]] = round(ts, 2) # in ms

        print("Reverberation time measurements completed.")

    def export_time_measurements_to_csv(self, filename: str):
        csv_path = os.path.join(self.result_folder, filename)
        self.rt_table.to_csv(csv_path)
        print(f"Reverberation time measurements exported to {csv_path}")


    def plot_octave_bands(self):
        plt.figure(figsize=(10, 4))
        plt.semilogx(self.freq.numpy(), self.spl.numpy())
        plt.title("Octave Band Levels")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.xlim([20, 20000])
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(self.result_folder, "octave_band_levels.png"))
        plt.close()

    def plot_band_signals(self):
        for i in range(self.xb.shape[0]):
            self.plot_ir(self.xb[i], title=f"Octave Band {self.freq[i].item():.1f} Hz Signal", filename=f"octave_band_{i+1}_signal")
            shroeder = self.compute_shroeder(self.xb[i])
            self.plot_ir(shroeder, title=f"Schroeder Curve for {self.freq[i].item():.1f} Hz Band", filename=f"schroeder_curve_band_{i+1}")

    def plot_surface_EDR(self): # this is shit at the moment
        EDR = self.lin2dB(self.compute_shroeder(self.xb))
        # X, Y = np.meshgrid(np.arange(EDR.shape[1]), self.freq.numpy())
        fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(111)
        # ax.contourf(Y, X, EDR.numpy(), cmap='viridis')
        # ax.contourf(np.arange(EDR.shape[1]),self.freq.numpy(), EDR.numpy(), cmap='viridis')
        # ax.set_yscale("log")
        # ax.set_xlim(0, 20e3)
        # ax.set_ylim(20, 20e3)
        plt.imshow(EDR, aspect='auto', origin='lower', extent=[0, EDR.shape[1], 20, 20e3])
        plt.title("Energy Decay Relief (EDR)")
        plt.xlabel("Samples")
        plt.ylabel("Frequency (Hz)")

        plt.savefig(os.path.join(self.result_folder, "energy_decay_relief.png"))
        plt.close()

    def compute_energy(self, signal, time_in_s: float):
        # here we compute the energy rather than the clarity. Clarity is a ratio between
        # early and late energy. We only compute the early energy here.
        # that gives us a measure in dB that can be compared across RIRs.
        split_index = int(time_in_s * self.sample_rate)
        if split_index >= signal.n_samples:
            print(f"Time for energy computation {time_in_s}s exceeds signal length {signal.n_samples/self.sample_rate}s. Using full signal length instead.")
            split_index = signal.n_samples - 1
        initial_delay = pf.dsp.find_impulse_response_start(signal)
        signal_no_delay = pf.dsp.time_shift(signal, -initial_delay, 'linear')


        signal_early = pf.Signal(signal_no_delay.time[0,:split_index], self.sample_rate)
        # signal_late  = pf.Signal(signal_no_delay.time[0,split_index:], self.sample_rate)

        energy_early = pf.dsp.energy(signal_early)
        # energy_late = pf.dsp.energy(signal_late)

        return 10 * np.log10(energy_early)

    def compute_shroeder(self, signal):
        sig_energy = torch.cumsum(signal.flip(dims=[0]) ** 2, dim=0).flip(dims=[0])
        return sig_energy

    def compute_iso3382_metrics(self, signal):
        ir_np = np.squeeze(signal.time)
        fs = self.sample_rate
        e_tot = np.sum(ir_np**2) + 1e-12
        
        # C80 (Clarity)
        i80 = int(0.08 * fs)
        e80 = np.sum(ir_np[:i80]**2)
        c80 = 10 * np.log10((e80 + 1e-12) / (e_tot - e80 + 1e-12))
        
        # D50 (Definition)
        i50 = int(0.05 * fs)
        e50 = np.sum(ir_np[:i50]**2)
        d50 = e50 / e_tot
        
        # Center Time (Ts) in ms
        t = np.arange(len(ir_np)) / fs
        ts = np.sum(t * ir_np**2) / e_tot * 1000.0
        
        return float(c80), float(d50), float(ts)

def compute_echo_density_profile(rir_signal, fs, window_ms=10.0, hop_ms=2.0):
    if hasattr(rir_signal, 'time'):
        rir = np.squeeze(rir_signal.time)
    else:
        rir = np.asarray(rir_signal).squeeze()
    win_samples = int(window_ms * fs / 1000)
    hop_samples = max(1, int(hop_ms * fs / 1000))
    n_frames = (len(rir) - win_samples) // hop_samples
    if n_frames < 1:
        return np.array([0.0]), np.array([0.0])
    time_axis = np.zeros(n_frames)
    ned_profile = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        segment = rir[start:start + win_samples]
        time_axis[i] = (start + win_samples / 2) / fs
        kurt = scipy_kurtosis(segment, fisher=False)
        if kurt > 0:
            ned_profile[i] = min(1.0, 3.0 / kurt)
        else:
            ned_profile[i] = 0.0
    return time_axis, ned_profile

def estimate_mixing_time(ned_profile, time_axis, threshold=0.85):
    count = 0
    for i in range(len(ned_profile)):
        if ned_profile[i] >= threshold:
            count += 1
            if count >= 3:
                return time_axis[i - 2] * 1000.0  # ms
        else:
            count = 0
    above_half = time_axis[ned_profile > 0.5]
    if len(above_half) > 0:
        return float(np.percentile(above_half, 80)) * 1000.0  # ms
    return 30.0  # fallback

def compute_drr(ir_signal, fs, t_direct_ms=2.5):
    """
    Compute Direct-to-Reverberant Ratio (DRR) in dB.
    DRR = 10 * log10(E_direct / E_reverberant)

    Args:
        ir_signal: pyfar Signal (onset-aligned) or 1D numpy array
        fs: sample rate
        t_direct_ms: direct sound window in ms (ISO 3382 uses 2.5ms)
    Returns:
        drr_db: float, DRR in dB
    """
    if hasattr(ir_signal, 'time'):
        ir_np = np.squeeze(ir_signal.time)
    else:
        ir_np = np.asarray(ir_signal).squeeze()

    n_direct = int(t_direct_ms * fs / 1000)
    n_direct = min(n_direct, len(ir_np))

    e_direct = np.sum(ir_np[:n_direct] ** 2)
    e_reverb = np.sum(ir_np[n_direct:] ** 2)

    eps = 1e-10
    drr_db = 10 * np.log10((e_direct + eps) / (e_reverb + eps))
    return float(drr_db)


def compute_edr(ir, fs, nfft=4096, hop=None):
    """Energy Decay Relief surface in dB, normalised to 0 dB at the onset per bin.

    Returns:
        edr_db: ndarray [K, T] EDR in dB
        freqs:  ndarray [K] frequency axis (Hz)
        times:  ndarray [T] time axis (s)
    """
    if hasattr(ir, "time"):
        x = np.squeeze(ir.time)
    else:
        x = np.asarray(ir).squeeze()
    x = torch.as_tensor(x, dtype=torch.float32)
    if hop is None:
        hop = nfft // 2
    win = torch.hann_window(nfft)
    spec = torch.stft(x, n_fft=nfft, hop_length=hop, window=win, return_complex=True)
    power = (spec.abs() ** 2)  # [K, T]
    edc = torch.flip(torch.cumsum(torch.flip(power, [-1]), dim=-1), [-1])
    eps = 1e-12
    edr_db = 10 * torch.log10(edc / (edc[:, :1] + eps) + eps)
    freqs = np.linspace(0, fs / 2, edr_db.shape[0])
    times = np.arange(edr_db.shape[1]) * hop / fs
    return edr_db.numpy(), freqs, times


def compute_edr_mae(target_ir, optimized_ir, fs, nfft=4096, hop=None,
                    db_lo=-5.0, db_hi=-35.0, f_lo=20.0, f_hi=None):
    """Mean absolute EDR error in dB within the [db_hi, db_lo] dB range of the target.

    Uses a soft sigmoid mask matching the training loss (slope 5).
    Returns: (mae_db, edr_target_db, edr_optimized_db, freqs, times).
    """
    edr_tgt, freqs, times = compute_edr(target_ir, fs, nfft=nfft, hop=hop)
    edr_opt, _, _ = compute_edr(optimized_ir, fs, nfft=nfft, hop=hop)

    # Align to shorter time axis
    T = min(edr_tgt.shape[1], edr_opt.shape[1])
    edr_tgt = edr_tgt[:, :T]
    edr_opt = edr_opt[:, :T]
    times = times[:T]

    # Frequency-bin restriction
    if f_hi is None:
        f_hi = fs / 2
    k_lo = max(1, int(f_lo / (fs / 2) * (edr_tgt.shape[0] - 1)))
    k_hi = min(edr_tgt.shape[0], int(f_hi / (fs / 2) * (edr_tgt.shape[0] - 1)))

    edr_tgt_band = edr_tgt[k_lo:k_hi]
    edr_opt_band = edr_opt[k_lo:k_hi]

    # Soft sigmoid mask on target EDR (matches training mask). Clip the
    # logits to keep np.exp from overflowing for EDR values far outside the
    # band — the sigmoid saturates either way.
    beta = 5.0
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    mask = _sigmoid(beta * (edr_tgt_band - db_hi)) * \
           _sigmoid(beta * (db_lo - edr_tgt_band))

    diff = np.abs(edr_opt_band - edr_tgt_band)
    mae = float((diff * mask).sum() / (mask.sum() + 1e-10))
    return mae, edr_tgt, edr_opt, freqs, times


def _as_array(ir) -> np.ndarray:
    """Coerce a pyrato Signal (has .time) or plain array to a 1-D numpy array."""
    if hasattr(ir, "time"):
        return np.asarray(np.squeeze(ir.time))
    return np.asarray(ir).squeeze()


def compute_spectral_centroid(ir, fs, nfft: int = 4096, hop: int | None = None) -> float:
    """Energy-weighted time-averaged spectral centroid of |STFT(ir)|. Returns Hz.

    Per-frame centroid is the magnitude-weighted mean frequency; the time
    average is weighted by per-frame energy so silent tails do not pollute
    the result.
    """
    if hop is None:
        hop = nfft // 4
    x = _as_array(ir).flatten().astype(np.float32)
    if len(x) < nfft:
        return float("nan")
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
    spec = np.abs(np.fft.rfft(frames * win, axis=1))  # (T, F)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    eps = 1e-12
    centroid_per_frame = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + eps)
    energy_per_frame = (spec ** 2).sum(axis=1)
    total_energy = float(energy_per_frame.sum() + eps)
    return float((centroid_per_frame * energy_per_frame).sum() / total_energy)


def compute_strength_g(ir, fs, reference_ir=None) -> float:
    """ISO 3382 sound strength G (broadband).

    Without reference_ir: returns relative G = 10·log10(∫|h|² dt). Two RIRs'
    values can be subtracted to compare strengths (absolute level cancels).

    With reference_ir: returns G in dB referenced to the reference IR's energy
    (typically a 10 m free-field response per ISO 3382-1).
    """
    x = _as_array(ir).flatten()
    energy = float(np.sum(x.astype(np.float64) ** 2))
    if reference_ir is None:
        return 10.0 * np.log10(energy + 1e-30)
    ref = _as_array(reference_ir).flatten()
    ref_energy = float(np.sum(ref.astype(np.float64) ** 2))
    return 10.0 * np.log10((energy + 1e-30) / (ref_energy + 1e-30))


def compute_spectral_ot(target_ir, optim_ir, fs, nfft: int = 4096) -> float:
    """1D Wasserstein distance between target and optim PSD (Welch-averaged).

    PSDs are normalised to unit mass before comparison so the result captures
    SHAPE differences in the spectrum (not absolute level). Returns the
    Wasserstein distance in Hz (lower = better spectral match).
    """
    from scipy.stats import wasserstein_distance

    def _psd(x):
        x = _as_array(x).flatten().astype(np.float32)
        nf = min(nfft, len(x))
        win = np.hanning(nf).astype(np.float32)
        hop = nf // 2
        n = max(1, 1 + (len(x) - nf) // hop)
        psd_sum = np.zeros(nf // 2 + 1)
        for i in range(n):
            seg = x[i * hop:i * hop + nf]
            if len(seg) < nf:
                break
            psd_sum += np.abs(np.fft.rfft(seg * win, n=nf)) ** 2
        return psd_sum / max(1, n)

    p_tgt = _psd(target_ir)
    p_opt = _psd(optim_ir)
    nf = (len(p_tgt) - 1) * 2
    freqs = np.fft.rfftfreq(nf, 1.0 / fs)
    eps = 1e-30
    p_tgt_norm = p_tgt / (p_tgt.sum() + eps)
    p_opt_norm = p_opt / (p_opt.sum() + eps)
    return float(wasserstein_distance(freqs, freqs,
                                       u_weights=p_tgt_norm, v_weights=p_opt_norm))


def compare_rir_metrics(target_path, optimized_path, output_dir):
    """Run ReverbAnalyzer on target and optimized RIRs, produce comparison CSV and plot."""

    # Analyze both RIRs
    target = ReverbAnalyzer(target_path)
    target.result_folder = output_dir
    target.compute_octave_bands()
    target.compute_energy_decay_curves()
    target.measure_reverberation_time()

    optimized = ReverbAnalyzer(optimized_path)
    optimized.result_folder = output_dir
    optimized.compute_octave_bands()
    optimized.compute_energy_decay_curves()
    optimized.measure_reverberation_time()

    # Find common frequency bands
    common_freqs = sorted(set(target.EDC_center_frequencies) & set(optimized.EDC_center_frequencies))
    if len(common_freqs) == 0:
        print("Warning: no common frequency bands between target and optimized RIR.")
        return None

    # Build comparison table for key metrics
    metrics = ['T30', 'EDT', 'C80', 'D50', 'Ts', 'E50', 'E80']
    rows = []
    for metric in metrics:
        if metric not in target.rt_table.index or metric not in optimized.rt_table.index:
            continue
        for freq in common_freqs:
            t_val = target.rt_table.at[metric, freq]
            o_val = optimized.rt_table.at[metric, freq]
            if pd.isna(t_val) or pd.isna(o_val):
                continue
            t_val = float(np.squeeze(t_val))
            o_val = float(np.squeeze(o_val))
            if np.isnan(t_val) or np.isnan(o_val) or np.isinf(t_val) or np.isinf(o_val):
                continue
            error = o_val - t_val
            pct = (error / t_val * 100) if t_val != 0 else float('nan')
            rows.append({
                'Metric': metric,
                'Frequency (Hz)': freq,
                'Target': round(t_val, 4),
                'Optimized': round(o_val, 4),
                'Error': round(error, 4),
                'Error (%)': round(pct, 1),
            })

    # Compute broadband DRR
    drr_tgt = compute_drr(target.rir, target.sample_rate)
    drr_opt = compute_drr(optimized.rir, optimized.sample_rate)
    drr_err = drr_opt - drr_tgt
    drr_pct = (drr_err / abs(drr_tgt) * 100) if drr_tgt != 0 else float('nan')
    rows.append({
        'Metric': 'DRR',
        'Frequency (Hz)': 'broadband',
        'Target': round(drr_tgt, 2),
        'Optimized': round(drr_opt, 2),
        'Error': round(drr_err, 2),
        'Error (%)': round(drr_pct, 1),
    })
    print(f"  DRR — Target: {drr_tgt:.2f} dB, Optimized: {drr_opt:.2f} dB, Error: {drr_err:+.2f} dB")

    # Compute Echo Density and Mixing Time
    t_axis_tgt, ned_tgt = compute_echo_density_profile(target.rir, target.sample_rate)
    tmix_tgt = estimate_mixing_time(ned_tgt, t_axis_tgt)
    
    t_axis_opt, ned_opt = compute_echo_density_profile(optimized.rir, optimized.sample_rate)
    tmix_opt = estimate_mixing_time(ned_opt, t_axis_opt)
    
    tmix_err = tmix_opt - tmix_tgt
    tmix_pct = (tmix_err / tmix_tgt * 100) if tmix_tgt != 0 else float('nan')
    rows.append({
        'Metric': 'Tmix',
        'Frequency (Hz)': 'broadband',
        'Target': round(tmix_tgt, 2),
        'Optimized': round(tmix_opt, 2),
        'Error': round(tmix_err, 2),
        'Error (%)': round(tmix_pct, 1),
    })
    
    min_frames = min(len(ned_tgt), len(ned_opt))
    ned_mae = float(np.mean(np.abs(ned_opt[:min_frames] - ned_tgt[:min_frames]))) if min_frames > 0 else float('nan')
    rows.append({
        'Metric': 'NED_MAE',
        'Frequency (Hz)': 'broadband',
        'Target': 0.0,
        'Optimized': round(ned_mae, 4),
        'Error': round(ned_mae, 4),
        'Error (%)': float('nan'),
    })
    print(f"  Tmix — Target: {tmix_tgt:.1f} ms, Optimized: {tmix_opt:.1f} ms, Error: {tmix_err:+.1f} ms")

    # EDR mean-absolute-error (dB) within the [-35, -5] dB range of the target.
    try:
        edr_mae, _, _, _, _ = compute_edr_mae(
            target.rir, optimized.rir, target.sample_rate
        )
        rows.append({
            'Metric': 'EDR_MAE',
            'Frequency (Hz)': 'broadband',
            'Target': 0.0,
            'Optimized': round(edr_mae, 3),
            'Error': round(edr_mae, 3),
            'Error (%)': float('nan'),
        })
        print(f"  EDR_MAE — {edr_mae:.3f} dB (target vs optimized, within [-35, -5] dB)")
    except Exception as e:
        print(f"  EDR_MAE failed: {e}")

    # ---- New metrics: spectral centroid, strength G, spectral OT ----
    try:
        sc_tgt = compute_spectral_centroid(target.rir, target.sample_rate)
        sc_opt = compute_spectral_centroid(optimized.rir, optimized.sample_rate)
        rows.append({
            'Metric': 'SpectralCentroid',
            'Frequency (Hz)': 'broadband',
            'Target': round(sc_tgt, 1),
            'Optimized': round(sc_opt, 1),
            'Error': round(sc_opt - sc_tgt, 1),
            'Error (%)': round((sc_opt - sc_tgt) / sc_tgt * 100, 1) if sc_tgt else float('nan'),
        })
        print(f"  SpectralCentroid — Target: {sc_tgt:.1f} Hz, Optimized: {sc_opt:.1f} Hz, Error: {sc_opt-sc_tgt:+.1f} Hz")
    except Exception as e:
        print(f"  SpectralCentroid failed: {e}")

    try:
        # Relative G — method-to-method comparison.
        g_tgt = compute_strength_g(target.rir, target.sample_rate)
        g_opt = compute_strength_g(optimized.rir, optimized.sample_rate)
        rows.append({
            'Metric': 'StrengthG_rel',
            'Frequency (Hz)': 'broadband',
            'Target': round(g_tgt, 2),
            'Optimized': round(g_opt, 2),
            'Error': round(g_opt - g_tgt, 2),
            'Error (%)': float('nan'),
        })
        print(f"  StrengthG_rel — Target: {g_tgt:+.2f} dB, Optimized: {g_opt:+.2f} dB, Error: {g_opt-g_tgt:+.2f} dB")
    except Exception as e:
        print(f"  StrengthG_rel failed: {e}")

    try:
        sot = compute_spectral_ot(target.rir, optimized.rir, target.sample_rate)
        rows.append({
            'Metric': 'SpectralOT',
            'Frequency (Hz)': 'broadband',
            'Target': 0.0,
            'Optimized': round(sot, 1),
            'Error': round(sot, 1),
            'Error (%)': float('nan'),
        })
        print(f"  SpectralOT — {sot:.1f} Hz (1D Wasserstein on normalized PSD)")
    except Exception as e:
        print(f"  SpectralOT failed: {e}")

    comparison = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'metrics_comparison.csv')
    comparison.to_csv(csv_path, index=False)
    print(f"Metrics comparison saved to {csv_path}")

    # Also save full tables
    target.export_time_measurements_to_csv('metrics_target.csv')
    optimized.export_time_measurements_to_csv('metrics_optimized.csv')

    # Plot T30, EDT, C80, DRR, and Echo Density comparison
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, metric in zip(axes[:3], ['T30', 'EDT', 'C80']):
        sub = comparison[comparison['Metric'] == metric]
        if sub.empty:
            ax.set_title(f'{metric} — no data')
            continue
        freqs = sub['Frequency (Hz)'].values
        ax.plot(freqs, sub['Target'].values, 'o-', label='Target', color='tab:blue')
        ax.plot(freqs, sub['Optimized'].values, 's--', label='Optimized', color='tab:red')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        if metric == 'C80':
            ax.set_ylabel('Clarity (dB)')
        else:
            ax.set_ylabel('Time (s)')
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # DRR bar chart
    ax_drr = axes[3]
    bars = ax_drr.bar(['Target', 'Optimized'], [drr_tgt, drr_opt],
                       color=['tab:blue', 'tab:red'], alpha=0.8, width=0.5)
    ax_drr.set_ylabel('DRR (dB)')
    ax_drr.set_title(f'DRR (error: {drr_err:+.1f} dB)')
    ax_drr.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, [drr_tgt, drr_opt]):
        ax_drr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f} dB', ha='center', va='bottom', fontsize=10)

    # Echo Density Profile plot
    ax_ned = axes[4]
    ax_ned.plot(t_axis_tgt * 1000, ned_tgt, color='tab:blue', label=f'Target ($T_{{mix}}$={tmix_tgt:.1f}ms)')
    ax_ned.plot(t_axis_opt * 1000, ned_opt, color='tab:red', linestyle='--', label=f'Optimized ($T_{{mix}}$={tmix_opt:.1f}ms)')
    ax_ned.set_xlabel('Time (ms)')
    ax_ned.set_ylabel('Normalized Echo Density')
    ax_ned.set_title(f'Echo Density (MAE: {ned_mae:.3f})')
    ax_ned.legend()
    ax_ned.grid(True, alpha=0.3)
    ax_ned.set_xlim(0, min(200, t_axis_tgt[-1]*1000 if len(t_axis_tgt)>0 else 200))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"Metrics comparison plot saved to metrics_comparison.png")

    # Print summary
    for metric in ['T30', 'EDT', 'C80']:
        sub = comparison[comparison['Metric'] == metric]
        if not sub.empty:
            mae = sub['Error'].abs().mean()
            unit = "dB" if metric == 'C80' else "s"
            print(f"  {metric} — Mean Absolute Error: {mae:.4f} {unit}")

    return comparison


def measure_rt60_continuous(ir, fs, nfft=4096, hop=None, t_val=30):
    """Measure RT60 at every STFT frequency bin via backward Schroeder integration.

    Computes per-bin EDC from the STFT power spectrogram, then fits a slope
    over the top `t_val` dB of the decay.  Returns reverberation time estimated
    via T<t_val> (ISO 3382-1): linear regression over [-5, -5-t_val] dB of
    the EDC, extrapolated to a 60 dB decay.

    Args:
        ir: 1-D numpy array, room impulse response
        fs: sample rate (Hz)
        nfft: STFT window length (default 4096)
        hop: STFT hop size (default nfft//4)
        t_val: The 'T' value to calculate (e.g., 10, 20, 30). The fit window
               is from -5 dB to (-5 - t_val) dB.

    Returns:
        freq_axis: (nfft//2+1,) array of frequency bins in Hz
        rt60: (nfft//2+1,) array of RT60 in seconds (NaN where fit failed)
    """
    from scipy.signal import stft as scipy_stft

    if hop is None:
        hop = nfft // 4
        
    db_lo = -5.0
    db_hi = -5.0 - t_val

    ir = np.asarray(ir, dtype=float).squeeze()
    f, t, Zxx = scipy_stft(ir, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    # Zxx: (n_freq, n_frames)
    power = np.abs(Zxx) ** 2

    n_freq = power.shape[0]
    rt60 = np.full(n_freq, np.nan)

    for fi in range(n_freq):
        p = power[fi, :]
        edc = np.cumsum(p[::-1])[::-1]
        edc_max = edc[0]
        if edc_max < 1e-30:
            continue
        edc_db = 10.0 * np.log10(edc / edc_max + 1e-30)

        mask = (edc_db <= db_lo) & (edc_db >= db_hi)
        if mask.sum() < 2:
            continue

        coeffs = np.polyfit(t[mask], edc_db[mask], 1)
        slope = coeffs[0]  # dB/s
        if slope >= 0:
            continue
        rt60[fi] = -60.0 / slope

    return f, rt60


def measure_rt60_decayfitnet(ir, fs, filter_frequencies=None):
    """Estimate RT60 per octave band using the DecayFitNet neural-network estimator.

    Wraps DecayFitNetToolbox (DecayFitNet/ submodule). Falls back to raising
    ImportError if the submodule is unavailable.

    Args:
        ir: 1-D numpy array, room impulse response
        fs: sample rate (Hz)
        filter_frequencies: list of octave-band centre frequencies (Hz).
            Defaults to [63, 125, 250, 500, 1000, 2000, 4000, 8000] filtered
            to those whose upper edge is below the Nyquist frequency.

    Returns:
        filter_frequencies: list of band centre frequencies used
        t60: (n_bands,) numpy array of T60 estimates in seconds
    """
    import sys
    import os
    import torch

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dfn_python_path = os.path.join(repo_root, "DecayFitNet", "python")
    if dfn_python_path not in sys.path:
        sys.path.insert(0, dfn_python_path)

    try:
        from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
    except ImportError as exc:
        raise ImportError(
            f"DecayFitNet not available (expected at {dfn_python_path}): {exc}"
        ) from exc

    ir = np.asarray(ir, dtype=float).squeeze()

    if filter_frequencies is None:
        all_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        filter_frequencies = [f for f in all_bands if f * 2 ** 0.5 < fs / 2]

    dfn = DecayFitNetToolbox(n_slopes=1, sample_rate=fs,
                             filter_frequencies=filter_frequencies)
    ir_tensor = torch.from_numpy(ir).float().unsqueeze(0)  # [1, n_samples]
    [t_vals, _a_vals, _n_vals], _ = dfn.estimate_parameters(
        ir_tensor, analyse_full_rir=True
    )
    if isinstance(t_vals, torch.Tensor):
        t60_per_band = t_vals[:, 0].numpy()  # first (only) slope per band
    else:
        t60_per_band = np.array(t_vals)[:, 0]

    return filter_frequencies, t60_per_band


def measure_snr_continuous(ir, fs, nfft=4096, hop=None, t60=None, t_multiplier=2.0):
    """Measure SNR at every STFT frequency bin using a robust noise estimator.

    Estimates noise floor by looking at the lowest percentiles of the active signal
    envelope, ignoring zero-padded tails.

    Args:
        ir: 1-D numpy array, room impulse response
        fs: sample rate (Hz)
        nfft: STFT window length (default 4096)
        hop: STFT hop size (default nfft//4)
        t60: (Ignored, kept for backward compatibility)
        t_multiplier: (Ignored, kept for backward compatibility)

    Returns:
        freq_axis: (nfft//2+1,) array of frequency bins in Hz
        snr_db: (nfft//2+1,) array of SNR in dB
    """
    from scipy.signal import stft as scipy_stft

    if hop is None:
        hop = nfft // 4

    ir = np.asarray(ir, dtype=float).squeeze()
    f, t, Zxx = scipy_stft(ir, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    power = np.abs(Zxx) ** 2

    n_freq = power.shape[0]
    snr_db = np.full(n_freq, np.nan)

    for fi in range(n_freq):
        p = power[fi, :]
        p_max = np.max(p)
        if p_max < 1e-30:
            continue
            
        # Ignore extreme digital silence / zero padding (> 120 dB below max)
        valid_p = p[p > 1e-12 * p_max]
        
        if len(valid_p) > 10:
            # The 10th percentile of the non-silent signal is a robust noise floor
            noise_floor = np.percentile(valid_p, 10)
        else:
            noise_floor = 0.0
            
        snr_db[fi] = 10.0 * np.log10(p_max / (noise_floor + 1e-30))

    return f, snr_db



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze room impulse responses.")
    parser.add_argument("rir_path", type=str, help="Path to the RIR file.")
    args = parser.parse_args()

    analyzer = ReverbAnalyzer(args.rir_path)
    # analyzer.plot_ir(analyzer.rir.time, title="Impulse Response after Pre-delay Removal", filename="impulse_response_no_predelay")
    analyzer.compute_octave_bands()
    analyzer.compute_energy_decay_curves()
    analyzer.measure_reverberation_time()
    analyzer.export_time_measurements_to_csv(os.path.splitext(os.path.basename(args.rir_path))[-2] + "_reverberation_times.csv")
    
    # analyzer.plot_octave_bands()
    # analyzer.plot_band_signals()
    # analyzer.plot_surface_EDR()

def measure_rt60_fractional_pyrato(ir, fs, num_fractions=6, t_val=30):
    """Estimate RT60 per fractional octave band using pyrato (Chu/Truncation).
    
    This provides a continuous-like robust curve without the noise issues of raw STFT.

    Args:
        ir: 1-D numpy array, room impulse response
        fs: sample rate (Hz)
        num_fractions: number of fractions per octave (e.g. 3 or 6)
        t_val: T value to measure (e.g. 20 or 30)

    Returns:
        freqs: (n_bands,) array of center frequencies
        rt60: (n_bands,) array of measured RT60 values (NaN where fit failed)
    """
    import pyfar as pf
    import pyrato as ra
    
    ir_np = np.asarray(ir, dtype=float).squeeze()
    
    # Convert to pyfar Signal
    signal = pf.Signal(ir_np, fs)
    
    # Remove pre-delay
    initial_delay = pf.dsp.find_impulse_response_start(signal)
    signal_no_delay = pf.dsp.time_shift(signal, -initial_delay, 'linear')
    
    # Filterbank
    filterbank = pf.dsp.filter.fractional_octave_bands(
        signal_no_delay, 
        num_fractions=num_fractions, 
        frequency_range=(20, 20000)
    )
    
    center_freqs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=num_fractions, frequency_range=(20, 20000)
    )[1]
    
    n_bands = filterbank.cshape[0]
    rt60_vals = np.full(n_bands, np.nan)
    
    for b in range(n_bands):
        band_sig = filterbank[b]
        
        edc = None
        # Try Truncation first, then Chu
        for method_name, method_fn in [
            ("truncation", lambda sig: ra.energy_decay_curve_truncation(sig)),
            ("chu", lambda sig: ra.energy_decay_curve_chu(sig))
        ]:
            try:
                edc = method_fn(band_sig)
                break
            except Exception:
                continue
                
        if edc is not None:
            try:
                t_str = f'T{t_val}'
                rt = ra.reverberation_time_linear_regression(edc, T=t_str)
                rt60_vals[b] = float(np.squeeze(rt))
            except Exception:
                pass

    return np.asarray(center_freqs), rt60_vals

def _robust_smooth_rt60(rt60, narrow=11, wide=31, max_dev=2.5):
    """Remove noise-floor spikes from per-bin RT60 measurements.

    1. Narrow median filter (size ``narrow``) removes isolated spikes.
    2. Wide median filter (size ``wide``) estimates the local trend.
    3. Bins where the narrow-filtered value exceeds ``max_dev`` times the
       local trend are rejected (set to NaN).

    Args:
        rt60: 1-D numpy array of RT60 values (may contain NaN).
        narrow: kernel size for the spike-removal median.
        wide: kernel size for the local-trend median.
        max_dev: maximum ratio of value/trend before rejection.

    Returns:
        Cleaned copy of *rt60* with rejected bins set to NaN.
    """
    from scipy.ndimage import median_filter

    out = rt60.copy()
    valid = np.isfinite(out)
    if valid.sum() < wide:
        return out

    # Fill NaN with global median (not 0) to avoid distorting edges
    global_med = np.nanmedian(out)
    filled = np.where(valid, out, global_med)

    narrow_med = median_filter(filled, size=narrow, mode='reflect')
    wide_med = median_filter(filled, size=wide, mode='reflect')

    # Reject bins that are far above the local trend
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(wide_med > 0, narrow_med / wide_med, 1.0)
    reject = valid & (ratio > max_dev)
    out[reject] = np.nan

    # Replace surviving values with the narrow-median-filtered version
    still_valid = valid & ~reject
    out[still_valid] = narrow_med[still_valid]

    return out


def measure_rt60_stft_pyrato(ir, fs, nfft=4096, hop=None, t_val=30,
                             robust=False):
    """Estimate reverberation time per STFT frequency bin using pyrato
    (Chu noise compensation).

    Returns RT estimated via T<t_val> per ISO 3382-1: linear regression
    over [-5, -5-t_val] dB of the EDC, extrapolated to a 60 dB decay.

    Args:
        ir: 1-D numpy array, room impulse response
        fs: sample rate (Hz)
        nfft: STFT window length (default 4096)
        hop: STFT hop size (default nfft//4)
        t_val: T value to measure (e.g. 20 or 30)
        robust: if True, use tighter SNR/sanity thresholds and apply
                post-processing spike removal via ``_robust_smooth_rt60``.

    Returns:
        freqs: (nfft//2+1,) array of frequency bins in Hz
        rt60: (nfft//2+1,) array of measured RT60 values (NaN where fit failed)
    """
    from scipy.signal import stft as scipy_stft
    import pyfar as pf
    import pyrato as ra

    if hop is None:
        hop = nfft // 4

    ir = np.asarray(ir, dtype=float).squeeze()

    f, t, Zxx = scipy_stft(ir, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    power = np.abs(Zxx) ** 2

    fs_env = fs / hop
    n_freq = power.shape[0]
    rt60 = np.full(n_freq, np.nan)

    t_str = f'T{t_val}'

    global_p_max = np.max(power)

    snr_margin = (t_val + 10.0) if robust else (t_val + 5.0)
    sanity_factor = 2.0 if robust else 4.0

    for fi in range(n_freq):
        p = power[fi, :]
        p_max = np.max(p)

        # Check if the bin has any energy
        if p_max < 1e-30:
            continue

        # Check if bin is too quiet globally (60 dB below global maximum power)
        if p_max < 1e-6 * global_p_max:
            continue

        # Robust noise floor estimation
        valid_p = p[p > 1e-12 * p_max]
        if len(valid_p) > 10:
            noise_est = np.percentile(valid_p, 10)
        else:
            noise_est = 0.0

        # SNR threshold check
        snr_db = 10 * np.log10(p_max / (noise_est + 1e-30))
        if snr_db < snr_margin:
            continue

        sig = pf.Signal(p, fs_env)

        try:
            edc = ra.energy_decay_curve_chu(
                sig,
                noise_level=np.array([noise_est]),
                is_energy=True,
                time_shift=False
            )
            rt = ra.reverberation_time_linear_regression(edc, T=t_str)
            val = float(np.squeeze(rt))

            # Sanity check: RT60 cannot be absurdly longer than the RIR itself.
            max_physically_possible_rt = (len(ir) / fs) * sanity_factor

            if 0 < val < max_physically_possible_rt:
                rt60[fi] = val
        except Exception:
            pass

    if robust:
        rt60 = _robust_smooth_rt60(rt60)

    return f, rt60
