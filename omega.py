import numpy as np


def compute_omega(series: np.ndarray, diffs: bool = False) -> float:
    """Computes spectral predictability Omega according to paper spec:

    - Subtracts DC component (mean)
    - Applies Hann taper window
    - Calculates one-sided PSD over k_bins = floor(t_len/2) bins (DC excluded)
    - Normalizes spectral entropy H(x) by h_max = log(k_bins)
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]

    if len(ts) < 2:
        return 0.0

    if diffs:
        ts = np.diff(ts)

    t_len = len(ts)
    if t_len < 2:
        return 0.0

    # Remove DC component
    ts_zero_mean = ts - np.mean(ts)

    # Apply Hann taper window
    hann_window = np.hanning(t_len)
    tapered_series = ts_zero_mean * hann_window

    #  Compute Real FFT (rfft returns DC at index 0, positive freqs at 1..k_bins)
    fft_vals = np.fft.rfft(tapered_series)

    # Exclude DC component (k = 1 ... k_bins, where k_bins = floor(t_len/2))
    fft_no_dc = fft_vals[1:]
    k_bins = len(fft_no_dc)

    if k_bins < 1:
        return 0.0

    #   power spectral density (PSD) 1-sided
    psd = np.abs(fft_no_dc) ** 2

    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    # Probability distribution over frequency bins
    p = psd / total_power
    p = p[p > 0]  # Filter zero-power bins for logarithm safety

    # Spectral Entropy & Normalized Predictability
    spectral_entropy = -np.sum(p * np.log(p))
    h_max = np.log(k_bins)

    if h_max <= 0:
        return 0.0

    omega = 1.0 - (spectral_entropy / h_max)

    return float(np.clip(omega, 0.0, 1.0))