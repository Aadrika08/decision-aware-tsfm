import numpy as np
from scipy import stats

# ---------------------------------------------------------------
# TEMP: this is a copy of Dayson's compute_omega, shared via Slack on
# 09/06/26. His omega.py is not yet pushed to GitHub (checked: not on
# main, not on any branch, no open PR as of this commit). Replace this
# block with `from omega import compute_omega` once #1 (K1) lands on
# main. Do not edit the body below independently of his file -- if a
# bug is found here, it must be fixed in his source and re-copied, so
# the two never silently diverge.
# ---------------------------------------------------------------
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

    ts_zero_mean = ts - np.mean(ts)
    hann_window = np.hanning(t_len)
    tapered_series = ts_zero_mean * hann_window

    fft_vals = np.fft.rfft(tapered_series)
    fft_no_dc = fft_vals[1:]
    k_bins = len(fft_no_dc)

    if k_bins < 1:
        return 0.0

    psd = np.abs(fft_no_dc) ** 2
    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    p = psd / total_power
    p = p[p > 0]

    spectral_entropy = -np.sum(p * np.log(p))
    h_max = np.log(k_bins)

    if h_max <= 0:
        return 0.0

    omega_val = 1.0 - (spectral_entropy / h_max)
    return float(np.clip(omega_val, 0.0, 1.0))


# ---------------------------------------------------------------
# Sanity check against the four synthetic targets from the proposal,
# using Dayson's actual compute_omega, to confirm nothing gets lost
# in translation before we build the matched pair on top of it.
# ---------------------------------------------------------------
def sanity_check():
    n = 2000
    t = np.arange(n)

    pure_sine = np.sin(2 * np.pi * t / 24)
    daily_weekly = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
    moderate_noise = np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1.5, n)
    white_noise = np.random.normal(0, 1, n)

    print("Sanity check with Dayson's compute_omega (expect ~0.87 / 0.79 / 0.32 / 0.06):")
    print(f"  pure sine       -> {compute_omega(pure_sine):.3f}")
    print(f"  daily+weekly    -> {compute_omega(daily_weekly):.3f}")
    print(f"  moderate noise  -> {compute_omega(moderate_noise):.3f}")
    print(f"  white noise     -> {compute_omega(white_noise):.3f}")
    print()


# ---------------------------------------------------------------
# Matched-Omega pair: same underlying deterministic signal (so spectral /
# second-order structure matches), different noise DISTRIBUTION on top
# (symmetric Gaussian vs. skewed, heavy-tailed), variance held equal.
# ---------------------------------------------------------------
def build_matched_pair(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    base_signal = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
    target_std = 1.0

    noise_a = rng.normal(0, target_std, n)  # symmetric, light-tailed

    raw_b = stats.skewnorm.rvs(a=8, size=n, random_state=rng)  # strong right skew
    raw_b = (raw_b - np.mean(raw_b)) / np.std(raw_b) * target_std  # match std exactly

    series_a = base_signal + noise_a
    series_b = base_signal + raw_b

    return series_a, series_b


def compare_pair():
    series_a, series_b = build_matched_pair()

    omega_a_levels = compute_omega(series_a, diffs=False)
    omega_b_levels = compute_omega(series_b, diffs=False)
    omega_a_diff = compute_omega(series_a, diffs=True)
    omega_b_diff = compute_omega(series_b, diffs=True)

    skew_a, kurt_a = stats.skew(series_a), stats.kurtosis(series_a)
    skew_b, kurt_b = stats.skew(series_b), stats.kurtosis(series_b)

    print("Matched-Omega pair result (using Dayson's compute_omega):")
    print(f"  Omega (levels):      A = {omega_a_levels:.4f}   B = {omega_b_levels:.4f}   diff = {abs(omega_a_levels - omega_b_levels):.5f}")
    print(f"  Omega (1st diffs):   A = {omega_a_diff:.4f}   B = {omega_b_diff:.4f}   diff = {abs(omega_a_diff - omega_b_diff):.5f}")
    print()
    print("  Distributions differ clearly despite matched Omega:")
    print(f"  Skewness:            A = {skew_a:.3f}   B = {skew_b:.3f}")
    print(f"  Excess kurtosis:     A = {kurt_a:.3f}   B = {kurt_b:.3f}")
    print()
    print("  Std dev (confirms second-order/variance structure is matched):")
    print(f"  Std:                 A = {np.std(series_a):.4f}   B = {np.std(series_b):.4f}")

    return series_a, series_b, {
        "omega_levels_a": omega_a_levels, "omega_levels_b": omega_b_levels,
        "omega_diff_a": omega_a_diff, "omega_diff_b": omega_b_diff,
        "skew_a": skew_a, "skew_b": skew_b,
        "kurt_a": kurt_a, "kurt_b": kurt_b,
    }


if __name__ == "__main__":
    sanity_check()
    compare_pair()
