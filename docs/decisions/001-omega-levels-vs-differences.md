# Decision Note 001: Omega Calculation on Raw Levels vs. First Differences

**Date:** 2026-09-07  
**Author:** Dayson Du  
**Parent Issue:** #1

---

## Context
We evaluated whether spectral predictability ($\Omega$) should be calculated directly on raw time series levels or on first-differenced series ($\Delta x_t = x_t - x_{t-1}$) across the GIFT-Eval benchmark.

## Synthetic Benchmark Verification
`compute_omega` was verified against four reference synthetic signals ($N=1000$):

| Test Signal | Target $\Omega$ | Computed $\Omega$ | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| Pure Sine Wave ($\sin(2\pi t/50) + \mathcal{N}(0, 0.05)$) | $\approx 0.87$ | 0.8515 | -0.0185 | Pass |
| Daily + Weekly Cycle ($\sin(2\pi t/24) + \sin(2\pi t/168)$) | $\approx 0.79$ | 0.7489 | -0.0411 | Pass |
| Moderately Noisy Signal ($\sin(2\pi t/24) + \mathcal{N}(0, 0.5)$) | $\approx 0.32$ | 0.4864 | +0.1664 | Deviation |
| White Noise ($\mathcal{N}(0, 1)$) | $\approx 0.06$ | 0.0668 | +0.0068 | Pass |

### Deviations Analysis
- **White Noise Lower Bound:** The computed value of 0.0668 matches the theoretical lower bound ($\approx 0.06$). This confirms proper balancing between the entropy normalization constant $\ln(K)$ and Hann window tapering loss corrections.
- **Moderately Noisy Signal (0.4864 vs ~0.32):** The score is higher than estimated because a single pure sinusoidal frequency at period 24 concentrates massive discrete energy into a single FFT bin, dominating the denominator relative to $\sigma=0.5$ Gaussian noise spread evenly across all 500 positive frequency bins.

## Empirical Findings on GIFT-Eval (`dev_manifest.json`)
- **Total Active Series Analyzed ($T \ge 1000$):** 9,055 series
- **Mean $\Omega_{\text{levels}}$:** 0.5871
- **Mean $\Omega_{\text{diffs}}$:** 0.1279
- **Mean Predictability Drop ($\Delta\Omega = \Omega_{\text{levels}} - \Omega_{\text{diffs}}$):** 0.4592 (Median: 0.5189)

### Categorical Breakdown
- **Trend/Seasonal Dominant ($\Omega_{\text{levels}} > \Omega_{\text{diffs}}$):** 8,371 series (92.4%) — Differencing strips low-frequency trends and seasonal cycles, which collapsing predictability.
- **Invariant / Level-Diff Balance ($|\Delta\Omega| \le 0.05$):** 653 series (7.2%) — Series where trend drift is minimal.
- **Velocity/Momentum Dominant ($\Omega_{\text{diffs}} > \Omega_{\text{levels}}$):** 31 series (0.3%) — High-level non-stationary noise removal isolates a cleaner high-frequency signal.

### The Random Walk Phenomenon (`m4_daily`)
Drifting series cause metric inflation on raw levels, demonstrated by `m4_daily` ($N=2,663$ series):
- **Mean $\Omega_{\text{levels}}$:** 0.7509
- **Mean $\Omega_{\text{diffs}}$:** 0.0560 (Drop: 0.6949)

Because daily financial series approximate Random Walks ($x_t = x_{t-1} + \epsilon_t$), their power spectral density follows a $1/f^2$ continuum concentrated at low frequencies. Raw levels misinterpret stochastic drift as high structure (0.7509), whereas first-differencing isolates the white-noise residuals $\epsilon_t$ (0.0560).

## Conclusion & Architectural Decision
- Primary Metric ($\Omega_{\text{levels}}$): Downstream tasks and matched-pair generators will compute $\Omega$ on raw levels by default. Time Series Foundation Models (TSFMs) evaluate un-differenced target sequences; keeping $\Omega$ on raw levels aligns feature space with model evaluation space and prevents high-pass filtering from destroying seasonal signals (e.g., `electricity_15T` dropping from 0.6525 to 0.1627).
- Diagnostic Guard Rule: To prevent random-walk series from deceiving model selection without penalizing seasonal data, series meeting the dual condition:
   $$\Omega_{\text{levels}} \ge 0.70 \quad \text{AND} \quad \Omega_{\text{diffs}} \le 0.10 \quad (\Delta\Omega > 0.50)$$
   are automatically tagged in `dev_manifest.json` as `Non-Stationary / Random Walk Dominant`.