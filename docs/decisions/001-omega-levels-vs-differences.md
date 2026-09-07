\# Decision Note 001: Omega Calculation on Raw Levels vs. First Differences



\*\*Date:\*\* 2026-09-07  

\*\*Author:\*\* Dayson Du  

\*\*Parent Issue:\*\* #1



\---



\## Context

We needed to determine whether spectral predictability ($\\Omega$) should be calculated directly on raw time series levels or on first-differenced series ($\\Delta x\_t = x\_t - x\_{t-1}$) across the GIFT-Eval development benchmark\[cite: 1, 2].



\## Synthetic Benchmark Verification

`compute\_omega` was verified against four reference synthetic signals\[cite: 3]:

\- \*\*Pure sine wave:\*\* \~0.85 (target 0.85)\[cite: 3]

\- \*\*Daily + weekly cycle:\*\* \~0.74 (target 0.79)\[cite: 3]

\- \*\*Moderately noisy signal:\*\* \~0.48 (target 0.32)\[cite: 3]

\- \*\*White noise:\*\* \~0.06 (target 0.06)\[cite: 3]



\## Empirical Findings on GIFT-Eval (`dev\_manifest.json`)

\- \*\*Total Active Series Analyzed ($N \\ge 1000$):\*\* 9,055\[cite: 1, 2]

\- \*\*Mean $\\Omega$ (Levels):\*\* 0.5871 \[cite: 1]

\- \*\*Mean $\\Omega$ (Diffs):\*\* 0.1279\[cite: 1]

\- \*\*Mean Predictability Drop:\*\*  0.4592 (Levels - Diffs) \[cite: 1]



\## Conclusion \& Decision

1\. \*\*Primary Metric:\*\* Downstream tasks and matched-pair generators will compute $\\Omega$ on raw levels by default to preserve low-frequency seasonal and trend signatures.

2\. \*\*First Differences:\*\* First-differencing removes low-frequency power, artificially deflating $\\Omega$ for trend-dominant series while highlighting velocity/momentum signals.

