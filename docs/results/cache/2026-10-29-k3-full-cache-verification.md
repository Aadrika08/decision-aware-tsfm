# Complete K3 cache verification — 2026-10-29

Owner: Aadrika. Fix: #81. Integration: #29; parent #16.

**Both K3 Omega variants now pass on the complete, unchanged pilot cache.**
This is producer execution of the actual consumer API. Veer and Abdullah's
personal acceptance remains a separate requirement of #29.

## Cause and change

The original reader at `e3b7e17e54e94439fc78a38057273473e6bc7985` timed out
after 900.13 seconds without a final report. Its `join_omega` filtered all
1,395,776 rows separately for each of 21,696 jobs just to obtain the first row's
identity. The cache's 21,696 Arrow chunks added overhead to every scan.

A read-only diagnostic on eight sampled jobs observed approximately 0.10 seconds
per full-table filter after the first 0.34-second call. Combining chunks reduced
those sampled filters to 0.005–0.011 seconds with identical selected rows. These
are diagnostic samples, not an end-to-end performance estimate.

The fix at `2890ab02f8daef591baa55a2f4febb573fcfcb5b` indexes the first row of each
job once and then looks up its provenance identity directly. The existing payload
validator already checks that every row in a job has consistent identity. The
index supports interleaved/reordered rows and preserves every source revision,
source hash, repository, Omega, actual-value and window check. Forecast files,
quantiles, schemas, selectors and experimental settings did not change.

## Complete cache-only check

The original probe script was reused byte-for-byte, SHA-256
`2521b8fc16062e8d26f467a0ff3a8f6f8e161a52196885a76fc51cdd8d5fc030`.
It calls the ordinary `load_k3_inputs` entry point on each variant, with fixture
opt-in disabled. No input conversion or validation bypass was introduced.

| Observation | Both `current` and `last_4096` |
|---|---:|
| Status | `ok` |
| Partitions / provenance-checked jobs | 21,696 |
| Forecast rows | 1,395,776 |
| Comparisons | 5,424 |
| Series / series-origins | 55 / 904 |
| Horizons | 1, 24, 168 |
| Seeds | 42, 43 |
| Quantiles | All eleven |
| Unused Omega origins / undefined Omega | 0 / 0 |

The sequential two-variant process exited **0**, with measured wall time
**94.37 seconds** and maximum process RSS **4,111,600 KiB**. It ran on the laptop's
AMD Ryzen 7 7735HS, Python 3.11.15, NumPy 1.26.4, pandas 2.2.3,
PyArrow 16.1.0 and SciPy 1.11.4. Arrow CPU/I/O thread counts were two and
BLAS/OpenMP thread counts one. Admission required at least 6 GiB available memory;
the invocation had a 900-second timeout. Neither torch nor statsforecast was
imported, and no forecast generation occurred.

The prior timeout was on the L4 machine's host CPU. The new wall time is not a
controlled speedup comparison between machines. Both observations and the
diagnostic samples are retained.

## Identity and reproducibility

The [full pilot release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/pilot-full-2026-10-29)
and original archives remain unchanged:

- Forecast archive: `4ec7b9ad45c921581562cc7bbb118c0988b318179fd4e5cfac97ae904cfc03a5`.
- Original evidence archive: `f983217fdce4538d9bc059576ed413d7521dd31478833ea4e84eb2fc0ac4bf99`.
- `handoff.json`: `82d2234855ba4b67df0aa0b2994946c1a46c70498375600378a6e6eb5b7aa0a9`.

The additional [K3 verification archive](https://github.com/Aadrika08/decision-aware-tsfm/releases/download/pilot-full-2026-10-29/pilot-k3-consumer-verification-20261029.tar.gz)
contains 18 files: original probe, bounded wrapper, source diff, diagnostic samples,
complete reader output, both test runs, environment records and file checksums.
Its SHA-256 is
`a2328f17a478c6f3fcb934914fc3669fe576563e173094ab87cc2efb6dfd9bae`.
The embedded README gives the exact guarded command and required environment.

Use the download instructions in [the full-pilot record](2026-10-29-full-pilot.md).
For this additional check, set `K3` to a clean checkout of `2890ab0`, with its full
SHA above, and run the same archived `probe_k3.py` using the existing environment.
Retain its stdout, stderr, exit status and `/usr/bin/time -v` report. Each owner
must record the actual reader revision rather than claim the old reader passed.

## Regression evidence

The new bounded-filter-work regression failed on the original implementation:
six jobs caused 72 row-visits through full-table filters against a 24-row work
budget. It passes with the index. Tests with reordered, discontiguous job rows
use different source hashes per series, and a mismatch on the final job still
raises the original source-provenance rejection. All **54 loader tests pass**.

The first full-suite invocation used the forecasting environment, which lacks
scikit-learn: 1,099 tests ran with five dependency errors and nine optional skips.
The five failing cases are preserved in the raw log. The repository-pinned
analysis packages were installed in a separate verification directory, leaving
the forecasting environment unchanged. The repeated full suite completed with
**1,090 passing tests and nine optional skips** (1,099 discovered), no errors or
failures, in 367.030 seconds. The skips require optional historical-release,
fixture, second-interpreter or real-input environment variables; their names and
reasons are retained in the log. The explicit full-cache checks above ran
separately. Both the initial dependency errors and the successful rerun are
retained in the supplemental evidence.

The original probe labels its API consumer as `Abdullah` in JSON. It was executed
by Aadrika's producer automation; that label is not Abdullah's acceptance.
No E1/E2/K3 scientific conclusion or final-study budget follows from this read.
