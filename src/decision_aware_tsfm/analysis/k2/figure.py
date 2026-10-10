"""Optional replication figure. matplotlib is not a pinned dependency, so it is
imported only when the figure is asked for."""
from __future__ import annotations

import os

import numpy as np

from .statistic import relative_delta, spearman

MATPLOTLIB_MISSING = "matplotlib not installed"


class FigureUnavailable(RuntimeError):
    """The figure cannot be drawn in this environment."""


def write_replication_figure(tables: dict, out_path: str) -> str:
    """tables: {metric name: family table joined to 'omega'} for the panels to draw."""
    try:
        import matplotlib
    except ImportError as exc:
        raise FigureUnavailable(MATPLOTLIB_MISSING) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(tables)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.8), squeeze=False)
    for ax, mname in zip(axes[0], names):
        t = tables[mname]
        for other, colour, label in [("deep", "#c0392b", "vs deep-learning"),
                                     ("statistical", "#2471a3", "vs statistical")]:
            d = relative_delta(t["zero-shot"], t[other])
            ok = np.isfinite(d)
            rho, p, _ = spearman(t, "zero-shot", other)
            ax.scatter(t.omega[ok], d[ok], s=38, color=colour, alpha=.8,
                       label=f"{label}  rho={rho:+.3f}, p={p:.3f}")
            z = np.polyfit(t.omega[ok], d[ok], 1)
            xs = np.linspace(0.05, 0.8, 50)
            ax.plot(xs, np.polyval(z, xs), color=colour, lw=1.2, ls="--", alpha=.7)
        ax.axhline(0, color="#555", lw=.8)
        ax.axvline(0.4, color="#999", lw=.8, ls=":")
        ax.axvline(0.5, color="#999", lw=.8, ls=":")
        ax.set_xlabel("Spectral predictability $\\Omega$")
        ax.set_ylabel("$\\Delta$ = 100(A-B)/A   negative = zero-shot better")
        ax.set_title(f"K2a replication on {mname}  (n={len(t)} datasets)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return os.path.abspath(out_path)
