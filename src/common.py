"""Shared helpers for the experiment scripts: output paths, plot style (spec 0.8), reference
curves and the bound of Theorem 1 (spec 0.2), and small reporting utilities."""
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, os.environ.get("AJKO_RESULTS", "results"))   # smoke runs redirect this
FLOOR = 1e-12
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

mpl.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 7.5,
                     "lines.linewidth": 1.6, "savefig.bbox": "tight", "pdf.fonttype": 42})


def outdir(name):
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def savefig(fig, d, name):
    fig.savefig(os.path.join(d, name + ".pdf"))
    fig.savefig(os.path.join(d, name + ".png"), dpi=200)
    plt.close(fig)


def floor(gap):
    return np.maximum(np.asarray(gap, float), FLOOR)


def add_references(ax, t, gap, t_fit=5, bound=None):
    """Dashed grey C1/t and C2/t^2 through gap[t_fit]; dotted bound of Theorem 1 if given."""
    t = np.asarray(t, float)
    g0 = max(gap[t_fit], FLOOR)
    ax.loglog(t, g0 * t_fit / t, "--", color="0.55", lw=1, label=r"$C_1/t$")
    ax.loglog(t, g0 * t_fit ** 2 / t ** 2, "--", color="0.3", lw=1, label=r"$C_2/t^2$")
    if bound is not None:
        ax.loglog(t, bound, ":", color="k", lw=1.2, label=r"bound $2W_{2,\nu}^2/(\gamma(t+1)^2)$")


def theorem_bound(w2nu0, gamma, t):
    return 2 * w2nu0 / (gamma * (np.asarray(t, float) + 1) ** 2)


def first_below(gaps, thr):
    """1-based iteration index of the first gap < thr, or None."""
    idx = np.nonzero(np.asarray(gaps) < thr)[0]
    return int(idx[0]) + 1 if idx.size else None


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.3e}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"
