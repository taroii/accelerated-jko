import faulthandler
import os
import sys
import time
from multiprocessing import Pool

faulthandler.enable(file=sys.stderr, all_threads=True)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, os.environ.get("AJKO_RESULTS", "results"))
FLOOR = 1e-12
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

mpl.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 7.5,
                     "lines.linewidth": 1.6, "savefig.bbox": "tight", "pdf.fonttype": 42})


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def gentle():
    try:
        os.nice(10)
    except AttributeError:
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
        except Exception:
            pass
    except OSError:
        pass


def pool(jobs):
    return Pool(jobs, initializer=gentle)


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
    t = np.asarray(t, float)
    g0 = max(gap[t_fit], FLOOR)
    ax.loglog(t, g0 * t_fit / t, "--", color="0.55", lw=1, label=r"$C_1/t$")
    ax.loglog(t, g0 * t_fit ** 2 / t ** 2, "--", color="0.3", lw=1, label=r"$C_2/t^2$")
    if bound is not None:
        ax.loglog(t, bound, ":", color="k", lw=1.2, label=r"bound $2W_{2,\nu}^2/(\gamma(t+1)^2)$")


def theorem_bound(w2nu0, gamma, t):
    return 2 * w2nu0 / (gamma * (np.asarray(t, float) + 1) ** 2)


def first_below(gaps, thr):
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


def run_seeds(work, seeds, cfg, jobs, cache, resume=False):
    os.makedirs(cache, exist_ok=True)
    path = lambda s: os.path.join(cache, f"seed_{s:03d}.npz")
    res, todo = {}, []
    for s in seeds:
        if resume and os.path.exists(path(s)):
            with np.load(path(s)) as f:
                res[s] = {k: f[k] for k in f.files}
        else:
            todo.append((s, cfg))

    name = os.path.join(os.path.basename(os.path.dirname(cache)), os.path.basename(cache))
    say(f"{name}: {len(todo)} seeds to run on {max(1, min(jobs, len(todo)))} processes"
        + (f" ({len(res)} reused)" if res else ""))
    t0, n_done = time.time(), [0]

    def done(s, out):
        np.savez_compressed(path(s), **out)
        res[s] = {k: np.asarray(v) for k, v in out.items()}
        n_done[0] += 1
        el = time.time() - t0
        left = len(todo) - n_done[0]
        say(f"{name}: seed {s} done — {len(res)}/{len(seeds)}, elapsed {_hms(el)}"
            + (f", about {_hms(el / n_done[0] * left)} left" if left else ""))

    if jobs > 1 and len(todo) > 1:
        with pool(min(jobs, len(todo))) as pl:
            for s, out in pl.imap_unordered(work, todo):
                done(s, out)
    else:
        for job in todo:
            done(*work(job))
    return [res[s] for s in seeds]


def stack(res, key):
    return np.stack([np.asarray(r[key], float) for r in res])


def band(a):
    a = np.asarray(a, float)
    with np.errstate(invalid="ignore"):
        lo, med, hi = np.nanpercentile(a, [2.5, 50, 97.5], axis=0, method="nearest")
    return med, lo, hi


def plot_band(ax, x, a, color, label, lo_floor=FLOOR, ls="-"):
    med, lo, hi = band(a)
    ok = np.isfinite(med)
    ax.plot(np.asarray(x)[ok], np.maximum(med[ok], lo_floor), ls, color=color, label=label)
    ok = np.isfinite(hi) & np.isfinite(lo)
    ax.fill_between(np.asarray(x)[ok], np.maximum(lo[ok], lo_floor), np.maximum(hi[ok], lo_floor),
                    color=color, alpha=0.18, lw=0)


def _boot(x, stat, B=2000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(0).integers(0, x.size, size=(B, x.size))
    with np.errstate(invalid="ignore"):
        lo, hi = np.percentile(stat(x[idx], axis=1), [2.5, 97.5], method="nearest")
    return float(stat(x)), float(lo), float(hi)


def boot_mean(x):
    return _boot(x, np.mean)


def boot_median(x):
    return _boot(x, lambda a, axis=None: np.percentile(a, 50, axis=axis, method="lower"))


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - hw), min(1.0, c + hw)


def ci(t, f="{:.3g}"):
    g = lambda v: "inf" if np.isinf(v) else f.format(v)
    return f"{g(t[0])} [{g(t[1])}, {g(t[2])}]"


def first_idx(mask):
    mask = np.asarray(mask)
    return np.where(mask.any(-1), mask.argmax(-1), -1)
