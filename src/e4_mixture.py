"""E4 — beyond the theory: a two-component Gaussian mixture target (not log-concave), three
reference measures, accelerated vs non-accelerated fixed-reference scheme.
Writes results/e4/{e4.npz, fig_e4_evolution.*, fig_e4_gap.*, results.md}. No arguments needed.

The inner problem is non-convex here: Algorithm 2 is run as is, the pointwise transport solve
is local (warm-started Newton), and the duality certificate is only a heuristic stopping rule."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from common import COLORS, FLOOR, RESULTS, floor, md_table, outdir, plt, savefig
from grid import Grid, Mix2, truncated_gaussian_weights
from pdhg import run_outer

GAMMA = 0.05
EPS0 = 1e-2
SNAP = (0, 5, 10, 20, 40, 80)
WEIGHTS = (0.5, 0.9)
INITS = {"A": "uniform", "B": "trunc. N((0.5,0.5), 0.15^2 I)", "C": "trunc. N((0.52,0.5), 0.15^2 I)"}
METHODS = {"accel": "accelerated", "mm": "non-accelerated"}


def reference_weights(grid, init):
    if init == "A":
        return None
    return truncated_gaussian_weights(grid, (0.5, 0.5) if init == "B" else (0.52, 0.5), 0.15)


def work(job):
    w, init, method, a = job
    grid = Grid(a["n"])
    tgt = Mix2(w, (0.3, 0.5), (0.7, 0.5), 0.06)
    wt = reference_weights(grid, init)
    snaps = tuple(s for s in SNAP if s <= a["T"])
    r = run_outer(grid, tgt, GAMMA, a["T"], method == "accel", (1.0, a["r"]), wt=wt, eps0=EPS0, cap=a["cap"],
                  snapshots=snaps)
    wts = np.ones((grid.n, grid.n)) if wt is None else wt
    left = float((wts * (r["X"][..., 0] < 0.5)).sum() * grid.h ** 2)
    out = dict(G=r["G"], inner=r["inner"], wall=r["wall"], cap_hit=r["cap_hit"], cert=r["cert"], fails=r["fails"],
               left=left, snap_t=np.array(snaps), snap_X=np.stack([r["snaps"][s][0] for s in snaps]),
               snap_rho=np.stack([r["snaps"][s][1] for s in snaps]))
    key = f"w{w}_{init}_{method}"
    np.savez_compressed(os.path.join(outdir("e4"), f"run_{key}.npz"), **out)
    print(f"[{key}] G_T = {r['G'][-1]:.6f}, left mass {left:.3f} (target {w}), inner its {r['inner'].sum()}, "
          f"cap hits {int(r['cap_hit'].sum())}, fails {r['fails']}, {r['wall'][-1]:.0f}s", flush=True)
    return key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel (wall-clock columns get noisier)")
    args = ap.parse_args()
    path = os.path.join(RESULTS, "e2", "r_best.json")
    found = os.path.exists(path)
    r_best = float(json.load(open(path))["r_best"]) if found else 10.0
    a = dict(vars(args), r=r_best)
    d = outdir("e4")
    jobs = [(w, init, m, a) for w in WEIGHTS for init in INITS for m in METHODS]
    if args.jobs > 1:
        with Pool(min(args.jobs, len(jobs))) as pool:
            res = dict(pool.map(work, jobs))
    else:
        res = dict(work(j) for j in jobs)

    T = args.T
    t = np.arange(T + 1)
    lines = [f"# E4 — beyond the theory (two-component mixture)\n\nn = {args.n}, gamma = {GAMMA}, T = {T}, target "
             f"MIX2(w, (0.3,0.5), (0.7,0.5), s = 0.06), inner solver (sig_L, sig_m) = (1, {r_best:g})"
             + ("" if found else " [r_best.json missing: default 10]") +
             f", schedule eps_t = {EPS0:g} (t+1)^-4.5, cap {args.cap}.\n\n"
             "**Outside the theory.** G is not convex along generalized geodesics, the inner problem is non-convex "
             "(the pointwise transport solve is a warm-started local Newton solve, and the 'certificate' uses a "
             "local maximisation for h2*, so it is not a valid bound). No scheme has a guarantee here. "
             "G = KL(x | q) with q normalised on the box; G* = the best value seen across all runs for that target.\n"]
    data = {}
    styles = {"A": "-", "B": "--", "C": ":"}
    fig, axes = plt.subplots(1, len(WEIGHTS), figsize=(10, 4), squeeze=False)
    for ax, w in zip(axes[0], WEIGHTS):
        Gs = min(res[f"w{w}_{i}_{m}"]["G"].min() for i in INITS for m in METHODS)
        tt = [k for k in (20, 80) if k <= T]
        rows = []
        for init in INITS:
            for (m, mlab), col in zip(METHODS.items(), COLORS):
                o = res[f"w{w}_{init}_{m}"]
                gap = o["G"] - Gs
                ax.loglog(t[1:], floor(gap[1:]), styles[init], color=col, label=f"{mlab}, init {init}")
                rows.append([f"{init} ({INITS[init]})", mlab] + [float(gap[k]) for k in tt] +
                            [f"{o['left']:.3f}", int(o["inner"].sum()), int(o["cap_hit"].sum()), int(o["fails"])])
                for k, v in o.items():
                    data[f"w{w}_{init}_{m}_{k}"] = v
        ax.set_xlabel("outer step $t$")
        ax.set_ylabel(r"$G(x_t)-G^\star$ (best value seen)")
        ax.set_title(f"mixture weight $w$ = {w}")
        ax.set_ylim(FLOOR / 3, None)
        lines.append(f"\n## w = {w}\n\nG* (best value seen) = {Gs:.8f}. Target mass in the left half-plane: {w}.\n\n")
        lines.append(md_table(["reference / init", "method"] + [f"gap at t={k}" for k in tt] +
                              [f"left mass at t={T}", "inner its", "steps at cap", "steps w/o admissible iterate"], rows))
        data[f"w{w}_Gstar"] = Gs
    axes[0, 0].legend(ncol=2)
    fig.tight_layout()
    savefig(fig, d, "fig_e4_gap")

    #  -- evolution for w = 0.9, init C
    snaps = [s for s in SNAP if s <= T]
    fig, axes = plt.subplots(2, len(snaps), figsize=(2.3 * len(snaps), 4.9), squeeze=False)
    vmax = max(np.log10(res[f"w0.9_C_{m}"]["snap_rho"]).max() for m in METHODS)
    for row, (m, mlab) in enumerate(METHODS.items()):
        o = res[f"w0.9_C_{m}"]
        for j, s in enumerate(snaps):
            ax = axes[row, j]
            sc = ax.scatter(o["snap_X"][j][..., 0].ravel(), o["snap_X"][j][..., 1].ravel(), s=1.2, lw=0,
                            c=np.log10(o["snap_rho"][j]).ravel(), cmap="viridis", vmin=vmax - 4, vmax=vmax)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"{mlab}, t = {s}", fontsize=8)
    fig.colorbar(sc, ax=axes, shrink=0.7, label=r"$\log_{10}$ density of $x_t$ at $S(x)$")
    savefig(fig, d, "fig_e4_evolution")

    lines.append("\nCollapse check: a run has collapsed to one mode if its left mass is near 0 or 1 rather than near w.\n")
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("".join(lines))
    np.savez_compressed(os.path.join(d, "e4.npz"), r_best=r_best, **data)
    for key in res:
        os.remove(os.path.join(d, f"run_{key}.npz"))


if __name__ == "__main__":
    main()
