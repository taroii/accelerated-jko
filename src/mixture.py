import argparse
import json
import os

import numpy as np

from common import (COLORS, FLOOR, RESULTS, boot_mean, ci, floor, md_table, outdir, plot_band, plt, pool, run_seeds, savefig,
                    say, stack, wilson)
from grid import Grid, Mix2, truncated_gaussian_weights
from pdhg import run_outer

GAMMA = 0.05
EPS0 = 1e-2
STALL = 1e-10
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
                  snapshots=snaps, stall=STALL)
    wts = np.ones((grid.n, grid.n)) if wt is None else wt
    left = float((wts * (r["X"][..., 0] < 0.5)).sum() * grid.h ** 2)
    out = dict(G=r["G"], inner=r["inner"], wall=r["wall"], cap_hit=r["cap_hit"], stalled=r["stalled"], cert=r["cert"],
               fails=r["fails"],
               left=left, snap_t=np.array(snaps), snap_X=np.stack([r["snaps"][s][0] for s in snaps]),
               snap_rho=np.stack([r["snaps"][s][1] for s in snaps]))
    key = f"w{w}_{init}_{method}"
    np.savez_compressed(os.path.join(outdir("e4"), f"run_{key}.npz"), **out)
    say(f"E4 canonical [{key}] G_T = {r['G'][-1]:.6f}, left mass {left:.3f} (target {w}), inner its {r['inner'].sum()}, "
          f"cap hits {int(r['cap_hit'].sum())}, stalled {int(r['stalled'].sum())}, fails {r['fails']}, {r['wall'][-1]:.0f}s")
    return key, out


def load_r_best():
    path = os.path.join(RESULTS, "e2", "r_best.json")
    found = os.path.exists(path)
    return (float(json.load(open(path))["r_best"]) if found else 10.0), found


def canonical(args):
    r_best, found = load_r_best()
    a = dict(n=args.n, T=args.T, cap=args.cap, r=r_best)
    d = outdir("e4")
    jobs = [(w, init, m, a) for w in WEIGHTS for init in INITS for m in METHODS]
    say(f"E4 canonical: {len(jobs)} runs x {args.T} outer steps (cap {args.cap} inner iterations each)")
    if args.jobs > 1:
        with pool(min(args.jobs, len(jobs))) as pl:
            res = dict(pl.map(work, jobs))
    else:
        res = dict(work(j) for j in jobs)

    T = args.T
    t = np.arange(T + 1)
    lines = [f"# E4 — beyond the theory (two-component mixture)\n\nn = {args.n}, gamma = {GAMMA}, T = {T}, target "
             f"MIX2(w, (0.3,0.5), (0.7,0.5), s = 0.06), inner solver (sig_L, sig_m) = (1, {r_best:g})"
             + ("" if found else " [r_best.json missing: default 10]") +
             f", schedule eps_t = {EPS0:g} (t+1)^-4.5, cap {args.cap}; an inner solve also ends ('stalled') once its "
             f"coefficients move by less than {STALL:g} (relative) over 25 iterations.\n\n"
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
                            [f"{o['left']:.3f}", int(o["inner"].sum()), int(o["cap_hit"].sum()), int(o["stalled"].sum()), int(o["fails"])])
                for k, v in o.items():
                    data[f"w{w}_{init}_{m}_{k}"] = v
        ax.set_xlabel("outer step $t$")
        ax.set_ylabel(r"$G(x_t)-G^\star$ (best value seen)")
        ax.set_title(f"mixture weight $w$ = {w}")
        ax.set_ylim(FLOOR / 3, None)
        lines.append(f"\n## w = {w}\n\nG* (best value seen) = {Gs:.8f}. Target mass in the left half-plane: {w}.\n\n")
        lines.append(md_table(["reference / init", "method"] + [f"gap at t={k}" for k in tt] +
                              [f"left mass at t={T}", "inner its", "steps at cap", "steps stalled", "steps w/o admissible iterate"], rows))
        data[f"w{w}_Gstar"] = Gs
    axes[0, 0].legend(ncol=2)
    fig.tight_layout()
    savefig(fig, d, "fig_e4_gap")

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
    np.savez_compressed(os.path.join(d, "e4.npz"), r_best=r_best, **data)
    for key in res:
        os.remove(os.path.join(d, f"run_{key}.npz"))
    return lines


ENS_INITS = {"A": "uniform reference", "C": "trunc. N((0.5 + off, 0.5), 0.15^2 I), off ~ N(0, 0.02^2)"}
RECOVER = 0.05


def ens_work(job):
    seed, a = job
    rng = np.random.default_rng([4, seed])
    w, delta, off = rng.uniform(0.6, 0.95), rng.uniform(0.15, 0.25), rng.normal(0, 0.02)
    grid = Grid(a["n"])
    tgt = Mix2(w, (0.5 - delta, 0.5), (0.5 + delta, 0.5), 0.06)
    out = dict(w=w, delta=delta, off=off)
    for init in ENS_INITS:
        wt = None if init == "A" else truncated_gaussian_weights(grid, (0.5 + off, 0.5), 0.15)
        wts = np.ones((grid.n, grid.n)) if wt is None else wt
        for m in METHODS:
            r = run_outer(grid, tgt, GAMMA, a["T"], m == "accel", (1.0, a["r"]), wt=wt, eps0=EPS0, cap=a["cap"], stall=STALL)
            out[f"{init}_{m}_G"] = r["G"]
            out[f"{init}_{m}_left"] = float((wts * (r["X"][..., 0] < 0.5)).sum() * grid.h ** 2)
            out[f"{init}_{m}_inner"] = r["inner"].sum()
            out[f"{init}_{m}_fails"] = r["fails"]
            say(f"E4 seed {seed} [{init}, {m}]: G_T = {r['G'][-1]:.5f}, left mass {out[f'{init}_{m}_left']:.3f} (w = {w:.3f}), "
                f"inner its {r['inner'].sum()}")
    return seed, out


def ensemble(args):
    d = outdir("e4")
    r_best, _ = load_r_best()
    T = args.T
    t = np.arange(T + 1)
    say(f"E4 ensemble: {args.seeds} instances x 2 references x 2 methods")
    res = run_seeds(ens_work, range(args.seeds), dict(n=args.n, T=T, cap=args.cap, r=r_best), args.jobs,
                    os.path.join(d, "seeds"), args.resume)
    n = len(res)
    w = stack(res, "w")
    keys = [(i, m) for i in ENS_INITS for m in METHODS]
    Gs = np.min([stack(res, f"{i}_{m}_G").min(axis=1) for i, m in keys], axis=0)[:, None]
    gap = {k: np.maximum(stack(res, f"{k[0]}_{k[1]}_G") - Gs, 0.0) for k in keys}
    rel = {k: gap[k] / gap[k][:, :1] for k in keys}
    tt = [k for k in (20, 80) if k <= T]
    lines = [f"\n# Ensemble over {n} random instances\n\nSeed = instance: mixture weight w ~ U(0.6, 0.95), modes at "
             f"(0.5 -+ delta, 0.5) with delta ~ U(0.15, 0.25), s = 0.06; references: {'; '.join(f'{k} = {v}' for k, v in ENS_INITS.items())}. "
             f"Both methods and both references run on the same instance; G* = best value seen across the four runs of that "
             f"instance. Mode recovery = mass in the left half-plane at t = {T} within +-{RECOVER} of w (Wilson 95% interval). "
             f"Gap columns: normalised gap (G(x_t) - G*)/(G(x_0) - G*), mean [95% bootstrap CI of the mean].\n\n"]
    rows = []
    for i, m in keys:
        left = stack(res, f"{i}_{m}_left")
        k = int((np.abs(left - w) <= RECOVER).sum())
        lo, hi = wilson(k, n)
        rows.append([i, METHODS[m], f"{k}/{n} [{lo:.2f}, {hi:.2f}]", ci(boot_mean(np.abs(left - w)))] +
                    [ci(boot_mean(rel[(i, m)][:, kk])) for kk in tt] + [int(stack(res, f"{i}_{m}_fails").sum())])
    lines.append(md_table(["reference", "method", "mode recovery rate", "abs(left mass - w)"] +
                          [f"normalised gap at t={kk}" for kk in tt] + ["steps w/o admissible iterate"], rows))
    lines.append("\nPaired log10(gap accelerated / gap non-accelerated), gaps floored at 1e-13 (negative = acceleration helps):\n\n")
    lg = {k: np.log10(np.maximum(gap[k], 1e-13)) for k in keys}
    lines.append(md_table(["reference"] + [f"t={kk}" for kk in tt],
                          [[i] + [ci(boot_mean((lg[(i, "accel")] - lg[(i, "mm")])[:, kk])) for kk in tt] for i in ENS_INITS]))
    fig, axes = plt.subplots(1, len(ENS_INITS), figsize=(10, 4), squeeze=False)
    for ax, i in zip(axes[0], ENS_INITS):
        for (m, mlab), col in zip(METHODS.items(), COLORS):
            plot_band(ax, t[1:], rel[(i, m)][:, 1:], col, mlab, lo_floor=1e-13)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("outer step $t$")
        ax.set_ylabel(r"$(G(x_t)-G^\star)/(G(x_0)-G^\star)$")
        ax.set_title(f"reference {i}, {n} instances (median, 2.5–97.5%)")
    axes[0, 0].legend()
    fig.tight_layout()
    savefig(fig, d, "fig_e4_ens")
    np.savez_compressed(os.path.join(d, "e4_ensemble.npz"), r_best=r_best, **{k: stack(res, k) for k in res[0]})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=20, help="random instances (0 = canonical configurations only)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (wall-clock columns get noisier)")
    ap.add_argument("--resume", action="store_true", help="reuse per-seed files of an interrupted ensemble")
    ap.add_argument("--skip-canonical", action="store_true")
    args = ap.parse_args()
    path = os.path.join(outdir("e4"), "results.md")
    lines = [] if args.skip_canonical else canonical(args)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    if args.seeds > 0:
        lines += ensemble(args)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    main()
