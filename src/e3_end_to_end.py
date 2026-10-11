"""E3 — end to end: Algorithm 1 with every proximal step solved by Algorithm 2, certified
schedule eps_t = eps0 (t+1)^-4.5 vs a fixed inner budget, accelerated vs non-accelerated.
Writes results/e3/{e3.npz, fig_e3.*, results.md}. No arguments needed (run E2 first for r_best)."""
import argparse
import json
import os

import numpy as np

from common import (COLORS, FLOOR, RESULTS, add_references, band, boot_mean, boot_median, ci, first_idx, floor,
                    md_table, outdir, plot_band, plt, pool, run_seeds, savefig, say, stack, theorem_bound)
from e2_inner_solver import random_gauss
from grid import Gauss, Grid, Quartic
from pdhg import run_outer

GAMMA = 0.05
EPS0 = 1e-2
RUNS = [("accel_cert", "accelerated, certified $\\varepsilon_t$", dict(accel=True)),
        ("mm_cert", "non-accelerated, certified $\\varepsilon_t$", dict(accel=False)),
        ("accel_fixed", "accelerated, 50 inner its", dict(accel=True, fixed=50)),
        ("mm_fixed", "non-accelerated, 50 inner its", dict(accel=False, fixed=50))]


def load_r_best():
    path = os.path.join(RESULTS, "e2", "r_best.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["r_best"]), True
    return 10.0, False


def make_target(name):
    return Gauss((0.5, 0.5), (np.sqrt(0.02), np.sqrt(0.05))) if name == "gauss" else Quartic((0.5, 0.5), 200.0)


def run_instance(tgt, a, name):
    """Reference run and the four runs of E3 on one target."""
    grid = Grid(a["n"])
    sig = (1.0, a["r"])
    out = {}
    say(f"E3 [{name}]: reference run, {a['ref_steps']} minimizing-movement steps")
    # reference: fixed-reference minimizing movement at a constant certificate
    ref = run_outer(grid, tgt, GAMMA, a["ref_steps"], False, sig, tol_const=a["ref_tol"], cap=a["ref_cap"])
    out["ref_G"] = ref["G"]
    out["ref_inner"] = ref["inner"]
    out["ref_cap_hit"] = ref["cap_hit"]
    out["Gstar"] = ref["G"].min()
    out["w2nu0"] = float(((ref["X"] - grid.X) ** 2).sum() * grid.h ** 2)
    say(f"E3 [{name}] reference: G* = {out['Gstar']:.12f}, W2nu0 = {out['w2nu0']:.6f}, "
          f"inner its {ref['inner'].sum()}, cap hits {int(ref['cap_hit'].sum())}")
    for key, _, kw in RUNS:
        r = run_outer(grid, tgt, GAMMA, a["T"], kw["accel"], sig, eps0=EPS0, cap=a["cap"], fixed=kw.get("fixed"))
        for f in ("G", "inner", "wall", "cap_hit", "cert"):
            out[f"{key}_{f}"] = r[f]
        out[f"{key}_fails"] = r["fails"]
        say(f"E3 [{name}] {key}: final gap {r['G'][-1] - out['Gstar']:.3e}, inner its {r['inner'].sum()}, "
              f"cap hits {int(r['cap_hit'].sum())}, {r['wall'][-1]:.0f}s")
    return out


def work(job):
    name, a = job
    out = run_instance(make_target(name), a, name)
    np.savez_compressed(os.path.join(outdir("e3"), f"e3_{name}.npz"), **out)
    return name, out


def canonical(args):
    """The two fixed targets of the spec."""
    r_best, found = load_r_best()
    a = dict(n=args.n, T=args.T, cap=args.cap, ref_steps=args.ref_steps, ref_tol=args.ref_tol, ref_cap=args.ref_cap,
             r=r_best)
    d = outdir("e3")
    names = ["gauss", "quartic"]
    jobs = [(nm, a) for nm in names]
    if args.jobs > 1:
        with pool(min(args.jobs, len(jobs))) as pl:
            res = dict(pl.map(work, jobs))
    else:
        res = dict(work(j) for j in jobs)

    T = args.T
    t = np.arange(T + 1)
    lines = [f"# E3 — end to end\n\nn = {args.n}, nu uniform, gamma = {GAMMA}, T = {T} outer steps, inner solver "
             f"(sig_L, sig_m) = (1, {r_best:g})" + ("" if found else " [r_best.json missing: default 10]") +
             f", eps_t = {EPS0:g} (t+1)^-4.5 on the projected certificate (h^2-weighted units), cap {args.cap}. "
             f"G* = min G over {args.ref_steps} minimizing-movement steps at certificate {args.ref_tol:g} "
             f"(cap {args.ref_cap}). G is normalised so that G = KL(x | q), q a density on the box.\n"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    data = {}
    for row, nm in enumerate(names):
        o = res[nm]
        Gs = o["Gstar"]
        bound = theorem_bound(o["w2nu0"], GAMMA, t[1:])
        gaps = {key: o[f"{key}_G"] - Gs for key, _, _ in RUNS}
        for (key, lab, _), col in zip(RUNS, COLORS):
            axes[row, 0].loglog(t[1:], floor(gaps[key][1:]), color=col, label=lab)
            axes[row, 1].loglog(np.cumsum(o[f"{key}_inner"]), floor(gaps[key][1:]), color=col, label=lab)
        add_references(axes[row, 0], t[1:], gaps["accel_cert"], bound=bound)
        axes[row, 0].set_xlabel("outer step $t$")
        axes[row, 1].set_xlabel("cumulative inner iterations")
        for ax in axes[row]:
            ax.set_ylim(FLOOR / 3, None)
            ax.set_ylabel(r"$G(x_t)-G^\star$")
            ax.set_title(nm)
        tt = [k for k in (10, 30, 60) if k <= T]
        lo = min(10, T // 2)
        sel = (t >= lo) & (t <= T) & (gaps["accel_cert"] > FLOOR)
        slope = float(np.polyfit(np.log(t[sel]), np.log(gaps["accel_cert"][sel]), 1)[0]) if sel.sum() > 2 else float("nan")
        lines.append(f"\n## Target: {nm}\n\nG* = {Gs:.12f}; W_{{2,nu}}^2(x_0, rho*) = {o['w2nu0']:.6f}; reference run used "
                     f"{int(o['ref_inner'].sum())} inner iterations, cap hit in {int(o['ref_cap_hit'].sum())} steps.\n\n")
        rows = [[lab.replace("$", "").replace("\\varepsilon_t", "eps_t")] + [float(gaps[key][k]) for k in tt]
                + [int(o[f"{key}_inner"].sum()), int(o[f"{key}_cap_hit"].sum()), float(gaps[key].min()),
                   f"{o[f'{key}_wall'][-1]:.0f}"] for key, lab, _ in RUNS]
        lines.append(md_table(["run"] + [f"gap at t={k}" for k in tt] +
                              ["total inner its", "steps at cap", "min gap (signed)", "wall (s)"], rows))
        lines.append(f"\n- empirical slope of the accelerated certified run, log gap vs log t on t in [{lo}, {T}]: {slope:.2f}\n"
                     f"- max_t gap/bound (accelerated, certified): {float((gaps['accel_cert'][1:] / bound).max()):.4f}\n"
                     f"- steps with no admissible inner iterate: " +
                     ", ".join(f"{key} {int(o[f'{key}_fails'])}" for key, _, _ in RUNS) + "\n")
        for k, v in o.items():
            data[f"{nm}_{k}"] = v
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, d, "fig_e3")
    np.savez_compressed(os.path.join(d, "e3.npz"), r_best=r_best, **data)
    for nm in names:
        os.remove(os.path.join(d, f"e3_{nm}.npz"))
    return lines


#  ------------------------------------------------------------------ ensemble over random instances
GFLOOR = 1e-13
REL = (1e-4, 1e-8)


def ens_work(job):
    seed, a = job
    out = run_instance(random_gauss(np.random.default_rng([3, seed])), a, f"seed {seed}")
    return seed, {k: np.asarray(v) for k, v in out.items() if not k.startswith("ref_")}


def ensemble(args):
    d = outdir("e3")
    r_best, _ = load_r_best()
    T = args.T
    t = np.arange(T + 1)
    a = dict(n=args.n, T=T, cap=args.cap, ref_steps=args.ref_steps, ref_tol=args.ref_tol, ref_cap=args.ref_cap, r=r_best)
    say(f"E3 ensemble: {args.seeds} instances x (reference + 4 runs)")
    res = run_seeds(ens_work, range(args.seeds), a, args.jobs, os.path.join(d, "seeds"), args.resume)
    n = len(res)
    Gs = stack(res, "Gstar")[:, None]
    gap = {key: np.maximum(stack(res, f"{key}_G") - Gs, 0.0) for key, _, _ in RUNS}
    rel = {key: gap[key] / gap[key][:, :1] for key in gap}
    lines = [f"\n# Ensemble over {n} random instances\n\nSeed = instance: Gaussian target with centre ~ U([0.35, 0.65]^2) and "
             f"variances log-uniform on [0.01, 0.06] (as in E2); nu uniform, X_0 = Id, all four runs on the same instance, "
             f"each with its own reference run for G*. Tables: mean [95% bootstrap CI of the mean]; inner iterations: median "
             f"[95% bootstrap CI of the median], runs that never reach the gap are right-censored. Gaps are floored at "
             f"{GFLOOR:g} before taking log-ratios.\n\n"]
    tt = [k for k in (10, 30, 60) if k <= T]
    lines.append("Normalised gap (G(x_t) - G*)/(G(x_0) - G*):\n\n")
    lines.append(md_table(["run"] + [f"t={k}" for k in tt],
                          [[lab.replace("$", "").replace("\\varepsilon_t", "eps_t")] + [ci(boot_mean(rel[key][:, k])) for k in tt]
                           for key, lab, _ in RUNS]))
    lg = {key: np.log10(np.maximum(gap[key], GFLOOR)) for key in gap}
    pairs = [("certified", lg["accel_cert"] - lg["mm_cert"]), ("50 inner its", lg["accel_fixed"] - lg["mm_fixed"])]
    lines.append("\nPaired log10(gap accelerated / gap non-accelerated) (negative = acceleration helps):\n\n")
    lines.append(md_table(["inner accuracy"] + [f"t={k}" for k in tt],
                          [[nm] + [ci(boot_mean(v[:, k])) for k in tt] for nm, v in pairs]))
    rows = []
    for key, lab, _ in RUNS:
        cum = np.cumsum(stack(res, f"{key}_inner"), axis=1)
        row = [lab.replace("$", "").replace("\\varepsilon_t", "eps_t")]
        for th in REL:
            idx = first_idx(rel[key][:, 1:] <= th)
            its = np.where(idx >= 0, cum[np.arange(n), np.maximum(idx, 0)], np.inf)
            row += [ci(boot_median(its)), f"{int((idx >= 0).sum())}/{n}"]
        row.append(f"{int(stack(res, f'{key}_cap_hit').sum())}")
        rows.append(row)
    lines.append("\nTotal inner iterations until the normalised outer gap first drops below a level:\n\n")
    lines.append(md_table(["run", "to 1e-4", "reached", "to 1e-8", "reached", "steps at cap (all instances)"], rows))
    bound = 2 * stack(res, "w2nu0")[:, None] / (GAMMA * (t[None, 1:] + 1.0) ** 2)
    u = (gap["accel_cert"][:, 1:] / bound).max(axis=1)
    lines.append(f"\n- accelerated certified run, max_t gap/bound of Theorem 1: max over instances {u.max():.4f}, "
                 f"instances with ratio <= 1: {int((u <= 1).sum())}/{n}\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (key, lab, _), col in zip(RUNS, COLORS):
        plot_band(axes[0], t[1:], rel[key][:, 1:], col, lab, lo_floor=GFLOOR)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("outer step $t$")
    axes[0].set_ylabel(r"$(G(x_t)-G^\star)/(G(x_0)-G^\star)$")
    axes[0].set_title(f"normalised gap, {n} instances (median, 2.5–97.5%)")
    axes[0].legend(loc="lower left")
    for (nm, v), col in zip(pairs, COLORS):
        med, lo, hi = band(v[:, 1:])
        axes[1].plot(t[1:], med, color=col, label=nm)
        axes[1].fill_between(t[1:], lo, hi, color=col, alpha=0.18, lw=0)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("outer step $t$")
    axes[1].set_ylabel(r"paired $\log_{10}$(gap accelerated / gap non-accelerated)")
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, d, "fig_e3_ens")
    np.savez_compressed(os.path.join(d, "e3_ensemble.npz"), r_best=r_best,
                        **{k: stack(res, k) for k in res[0] if np.asarray(res[0][k]).dtype != object})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--ref-steps", type=int, default=1500)
    ap.add_argument("--ref-tol", type=float, default=1e-10)
    ap.add_argument("--ref-cap", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=20, help="random instances (0 = canonical targets only)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (wall-clock columns get noisier)")
    ap.add_argument("--resume", action="store_true", help="reuse per-seed files of an interrupted ensemble")
    ap.add_argument("--skip-canonical", action="store_true")
    args = ap.parse_args()
    path = os.path.join(outdir("e3"), "results.md")
    lines = [] if args.skip_canonical else canonical(args)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    if args.seeds > 0:
        lines += ensemble(args)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    main()
