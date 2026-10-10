"""E3 — end to end: Algorithm 1 with every proximal step solved by Algorithm 2, certified
schedule eps_t = eps0 (t+1)^-4.5 vs a fixed inner budget, accelerated vs non-accelerated.
Writes results/e3/{e3.npz, fig_e3.*, results.md}. No arguments needed (run E2 first for r_best)."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from common import COLORS, FLOOR, RESULTS, add_references, floor, md_table, outdir, plt, savefig, theorem_bound
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


def work(job):
    name, a = job
    grid = Grid(a["n"])
    tgt = make_target(name)
    sig = (1.0, a["r"])
    out = {}
    # reference: fixed-reference minimizing movement at a constant certificate
    ref = run_outer(grid, tgt, GAMMA, a["ref_steps"], False, sig, tol_const=a["ref_tol"], cap=a["ref_cap"])
    out["ref_G"] = ref["G"]
    out["ref_inner"] = ref["inner"]
    out["ref_cap_hit"] = ref["cap_hit"]
    out["Gstar"] = ref["G"].min()
    out["w2nu0"] = float(((ref["X"] - grid.X) ** 2).sum() * grid.h ** 2)
    print(f"[{name}] reference: G* = {out['Gstar']:.12f}, W2nu0 = {out['w2nu0']:.6f}, "
          f"inner its {ref['inner'].sum()}, cap hits {int(ref['cap_hit'].sum())}", flush=True)
    for key, _, kw in RUNS:
        r = run_outer(grid, tgt, GAMMA, a["T"], kw["accel"], sig, eps0=EPS0, cap=a["cap"], fixed=kw.get("fixed"))
        for f in ("G", "inner", "wall", "cap_hit", "cert"):
            out[f"{key}_{f}"] = r[f]
        out[f"{key}_fails"] = r["fails"]
        print(f"[{name}] {key}: final gap {r['G'][-1] - out['Gstar']:.3e}, inner its {r['inner'].sum()}, "
              f"cap hits {int(r['cap_hit'].sum())}, {r['wall'][-1]:.0f}s", flush=True)
    np.savez_compressed(os.path.join(outdir("e3"), f"e3_{name}.npz"), **out)
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--ref-steps", type=int, default=1500)
    ap.add_argument("--ref-tol", type=float, default=1e-10)
    ap.add_argument("--ref-cap", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1, help="targets in parallel (wall-clock columns get noisier)")
    args = ap.parse_args()
    r_best, found = load_r_best()
    a = dict(vars(args), r=r_best)
    d = outdir("e3")
    names = ["gauss", "quartic"]
    jobs = [(nm, a) for nm in names]
    if args.jobs > 1:
        with Pool(min(args.jobs, len(jobs))) as pool:
            res = dict(pool.map(work, jobs))
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
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("".join(lines))
    np.savez_compressed(os.path.join(d, "e3.npz"), r_best=r_best, **data)
    for nm in names:
        os.remove(os.path.join(d, f"e3_{nm}.npz"))


if __name__ == "__main__":
    main()
