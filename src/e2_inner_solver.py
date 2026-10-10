"""E2 — the inner solver: one restricted proximal step, scalar-step Chambolle–Pock vs the
two-channel spectrally preconditioned PDHG, channel-ratio sweep, balanced steps, and n-scaling.
Writes results/e2/{e2.npz, fig_e2.pdf/png, results.md, r_best.json}. No arguments needed."""
import argparse
import json
import os

import numpy as np

from common import COLORS, FLOOR, first_below, floor, md_table, outdir, plt, savefig
from grid import Gauss, Grid, Quartic
from pdhg import ProxProblem, pdhg

GAMMA = 0.05
THETA = 0.99
THRESH = (1e-2, 1e-4, 1e-6, 1e-8)
RATIOS = (0.1, 1.0, 10.0, 100.0)


def targets():
    return [Gauss((0.5, 0.5), (np.sqrt(0.02), np.sqrt(0.05))), Quartic((0.5, 0.5), 200.0)]


def field_Y(g):
    """Y = x + 0.08 grad( sin(2 pi x) cos(4 pi y) + 0.5 cos(2 pi y) ), gradient computed analytically."""
    x, y = g.X[..., 0], g.X[..., 1]
    gx = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y)
    gy = -4 * np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y) - np.pi * np.sin(2 * np.pi * y)
    return g.X + 0.08 * np.stack([gx, gy], -1)


def reference(prob, n_ref):
    """J* = min J over a long preconditioned run with ratio 10; also the converged primal and duals."""
    ref = pdhg(prob, 1.0, 10.0, n_ref, theta=THETA, record=True)
    return ref["J"].min(), ref


def summarize(run, Jstar, h2):
    gap = (run["J"] - Jstar) * h2
    its = [first_below(gap, th) for th in THRESH]
    cs = run["certs"]
    ratios, specs = [], []
    for k in its:
        j = None if k is None else np.searchsorted(cs[:, 0], k)
        if j is None or j >= len(cs):
            ratios.append(None)
            specs.append(None)
        else:
            gk = max((cs[j, 3] - Jstar) * h2, 1e-16)
            ratios.append(cs[j, 2] / gk)
            specs.append(cs[j, 1] / gk)
    return dict(gap=gap, its=its, ratio=ratios, ratio_spec=specs, certs=cs, time=run["time"], sig=run["sig"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-big", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--ref-iters", type=int, default=20000)
    args = ap.parse_args()
    d = outdir("e2")
    grid = Grid(args.n)
    data, lines, R = {}, [], {}
    lines.append(f"# E2 — inner solver\n\nn = {args.n}, gamma = {GAMMA}, theta = {THETA}, N = {args.iters} iterations per run, "
                 f"reference J* from {args.ref_iters} preconditioned iterations (ratio 10). Gaps and certificates are in "
                 f"h^2-weighted units (J h^2).\n")
    balanced = {}
    for tgt in targets():
        prob = ProxProblem(grid, tgt, GAMMA, field_Y(grid))
        Jstar, ref = reference(prob, args.ref_iters)
        st = ref["state"]
        psi_c = ref["last"]["c"]
        sL = np.sqrt(THETA) * np.linalg.norm(st["Lam"]) / np.linalg.norm(grid.K1c(psi_c))
        sm = np.sqrt(THETA) * np.linalg.norm(st["mu"]) / np.linalg.norm(grid.K2c(psi_c))
        balanced[tgt.name] = (float(sL), float(sm))
        runs = {"scalar": dict(scalar=True, s=(None, None))}
        for r in RATIOS:
            runs[f"precond r={r:g}"] = dict(s=(1.0, r))
        runs["precond (0.1, 1)"] = dict(s=(0.1, 1.0))
        runs["precond (10, 100)"] = dict(s=(10.0, 100.0))
        runs["balanced eq.(15)"] = dict(s=(sL, sm))
        R[tgt.name] = {}
        for name, kw in runs.items():
            run = pdhg(prob, kw["s"][0], kw["s"][1], args.iters, theta=THETA, scalar=kw.get("scalar", False),
                       record=True)
            R[tgt.name][name] = summarize(run, Jstar, prob.h2)
            print(f"[{tgt.name}] {name:20s} its to {THRESH}: {R[tgt.name][name]['its']}  ({run['time']:.1f}s)", flush=True)
        data[f"{tgt.name}_Jstar"] = Jstar * prob.h2
        data[f"{tgt.name}_psi_star"] = grid.synth(psi_c)
        data[f"{tgt.name}_ref_final_cert"] = np.array([ref["cert_spec"], ref["cert"]])

    # r_best = argmin iterations-to-1e-6 over the ratio sweep (summed over the two targets)
    def score(r):
        its = [R[t.name][f"precond r={r:g}"]["its"][2] for t in targets()]
        return np.inf if any(i is None for i in its) else sum(its)
    r_best = min(RATIOS, key=score)
    with open(os.path.join(d, "r_best.json"), "w") as f:
        json.dump({"r_best": r_best, "sig_L": 1.0, "sig_m": r_best}, f)

    # n-scaling run with the best ratio
    gbig = Grid(args.n_big)
    for tgt in targets():
        prob = ProxProblem(gbig, tgt, GAMMA, field_Y(gbig))
        Jstar, _ = reference(prob, args.ref_iters)
        run = pdhg(prob, 1.0, r_best, args.iters, theta=THETA, record=True)
        R[tgt.name][f"n={args.n_big}, r={r_best:g}"] = summarize(run, Jstar, prob.h2)
        print(f"[{tgt.name}] n={args.n_big} r={r_best:g}: {R[tgt.name][f'n={args.n_big}, r={r_best:g}']['its']}", flush=True)

    #  -- results.md and npz
    for tgt in targets():
        lines.append(f"\n## Target: {tgt.name}\n\nIterations to gap < threshold (None = not reached in {args.iters}):\n")
        rows = [[name, f"{s['sig'][0]:.4g}", f"{s['sig'][1]:.4g}"] + s["its"] + [float(max(s["gap"][-1], 0.0)), f"{s['time']:.1f}"]
                for name, s in R[tgt.name].items()]
        lines.append(md_table(["run", "sig_L", "sig_m"] + [f"<{t:g}" for t in THRESH] + ["final gap", "time (s)"], rows))
        lines.append("\nCertificate / true gap at the first certificate evaluation (every 25 its) after each threshold "
                     "is crossed — projected certificate (spec certificate in parentheses):\n")
        rows = [[name] + ["None" if a is None else f"{a:.2e} ({b:.2e})" for a, b in zip(s["ratio"], s["ratio_spec"])]
                for name, s in R[tgt.name].items()]
        lines.append(md_table(["run"] + [f"<{t:g}" for t in THRESH], rows))
        sL, sm = balanced[tgt.name]
        cf = data[f"{tgt.name}_ref_final_cert"]
        lines.append(f"\nBalanced steps of eq. (15): sig_L = {sL:.4g}, sig_m = {sm:.4g}, ratio sig_m/sig_L = {sm / sL:.4g}.\n"
                     f"\nReference run, final certificates: spec (unprojected) = {cf[0]:.3e}, projected = {cf[1]:.3e}; "
                     f"J* h^2 = {data[f'{tgt.name}_Jstar']:.12f}.\n")
        for name, s in R[tgt.name].items():
            key = f"{tgt.name}__{name}".replace(" ", "_")
            data[key + "__gap"] = s["gap"]
            data[key + "__certs"] = s["certs"]
    lines.append(f"\n## Channel ratio\n\nr_best = {r_best:g} (argmin of iterations to 1e-6 over sig_L = 1, sig_m = r, "
                 f"summed over the two targets); written to r_best.json and used by E3/E4.\n")
    lines.append("\n## Deviation from the spec: the certificate\n\n"
                 "The certificate of Remark 4 as specified, `J - (-h1* - h2* - R ||B^T Lam + mu||_1)`, is a valid upper "
                 "bound on the suboptimality but does **not** vanish: at a saddle point only `K2^T (B^T Lam + mu) = 0` "
                 "holds, and `r = B^T Lam + mu` keeps a nonzero part orthogonal to gradient fields (see the 'spec' "
                 "values above, which stay O(1)). Since `<K2 psi, r> = <K2 psi, P r>` with `P` the orthogonal projection "
                 "onto range(K2) (one extra DCT pair), `R ||P r||_1` is an equally valid bound and it does vanish. "
                 "All stopping rules (E3, E4) use this projected certificate; both are recorded here.\n")
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("".join(lines))
    np.savez_compressed(os.path.join(d, "e2.npz"), r_best=r_best, **data)

    #  -- figure
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for row, tgt in enumerate(targets()):
        ax = axes[row, 0]
        for (name, lab), col in zip([("scalar", "scalar-step PDHG")] + [(f"precond r={r:g}", rf"preconditioned, $\sigma_\mu/\sigma_\Lambda={r:g}$")
                                                                         for r in (1.0, 10.0, 100.0)], COLORS):
            gp = R[tgt.name][name]["gap"]
            k = np.arange(1, len(gp) + 1)
            ok = np.isfinite(gp)
            ax.loglog(k[ok], floor(gp[ok]), color=col, label=lab)
        ax.set_ylim(FLOOR / 3, None)
        ax.set_xlabel("inner iteration $k$")
        ax.set_ylabel(r"$J(\psi^k)-J^\star$")
        ax.set_title(f"{tgt.name}: objective gap")
        ax.legend()
        ax = axes[row, 1]
        s = R[tgt.name]["precond r=10"]
        gp = s["gap"]
        k = np.arange(1, len(gp) + 1)
        ok = np.isfinite(gp)
        ax.loglog(k[ok], floor(gp[ok]), color=COLORS[2], label="true gap")
        ax.loglog(s["certs"][:, 0], floor(s["certs"][:, 2]), "o-", ms=2.5, color=COLORS[0], label="certificate (projected)")
        ax.loglog(s["certs"][:, 0], floor(s["certs"][:, 1]), "s--", ms=2.5, color="0.5", label="certificate (Remark 4 as written)")
        ax.set_ylim(FLOOR / 3, None)
        ax.set_xlabel("inner iteration $k$")
        ax.set_title(rf"{tgt.name}: certificate vs gap, $\sigma_\mu/\sigma_\Lambda=10$")
        ax.legend()
    fig.tight_layout()
    savefig(fig, d, "fig_e2")
    print("r_best =", r_best)


if __name__ == "__main__":
    main()
