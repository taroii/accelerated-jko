"""E2 — the inner solver: one restricted proximal step, scalar-step Chambolle–Pock vs the
two-channel spectrally preconditioned PDHG, channel-ratio sweep, balanced steps, and n-scaling.
Writes results/e2/{e2.npz, fig_e2.pdf/png, results.md, r_best.json}. No arguments needed."""
import argparse
import json
import os

import numpy as np

from common import (COLORS, FLOOR, band, boot_median, ci, first_below, floor, md_table, outdir, plot_band, plt,
                    run_seeds, savefig, say, stack)
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


def canonical(args):
    """The fixed instance of the spec; fixes r_best."""
    d = outdir("e2")
    grid = Grid(args.n)
    data, lines, R = {}, [], {}
    lines.append(f"# E2 — inner solver\n\nn = {args.n}, gamma = {GAMMA}, theta = {THETA}, N = {args.iters} iterations per run, "
                 f"reference J* from {args.ref_iters} preconditioned iterations (ratio 10). Gaps and certificates are in "
                 f"h^2-weighted units (J h^2).\n")
    balanced = {}
    for tgt in targets():
        prob = ProxProblem(grid, tgt, GAMMA, field_Y(grid))
        say(f"E2 canonical [{tgt.name}]: reference run, {args.ref_iters} iterations")
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
        say(f"E2 canonical [{tgt.name}]: n = {args.n_big} scaling run (reference + {args.iters} iterations)")
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
    return lines


#  ------------------------------------------------------------------ ensemble over random instances
CONFIGS = [("scalar", "scalar-step PDHG", dict(scalar=True, s=(None, None)))] + \
          [(f"r{r:g}", rf"preconditioned, $\sigma_\mu/\sigma_\Lambda={r:g}$", dict(s=(1.0, r))) for r in RATIOS] + \
          [("s0.1_1", "preconditioned (0.1, 1)", dict(s=(0.1, 1.0))), ("s10_100", "preconditioned (10, 100)", dict(s=(10.0, 100.0))),
           ("balanced", "balanced eq. (15)", dict())]


def random_field(g, rng, K=6, amp=0.08):
    """Y = x + amp (grad phi + curl chi): phi, chi random cosine series with coefficients N(0,1)/|k|^3,
    |k|_inf <= K, each scaled to unit sup norm on the grid. The curl part makes Y a non-gradient field."""
    k = np.arange(K + 1)
    kn = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    kn[0, 0] = np.inf
    x = g.X[:, 0, 0]
    C, S = np.cos(np.pi * np.outer(k, x)), np.sin(np.pi * np.outer(k, x))

    def potential():
        c = rng.normal(size=(K + 1, K + 1)) / kn ** 3
        phi = np.einsum("kl,ki,lj->ij", c, C, C)
        dx = np.einsum("kl,ki,lj->ij", -np.pi * k[:, None] * c, S, C)
        dy = np.einsum("kl,ki,lj->ij", -np.pi * k[None, :] * c, C, S)
        return dx / np.abs(phi).max(), dy / np.abs(phi).max()

    px, py = potential()
    cx, cy = potential()
    return g.X + amp * np.stack([px + cy, py - cx], -1)


def random_gauss(rng):
    """Centre m ~ U([0.35, 0.65]^2), variances log-uniform on [0.01, 0.06]."""
    m = rng.uniform(0.35, 0.65, 2)
    var = np.exp(rng.uniform(np.log(0.01), np.log(0.06), 2))
    return Gauss(m, np.sqrt(var))


def ens_work(job):
    seed, a = job
    rng = np.random.default_rng([2, seed])
    grid = Grid(a["n"])
    prob = ProxProblem(grid, random_gauss(rng), GAMMA, random_field(grid, rng))
    Jstar, ref = reference(prob, a["ref_iters"])
    z = np.zeros((grid.n, grid.n))
    J0 = prob.values(grid.K1c(z), grid.K2c(z))[1]
    st, psi_c = ref["state"], ref["last"]["c"]
    bal = (np.sqrt(THETA) * np.linalg.norm(st["Lam"]) / np.linalg.norm(grid.K1c(psi_c)),
           np.sqrt(THETA) * np.linalg.norm(st["mu"]) / np.linalg.norm(grid.K2c(psi_c)))
    out = dict(balanced_sig=np.array(bal), ref_cert=np.array([ref["cert_spec"], ref["cert"]]), gap0=(J0 - Jstar) * prob.h2)
    for key, _, kw in CONFIGS:
        s = kw.get("s", bal)
        run = pdhg(prob, s[0], s[1], a["iters"], theta=THETA, scalar=kw.get("scalar", False), record=True)
        rel = (run["J"] - Jstar) / (J0 - Jstar)                       # gap normalised by the initial gap
        cs = run["certs"]
        its, ratio = [], []
        for th in THRESH:
            k = first_below(rel, th)
            its.append(np.inf if k is None else k)
            j = None if k is None else np.searchsorted(cs[:, 0], k)
            ratio.append(np.nan if j is None or j >= len(cs) else cs[j, 2] / max((cs[j, 3] - Jstar) * prob.h2, 1e-16))
        out[f"{key}_rel"] = rel
        out[f"{key}_its"] = np.array(its)
        out[f"{key}_ratio"] = np.array(ratio)
    return seed, out


def ensemble(args):
    d = outdir("e2")
    say(f"E2 ensemble: {args.seeds} instances x {len(CONFIGS)} solver configurations")
    res = run_seeds(ens_work, range(args.seeds), dict(n=args.n, iters=args.iters, ref_iters=args.ref_iters), args.jobs,
                    os.path.join(d, "seeds"), args.resume)
    n = len(res)
    lines = [f"\n# Ensemble over {n} random instances\n\nSeed = instance: Gaussian target with centre ~ U([0.35, 0.65]^2) and "
             "variances log-uniform on [0.01, 0.06]; field Y = x + 0.08 (grad phi + curl chi) with random cosine-series "
             "potentials (coefficients N(0,1)/|k|^3, |k|_inf <= 6, unit sup norm), so Y is not a gradient. All solver "
             f"configurations run on the same instance (n = {args.n}, N = {args.iters}). Thresholds are on the gap normalised "
             "by the initial gap J(0) - J*. Iterations: median [95% bootstrap CI of the median]; a run that never reaches the "
             "threshold is right-censored (inf).\n\n"]
    rows, rrows, data = [], [], {}
    for key, lab, _ in CONFIGS:
        its = stack(res, f"{key}_its")
        ratio = stack(res, f"{key}_ratio")
        data[f"{key}_its"], data[f"{key}_rel"], data[f"{key}_ratio"] = its, stack(res, f"{key}_rel"), ratio
        row, rrow = [lab.replace("$", "")], [lab.replace("$", "")]
        for j in (1, 3):
            row += [ci(boot_median(its[:, j])), f"{int(np.isfinite(its[:, j]).sum())}/{n}"]
            r = ratio[:, j][np.isfinite(ratio[:, j])]
            rrow.append("—" if r.size == 0 else "{:.2e} [{:.1e}, {:.1e}]".format(*band(r)))
        rows.append(row)
        rrows.append(rrow)
    lines.append(md_table(["configuration", "its to 1e-4", "reached", "its to 1e-8", "reached"], rows))
    lines.append("\nProjected certificate / true gap at the first certificate evaluation after the threshold is crossed "
                 "(median [2.5, 97.5 percentile] over instances):\n\n")
    lines.append(md_table(["configuration", "at 1e-4", "at 1e-8"], rrows))
    bal = stack(res, "balanced_sig")
    rc = stack(res, "ref_cert")
    med6 = {r: boot_median(stack(res, f"r{r:g}_its")[:, 2])[0] for r in RATIOS}
    lines.append(f"\n- balanced ratio sig_m/sig_L of eq. (15): median {np.median(bal[:, 1] / bal[:, 0]):.3g}, "
                 f"range [{(bal[:, 1] / bal[:, 0]).min():.3g}, {(bal[:, 1] / bal[:, 0]).max():.3g}]\n"
                 f"- ratio with the smallest median iterations to 1e-6 over the ensemble: {min(med6, key=med6.get):g} "
                 f"(medians: " + ", ".join(f"r={r:g}: {v:g}" for r, v in med6.items()) + ")\n"
                 f"- converged reference runs: certificate as written in Remark 4, median {np.median(rc[:, 0]):.3e} "
                 f"(min {rc[:, 0].min():.3e}); projected certificate, median {np.median(rc[:, 1]):.3e} (max {rc[:, 1].max():.3e})\n")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    k = np.arange(1, args.iters + 1)
    for (key, lab, _), col in zip([c for c in CONFIGS if c[0] in ("scalar", "r1", "r10", "r100")], COLORS):
        plot_band(axes[0], k, data[f"{key}_rel"], col, lab)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(FLOOR / 3, None)
    axes[0].set_xlabel("inner iteration $k$")
    axes[0].set_ylabel(r"$(J(\psi^k)-J^\star)/(J(0)-J^\star)$")
    axes[0].set_title(f"normalised gap, {n} instances (median, 2.5–97.5%)")
    axes[0].legend()
    for i, (key, lab, _) in enumerate(CONFIGS):
        for j, col, off in ((1, COLORS[0], -0.15), (3, COLORS[1], 0.15)):
            m, lo, hi = boot_median(data[f"{key}_its"][:, j])
            if np.isfinite(m):
                hi = hi if np.isfinite(hi) else args.iters
                axes[1].errorbar(m, i + off, xerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=col, capsize=2)
    axes[1].set_yticks(range(len(CONFIGS)))
    axes[1].set_yticklabels([lab for _, lab, _ in CONFIGS], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("iterations (median, 95% CI; no marker = not reached)")
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLORS[0], label="to normalised gap $10^{-4}$"),
                            plt.Line2D([], [], marker="o", ls="", color=COLORS[1], label="to normalised gap $10^{-8}$")])
    fig.tight_layout()
    savefig(fig, d, "fig_e2_ens")
    np.savez_compressed(os.path.join(d, "e2_ensemble.npz"), **data)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-big", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--ref-iters", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=50, help="random instances (0 = canonical instance only)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="reuse per-seed files of an interrupted ensemble")
    ap.add_argument("--skip-canonical", action="store_true")
    args = ap.parse_args()
    path = os.path.join(outdir("e2"), "results.md")
    lines = [] if args.skip_canonical else canonical(args)
    with open(path, "w") as f:
        f.write("".join(lines))
    if args.seeds > 0:
        lines += ensemble(args)
        with open(path, "w") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    main()
