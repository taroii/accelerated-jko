"""E1 — rates with exact steps (centered Gaussians in d = 2): Algorithm 1 vs the fixed-reference
minimizing movement, the moving-reference scheme and plain JKO, plus a sweep over the
strong-convexity parameter. Writes results/e1/{e1.npz, fig_e1.*, fig_e1b.*, results.md}."""
import argparse
import os

import numpy as np

import gaussian as ga
from common import (COLORS, FLOOR, add_references, band, boot_mean, ci, floor, md_table, outdir, plot_band, plt,
                    run_seeds, savefig, say, stack, theorem_bound, wilson)

GAMMA = 0.5
METHODS = [("accel", "Alg. 1 (accelerated, fixed reference)"),
           ("mm", r"fixed-reference minimizing movement ($\alpha_t\equiv1$)"),
           ("moving", "moving-reference scheme"),
           ("jko", "plain JKO")]


def reference_cov():
    c, s = np.cos(np.pi / 5), np.sin(np.pi / 5)
    Rm = np.array([[c, -s], [s, c]])
    return Rm @ np.diag([6.25, 2.25]) @ Rm.T


def run_all(Zc, Q, T, methods=("accel", "mm", "moving", "jko")):
    out = {}
    if "accel" in methods:
        out["accel"], out["zdist"], out["kkt_accel"] = ga.run_fixed_reference(Zc, Q, GAMMA, T, True)
    if "mm" in methods:
        out["mm"], _, out["kkt_mm"] = ga.run_fixed_reference(Zc, Q, GAMMA, T, False)
    if "moving" in methods:
        out["moving"], out["res_moving"] = ga.run_moving_reference(Zc, Q, GAMMA, T)
    if "jko" in methods:
        out["jko"], out["res_jko"] = ga.run_jko(Zc, Q, GAMMA, T)
    return out


def canonical(args):
    """The fixed instance of the spec."""
    T = args.T
    d = outdir("e1")
    Zc = reference_cov()
    cases = {"WEAK": np.diag([25.0, 10.0]), "STRONG": np.diag([1.0, 0.5])}
    t = np.arange(T + 1)
    data = {}
    lines = [f"# E1 — rates with exact steps (Gaussians)\n\nd = 2, nu = N(0, Zc), Zc = R(pi/5) diag(6.25, 2.25) R(pi/5)^T, "
             f"gamma = {GAMMA}, T = {T}. gap_t = KL(x_t | q), G* = 0.\n"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (cname, Q) in zip(axes, cases.items()):
        say(f"E1 canonical: {cname} target, 4 methods x {T} steps")
        out = run_all(Zc, Q, T)
        P = ga.ot_map(Zc, Q)
        w2 = ga.norm2_nu(np.eye(2) - P, Zc)
        bound = theorem_bound(w2, GAMMA, t[1:])
        ratio = float((out["accel"][1:] / bound).max())
        zr = float((out["zdist"] / np.sqrt(w2)).max())
        assert ratio <= 1.0, f"{cname}: gap exceeds the bound of Theorem 1 (ratio {ratio}) — tighten the prox solver"
        for (key, lab), col in zip(METHODS, COLORS):
            ax.loglog(t[1:], floor(out[key][1:]), color=col, label=lab)
            data[f"{cname}_{key}"] = out[key]
        add_references(ax, t[1:], out["accel"], bound=bound)
        ax.set_ylim(FLOOR / 3, None)
        ax.set_xlabel("outer step $t$")
        ax.set_ylabel(r"$G(x_t)-G^\star$")
        ax.set_title(f"{cname}: $\\Sigma_q$ = diag({Q[0, 0]:g}, {Q[1, 1]:g})")
        data[f"{cname}_zdist"] = out["zdist"]
        data[f"{cname}_bound"] = bound
        tt = [k for k in (10, 100, 300) if k <= T]
        lines.append(f"\n## {cname}: Q = diag({Q[0, 0]:g}, {Q[1, 1]:g})\n\n")
        lines.append(md_table(["method"] + [f"gap at t={k}" for k in tt],
                              [[lab.replace("$", "")] + [float(out[key][k]) for k in tt] for key, lab in METHODS]))
        lines.append(f"\n- W_2^2(nu, rho*) = ||I - P||_nu^2 = {w2:.6f}\n"
                     f"- max_t gap/bound for Alg. 1: {ratio:.4f} (must be <= 1)\n"
                     f"- max_t ||Z_t - P||_nu / ||I - P||_nu: {zr:.4f} (Theorem 1: <= 1)\n"
                     f"- max KKT residual of the restricted prox: accelerated {out['kkt_accel']:.1e}, "
                     f"minimizing movement {out['kkt_mm']:.1e}\n"
                     f"- max residual of the JKO solves: moving reference {out['res_moving']:.1e}, plain JKO {out['res_jko']:.1e}\n")
        print(f"{cname}: gap/bound max {ratio:.4f}, |Z-P| ratio max {zr:.4f}, kkt {out['kkt_accel']:.1e}", flush=True)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, d, "fig_e1")

    #  -- lambda sweep: Q = (1/lam) I
    lams = np.logspace(-3, 0, 13)
    N = args.sweep_steps
    final = {k: [] for k in ("accel", "mm", "jko")}
    say(f"E1 canonical: lambda sweep, {len(lams)} values x {N} steps")
    for lam in lams:
        out = run_all(Zc, np.eye(2) / lam, N, methods=("accel", "mm", "jko"))
        for k in final:
            final[k].append(out[k][-1])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for (key, lab), col in zip(METHODS, COLORS):
        if key in final:
            ax.loglog(lams, floor(final[key]), "o-", ms=3, color=col, label=lab)
            data[f"sweep_{key}"] = np.array(final[key])
    ax.set_ylim(FLOOR / 3, None)
    ax.set_xlabel(r"strong-convexity parameter $\lambda$ ($\Sigma_q=\lambda^{-1}I$)")
    ax.set_ylabel(f"gap after {N} steps")
    ax.legend()
    savefig(fig, d, "fig_e1b")
    data["sweep_lam"] = lams
    lines.append(f"\n## Lambda sweep: Q = (1/lam) I, gap after {N} steps\n\n")
    lines.append(md_table(["lam", "Alg. 1", "min. movement", "plain JKO"],
                          [[f"{l:.3e}", float(a), float(b), float(c)] for l, a, b, c in
                           zip(lams, final["accel"], final["mm"], final["jko"])]))
    np.savez_compressed(os.path.join(d, "e1.npz"), t=t, Zc=Zc, **data)
    return lines


#  ------------------------------------------------------------------ ensemble over random instances
REGIMES = {"weak": (0.02, 0.2), "strong": (0.5, 5.0)}      # range of the target precisions lambda_i
UFLOOR = 1e-14


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def random_reference(rng):
    """Zc = R(theta) diag(e^a1, e^a2) R(theta)^T, theta ~ U(0, pi), a_i ~ U(log 1, log 10)."""
    Rm = rot(rng.uniform(0, np.pi))
    return Rm @ np.diag(np.exp(rng.uniform(0, np.log(10), 2))) @ Rm.T


def ens_work(job):
    """One seed = one reference and one target per regime; all four methods run on that instance."""
    seed, a = job
    rng = np.random.default_rng([1, seed])
    Zc = random_reference(rng)
    out = {}
    for reg, (lo, hi) in REGIMES.items():
        Rq = rot(rng.uniform(0, np.pi))
        lam = np.exp(rng.uniform(np.log(lo), np.log(hi), 2))
        Q = Rq @ np.diag(1 / lam) @ Rq.T
        r = run_all(Zc, Q, a["T"])
        w2 = ga.norm2_nu(np.eye(2) - ga.ot_map(Zc, Q), Zc)
        for key, _ in METHODS:
            out[f"{reg}_{key}"] = r[key]
        out[f"{reg}_w2"] = w2
        out[f"{reg}_zratio"] = np.nanmax(r["zdist"]) / np.sqrt(w2)
        out[f"{reg}_kkt"] = max(r["kkt_accel"], r["kkt_mm"])
    return seed, out


def sweep_work(job):
    seed, a = job
    Zc = random_reference(np.random.default_rng([11, seed]))
    out = {k: [] for k in ("accel", "mm", "jko", "w2")}
    for lam in a["lams"]:
        Q = np.eye(2) / lam
        r = run_all(Zc, Q, a["sweep_steps"], methods=("accel", "mm", "jko"))
        for k in ("accel", "mm", "jko"):
            out[k].append(r[k][-1])
        out["w2"].append(ga.norm2_nu(np.eye(2) - ga.ot_map(Zc, Q), Zc))
    return seed, {k: np.array(v) for k, v in out.items()}


def ensemble(args):
    d = outdir("e1")
    T = args.T
    t = np.arange(T + 1)
    say(f"E1 ensemble: {args.seeds} instances x 2 regimes x 4 methods")
    res = run_seeds(ens_work, range(args.seeds), dict(T=T), args.jobs, os.path.join(d, "seeds"), args.resume)
    n = len(res)
    lines = [f"\n# Ensemble over {n} random instances\n\nSeed = instance: Zc = R(theta) diag(e^a1, e^a2) R(theta)^T with "
             "theta ~ U(0, pi), a_i ~ U(0, log 10); Q = R(theta') diag(1/lam1, 1/lam2) R(theta')^T with lam_i log-uniform "
             "on [0.02, 0.2] (weak) or [0.5, 5] (strong). All four methods run on the same instance. Reported quantity: the "
             "bound utilisation u_t = gap_t gamma (t+1)^2 / (2 W_{2,nu}^2(x_0, rho*)); Theorem 1 says u_t <= 1 for Alg. 1 "
             "on every instance. Tables: mean [95% bootstrap CI of the mean, 2000 resamples]; figures: median and "
             f"2.5-97.5 percentile band. Gaps are floored at {UFLOOR:g} before taking log-ratios.\n"]
    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tt = [k for k in (10, 100, 300) if k <= T]
    for ax, reg in zip(axes, REGIMES):
        w2 = stack(res, f"{reg}_w2")[:, None]
        scale = GAMMA * (t[None, :] + 1.0) ** 2 / (2 * w2)
        U = {key: np.maximum(stack(res, f"{reg}_{key}"), 0.0) * scale for key, _ in METHODS}   # roundoff-negative gaps -> 0
        for (key, lab), col in zip(METHODS, COLORS):
            plot_band(ax, t[1:], U[key][:, 1:], col, lab, lo_floor=UFLOOR)
            data[f"{reg}_{key}_u"] = U[key]
        ax.axhline(1.0, ls=":", color="k", lw=1.2, label="Theorem 1: $u_t\\leq 1$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(UFLOOR / 3, 10)
        ax.set_xlabel("outer step $t$")
        ax.set_ylabel(r"bound utilisation $u_t$")
        ax.set_title(f"{reg} regime, {n} instances")
        umax = np.nanmax(U["accel"][:, 1:], axis=1)
        ok = int((umax <= 1).sum())
        fails = {key: int(np.isnan(U[key]).any(axis=1).sum()) for key, _ in METHODS}
        lines.append(f"\n## {reg} regime\n\n")
        lines.append(md_table(["method"] + [f"u at t={k}" for k in tt],
                              [[lab.replace("$", "")] + [ci(boot_mean(U[key][:, k])) for k in tt] for key, lab in METHODS]))
        la = np.log10(np.maximum(stack(res, f"{reg}_accel"), UFLOOR))
        lm = np.log10(np.maximum(stack(res, f"{reg}_mm"), UFLOOR))
        lines.append(f"\n- Alg. 1: max over instances and t of u_t = {np.nanmax(umax):.4f}; instances with u_t <= 1 for all t: "
                     f"{ok}/{n} (Wilson 95%: [{wilson(ok, n)[0]:.3f}, {wilson(ok, n)[1]:.3f}])\n"
                     f"- max over instances and t of ||Z_t - P||_nu / ||I - P||_nu: {np.nanmax(stack(res, f'{reg}_zratio')):.6f}\n"
                     f"- paired log10(gap accelerated / gap minimizing movement): " +
                     "; ".join(f"t={k}: {ci(boot_mean((la - lm)[:, k]))}" for k in tt) + "\n"
                     f"- max KKT residual of the restricted prox over all instances: {np.nanmax(stack(res, f'{reg}_kkt')):.1e}\n"
                     f"- instances in which a solve failed (run truncated): " + ", ".join(f"{k} {v}" for k, v in fails.items()) + "\n")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, d, "fig_e1_ens")

    if args.sweep_seeds > 0:
        lams = np.logspace(-3, 0, 13)
        say(f"E1 ensemble: lambda sweep, {args.sweep_seeds} references x {len(lams)} values")
        sres = run_seeds(sweep_work, range(args.sweep_seeds), dict(lams=lams, sweep_steps=args.sweep_steps), args.jobs,
                         os.path.join(d, "seeds_sweep"), args.resume)
        N = args.sweep_steps
        scale = GAMMA * (N + 1.0) ** 2 / (2 * stack(sres, "w2"))
        fig, ax = plt.subplots(figsize=(5, 3.8))
        rows = []
        for (key, lab), col in zip(METHODS, COLORS):
            if key in ("accel", "mm", "jko"):
                Uf = np.maximum(stack(sres, key), 0.0) * scale
                plot_band(ax, lams, Uf, col, lab, lo_floor=UFLOOR)
                data[f"sweep_{key}_u"] = Uf
                rows.append((lab, Uf))
        ax.axhline(1.0, ls=":", color="k", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(UFLOOR / 3, None)
        ax.set_xlabel(r"strong-convexity parameter $\lambda$ ($\Sigma_q=\lambda^{-1}I$)")
        ax.set_ylabel(f"bound utilisation after {N} steps")
        ax.legend()
        savefig(fig, d, "fig_e1b_ens")
        lines.append(f"\n## Lambda sweep, {len(sres)} random references per lambda: u after {N} steps (median [2.5, 97.5 percentile])\n\n")
        fmtb = lambda a: "{:.2e} [{:.1e}, {:.1e}]".format(*band(a))
        lines.append(md_table(["lam"] + [lab.replace("$", "") for lab, _ in rows],
                              [[f"{l:.3e}"] + [fmtb(Uf[:, i]) for _, Uf in rows] for i, l in enumerate(lams)]))
    np.savez_compressed(os.path.join(d, "e1_ensemble.npz"), t=t, **data)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--sweep-steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=100, help="random instances (0 = canonical instance only)")
    ap.add_argument("--sweep-seeds", type=int, default=50, help="random references per lambda in the sweep")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="reuse per-seed files of an interrupted ensemble")
    ap.add_argument("--skip-canonical", action="store_true")
    args = ap.parse_args()
    path = os.path.join(outdir("e1"), "results.md")
    lines = [] if args.skip_canonical else canonical(args)
    with open(path, "w") as f:
        f.write("".join(lines))
    if args.seeds > 0:
        lines += ensemble(args)
        with open(path, "w") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    main()
