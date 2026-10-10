"""E1 — rates with exact steps (centered Gaussians in d = 2): Algorithm 1 vs the fixed-reference
minimizing movement, the moving-reference scheme and plain JKO, plus a sweep over the
strong-convexity parameter. Writes results/e1/{e1.npz, fig_e1.*, fig_e1b.*, results.md}."""
import argparse
import os

import numpy as np

import gaussian as ga
from common import COLORS, FLOOR, add_references, floor, md_table, outdir, plt, savefig, theorem_bound

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--sweep-steps", type=int, default=200)
    args = ap.parse_args()
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
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("".join(lines))
    np.savez_compressed(os.path.join(d, "e1.npz"), t=t, Zc=Zc, **data)


if __name__ == "__main__":
    main()
