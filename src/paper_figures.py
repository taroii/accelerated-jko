"""Figures and tables for the paper, built from the saved results of E1-E3 (nothing is re-run).
Writes results/figures/: fig_main_*.*, fig_app_*.*, tab_*.tex and tables.md. No arguments needed."""
import os

import numpy as np

from common import RESULTS, boot_median, outdir, plt, savefig, say
from e1_gaussian_rates import GAMMA, first_step

ACC, BASE, GREY, DARK = "#0072B2", "#D55E00", "#8C8C8C", "#222222"
RATIO_COLS = {"1": "#9ECAE1", "10": "#3182BD", "100": "#08306B"}
NOISE = 1e-10            # gaps below this are round-off; axes stop here
FULL = 5.5               # text width of the paper (inches)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False})


def load(name):
    return np.load(os.path.join(RESULTS, name), allow_pickle=True)


def clip(g):
    return np.maximum(np.asarray(g, float), NOISE)


def noise_band(ax, label=True, left=False):
    """Mark the region where gaps are numerical round-off."""
    ax.axhspan(NOISE, 10 * NOISE, color="0.88", lw=0, zorder=0)
    if label:
        ax.text(0.02 if left else 0.98, 0.02, "round-off", transform=ax.transAxes, ha="left" if left else "right",
                va="bottom", fontsize=6, color="0.4")


#  ------------------------------------------------------------------ main-body figures
L_ACC, L_BASE, L_MM = "accelerated (Alg. 1)", "JKO", r"non-accelerated ($\alpha_t\equiv1$)"


def fig_main_outer(d, flat, ens):
    """Acceleration against plain JKO: one flat target, and the gain against the curvature of the target."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    # (a) flat target: gap against outer steps
    ax = axes[0]
    lam = float(flat["lams"][0])
    FLAT_T = int(flat["T"])
    t = np.arange(1, FLAT_T + 1)
    a, m = flat[f"accel_{lam:g}"][1:], flat[f"jko_{lam:g}"][1:]
    best = np.minimum.accumulate(a)
    bound = 2 * float(flat[f"w2_{lam:g}"]) / (GAMMA * (t + 1.0) ** 2)
    ax.loglog(t, bound, ":", color=DARK, lw=1.0, label="guarantee of Theorem 1")
    ax.loglog(t, m, "--", color=BASE, lw=1.8, label=L_BASE)
    ax.loglog(t, a, color=ACC, lw=0.6, alpha=0.4, label="accelerated, every iterate")
    ax.loglog(t, best, color=ACC, lw=2.2, label=L_ACC + ", best so far")
    ax.set_ylim(1e-8, 3e3)
    ax.set_xlim(1, FLAT_T)
    ax.set_xlabel("proximal steps $t$")
    ax.set_ylabel(r"objective gap $\mathrm{KL}(x_t\,\|\,q)$")
    ax.set_title(rf"(a) a flat target ($\lambda=10^{{{int(np.log10(lam))}}}$)")
    # (b) gap / bound after a fixed budget against the curvature of the target, 50 random references
    ax = axes[1]
    lams = np.geomspace(1e-3, 1.0, ens["sweep_accel_u"].shape[1])
    keep = lams <= 0.1 * 1.001
    ax.axhline(1.0, color=DARK, ls=":", lw=1.0)
    for key, col, ls, lw in (("jko", BASE, "--", 1.8), ("accel", ACC, "-", 2.2)):
        lo, med, hi = np.percentile(clip(ens[f"sweep_{key}_u"]), [2.5, 50, 97.5], axis=0)
        ax.fill_between(lams[keep], lo[keep], hi[keep], color=col, alpha=0.18, lw=0)
        ax.loglog(lams[keep], med[keep], ls, color=col, lw=lw, marker="o", ms=2.5)
    ax.text(0.1, 1.3, "guarantee of Theorem 1", fontsize=6.5, color=DARK, va="bottom", ha="right")
    ax.set_ylim(NOISE, 30)
    noise_band(ax, left=True)
    ax.set_xlabel(r"curvature of the target, $\lambda$   ($\leftarrow$ flatter)")
    ax.set_ylabel("gap / guarantee after 200 steps")
    ax.set_title("(b) the gain grows as the target flattens")
    for ax in axes:
        ax.grid(True, which="major", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend([h[i] for i in (3, 2, 1, 0)], [l[i] for i in (3, 2, 1, 0)], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    savefig(fig, d, "fig_main_outer")


def fig_main_solver(d, e2e, e3e, r):
    """(a) one proximal step on 50 random instances: scalar-step PDHG against Algorithm 2;
    (b) the whole method on 20 random instances, 50 inner iterations per step: gap against total inner iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    ax = axes[0]
    K = 400
    k = np.arange(1, K + 1)
    for key, col, ls, lw, lab in (("scalar", BASE, "--", 1.8, "scalar-step PDHG"),
                                  ("r1", "#9ECAE1", "-", 1.2, r"preconditioned (Alg. 2), $\sigma_\mu/\sigma_\Lambda=1$"),
                                  (f"r{r}", ACC, "-", 2.2, rf"preconditioned (Alg. 2), $\sigma_\mu/\sigma_\Lambda={r}$")):
        lo, med, hi = np.percentile(clip(ens_rel(e2e, key)[:, :K]), [2.5, 50, 97.5], axis=0)
        ax.fill_between(k, lo, hi, color=col, alpha=0.18, lw=0)
        ax.semilogy(k, med, ls, color=col, lw=lw, label=lab)
    noise_band(ax)
    ax.set_ylim(NOISE, 5)
    ax.set_xlim(0, K)
    ax.set_xlabel("inner iterations")
    ax.set_ylabel("relative gap")
    ax.set_title("(a) one proximal step")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    ax = axes[1]
    if e3e is not None:
        Gs = e3e["Gstar"][:, None]
        for key, col, ls, lw, lab in (("mm_fixed", BASE, "--", 1.8, L_MM), ("accel_fixed", ACC, "-", 2.2, L_ACC)):
            g = np.maximum(e3e[f"{key}_G"] - Gs, 0.0)
            rel = clip(g / g[:, :1])[:, 1:]
            its = np.cumsum(e3e[f"{key}_inner"], axis=1).mean(0)
            lo, med, hi = np.percentile(rel, [2.5, 50, 97.5], axis=0)
            ax.fill_between(its, lo, hi, color=col, alpha=0.18, lw=0)
            ax.semilogy(its, med, ls, color=col, lw=lw, label=lab)
        ax.set_xlim(0, its[-1])
        ax.set_ylim(1e-8, 2)
    ax.set_xlabel("total inner iterations (50 per proximal step)")
    ax.set_ylabel("normalised objective gap")
    ax.set_title("(b) the whole method")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout(w_pad=1.5)
    savefig(fig, d, "fig_main_solver")


def ens_rel(e2e, key):
    return np.asarray(e2e[f"{key}_rel"], float)


#  ------------------------------------------------------------------ appendix figures
def fig_app_e1_methods(d, e1):
    """All four schemes on the two fixed targets: the two accelerated ones coincide, as do the two others."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    t = e1["t"][1:]
    for ax, name, title, logx, tmax in ((axes[0], "WEAK", r"weakly convex target, $\Sigma_q=\mathrm{diag}(25,10)$", True, 300),
                                        (axes[1], "STRONG", r"strongly convex target, $\Sigma_q=\mathrm{diag}(1,0.5)$", False, 40)):
        s = slice(0, tmax)
        ax.plot(t[s], e1[f"{name}_bound"][s], ":", color=GREY, lw=1.1, label="Theorem 1 bound")
        ax.plot(t[s], clip(e1[f"{name}_mm"][1:][s]), color=BASE, lw=2.4, alpha=0.55, label=r"non-accelerated, fixed reference ($\alpha_t\equiv1$)")
        ax.plot(t[s], clip(e1[f"{name}_jko"][1:][s]), "--", color="#7A2E00", lw=0.9, label="JKO")
        ax.plot(t[s], clip(e1[f"{name}_accel"][1:][s]), color=ACC, lw=2.4, alpha=0.55, label="accelerated, fixed reference (Alg. 1)")
        ax.plot(t[s], clip(e1[f"{name}_moving"][1:][s]), "--", color="#08306B", lw=0.9, label="accelerated, moving reference")
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        noise_band(ax, label=not logx)
        ax.set_ylim(NOISE, None)
        ax.set_xlabel("outer step $t$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"objective gap $G(x_t)-G^\star$")
    h, l = axes[0].get_legend_handles_labels()
    order = [3, 4, 1, 2, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    savefig(fig, d, "fig_app_e1_methods")


def fig_app_e1_ensemble(d, ens):
    """Gap divided by the bound of Theorem 1 over 100 random instances."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    t = ens["t"][1:]
    for ax, reg, title in ((axes[0], "weak", "weakly convex targets"), (axes[1], "strong", "strongly convex targets")):
        ax.axhline(1.0, color=DARK, ls=":", lw=1.0, label="guarantee of Theorem 1")
        for key, col, lab in (("jko", BASE, L_BASE), ("accel", ACC, L_ACC)):
            u = clip(ens[f"{reg}_{key}_u"][:, 1:])
            lo, med, hi = np.percentile(u, [2.5, 50, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(t, med, color=col, lw=1.8, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        noise_band(ax, label=ax is axes[1])
        ax.set_ylim(NOISE, 5)
        umax = ens[f"{reg}_accel_u"][:, 1:].max()
        ax.set_title(f"{ens[f'{reg}_accel_u'].shape[0]} {title}")
        say(f"E1 ensemble, {reg}: largest gap/guarantee for Alg. 1 over t >= 1 and all instances = {umax:.3f}")
        ax.set_xlabel("outer step $t$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("gap / guarantee of Theorem 1")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, d, "fig_app_e1_ensemble")


def fig_app_e2(d, e2):
    """Channel-ratio sweep on both targets, and the certificate against the true gap."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    K = 600
    k = np.arange(1, K + 1)
    for ax, tgt, title in ((axes[0], "gauss", "Gaussian target"), (axes[1], "quartic", "quartic target")):
        ax.semilogy(k, clip(e2[f"{tgt}__scalar__gap"][:K]), "--", color=BASE, lw=1.8, label="scalar-step PDHG")
        for r, col in RATIO_COLS.items():
            ax.semilogy(k, clip(e2[f"{tgt}__precond_r={r}__gap"][:K]), color=col, label=rf"preconditioned, $\sigma_\mu/\sigma_\Lambda={r}$")
        ax.semilogy(k, clip(e2[f"{tgt}__balanced_eq.(15)__gap"][:K]), ":", color=DARK, lw=1.2, label="preconditioned, balanced steps")
        noise_band(ax, label=False)
        ax.set_ylim(NOISE, 10)
        ax.set_xlim(0, K)
        ax.set_xlabel("PDHG iterations")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("gap of the proximal step")
    ax = axes[2]
    r = f"{float(e2['r_best']):g}"
    cs = e2[f"gauss__precond_r={r}__certs"]
    cs = cs[cs[:, 0] <= K]
    ax.semilogy(k, clip(e2[f"gauss__precond_r={r}__gap"][:K]), color=RATIO_COLS[r], lw=2.0, label="true gap")
    ax.semilogy(cs[:, 0], clip(cs[:, 2]), "o-", color=DARK, ms=2, lw=1.0, label="certificate")
    noise_band(ax)
    ax.set_ylim(NOISE, 10)
    ax.set_xlim(0, K)
    ax.set_xlabel("PDHG iterations")
    ax.set_title("certificate vs. true gap")
    ax.legend(loc="center right", fontsize=6, bbox_to_anchor=(1.02, 0.6), handlelength=1.5)
    ax.grid(True, alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    savefig(fig, d, "fig_app_e2")


def fig_app_e3(d, e3):
    """End to end on the two fixed targets: gap against the total number of inner iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    runs = (("mm_cert", BASE, "-", "non-accelerated, certified accuracy"), ("mm_fixed", BASE, "--", "non-accelerated, 50 inner iterations"),
            ("accel_cert", ACC, "-", "accelerated (Alg. 1), certified accuracy"), ("accel_fixed", ACC, "--", "accelerated (Alg. 1), 50 inner iterations"))
    for ax, tgt, title in ((axes[0], "gauss", "Gaussian target"), (axes[1], "quartic", "quartic target")):
        Gs = float(e3[f"{tgt}_Gstar"])
        for key, col, ls, lab in runs:
            ax.semilogy(np.cumsum(e3[f"{tgt}_{key}_inner"]) / 1e3, clip(e3[f"{tgt}_{key}_G"][1:] - Gs), ls, color=col, lw=1.5, label=lab)
        ax.set_ylim(1e-9, 3)
        ax.set_xlim(0, None)
        ax.set_xlabel(r"total inner iterations ($\times10^3$)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"objective gap $G(x_t)-G^\star$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend([h[i] for i in (2, 3, 0, 1)], [l[i] for i in (2, 3, 0, 1)], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    savefig(fig, d, "fig_app_e3")


#  ------------------------------------------------------------------ tables
def bold(s, on, tex):
    if not on:
        return s
    if tex and s.startswith("$"):
        return rf"$\mathbf{{{s[1:-1]}}}$"
    return rf"\textbf{{{s}}}" if tex else f"**{s}**"


def sci(x, tex):
    """1.4e-03 -> 1.4 x 10^-3."""
    if x < NOISE:
        return r"$<10^{-10}$" if tex else "<1e-10"
    m, e = f"{x:.1e}".split("e")
    if int(e) == 0:
        return f"${m}$" if tex else m
    return rf"${m}\times10^{{{int(e)}}}$" if tex else f"{m}e{int(e)}"


def tab_outer(flat, tex):
    """Outer scheme with exact steps: steps to a tolerance and gap after 100 steps."""
    rows = []
    FLAT_T = int(flat["T"])
    for lam in flat["lams"]:
        a, m = flat[f"accel_{lam:g}"], flat[f"jko_{lam:g}"]
        cells = [rf"$10^{{{int(np.log10(lam))}}}$" if tex else f"{lam:g}"]
        for thr in (1e-2, 1e-4):
            sa, sm = first_step(a, thr), first_step(m, thr)
            fa = str(sa) if sa is not None else (rf"$>{FLAT_T}$" if tex else f">{FLAT_T}")
            fm = str(sm) if sm is not None else (rf"$>{FLAT_T}$" if tex else f">{FLAT_T}")
            cells += [bold(fa, sa is not None and (sm is None or sa <= sm), tex), bold(fm, sm is not None and (sa is None or sm <= sa), tex)]
        cells += [bold(sci(a[100], tex), a[100] <= m[100], tex), bold(sci(m[100], tex), m[100] <= a[100], tex)]
        rows.append(cells)
    return rows


def tab_inner(e2e, tex):
    """Inner solver over 50 random instances: median iterations to a relative gap."""
    pre = "preconditioned, " + (r"$(\sigma_\Lambda,\sigma_\mu)=$ " if tex else "(sig_L, sig_m) = ")
    confs = (("scalar", "scalar-step PDHG"), ("r1", pre + "(1, 1)"), ("r10", pre + "(1, 10)"), ("r100", pre + "(1, 100)"),
             ("s10_100", pre + "(10, 100)"), ("balanced", "preconditioned, balanced steps" + (r" \eqref{eq:balance}" if tex else "")))
    med = {}
    for key, _ in confs:
        its = e2e[f"{key}_its"].astype(float)
        med[key] = [np.median(np.where(np.isfinite(its[:, j]), its[:, j], np.inf)) for j in range(4)]
    best = [min(med[k][j] for k, _ in confs) for j in range(4)]
    rows = []
    for key, lab in confs:
        its = e2e[f"{key}_its"].astype(float)
        n = its.shape[0]
        cells = [lab]
        for j in (1, 2, 3):
            reached = int(np.isfinite(its[:, j]).sum())
            if reached < n / 2:
                cells.append((r"$>3000$" if tex else ">3000") + f" ({reached}/{n})")
            else:
                lo, hi = boot_median(np.where(np.isfinite(its[:, j]), its[:, j], np.inf))[1:]
                cells.append(bold(f"{med[key][j]:.0f}", med[key][j] == best[j], tex) + f" [{lo:.0f}, {hi:.0f}]")
        rows.append(cells)
    return rows


def tab_e3(e3e, tex):
    """End to end over 20 random instances: median normalised gap after t steps, and inner iterations spent."""
    T = e3e["accel_fixed_G"].shape[1] - 1
    tt = [k for k in (50, 100, 300) if k <= T]
    Gs = e3e["Gstar"][:, None]
    rel = {}
    for key in ("accel_fixed", "mm_fixed", "accel_cert", "mm_cert"):
        g = np.maximum(e3e[f"{key}_G"] - Gs, 0.0)
        rel[key] = np.median(g / g[:, :1], axis=0)
    rows = []
    for acc, mm, lab in (("accel_fixed", "mm_fixed", "50 inner iterations per step"), ("accel_cert", "mm_cert", "certified accuracy")):
        for key, other, name in ((acc, mm, "accelerated (Alg. 1)"), (mm, acc, "non-accelerated")):
            its = np.median(e3e[f"{key}_inner"].sum(1))
            rows.append([lab, name] + [bold(sci(rel[key][k], tex), rel[key][k] <= rel[other][k], tex) for k in tt]
                        + [f"{its:,.0f}".replace(",", r"\,") if tex else f"{its:.0f}"])
    return tt, rows


def write_tables(d, flat, e2e, e3e):
    n = e2e["scalar_its"].shape[0]
    h_out = ["curvature lambda", "steps to gap 1e-2: accelerated", "JKO", "steps to gap 1e-4: accelerated", "JKO", "gap after 100 steps: accelerated", "JKO"]
    h_in = ["inner solver", "to 1e-4", "to 1e-6", "to 1e-8"]
    md = ["# Tables for the paper\n\n## Outer scheme, exact steps (Gaussian target N(0, I/lambda))\n\n",
          "| " + " | ".join(h_out) + " |\n", "|" + "---|" * len(h_out) + "\n"]
    md += ["| " + " | ".join(r) + " |\n" for r in tab_outer(flat, False)]
    md += [f"\n## Inner solver: iterations to a relative gap, median over {n} random instances [95% CI]\n\n",
           "| " + " | ".join(h_in) + " |\n", "|" + "---|" * len(h_in) + "\n"]
    md += ["| " + " | ".join(r) + " |\n" for r in tab_inner(e2e, False)]
    if e3e is not None:
        tt, rows = tab_e3(e3e, False)
        h3 = ["inner accuracy", "outer scheme"] + [f"gap at t={k}" for k in tt] + ["inner iterations"]
        md += [f"\n## End to end: median normalised gap over {e3e['Gstar'].shape[0]} random instances\n\n",
               "| " + " | ".join(h3) + " |\n", "|" + "---|" * len(h3) + "\n"]
        md += ["| " + " | ".join(r) + " |\n" for r in rows]
        tt, rows = tab_e3(e3e, True)
        tex = [r"% requires booktabs", r"\begin{tabular}{ll" + "r" * len(tt) + "r}", r"\toprule",
               rf" & & \multicolumn{{{len(tt)}}}{{c}}{{normalised gap after $t$ steps}} & inner \\", rf"\cmidrule(lr){{3-{2 + len(tt)}}}",
               "inner accuracy & outer scheme & " + " & ".join(f"$t={k}$" for k in tt) + r" & iterations \\", r"\midrule"]
        tex += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(d, "tab_e3.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
    with open(os.path.join(d, "tables.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    tex = [r"% requires booktabs", r"\begin{tabular}{c rr rr rr}", r"\toprule",
           r" & \multicolumn{2}{c}{steps to gap $10^{-2}$} & \multicolumn{2}{c}{steps to gap $10^{-4}$} & \multicolumn{2}{c}{gap after 100 steps} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"$\lambda$ & accel. & JKO & accel. & JKO & accel. & JKO \\", r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in tab_outer(flat, True)] + [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(d, "tab_outer.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    tex = [r"% requires booktabs", r"\begin{tabular}{l rrr}", r"\toprule",
           r" & \multicolumn{3}{c}{iterations to relative gap} \\", r"\cmidrule(lr){2-4}",
           r"inner solver & $10^{-4}$ & $10^{-6}$ & $10^{-8}$ \\", r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in tab_inner(e2e, True)] + [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(d, "tab_inner.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def load_e3(name):
    """E3 results, or None if they predate the small-gamma protocol (rerun E3)."""
    path = os.path.join(RESULTS, name)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if "gamma" in z.files:
            return z
    say(f"{name}: missing or from the old protocol — rerun E3; its figures and table are skipped")
    return None


def main():
    if not os.path.exists(os.path.join(RESULTS, "e1", "e1_flat.npz")):
        raise SystemExit("results/e1/e1_flat.npz is missing: rerun E1 (RESUME=1 EXPS=e1 ./scripts/run_all.sh)")
    d = outdir("figures")
    e1, e1e, flat, e2, e2e = (load(f) for f in ("e1/e1.npz", "e1/e1_ensemble.npz", "e1/e1_flat.npz", "e2/e2.npz", "e2/e2_ensemble.npz"))
    e3, e3e = load_e3("e3/e3.npz"), load_e3("e3/e3_ensemble.npz")
    r = f"{float(e2['r_best']):g}"
    fig_main_outer(d, flat, e1e)
    fig_main_solver(d, e2e, e3e, r)
    fig_app_e1_methods(d, e1)
    fig_app_e1_ensemble(d, e1e)
    fig_app_e2(d, e2)
    if e3 is not None:
        fig_app_e3(d, e3)
    write_tables(d, flat, e2e, e3e)
    say(f"wrote figures and tables to {d}")


if __name__ == "__main__":
    main()
