"""E5 — step-size sweep: is the end-to-end gain of E3 still there when gamma is chosen to minimise
the total work? E5a: inner iterations of one proximal step against gamma (the 50 instances of E2).
E5b: total inner iterations to a normalised outer gap against gamma, accelerated vs non-accelerated,
for several inner budgets (the 20 targets of E3 and the quartic target).
Writes results/e5/{e5a_raw.npz, e5b_raw.npz, tab_e5a.tex, tab_e5b.tex, fig_e5a.*, fig_e5.*,
fig_e5_budgets.*, e5_summary.txt}. Stages run in the order of the spec and the outputs are rewritten
after each one, so partial results can be read while later stages run."""
import argparse
import os
import time

import numpy as np

from common import RESULTS, boot_median, outdir, plt, run_seeds, savefig, say, stack
from e2_inner_solver import THETA, random_field, random_gauss, reference
from e3_end_to_end import GAMMA_REF, load_r_best, make_target
from grid import Grid
from paper_figures import ACC, BASE, DARK, FULL, bold
from pdhg import ProxProblem, pdhg, run_outer

GAMMAS = (5e-4, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1)
BUDGETS = (10, 25, 50, 100)
B_MAIN = 50                    # the inner budget of E3 and of the paper
TOL_A = (1e-4, 1e-6, 1e-8)     # relative gap of one proximal step (E5a)
TOL_B = (1e-2, 1e-4, 1e-6)     # normalised outer gap (E5b)
SCHEMES = (("accel", "accelerated", True), ("mm", "non-accelerated", False))
REF_CERT = 1e-9                # E5a: relative certificate at which a reference run for J* counts as converged
STOP = ("reached 1e-6", "inner-iteration budget", "stalled", "T_max")


#  ------------------------------------------------------------------ E5a: one proximal step against gamma
def work_a(job):
    """One E2 instance (same seed, same q and Y), every gamma: iterations to a relative gap with the
    fixed steps (1, r) and with the balanced steps of eq. (15) computed from the reference solution."""
    seed, a = job
    rng = np.random.default_rng([2, seed])
    grid = Grid(a["n"])
    tgt = random_gauss(rng)
    Y = random_field(grid, rng)
    z = np.zeros((grid.n, grid.n))
    ng = len(a["gammas"])
    out = dict(its_fixed=np.full((ng, len(TOL_A)), np.inf), its_bal=np.full((ng, len(TOL_A)), np.inf),
               sig_bal=np.zeros((ng, 2)), inadm_fixed=np.zeros(ng), inadm_bal=np.zeros(ng), wall=np.zeros(ng),
               ref_cert=np.zeros(ng), ref_shift=np.zeros(ng), ref_runs=np.zeros(ng))
    for i, gam in enumerate(a["gammas"]):
        t0 = time.perf_counter()
        prob = ProxProblem(grid, tgt, gam, Y)
        J0 = prob.values(grid.K1c(z), grid.K2c(z))[1]
        balanced = lambda rf: (np.sqrt(THETA) * np.linalg.norm(rf["state"]["Lam"]) / np.linalg.norm(grid.K1c(rf["last"]["c"])),
                               np.sqrt(THETA) * np.linalg.norm(rf["state"]["mu"]) / np.linalg.norm(grid.K2c(rf["last"]["c"])))
        # reference as in E2 (ratio 10); at small gamma that run is not converged in ref_iters, so further
        # reference runs follow (the steps of the experiment, then balanced steps) until the certificate is small
        Jref, ref = reference(prob, a["ref_iters"])
        n_ref = 1
        for steps in ((1.0, a["r"]), None):
            if ref["cert"] <= REF_CERT * (J0 - Jref) * prob.h2:
                break
            s2 = balanced(ref) if steps is None else steps
            r2 = pdhg(prob, s2[0], s2[1], a["ref_iters"], theta=THETA, record=True)
            n_ref += 1
            if r2["last"] is not None and r2["J"].min() < Jref:
                Jref, ref = r2["J"].min(), r2
        bal = balanced(ref)
        runs = {key: pdhg(prob, s[0], s[1], a["iters"], theta=THETA, record=True, cert_every=None)
                for key, s in (("fixed", (1.0, a["r"])), ("bal", bal))}
        Jstar = min([Jref] + [r["J"].min() for r in runs.values()])
        # J* is known only up to the reference's certificate and to how far a measured run went below it:
        # a tolerance finer than that is undetermined for this instance (NaN), not reached or censored
        out["ref_cert"][i] = ref["cert"] / ((J0 - Jstar) * prob.h2)
        out["ref_shift"][i] = (Jref - Jstar) / (J0 - Jstar)
        out["ref_runs"][i] = n_ref
        for key, r in runs.items():
            rel = (r["J"] - Jstar) / (J0 - Jstar)
            for j, th in enumerate(TOL_A):
                if out["ref_cert"][i] > th or out["ref_shift"][i] > 0.1 * th:
                    out[f"its_{key}"][i, j] = np.nan
                    continue
                hit = np.nonzero(rel < th)[0]
                if hit.size:
                    out[f"its_{key}"][i, j] = hit[0] + 1
            adm = np.isfinite(r["J"])
            out[f"inadm_{key}"][i] = (~adm[adm.argmax():]).sum() if adm.any() else adm.size   # after the first admissible one
        out["sig_bal"][i] = bal
        out["wall"][i] = time.perf_counter() - t0
    return seed, out


#  ------------------------------------------------------------------ E5b: total work against gamma
def sweep(tgt, Gstar, a, b, name):
    """Both outer schemes at every gamma with `b` inner iterations per step, on one target."""
    grid = Grid(a["n"])
    sig = (1.0, a["r"])
    ng, T = len(a["gammas"]), a["T_max"]
    out = dict(G=np.full((ng, 2, T + 1), np.nan), steps=np.zeros((ng, 2)), stop=np.zeros((ng, 2)), fails=np.zeros((ng, 2)),
               wall=np.zeros((ng, 2)), hit_steps=np.full((ng, 2, len(TOL_B)), np.inf), Gstar=Gstar, b=b)
    for i, gam in enumerate(a["gammas"]):
        for j, (key, _, accel) in enumerate(SCHEMES):
            code = [3]

            def stop(G, inner):
                rel = (np.asarray(G) - Gstar) / (G[0] - Gstar)
                t = len(inner)
                if rel[-1] <= TOL_B[-1]:
                    code[0] = 0
                elif sum(inner) > a["budget"]:
                    code[0] = 1
                elif t >= a["stall_steps"] and rel.min() > 0.5 * rel[:t + 1 - a["stall_steps"]].min():
                    code[0] = 2
                return code[0] != 3

            t0 = time.perf_counter()
            r = run_outer(grid, tgt, gam, T, accel, sig, fixed=b, stop=stop)
            G = r["G"]
            rel = (G - Gstar) / (G[0] - Gstar)
            out["G"][i, j, :len(G)] = G
            out["steps"][i, j], out["stop"][i, j], out["fails"][i, j] = len(G) - 1, code[0], r["fails"]
            out["wall"][i, j] = time.perf_counter() - t0
            for k, th in enumerate(TOL_B):
                hit = np.nonzero(rel <= th)[0]
                if hit.size:
                    out["hit_steps"][i, j, k] = hit[0]
            say(f"E5b [{name}, b = {b}] gamma = {gam:g}, {key}: {len(G) - 1} steps, final normalised gap {rel[-1]:.2e} "
                f"({STOP[code[0]]}), {out['wall'][i, j]:.0f}s")
    out["G0"] = out["G"][0, 0, 0]
    return out


def ref_Gstar(tgt, a, name):
    """G* as in E3: the smallest value over non-accelerated steps with gamma = 0.05 at a constant certificate."""
    say(f"E5b [{name}]: no saved G*, reference run of {a['ref_steps']} steps")
    ref = run_outer(Grid(a["n"]), tgt, GAMMA_REF, a["ref_steps"], False, (1.0, a["r"]), tol_const=a["ref_tol"], cap=a["ref_cap"])
    return float(ref["G"].min())


def work_gauss(job):
    seed, a = job
    tgt = random_gauss(np.random.default_rng([3, seed]))          # the targets of the E3 ensemble
    Gs = a["Gstar"][seed] if a["Gstar"] is not None else ref_Gstar(tgt, a, f"seed {seed}")
    out = sweep(tgt, float(Gs), a, a["b"], f"seed {seed}")
    out["lam"] = float(1.0 / tgt.s2.max())                         # strong-convexity modulus of V
    return seed, out


def work_quartic(job):
    b, a = job                                                     # the "seed" is the inner budget
    tgt = make_target("quartic")
    Gs = a["Gstar_q"] if a["Gstar_q"] is not None else ref_Gstar(tgt, a, "quartic")
    return b, sweep(tgt, float(Gs), a, b, "quartic")


def saved_Gstar(args):
    """G* of E3 for the same targets (same grid); None if E3 has not been run at this resolution."""
    fe, fc = (os.path.join(RESULTS, "e3", f) for f in ("e3_ensemble.npz", "e3.npz"))
    if args.n != 64 or not (os.path.exists(fe) and os.path.exists(fc)):
        return None, None
    g = np.load(fe)["Gstar"]
    return (g if len(g) >= args.seeds_b else None), float(np.load(fc)["quartic_Gstar"])


#  ------------------------------------------------------------------ reporting
def med_ci(x, cap, tex):
    """Median [95% bootstrap CI] of iteration counts with inf = censored, and the number reached."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]                                            # undetermined instances (E5a) are left out
    k, n = int(np.isfinite(x).sum()), x.size
    if n == 0:
        return "n/a", np.nan
    if k < n / 2:
        return (rf"$>{cap:,}$".replace(",", r"\,") if tex else f">{cap}") + f" ({k}/{n})", np.inf
    m, lo, hi = boot_median(x)
    f = lambda v: (r"$\infty$" if tex else "inf") if np.isinf(v) else f"{v:,.0f}".replace(",", r"\," if tex else "")
    return f"{f(m)} [{f(lo)}, {f(hi)}]" + ("" if k == n else f" ({k}/{n})"), m


def gam_tex(g):
    m, e = f"{g:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def report_a(d, res, args, r):
    n = len(res)
    gam = np.array(GAMMAS)
    data = {k: stack(res, k) for k in res[0]}
    np.savez_compressed(os.path.join(d, "e5a_raw.npz"), gammas=gam, tol=np.array(TOL_A), iters=args.iters, **data)
    blocks = (("its_fixed", rf"$(\sigma_\Lambda,\sigma_\mu)=(1,{r:g})$", f"(sig_L, sig_m) = (1, {r:g})"),
              ("its_bal", r"balanced steps \eqref{eq:balance}", "balanced steps (15)"))
    tex = [r"% requires booktabs", r"\begin{tabular}{l rrr}", r"\toprule", r" & \multicolumn{3}{c}{iterations to relative gap} \\",
           r"\cmidrule(lr){2-4}", r"$\gamma$ & $10^{-4}$ & $10^{-6}$ & $10^{-8}$ \\"]
    txt = [f"E5a — one proximal step against gamma: {n} instances of E2, n = {args.n}, at most {args.iters} iterations, "
           "median [95% bootstrap CI] iterations to a relative gap (k/n = instances that reached it)\n"]
    for key, ltex, ltxt in blocks:
        tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{ltex}}}}} \\"]
        txt.append(f"\n{ltxt}\n  gamma     " + "".join(f"to {t:g}".ljust(26) for t in TOL_A) + "\n")
        for i, g in enumerate(gam):
            tex.append(gam_tex(g) + " & " + " & ".join(med_ci(data[key][:, i, j], args.iters, True)[0] for j in range(3)) + r" \\")
            txt.append(f"  {g:<9g} " + "".join(med_ci(data[key][:, i, j], args.iters, False)[0].ljust(26) for j in range(3)) + "\n")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(d, "tab_e5a.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    sb = data["sig_bal"]
    txt.append("\nbalanced steps, median over instances (sig_L, sig_m, ratio):\n" + "".join(
        f"  gamma = {g:<7g} {np.median(sb[:, i, 0]):.3g}, {np.median(sb[:, i, 1]):.4g}, {np.median(sb[:, i, 1] / sb[:, i, 0]):.4g}\n"
        for i, g in enumerate(gam)))
    und = np.isnan(data["its_fixed"]).sum(0)
    txt.append("\nreference for J*: instances whose J* is too uncertain for a tolerance (left out of that entry; k/n counts the rest), "
               "largest relative certificate, largest relative amount by which a measured run went below the reference, reference runs used\n"
               + "".join(f"  gamma = {g:<7g} undetermined at {TOL_A}: {tuple(int(v) for v in und[i])} of {n}; certificate {data['ref_cert'][:, i].max():.1e}; "
                         f"shift {data['ref_shift'][:, i].max():.1e}; reference runs {data['ref_runs'][:, i].mean():.2f} on average\n" for i, g in enumerate(gam)))
    txt.append(f"\niterates with I + K1 psi not positive definite after the first admissible one, summed over instances and gamma: "
               f"fixed steps {int(data['inadm_fixed'].sum())}, balanced {int(data['inadm_bal'].sum())}\n"
               f"wall-clock per instance and gamma (reference + two runs): median {np.median(data['wall']):.0f} s\n")
    fig, ax = plt.subplots(figsize=(FULL / 2, 2.4))
    for key, col, ls, lab in (("its_fixed", ACC, "-", rf"$(\sigma_\Lambda,\sigma_\mu)=(1,{r:g})$"), ("its_bal", DARK, "--", "balanced steps")):
        x = np.minimum(data[key][:, :, 1], args.iters)
        with np.errstate(all="ignore"):
            lo, med, hi = np.nanpercentile(x, [2.5, 50, 97.5], axis=0)
        ax.fill_between(gam, lo, hi, color=col, alpha=0.15, lw=0)
        ax.loglog(gam, med, ls, color=col, marker="o", ms=2.5, label=lab)
    ax.axhline(args.iters, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel(r"step size $\gamma$")
    ax.set_ylabel(r"inner iterations to relative gap $10^{-6}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    savefig(fig, d, "fig_e5a")
    return txt


def costs(res, b):
    """Total inner iterations to each tolerance, shape (targets, gamma, scheme, tol); inf = not reached."""
    return stack(res, "hit_steps") * b


def paired(c_mm, c_acc):
    """Median [95% CI] of log10(cost non-accelerated / cost accelerated) over targets where both are finite."""
    ok = np.isfinite(c_mm) & np.isfinite(c_acc)
    if not ok.any():
        return "no target reached by both schemes", ok.sum()
    m, lo, hi = boot_median(np.log10(c_mm[ok] / c_acc[ok]))
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] (factor {10 ** m:.2f}) over {int(ok.sum())}/{ok.size} targets", ok.sum()


def report_b(d, gauss, quartic, args):
    """gauss: {b: list of per-target dicts}; quartic: {b: dict}. Either may be partial."""
    gam = np.array(GAMMAS)
    cap = args.budget
    txt = [f"\n\nE5b — total inner iterations to a normalised outer gap against gamma. n = {args.n}, at most {args.T_max} outer steps, "
           f"{cap} inner iterations, or until the best gap has not halved in {args.stall_steps} steps. Inner budgets available: "
           f"Gaussian targets {sorted(gauss)}, quartic {sorted(quartic)}.\n"]
    raw = {}
    for b, res in gauss.items():
        for k in ("G", "steps", "stop", "fails", "wall", "hit_steps"):
            raw[f"gauss_b{b}_{k}"] = stack(res, k)
        raw["gauss_Gstar"], raw["gauss_lam"] = stack(res, "Gstar"), stack(res, "lam")
    for b, o in quartic.items():
        for k in ("G", "steps", "stop", "fails", "wall", "hit_steps"):
            raw[f"quartic_b{b}_{k}"] = o[k]
        raw["quartic_Gstar"] = o["Gstar"]
    np.savez_compressed(os.path.join(d, "e5b_raw.npz"), gammas=gam, tol=np.array(TOL_B), **raw)

    fmt = lambda v: "censored" if np.isinf(v) else f"{v:.0f}"
    # --- per gamma at the main budget
    if B_MAIN in gauss:
        res = gauss[B_MAIN]
        n = len(res)
        C = costs(res, B_MAIN)
        S = stack(res, "hit_steps")
        lam = stack(res, "lam")
        txt.append(f"\nGaussian targets ({n}), b = {B_MAIN}: median total inner iterations [95% CI] (k/n reached), then median outer steps\n")
        for k in (1, 2):
            txt.append(f"\n  to normalised gap {TOL_B[k]:g}\n  gamma     " + "accelerated".ljust(34) + "non-accelerated".ljust(34)
                       + "steps acc / non-acc    median 1/(gamma lam)\n")
            for i, g in enumerate(gam):
                st = [np.median(S[:, i, j, k]) for j in (0, 1)]
                txt.append(f"  {g:<9g} " + "".join(med_ci(C[:, i, j, k], cap, False)[0].ljust(34) for j in (0, 1))
                           + f"{fmt(st[0])} / {fmt(st[1])}".ljust(23) + f"{np.median(1 / (g * lam)):.1f}\n")
        stop, fails = stack(res, "stop"), stack(res, "fails")
        txt.append("\n  how the runs ended (accelerated / non-accelerated), and steps without an admissible inner iterate:\n")
        for i, g in enumerate(gam):
            txt.append(f"  gamma = {g:<7g} " + "; ".join(f"{STOP[c]}: {int((stop[:, i, 0] == c).sum())} / {int((stop[:, i, 1] == c).sum())}"
                                                          for c in range(4)) + f"; inadmissible steps {int(fails[:, i, 0].sum())} / {int(fails[:, i, 1].sum())}\n")
        txt.append(f"\n  wall-clock per run: median {np.median(stack(res, 'wall')):.0f} s, total {stack(res, 'wall').sum() / 3600:.1f} CPU-hours\n")
    # --- best over gamma (and over the budget)
    txt.append("\nBest cost per target: min over gamma of total inner iterations; paired log10(non-accelerated / accelerated), "
               "median [95% bootstrap CI] (positive = acceleration needs less work)\n")
    best_rows = {}
    for k in (1, 2):
        if B_MAIN in gauss:
            C = costs(gauss[B_MAIN], B_MAIN)[..., k].min(1)                      # (targets, scheme)
            best_rows[k] = C
            arg = costs(gauss[B_MAIN], B_MAIN)[..., k].argmin(1)
            txt.append(f"  to {TOL_B[k]:g}, b = {B_MAIN} only, Gaussian targets: {paired(C[:, 1], C[:, 0])[0]}\n"
                       f"      median best cost: accelerated {med_ci(C[:, 0], cap, False)[0]}, non-accelerated {med_ci(C[:, 1], cap, False)[0]}; "
                       "median best gamma over the targets that reached it: " + ", ".join(
                           f"{nm} " + (f"{np.median(gam[arg[:, j]][np.isfinite(C[:, j])]):g}" if np.isfinite(C[:, j]).any() else "n/a")
                           for j, nm in enumerate(("accelerated", "non-accelerated"))) + "\n")
        if len(gauss) > 1:
            n = min(len(r) for r in gauss.values())
            C = np.min([costs(gauss[b][:n], b)[..., k].min(1) for b in gauss], axis=0)
            txt.append(f"  to {TOL_B[k]:g}, min over gamma and b in {sorted(gauss)}, Gaussian targets: {paired(C[:, 1], C[:, 0])[0]}\n"
                       f"      median best cost: accelerated {med_ci(C[:, 0], cap, False)[0]}, non-accelerated {med_ci(C[:, 1], cap, False)[0]}\n")
        if quartic:
            for bs in ([B_MAIN] if B_MAIN in quartic else []) + [sorted(quartic)]:
                bs = bs if isinstance(bs, list) else [bs]
                C = np.min([quartic[b]["hit_steps"][..., k].min(0) * b for b in bs], axis=0)
                r = "censored" if not np.isfinite(C).all() else f"log10 ratio {np.log10(C[1] / C[0]):+.2f} (factor {C[1] / C[0]:.2f})"
                txt.append(f"  to {TOL_B[k]:g}, quartic target, b in {bs}: accelerated {fmt(C[0])}, non-accelerated {fmt(C[1])}, {r}\n")
    if quartic:
        txt.append("\nQuartic target: total inner iterations (accelerated / non-accelerated) and how the runs ended\n")
        for b in sorted(quartic):
            o = quartic[b]
            txt.append(f"\n  b = {b}\n  gamma     " + "".join(f"to {t:g}".ljust(22) for t in TOL_B[1:]) + "ended (acc / non-acc), inadmissible steps\n")
            for i, g in enumerate(gam):
                txt.append(f"  {g:<9g} " + "".join(f"{fmt(o['hit_steps'][i, 0, k] * b)} / {fmt(o['hit_steps'][i, 1, k] * b)}".ljust(22) for k in (1, 2))
                           + f"{STOP[int(o['stop'][i, 0])]} / {STOP[int(o['stop'][i, 1])]}, {int(o['fails'][i, 0])} / {int(o['fails'][i, 1])}\n")

    # --- table: rows gamma, b = B_MAIN
    if B_MAIN in gauss:
        C = costs(gauss[B_MAIN], B_MAIN)
        Q = quartic[B_MAIN]["hit_steps"] * B_MAIN if B_MAIN in quartic else None
        nq = 4 if Q is not None else 0
        tex = [r"% requires booktabs", r"\begin{tabular}{l rrrr" + "r" * nq + "}", r"\toprule",
               rf" & \multicolumn{{4}}{{c}}{{Gaussian targets (median over {C.shape[0]})}}" + (r" & \multicolumn{4}{c}{quartic target}" if nq else "") + r" \\",
               r"\cmidrule(lr){2-5}" + (r"\cmidrule(lr){6-9}" if nq else ""),
               r" & \multicolumn{2}{c}{to $10^{-4}$} & \multicolumn{2}{c}{to $10^{-6}$}" * (2 if nq else 1) + r" \\",
               r"$\gamma$" + r" & accel. & non-acc." * (4 if nq else 2) + r" \\", r"\midrule"]
        qf = lambda v: "--" if np.isinf(v) else f"{v:,.0f}".replace(",", r"\,")

        def cells(pairs_g, pairs_q):
            row = []
            for (sa, ma), (sm, mm) in pairs_g:
                row += [bold(sa, ma <= mm and np.isfinite(ma), True), bold(sm, mm <= ma and np.isfinite(mm), True)]
            for va, vm in pairs_q:
                row += [bold(qf(va), va <= vm and np.isfinite(va), True), bold(qf(vm), vm <= va and np.isfinite(vm), True)]
            return row

        short = lambda x: (lambda s, m: (s.split(" [")[0] + (" (" + s.split(" (")[1] if " (" in s and "[" in s else ""), m))(*med_ci(x, cap, True))
        for i, g in enumerate(gam):
            pg = [(short(C[:, i, 0, k]), short(C[:, i, 1, k])) for k in (1, 2)]
            pq = [(Q[i, 0, k], Q[i, 1, k]) for k in (1, 2)] if nq else []
            tex.append(gam_tex(g) + " & " + " & ".join(cells(pg, pq)) + r" \\")
        pg = [(short(best_rows[k][:, 0]), short(best_rows[k][:, 1])) for k in (1, 2)]
        pq = [(Q[:, 0, k].min(), Q[:, 1, k].min()) for k in (1, 2)] if nq else []
        tex += [r"\midrule", r"best over $\gamma$ & " + " & ".join(cells(pg, pq)) + r" \\"]
        ratio = "; ".join(f"to $10^{{{int(np.log10(TOL_B[k]))}}}$: " + paired(best_rows[k][:, 1], best_rows[k][:, 0])[0].split(" (factor")[0] for k in (1, 2))
        tex += [rf"\multicolumn{{{5 + nq}}}{{l}}{{\footnotesize paired $\log_{{10}}$(non-acc./accel.) of the best cost, Gaussian targets, median [95\% CI]: {ratio}}} \\",
                r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(d, "tab_e5b.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")

    # --- figure: cost against gamma at the main budget
    def curves(ax, cost, band):
        """cost: (targets, gamma, scheme, tol)."""
        for j, (col, lab) in enumerate(((ACC, "accelerated (Alg. 1)"), (BASE, "non-accelerated"))):
            for k, ls in ((1, "-"), (2, "--")):
                cen = np.isinf(np.median(cost[:, :, j, k], axis=0))          # fewer than half of the targets reached it
                x = np.minimum(cost[:, :, j, k], cap)
                lo, med, hi = np.percentile(x, [2.5, 50, 97.5], axis=0)
                med = np.where(cen, cap, med)
                if band and k == 1:
                    ax.fill_between(gam, lo, hi, color=col, alpha=0.15, lw=0)
                ax.loglog(gam, np.where(cen, np.nan, med), ls, color=col, lw=2.0 if j == 0 else 1.6, marker="o", ms=2.5,
                          label=f"{lab}, to $10^{{{int(np.log10(TOL_B[k]))}}}$")
                ax.loglog(gam[cen], med[cen], "o", color=col, mfc="none", ms=4)
        ax.axhline(cap, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel(r"step size $\gamma$")
        ax.grid(True, which="major", alpha=0.25)

    if B_MAIN in gauss:
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
        curves(axes[0], costs(gauss[B_MAIN], B_MAIN), True)
        axes[0].set_title(f"(a) {len(gauss[B_MAIN])} Gaussian targets")
        axes[0].set_ylabel("total inner iterations")
        if B_MAIN in quartic:
            curves(axes[1], quartic[B_MAIN]["hit_steps"][None] * B_MAIN, False)
        axes[1].set_title("(b) quartic target")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        savefig(fig, d, "fig_e5")
    if quartic:
        bs = sorted(quartic)
        fig, axes = plt.subplots(1, len(bs), figsize=(FULL, 2.2), sharey=True, squeeze=False)
        for ax, b in zip(axes[0], bs):
            curves(ax, quartic[b]["hit_steps"][None] * b, False)
            ax.set_title(f"{b} inner iterations per step")
        axes[0, 0].set_ylabel("total inner iterations")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=6)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        savefig(fig, d, "fig_e5_budgets")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5000, help="E5a: iteration budget of one proximal step")
    ap.add_argument("--ref-iters", type=int, default=20000, help="E5a: iterations of the reference run for J*")
    ap.add_argument("--seeds-a", type=int, default=50, help="E5a: instances of E2")
    ap.add_argument("--seeds-b", type=int, default=20, help="E5b: Gaussian targets of E3")
    ap.add_argument("--T-max", dest="T_max", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=100000, help="E5b: total inner iterations per run")
    ap.add_argument("--stall-steps", dest="stall_steps", type=int, default=500)
    ap.add_argument("--stages", default="a,b50,quartic,rest", help="comma-separated: a, b50, quartic, rest")
    ap.add_argument("--ref-steps", type=int, default=1500)
    ap.add_argument("--ref-tol", type=float, default=1e-10)
    ap.add_argument("--ref-cap", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="reuse per-seed files of an interrupted run")
    ap.add_argument("--seeds", type=int, default=None, help="overrides --seeds-a and --seeds-b (smoke tests)")
    args = ap.parse_args()
    if args.seeds is not None:
        args.seeds_a = args.seeds_b = args.seeds
    d = outdir("e5")
    r, found = load_r_best()
    if not found and args.n == 64:
        raise SystemExit("results/e2/r_best.json is missing: E5 must use the channel ratio of E2/E3")
    Gs, Gs_q = saved_Gstar(args)
    say(f"E5: (sig_L, sig_m) = (1, {r:g}); G* " + ("reused from results/e3" if Gs is not None else "recomputed (no E3 results at this resolution)"))
    cfg = dict(n=args.n, r=r, gammas=GAMMAS, T_max=args.T_max, budget=args.budget, stall_steps=args.stall_steps, Gstar=Gs, Gstar_q=Gs_q,
               ref_steps=args.ref_steps, ref_tol=args.ref_tol, ref_cap=args.ref_cap)
    stages = args.stages.split(",")
    txt_a, gauss, quartic = [], {}, {}

    def load(sub, seeds):
        """Per-seed files of an earlier stage or run, if all are there."""
        paths = [os.path.join(d, sub, f"seed_{s:03d}.npz") for s in seeds]
        if not all(os.path.exists(p) for p in paths):
            return None
        return [dict(np.load(p)) for p in paths]

    def write():
        with open(os.path.join(d, "e5_summary.txt"), "w", encoding="utf-8") as f:
            f.write("".join(txt_a + (report_b(d, gauss, quartic, args) if gauss or quartic else [])))

    for stage in ("a", "b50", "quartic", "rest"):
        run = stage in stages
        if stage == "a":
            res = (run_seeds(work_a, range(args.seeds_a), dict(n=args.n, r=r, gammas=GAMMAS, iters=args.iters, ref_iters=args.ref_iters),
                             args.jobs, os.path.join(d, "seeds_a"), args.resume) if run else load("seeds_a", range(args.seeds_a)))
            if res is not None:
                txt_a = report_a(d, res, args, r)
        elif stage == "quartic":
            res = (run_seeds(work_quartic, BUDGETS, cfg, args.jobs, os.path.join(d, "seeds_quartic"), args.resume) if run
                   else load("seeds_quartic", BUDGETS))
            if res is not None:
                quartic = dict(zip(BUDGETS, res))
        else:
            for b in ((B_MAIN,) if stage == "b50" else tuple(x for x in BUDGETS if x != B_MAIN)):
                res = (run_seeds(work_gauss, range(args.seeds_b), dict(cfg, b=b), args.jobs, os.path.join(d, f"seeds_b{b}"), args.resume)
                       if run else load(f"seeds_b{b}", range(args.seeds_b)))
                if res is not None:
                    gauss[b] = res
                    write()
        write()
        if run:
            say(f"E5 stage '{stage}' done; outputs in {d} are up to date")


if __name__ == "__main__":
    main()
