import argparse
import os
import time

import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

import experiment_io as eio
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("Standard JKO")
RED = color_for("Accelerated JKO")
GREEN = "#009E73"
PURPLE = "#CC79A7"
os.makedirs("images", exist_ok=True)

UNIFORM_BARRIER = 1e5


#  Monotone quantile parameterization: Q_0 free, Q'_k = exp(s_k),
#  Q_j = Q_0 + (1/M) cumsum(exp(s)).
def unpack(theta, M):
    Q0 = theta[0]
    e = np.exp(np.clip(theta[1:], -60, 40))
    Q = Q0 + np.concatenate([[0.0], np.cumsum(e)]) / M
    return Q, e


def theta_from_Q(Q, M):
    dQ = np.diff(Q)
    return np.concatenate([[Q[0]], np.log(np.clip(M * dQ, 1e-12, None))])


def isotonic(y):
    n = len(y)
    lvl = list(map(float, y))
    lw = [1.0] * n
    idx = list(range(n + 1))
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] > lvl[i + 1]:
            lvl[i] = (lw[i] * lvl[i] + lw[i + 1] * lvl[i + 1]) / (lw[i] + lw[i + 1])
            lw[i] += lw[i + 1]
            del lvl[i + 1], lw[i + 1], idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty(n)
    for k in range(len(lvl)):
        out[idx[k]:idx[k + 1]] = lvl[k]
    return out


#  KL targets: potential V = -log q and derivative, plus a reference x-grid.
def make_kl_target(spec):
    kind = spec["kind"]
    if kind == "quartic":
        V, Vp, xref = lambda x: x**4 / 4, lambda x: x**3, np.linspace(-6, 6, 200001)
    elif kind == "cubic":
        V = lambda x: np.abs(x) ** 3 / 3
        Vp = lambda x: x * np.abs(x)
        xref = np.linspace(-8, 8, 200001)
    elif kind == "uniform":
        b = UNIFORM_BARRIER
        V = lambda x: b * np.maximum(np.abs(x) - 1, 0.0) ** 2
        Vp = lambda x: 2 * b * np.maximum(np.abs(x) - 1, 0.0) * np.sign(x)
        xref = np.linspace(-1, 1, 200001)
    elif kind == "gauss":
        var = spec["var"]
        V, Vp = lambda x: x**2 / (2 * var), lambda x: x / var
        s = np.sqrt(var)
        xref = np.linspace(-8 * s, 8 * s, 200001)
    elif kind == "flatvalley":
        R = spec["R"]
        V = lambda x: np.maximum(np.abs(x) - R, 0.0) ** 4 / 4
        Vp = lambda x: np.maximum(np.abs(x) - R, 0.0) ** 3 * np.sign(x)
        xref = np.linspace(-(R + 8), R + 8, 400001)
    elif kind == "doublewell":
        mu, sg = spec["mu"], spec["sigma"]
        lq = lambda x: np.logaddexp(norm.logpdf(x, mu, sg),
                                    norm.logpdf(x, -mu, sg)) + np.log(0.5)
        V = lambda x: -lq(x)

        def Vp(x):
            a, b = norm.logpdf(x, mu, sg), norm.logpdf(x, -mu, sg)
            wa = np.exp(a - np.logaddexp(a, b))
            return -(wa * (-(x - mu) / sg**2) + (1 - wa) * (-(x + mu) / sg**2))

        xref = np.linspace(-8, 8, 200001)
    else:
        raise ValueError(kind)
    return V, Vp, xref


def target_quantile(V, xref, u):
    logp = -V(xref)
    logp -= logp.max()
    p = np.exp(logp)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xref))])
    cdf /= cdf[-1]
    return np.interp(u, cdf, xref)


def kl_free_energy(Q, V, M):
    return V(Q).mean() - np.log(M * np.diff(Q)).sum() / M


#  Energies return (E_Q value, dE_Q/dQ, E_s value, dE_s/ds).
def kl_energy(V, Vp):
    def f(Q, s, M):
        return V(Q).mean(), Vp(Q) / M, -s.sum() / M, -np.ones(len(s)) / M
    return f


def potential_energy(V, Vp):
    def f(Q, s, M):
        return V(Q).mean(), Vp(Q) / M, 0.0, np.zeros(len(s))
    return f


def interaction_energy():
    def f(Q, s, M):
        D = Q[:, None] - Q[None, :]
        return 0.5 * (D**4).mean(), (4 * D**3).sum(1) / M**2, 0.0, np.zeros(len(s))
    return f


def prox_step(y, theta0, energy, gamma, M):
    def obj(theta):
        Q, e = unpack(theta, M)
        EQ, gEQ, Es, gEs = energy(Q, theta[1:], M)
        F = EQ + Es + ((Q - y) ** 2).mean() / (2 * gamma)
        gQ = gEQ + (Q - y) / (gamma * M)
        suf = np.cumsum(gQ[::-1])[::-1]
        grad = np.empty_like(theta)
        grad[0] = suf[0]
        grad[1:] = e * suf[1:] / M + gEs
        return F, grad

    res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-10})
    return unpack(res.x, M)[0], res.x


def run_pair(energy, gap, w2ref, Q0, gamma, N, M, label, early_stop=None):
    def run_std():
        Q = Q0.copy()
        th = theta_from_Q(Q, M)
        g, rows = [gap(Q)], []
        for t in range(N):
            t1 = time.perf_counter()
            Q, th = prox_step(Q, th, energy, gamma, M)
            t2 = time.perf_counter()
            g.append(gap(Q))
            rows.append({"block": t + 1, "method": "std", "gap": g[-1],
                         "t_prox": t2 - t1, "t_momentum": 0.0,
                         "t_eval": time.perf_counter() - t2})
            if early_stop and g[-1] < early_stop:
                break
        return np.array(g), rows

    def run_acc():
        Qx, Qz = Q0.copy(), Q0.copy()
        th = theta_from_Q(Qx, M)
        g, w2 = [gap(Qx)], [w2ref(Qz)]
        rows = []
        viol, mviol, mproj, mdelta = 0, 0.0, 0.0, 0.0
        for t in range(N):
            a = 3.0 / (t + 3.0)
            tm = time.perf_counter()
            Qy = (1 - a) * Qx + a * Qz
            tp = time.perf_counter()
            Qxn, th = prox_step(Qy, theta_from_Q(Qy, M), energy, gamma, M)
            tp2 = time.perf_counter()
            Qzr = (Qxn - (1 - a) * Qx) / a
            d = np.diff(Qzr)
            v = float(np.maximum(-d, 0).max()) if len(d) else 0.0
            if v > 1e-12:
                viol += 1
                mviol = max(mviol, v)
                Qzn = isotonic(Qzr)
                mproj = max(mproj, float(np.abs(Qzn - Qzr).max()))
            else:
                Qzn = Qzr
            tmom = time.perf_counter()
            # map defect at off-node midpoints: composed map (z->y->x) vs direct OT map (z->x)
            pmid = 0.5 * (Qz[:-1] + Qz[1:])
            comp = np.interp(np.interp(pmid, Qz, Qy), Qy, Qxn)
            direct = np.interp(pmid, Qz, Qxn)
            scale = max(float(np.abs(Qxn).max()), 1e-12)
            mdelta = max(mdelta, float(np.abs(comp - direct).max()) / scale)
            Qx, Qz = Qxn, Qzn
            te = time.perf_counter()
            g.append(gap(Qx))
            w2.append(w2ref(Qz))
            rows.append({"block": t + 1, "method": "acc", "gap": g[-1],
                         "w2sq_z_rho": w2[-1], "t_prox": tp2 - tp,
                         "t_momentum": (tp - tm) + (tmom - tp2),
                         "t_eval": time.perf_counter() - te})
            if early_stop and g[-1] < early_stop:
                break
        nsteps = len(g) - 1
        checks = {"frac_steps_with_monotonicity_violation": viol / max(nsteps, 1),
                  "max_violation_magnitude": mviol,
                  "max_projection_magnitude": mproj,
                  "max_delta_t_1d": mdelta,
                  "sup_t_W2sq_z_rho": float(np.max(w2))}
        return np.array(g), np.array(w2), rows, checks

    gs, rs = run_std()
    ga, w2a, ra, checks = run_acc()
    return dict(g_std=gs, g_acc=ga, w2_acc=w2a, checks=checks, rows=rs + ra)


#  Fitters
def fit_loglog(g, lo, hi):
    t = np.arange(lo, hi + 1)
    y = g[lo:hi + 1]
    pos = g[g > 0]
    floor = 50 * pos.min() if len(pos) else 0.0   # exclude discretization plateau
    m = y > max(1e-11, floor)
    return float(np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0]) if m.sum() > 2 else float("nan")


def fit_exp_rate(g):
    t = np.arange(len(g))
    m = (g > 1e-11) & (t > 0)
    if m.sum() < 2:
        return float("nan")
    return float(-np.polyfit(t[m], np.log(g[m]), 1)[0])


def blocks_to(g, thr):
    w = np.where(g < thr)[0]
    return int(w[0]) if len(w) else -1


#  E1b: KL targets — exponential decay + blocks-to-threshold
def run_e1b(cfg):
    ic = cfg["e1b"]["init"]
    thr = cfg["e1b"]["thresholds"]
    M, gamma, N = cfg["M"], cfg["gamma"], cfg["n_blocks"]
    u = (np.arange(M) + 0.5) / M
    out = {}
    for spec in cfg["e1b"]["targets"]:
        tid = spec["id"]
        V, Vp, xref = make_kl_target(spec)
        Qstar = target_quantile(V, xref, u)
        Gmin = kl_free_energy(Qstar, V, M)
        gap = lambda Q: max(kl_free_energy(Q, V, M) - Gmin, 0.0)
        w2ref = lambda Q: float(((Q - Qstar) ** 2).mean())
        Q0 = ic["mean"] + ic["std"] * norm.ppf(u)
        r = run_pair(kl_energy(V, Vp), gap, w2ref, Q0, gamma, N, M, tid)
        s = {"target": tid, "lam_class": spec["lam_class"],
             "initial_gap": float(r["g_std"][0]),
             "final_kl_std": float(r["g_std"][-1]), "final_kl_acc": float(r["g_acc"][-1]),
             "exp_rate_std": fit_exp_rate(r["g_std"]), "exp_rate_acc": fit_exp_rate(r["g_acc"]),
             "slope_std_transient": fit_loglog(r["g_std"], max(1, N // 10), N),
             "slope_acc_transient": fit_loglog(r["g_acc"], max(1, N // 10), N),
             "gap_ratio_at_T": float(r["g_std"][-1] / max(r["g_acc"][-1], 1e-300)),
             **r["checks"]}
        for th in thr:
            k = f"{-int(round(np.log10(th)))}"
            s[f"blocks_to_1e-{k}_std"] = blocks_to(r["g_std"], th)
            s[f"blocks_to_1e-{k}_acc"] = blocks_to(r["g_acc"], th)
        eio.save_config(f"e1b_{tid}", {**{k: cfg[k] for k in ("gamma", "M", "n_blocks")}, "init": ic, "target": spec})
        eio.save_metrics(f"e1b_{tid}", r["rows"])
        eio.save_summary(f"e1b_{tid}", s)
        _plot_kl(tid, r, N, semilog=True)
        out[tid] = s
        print(f"[e1b {tid}] exp_rate std={s['exp_rate_std']:.3f} acc={s['exp_rate_acc']:.3f} "
              f"finalKL std={s['final_kl_std']:.2e} acc={s['final_kl_acc']:.2e} "
              f"b2(1e-3) std/acc={s.get('blocks_to_1e-3_std')}/{s.get('blocks_to_1e-3_acc')} "
              f"viol={s['frac_steps_with_monotonicity_violation']:.3f} maxδ={s['max_delta_t_1d']:.1e}")
    return out


#  E1a: hold λ=0, drive the spectral gap to zero via V_R.
#  Robust metric: blocks-to-threshold (std ~ 1/gap, acc ~ 1/sqrt(gap)).
def _measured_gap(g):
    lg = np.log(np.clip(g, 1e-16, None))
    t = np.arange(len(g))
    m = (g > 1e-8) & (g < 0.5 * g[0]) & (t > 0)
    return float(-np.polyfit(t[m], lg[m], 1)[0]) if m.sum() > 2 else float("nan")


def run_e1a(cfg):
    a = cfg["e1a"]
    M, gamma, N = a["M"], cfg["gamma"], a["n_blocks"]
    es = a["early_stop"]
    thr = a["thresholds"]
    u = (np.arange(M) + 0.5) / M
    rows_all, out = [], {}
    for R in a["R_values"]:
        spec = {"kind": "flatvalley", "R": R}
        V, Vp, xref = make_kl_target(spec)
        Qstar = target_quantile(V, xref, u)
        Gmin = kl_free_energy(Qstar, V, M)
        gap = lambda Q: max(kl_free_energy(Q, V, M) - Gmin, 0.0)
        w2ref = lambda Q: float(((Q - Qstar) ** 2).mean())
        istd = a["init_std_base"] + a["init_std_slope"] * R
        Q0 = a["init_mean"] + istd * norm.ppf(u)
        t0 = time.time()
        r = run_pair(kl_energy(V, Vp), gap, w2ref, Q0, gamma, N, M, f"R{R}", early_stop=es)
        mg = _measured_gap(r["g_std"])
        s = {"R": R, "init_mean": a["init_mean"], "init_std": istd,
             "initial_kl": float(r["g_std"][0]), "measured_gap": mg,
             "sup_W2sq": r["checks"]["sup_t_W2sq_z_rho"],
             "max_delta_t_1d": r["checks"]["max_delta_t_1d"],
             "frac_steps_with_monotonicity_violation": r["checks"]["frac_steps_with_monotonicity_violation"],
             "wall_seconds": time.time() - t0}
        for th in thr:
            k = f"{-int(round(np.log10(th)))}"
            bs, ba = blocks_to(r["g_std"], th), blocks_to(r["g_acc"], th)
            s[f"blocks_std_1e-{k}"] = bs
            s[f"blocks_acc_1e-{k}"] = ba
            s[f"blocks_ratio_1e-{k}"] = float(bs / ba) if (bs > 0 and ba > 0) else float("nan")
        for row in r["rows"]:
            rows_all.append({"R": R, **row})
        out[R] = {**s, "_g_std": r["g_std"], "_g_acc": r["g_acc"]}
        print(f"[e1a R={R:2d}] gap~{mg:.2e}  blocks->1e-3 std={s['blocks_std_1e-3']} "
              f"acc={s['blocks_acc_1e-3']} ratio={s.get('blocks_ratio_1e-3'):.1f}  "
              f"maxδ={s['max_delta_t_1d']:.1e} ({s['wall_seconds']:.0f}s)")
    eio.save_config("e1a", {"gamma": gamma, **a})
    eio.save_metrics("e1a", rows_all)
    eio.save_summary("e1a", {str(R): {k: v for k, v in out[R].items() if not k.startswith("_")} for R in out})
    _plot_e1a(out, thr)
    return out


#  E1c: non-diffusive functionals (no entropy → no spectral gap → power laws)
def run_e1c(cfg):
    M = cfg["e1c"]["M"]
    gamma, N = cfg["gamma"], cfg["n_blocks"]
    ic = cfg["e1c"]["init"]
    u = (np.arange(M) + 0.5) / M
    Q0 = ic["mean"] + ic["std"] * norm.ppf(u)
    out = {}

    # (i) potential energy only, V = x^4/4, minimizer delta_0
    V, Vp = lambda x: x**4 / 4, lambda x: x**3
    gap = lambda Q: float((Q**4 / 4).mean())
    w2ref = lambda Q: float((Q**2).mean())
    r = run_pair(potential_energy(V, Vp), gap, w2ref, Q0, gamma, N, M, "potential")
    lo = max(1, N // 10)
    si = {"functional": "potential_only",
          "slope_std_potential_only": fit_loglog(r["g_std"], lo, N),
          "slope_acc_potential_only": fit_loglog(r["g_acc"], lo, N),
          "final_std": float(r["g_std"][-1]), "final_acc": float(r["g_acc"][-1]),
          **r["checks"]}
    eio.save_config("e1c_potential", {"gamma": gamma, "M": M, "n_blocks": N, "init": ic, "V": "x^4/4"})
    eio.save_metrics("e1c_potential", r["rows"])
    eio.save_summary("e1c_potential", si)
    _plot_kl("potential_only", r, N, semilog=False)
    out["potential"] = si
    print(f"[e1c potential] slope std={si['slope_std_potential_only']:.2f} "
          f"acc={si['slope_acc_potential_only']:.2f} maxδ={si['max_delta_t_1d']:.1e}")

    # (ii) interaction energy, W(z)=|z|^4, minimizer delta_0
    gap2 = lambda Q: float(0.5 * ((Q[:, None] - Q[None, :]) ** 4).mean())
    r2 = run_pair(interaction_energy(), gap2, w2ref, Q0, gamma, N, M, "interaction")
    sj = {"functional": "interaction",
          "slope_std_interaction": fit_loglog(r2["g_std"], lo, N),
          "slope_acc_interaction": fit_loglog(r2["g_acc"], lo, N),
          "final_std": float(r2["g_std"][-1]), "final_acc": float(r2["g_acc"][-1]),
          **r2["checks"]}
    eio.save_config("e1c_interaction", {"gamma": gamma, "M": M, "n_blocks": N, "init": ic, "W": "|z|^4"})
    eio.save_metrics("e1c_interaction", r2["rows"])
    eio.save_summary("e1c_interaction", sj)
    _plot_kl("interaction", r2, N, semilog=False)
    out["interaction"] = sj
    print(f"[e1c interaction] slope std={sj['slope_std_interaction']:.2f} "
          f"acc={sj['slope_acc_interaction']:.2f} maxδ={sj['max_delta_t_1d']:.1e}")
    return out


def _plot_kl(tid, r, N, semilog):
    t = np.arange(N + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    plot = a1.semilogy if semilog else a1.loglog
    xx = t if semilog else np.arange(1, N + 1)
    ys, ya = (r["g_std"], r["g_acc"]) if semilog else (r["g_std"][1:], r["g_acc"][1:])
    plot(xx, np.maximum(ys, 1e-16), color=BLUE, label="Standard JKO")
    plot(xx, np.maximum(ya, 1e-16), color=RED, label="Accelerated JKO")
    plot(xx, np.maximum(np.minimum.accumulate(ya), 1e-16), ":", color=RED, lw=1.2,
         label="Accelerated running min")
    if not semilog:
        lo = max(1, N // 10)
        ref = np.array([lo, N], dtype=float)
        for g, c in [(r["g_std"], BLUE), (r["g_acc"], RED)]:
            s = fit_loglog(g, lo, N)
            a1.loglog(ref, g[lo] * (ref / lo) ** s, "--", color=c, lw=1.0, alpha=0.7,
                      label=f"fit $t^{{{s:.1f}}}$")
    a1.set_xlabel("Block $t$")
    a1.set_ylabel(r"gap $G(\rho_t)-G(\rho^*)$")
    a1.set_title(f"(a) {tid}")
    a1.legend(loc="best")
    a1.grid(True, which="both")

    a2.semilogx(t, r["w2_acc"], color=RED, label=r"$W_2^2(z_t,\rho^*)$")
    a2.axhline(r["checks"]["sup_t_W2sq_z_rho"], ls="--", color="gray", lw=1.0, label="sup")
    a2.set_xlabel("Block $t$")
    a2.set_ylabel(r"$W_2^2(z_t,\rho^*)$")
    a2.set_title(f"(b) momentum boundedness — {tid}")
    a2.legend(loc="best")
    a2.grid(True, which="both")
    plt.tight_layout()
    fig.savefig(f"images/e1_{tid}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_e1a(out, thr):
    Rs = sorted(out)
    gaps = [out[R]["measured_gap"] for R in Rs]
    tau = min(thr)
    k = f"{-int(round(np.log10(tau)))}"
    bs = [out[R][f"blocks_std_1e-{k}"] for R in Rs]
    ba = [out[R][f"blocks_acc_1e-{k}"] for R in Rs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Rs)))
    for R, c in zip(Rs, cmap):
        g = out[R]["_g_std"]
        a1.semilogy(np.arange(len(g)), np.maximum(g, 1e-16), color=c, label=f"R={R} std")
        ga = out[R]["_g_acc"]
        a1.semilogy(np.arange(len(ga)), np.maximum(ga, 1e-16), "--", color=c)
    a1.set_xlabel("Block $t$")
    a1.set_ylabel(r"KL$(\rho_t\|q_R)$")
    a1.set_title(r"(a) solid=std, dashed=acc; $\lambda=0$ fixed")
    a1.legend(loc="upper right", ncol=2, fontsize=6)
    a1.grid(True, which="both")
    bs = [b if b > 0 else np.nan for b in bs]
    ba = [b if b > 0 else np.nan for b in ba]
    a2.loglog(gaps, bs, "o-", color=BLUE, label="Standard JKO")
    a2.loglog(gaps, ba, "s-", color=RED, label="Accelerated JKO")
    for x, R, y in zip(gaps, Rs, bs):
        if not np.isnan(y):
            a2.annotate(f"R={R}", (x, y), fontsize=7)
    a2.set_xlabel("measured spectral gap")
    a2.set_ylabel(f"blocks to KL $< 10^{{-{k}}}$")
    a2.set_title("(b) cost vs gap: std$\\propto1/$gap, acc$\\propto1/\\sqrt{{}}$gap")
    a2.legend(loc="best")
    a2.invert_xaxis()
    a2.grid(True, which="both")
    plt.tight_layout()
    fig.savefig("images/e1a_gap_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


#  E5/E6: adaptive restart and monotone-fallback safeguard for accelerated JKO
def _momentum(Qxn, Qx, a):
    Qzr = (Qxn - (1 - a) * Qx) / a
    if len(Qzr) > 1 and np.maximum(-np.diff(Qzr), 0).max() > 1e-12:
        Qzr = isotonic(Qzr)
    return Qzr


def run_acc_restart(energy, gap, Q0, gamma, N, M, restart=None):
    Qx = Qz = Q0.copy()
    th = theta_from_Q(Qx, M)
    g, k, restarts = [gap(Qx)], 0, 0
    for t in range(N):
        a = 3.0 / (k + 3.0)
        Qy = (1 - a) * Qx + a * Qz
        Qxn, th = prox_step(Qy, theta_from_Q(Qy, M), energy, gamma, M)
        Qzn = _momentum(Qxn, Qx, a)
        if restart == "function":
            trig = gap(Qxn) > g[-1]
        elif restart == "gradient":                    # O'Donoghue-Candes
            trig = float(((Qy - Qxn) * (Qxn - Qx)).mean()) > 0
        else:
            trig = False
        if trig:
            restarts += 1
            k, Qz = 0, Qxn.copy()
        else:
            k, Qz = k + 1, Qzn
        Qx = Qxn
        g.append(gap(Qx))
    return np.array(g), restarts


def run_safeguard(energy, gap, Q0, gamma, N, M):
    Qx = Qz = Q0.copy()
    th = theta_from_Q(Qx, M)
    g, k, fb = [gap(Qx)], 0, 0
    for t in range(N):
        a = 3.0 / (k + 3.0)
        Qy = (1 - a) * Qx + a * Qz
        Qacc, _ = prox_step(Qy, theta_from_Q(Qy, M), energy, gamma, M)
        Qplain, thp = prox_step(Qx, th, energy, gamma, M)
        if gap(Qacc) <= gap(Qplain):
            Qx, Qz, th, k = Qacc, _momentum(Qacc, Qx, a), theta_from_Q(Qacc, M), k + 1
        else:
            fb += 1
            Qx, Qz, th, k = Qplain, Qplain.copy(), thp, 0
        g.append(gap(Qx))
    return np.array(g), fb


def run_e56(cfg):
    M, gamma, N = cfg["M"], cfg["gamma"], cfg["n_blocks"]
    ic = cfg["e1b"]["init"]
    u = (np.arange(M) + 0.5) / M
    targets = {"quartic": {"kind": "quartic"},
               "gauss_lam1e-3": {"kind": "gauss", "var": 1000.0}}
    for tid, spec in targets.items():
        V, Vp, xref = make_kl_target(spec)
        Qstar = target_quantile(V, xref, u)
        Gmin = kl_free_energy(Qstar, V, M)
        gap = lambda Q: max(kl_free_energy(Q, V, M) - Gmin, 0.0)
        en = kl_energy(V, Vp)
        Q0 = ic["mean"] + ic["std"] * norm.ppf(u)
        g_base, _ = run_acc_restart(en, gap, Q0, gamma, N, M, None)
        g_fun, rc_fun = run_acc_restart(en, gap, Q0, gamma, N, M, "function")
        g_grad, rc_grad = run_acc_restart(en, gap, Q0, gamma, N, M, "gradient")
        g_sg, fb = run_safeguard(en, gap, Q0, gamma, N, M)
        better = "function" if g_fun[-1] <= g_grad[-1] else "gradient"
        s = {"target": tid, "final_gap_acc": float(g_base[-1]),
             "restart_count_function": rc_fun, "final_gap_restart_function": float(g_fun[-1]),
             "restart_count_gradient": rc_grad, "final_gap_restart_gradient": float(g_grad[-1]),
             "better_restart_rule": better,
             "restart_count": rc_fun if better == "function" else rc_grad,
             "final_gap_restart": float(g_fun[-1] if better == "function" else g_grad[-1]),
             "fallback_frac": fb / N, "final_gap_safeguard": float(g_sg[-1])}
        eio.save_config(f"e56_{tid}", {"gamma": gamma, "M": M, "n_blocks": N, "init": ic, "target": spec})
        eio.save_metrics(f"e56_{tid}", [{"block": t, "gap_acc": g_base[t], "gap_restart_function": g_fun[t],
                                         "gap_restart_gradient": g_grad[t], "gap_safeguard": g_sg[t]}
                                        for t in range(N + 1)])
        eio.save_summary(f"e56_{tid}", s)
        t = np.arange(N + 1)
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.semilogy(t, np.maximum(g_base, 1e-16), color=RED, alpha=0.7, label="Accelerated (no restart)")
        ax.semilogy(t, np.maximum(g_fun, 1e-16), color=GREEN, label=f"Function restart ({rc_fun})")
        ax.semilogy(t, np.maximum(g_grad, 1e-16), color=PURPLE, label=f"Gradient restart ({rc_grad})")
        ax.semilogy(t, np.maximum(g_sg, 1e-16), ":", color="k", label=f"Safeguarded (fallback {fb}/{N})")
        ax.set_xlabel("Block $t$")
        ax.set_ylabel(r"gap $G(\rho_t)-G(\rho^*)$")
        ax.set_title(f"E5/E6 restart & safeguard — {tid}")
        ax.legend(loc="upper right")
        ax.grid(True, which="both")
        plt.tight_layout()
        fig.savefig(f"images/e56_{tid}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[e56 {tid}] restarts fun/grad={rc_fun}/{rc_grad} (better={better}) fallback={fb}/{N} "
              f"finalgap acc={g_base[-1]:.2e} restart={s['final_gap_restart']:.2e} safeguard={g_sg[-1]:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--exp", default="all", choices=["e1a", "e1b", "e1c", "e56", "all"])
    ap.add_argument("--blocks", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.blocks:
        cfg["n_blocks"] = args.blocks
    np.random.seed(cfg["seed"])
    if args.exp in ("e1a", "all"):
        run_e1a(cfg)
    if args.exp in ("e1b", "all"):
        run_e1b(cfg)
    if args.exp in ("e1c", "all"):
        run_e1c(cfg)
    if args.exp in ("e56", "all"):
        run_e56(cfg)


if __name__ == "__main__":
    main()
