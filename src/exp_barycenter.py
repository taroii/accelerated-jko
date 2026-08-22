import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_banded
from scipy.optimize import minimize

import jko


# ============================================================ ported numerics
class Barycenter:
    """Entropic barycenter dual on [0,1]. Randomized per seed: marginal centres ~
    U(0.15,0.85), widths ~ U(0.05,0.09), weights ~ Dirichlet. Poisson solve and dual
    objective are unchanged from the reference implementation."""

    def __init__(self, n=60, m=4, eps=0.008, seed=0):
        self.n, self.m, self.eps = n, m, eps
        self.x = np.linspace(0, 1, n)
        self.h = self.x[1] - self.x[0]
        rng = np.random.default_rng(seed)
        centres = rng.uniform(0.15, 0.85, m)
        sigmas = rng.uniform(0.05, 0.09, m)
        self.MU = [self._bump(c, s) for c, s in zip(centres, sigmas)]
        self.logMU = [np.log(mu + 1e-20) for mu in self.MU]
        w = rng.dirichlet(np.ones(m))
        self.w = w / w.sum()
        self.log_nu = np.log(np.full(n, 1.0 / n))
        self.C = 0.5 * (self.x[:, None] - self.x[None, :]) ** 2

    def _bump(self, c, s):
        d = np.exp(-0.5 * ((self.x - c) / s) ** 2)
        return d / d.sum()

    def poisson(self, rhs):
        n, h = self.n, self.h
        ab = np.zeros((3, n))
        ab[1] = 2.0 / h**2
        ab[0, 1:] = -1.0 / h**2
        ab[2, :-1] = -1.0 / h**2
        ab[1, 0] = 1.0 / h**2
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0
        r = rhs.copy()
        r[-1] = 0.0
        u = solve_banded((1, 1), ab, r)
        return u - u.mean()

    def h1_ip(self, a, b):
        return float((np.diff(a) * np.diff(b)).sum() / self.h)

    def A(self, v):                      # pseudo-Laplacian: grad of 0.5||v||^2_H1
        w = np.diff(v)
        out = np.zeros(self.n)
        out[:-1] -= w
        out[1:] += w
        return out / self.h

    def I_val(self, fs):
        val = 0.0
        for i in range(self.m):
            fi = fs[i]
            gi = -self.eps * logsumexp(self.logMU[i][:, None] + fi[:, None] / self.eps
                                       - self.C / self.eps, axis=0)
            val += self.w[i] * ((fi * self.MU[i]).sum() + (gi / self.n).sum())
        return float(val)

    def residual(self, fs):              # Euclidean gradient of I w.r.t. each f_i
        res = []
        for i in range(self.m):
            fi = fs[i]
            gi = -self.eps * logsumexp(self.logMU[i][:, None] + fi[:, None] / self.eps
                                       - self.C / self.eps, axis=0)
            lp = (self.logMU[i][:, None] + fi[:, None] / self.eps + gi[None, :] / self.eps
                  - self.C / self.eps + self.log_nu[None, :])
            lp -= logsumexp(lp)
            pi = np.exp(lp)
            res.append(self.MU[i] - pi.sum(axis=1))
        return res

    def I_grad(self, fs):                # Sobolev (H1) gradient
        return [self.poisson(r) for r in self.residual(fs)]


def sga(prob, N, eta, Istar):
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    gaps = []
    for _ in range(N):
        gs = prob.I_grad(fs)
        fs = [fs[i] + eta * gs[i] for i in range(prob.m)]
        gaps.append(max(Istar - prob.I_val(fs), 1e-14))
    return np.array(gaps)


def asga(prob, N, eta, Istar):
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    zs = [f.copy() for f in fs]
    gaps = []
    for t in range(1, N + 1):
        a = 3.0 / (t + 3.0)
        ys = [(1 - a) * fs[i] + a * zs[i] for i in range(prob.m)]
        gs = prob.I_grad(ys)
        fs = [ys[i] + eta * gs[i] for i in range(prob.m)]
        zs = [zs[i] + (eta / a) * gs[i] for i in range(prob.m)]
        gaps.append(max(Istar - prob.I_val(fs), 1e-14))
    return np.array(gaps)


def _exact_prox(prob, ys, eta):
    n, m = prob.n, prob.m

    def obj(flat):
        fs = [flat[i * n:(i + 1) * n] for i in range(m)]
        val = -prob.I_val(fs)
        res = prob.residual(fs)
        grad = np.zeros(m * n)
        for i in range(m):
            diff = fs[i] - ys[i]
            val += prob.w[i] * prob.h1_ip(diff, diff) / (2 * eta)
            grad[i * n:(i + 1) * n] = -res[i] + prob.w[i] * prob.A(diff) / eta
        return val, grad

    x0 = np.concatenate(ys)
    r = minimize(obj, x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-18})
    return [r.x[i * n:(i + 1) * n] for i in range(m)], r.nit


def asga_exact_prox(prob, N, eta, Istar):
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    zs = [f.copy() for f in fs]
    gaps, inner = [], []
    for t in range(1, N + 1):
        a = 3.0 / (t + 3.0)
        ys = [(1 - a) * fs[i] + a * zs[i] for i in range(prob.m)]
        fs, nit = _exact_prox(prob, ys, eta)
        inner.append(nit)
        zs = [zs[i] + (1.0 / a) * (fs[i] - ys[i]) for i in range(prob.m)]
        gaps.append(max(Istar - prob.I_val(fs), 1e-14))
    return np.array(gaps), np.array(inner)


def empirical_L(prob, eps=1e-4, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    f0 = [np.zeros(prob.n) for _ in range(prob.m)]
    g0 = prob.I_grad(f0)
    v = [rng.standard_normal(prob.n) for _ in range(prob.m)]

    def hess(vv):
        gp = prob.I_grad([f0[i] + eps * vv[i] for i in range(prob.m)])
        return [(gp[i] - g0[i]) / eps for i in range(prob.m)]

    def ip(a, b):
        return sum(prob.h1_ip(a[i], b[i]) for i in range(prob.m))

    lam = 0.0
    for _ in range(iters):
        hv = hess(v)
        lam = abs(ip(v, hv) / max(ip(v, v), 1e-30))
        nrm = np.sqrt(max(ip(hv, hv), 1e-30))
        v = [hv[i] / nrm for i in range(prob.m)]
    return lam


def certified_L(prob):
    # ||Hess I||_H1 <= C_P * ||Hess I||_eucl ; C_P = 1/pi^2 (Neumann Poincare on [0,1]),
    # euclidean Hessian bounded by (max marginal density)/eps.
    rho_max = max(mu.max() / prob.h for mu in prob.MU)
    return rho_max / (np.pi**2 * prob.eps)


def _long_ref(prob, N=3000):
    L = empirical_L(prob)
    eta = 1.0 / L
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    zs = [f.copy() for f in fs]
    for t in range(1, N + 1):
        a = 3.0 / (t + 3.0)
        ys = [(1 - a) * fs[i] + a * zs[i] for i in range(prob.m)]
        gs = prob.I_grad(ys)
        fs = [ys[i] + eta * gs[i] for i in range(prob.m)]
        zs = [zs[i] + (eta / a) * gs[i] for i in range(prob.m)]
    return fs


# ============================================================ slope fit
_FLOOR = 1e-11        # numerical floor (gaps are clipped at 1e-14); excluded from the fit


def _slope(gap, decade=10.0, floor=_FLOOR):
    """Log-log slope of the duality gap over the last decade of iterations,
    excluding samples that have hit the numerical floor."""
    gap = np.asarray(gap, float)
    N = len(gap)
    t = np.arange(1, N + 1)
    lo = max(2, int(N / decade))
    mk = (t >= lo) & (gap > floor)
    if mk.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(t[mk]), np.log(gap[mk]), 1)[0])


# ============================================================ instance runners
REP = (0.008, 60, 4)          # representative cell (left-panel curves)


def _run_instance(eps, n, m, seed, N, Nref, want_curves):
    prob = Barycenter(n=n, m=m, eps=eps, seed=seed)
    L_emp = empirical_L(prob, seed=seed)
    L_cert = certified_L(prob)
    Istar = prob.I_val(_long_ref(prob, Nref)) + 1e-12
    rows, info, curves = [], {"L_emp": L_emp, "L_cert": L_cert}, {}
    for step, L in (("emp", L_emp), ("cert", L_cert)):
        eta = 1.0 / L
        g_sga = sga(prob, N, eta, Istar)
        g_asga = asga(prob, N, eta, Istar)
        for method, g in (("SGA", g_sga), ("ASGA", g_asga)):
            sl = _slope(g)
            rows.append({"arm": "main", "seed": seed, "eps": eps, "n": n, "m": m,
                         "method": method, "stepsize": step, "slope": sl,
                         "final_gap": float(g[-1]), "L_emp": L_emp, "L_cert": L_cert,
                         "eta": eta, "inner_iters_mean": ""})
            info[(method, step)] = sl
        if want_curves and step == "emp":
            curves["SGA"], curves["ASGA"] = g_sga, g_asga
    return rows, info, curves


def _istar_exactprox(prob, Nref, eta):
    """Tight reference for the exact-prox cell: best of a long explicit run and a long
    exact-prox run (exact-prox can overshoot the explicit reference)."""
    best = prob.I_val(_long_ref(prob, Nref))
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    zs = [f.copy() for f in fs]
    for t in range(1, Nref + 1):
        a = 3.0 / (t + 3.0)
        ys = [(1 - a) * fs[i] + a * zs[i] for i in range(prob.m)]
        fs, _ = _exact_prox(prob, ys, eta)
        zs = [zs[i] + (1.0 / a) * (fs[i] - ys[i]) for i in range(prob.m)]
        best = max(best, prob.I_val(fs))
    return best + 1e-12


def _run_exactprox(eps, seed, N, Nref):
    prob = Barycenter(n=30, m=2, eps=eps, seed=seed)
    L_emp = empirical_L(prob, seed=seed)
    L_cert = certified_L(prob)
    rows = []
    for step, L in (("emp", L_emp), ("cert", L_cert)):
        eta = 1.0 / L
        Istar = _istar_exactprox(prob, Nref, eta)
        g_ex = asga(prob, N, eta, Istar)
        g_ep, inner = asga_exact_prox(prob, N, eta, Istar)
        base = {"arm": "exactprox", "seed": seed, "eps": eps, "n": 30, "m": 2,
                "stepsize": step, "L_emp": L_emp, "L_cert": L_cert, "eta": eta}
        rows.append({**base, "method": "ASGA", "slope": _slope(g_ex),
                     "final_gap": float(g_ex[-1]), "inner_iters_mean": ""})
        rows.append({**base, "method": "ASGA-exactprox", "slope": _slope(g_ep),
                     "final_gap": float(g_ep[-1]), "inner_iters_mean": float(inner.mean())})
    return rows


# ============================================================ config
def make_cfg(args):
    if args.quick:
        return {"eps": [0.008], "ns": [60], "ms": [4], "seeds": 2, "N": 120, "Nref": 800,
                "ep_eps": 0.008, "ep_seeds": 2, "ep_N": 60, "ep_ref": 60, "quick": True}
    return {"eps": [0.002, 0.004, 0.008, 0.016], "ns": [60, 120, 240], "ms": [2, 4, 8],
            "seeds": args.seeds, "N": 400, "Nref": 3000,
            "ep_eps": 0.008, "ep_seeds": min(args.seeds, 3), "ep_N": 300, "ep_ref": 200,
            "quick": False,
            "note": "reported quantity is the fitted-slope distribution over seeds per cell"}


# ============================================================ summary
_INT = {"seed", "n", "m"}
_FLOATS = {"eps", "slope", "final_gap", "L_emp", "L_cert", "eta", "inner_iters_mean"}


def _coerce(r):
    o = dict(r)
    for k in _INT:
        if k in o and o[k] not in ("", None):
            o[k] = int(float(o[k]))
    for k in _FLOATS:
        if k in o:
            o[k] = float(o[k]) if o[k] not in ("", None) else None
    return o


def _cellkey(eps, n, m):
    return f"eps{eps:g}_n{n}_m{m}"


def _epskey(eps):
    return f"{eps:g}"


def _dist(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"median": float("nan"), "iqr_lo": float("nan"), "iqr_hi": float("nan"), "n": 0}
    return {"median": float(np.median(v)),
            "iqr_lo": float(np.percentile(v, 25)),
            "iqr_hi": float(np.percentile(v, 75)), "n": int(len(v))}


def _fit_curve(iters, med):
    lo = max(2, int(len(iters) / 10.0))
    mk = (iters >= lo) & (med > _FLOOR)
    if mk.sum() < 3:
        return {"exponent": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return jko.bootstrap_exponent(iters[mk], med[mk])


def summarize(rows, rep):
    rows = [_coerce(r) for r in rows]
    main = [r for r in rows if r.get("arm") == "main"]

    # ---- per-cell slope distributions + paired ASGA-vs-SGA advantage (empirical step)
    by = defaultdict(lambda: defaultdict(dict))
    Lmap = defaultdict(dict)
    for r in main:
        key = (r["eps"], r["n"], r["m"])
        by[key][r["seed"]][(r["method"], r["stepsize"])] = r["slope"]
        Lmap[key][r["seed"]] = (r["L_emp"], r["L_cert"])

    cells = {}
    all_ratio, all_eemp, all_ecert = [], [], []
    for key, seeds_d in by.items():
        sga_s, asga_s, diff, ratios, eemp, ecert = [], [], [], [], [], []
        for s, md in seeds_d.items():
            sg, ag = md.get(("SGA", "emp")), md.get(("ASGA", "emp"))
            if sg is not None and ag is not None and np.isfinite(sg) and np.isfinite(ag):
                sga_s.append(sg); asga_s.append(ag); diff.append(sg - ag)  # advantage = steeper ASGA
            Le, Lc = Lmap[key][s]
            ratios.append(Lc / Le); eemp.append(1.0 / Le); ecert.append(1.0 / Lc)
        all_ratio += ratios; all_eemp += eemp; all_ecert += ecert
        cells[_cellkey(*key)] = {
            "sga_slope": _dist(sga_s), "asga_slope": _dist(asga_s),
            "paired_advantage": jko.paired_summary(diff) if diff else {},
            "L_ratio_median": float(np.median(ratios)),
            "eta_emp_median": float(np.median(eemp)),
            "eta_cert_median": float(np.median(ecert))}

    # ---- ASGA slope distribution across eps at both step sizes (right panel)
    by_eps = {}
    for eps in sorted(set(r["eps"] for r in main)):
        d = {}
        for step in ("emp", "cert"):
            sl = [r["slope"] for r in main
                  if r["method"] == "ASGA" and r["stepsize"] == step and r["eps"] == eps]
            d[step] = _dist(sl)
        by_eps[_epskey(eps)] = d

    # ---- exact-prox vs explicit
    ep = [r for r in rows if r.get("arm") == "exactprox"]
    ep_sum = {}
    for step in ("emp", "cert"):
        ex = [r["slope"] for r in ep if r["method"] == "ASGA" and r["stepsize"] == step]
        px = [r["slope"] for r in ep if r["method"] == "ASGA-exactprox" and r["stepsize"] == step]
        inner = [r["inner_iters_mean"] for r in ep
                 if r["method"] == "ASGA-exactprox" and r["stepsize"] == step
                 and r["inner_iters_mean"] is not None]
        ep_sum[step] = {"explicit_slope": _dist(ex), "exactprox_slope": _dist(px),
                        "inner_iters_mean": float(np.mean(inner)) if inner else None}

    summary = {
        "n_cells": len(cells),
        "representative_cell": _cellkey(*REP),
        "cells": cells,
        "asga_slope_by_eps": by_eps,
        "exact_prox": ep_sum,
        "L_ratio_overall_median": float(np.median(all_ratio)) if all_ratio else float("nan"),
        "eta_emp_overall_median": float(np.median(all_eemp)) if all_eemp else float("nan"),
        "eta_cert_overall_median": float(np.median(all_ecert)) if all_ecert else float("nan"),
    }

    # ---- representative curves + scaling fit (median gap ~ t^a with bootstrap CI)
    if rep and rep.get("iters"):
        it = np.asarray(rep["iters"], float)
        sga_all = np.asarray(rep["sga_all"], float)
        asga_all = np.asarray(rep["asga_all"], float)
        summary["rep_curves"] = {"iters": it.tolist(),
                                 "sga_all": sga_all.tolist(), "asga_all": asga_all.tolist()}
        summary["rep_scaling_fit"] = {
            "sga": _fit_curve(it, np.median(sga_all, axis=0)),
            "asga": _fit_curve(it, np.median(asga_all, axis=0))}
    else:
        summary["rep_curves"] = {}
        summary["rep_scaling_fit"] = {}
    return summary


# ============================================================ figure
def make_figure(summary, path):
    fig, (axL, axR) = jko.figure(ncols=2, full=True)

    rc = summary.get("rep_curves", {})
    if rc and rc.get("iters"):
        it = np.asarray(rc["iters"], float)
        sga_all = np.asarray(rc["sga_all"], float)
        asga_all = np.asarray(rc["asga_all"], float)
        jko.median_iqr(axL, it, sga_all, jko.BLUE, label="SGA")
        jko.median_iqr(axL, it, asga_all, jko.ORANGE, label="ASGA")
        axL.set_xscale("log"); axL.set_yscale("log")
        lo = max(1, len(it) // 10)
        tt = it[lo:]
        smed, amed = np.median(sga_all, axis=0), np.median(asga_all, axis=0)
        c1, c2 = smed[lo] * it[lo], amed[lo] * it[lo] ** 2
        axL.plot(tt, c1 / tt, ls="--", color="0.6", lw=0.8)
        axL.plot(tt, c2 / tt ** 2, ls="--", color="0.6", lw=0.8)
        axL.text(tt[-1], c1 / tt[-1], r"$t^{-1}$", color="0.5", fontsize=7,
                 ha="right", va="bottom")
        axL.text(tt[-1], c2 / tt[-1] ** 2, r"$t^{-2}$", color="0.5", fontsize=7,
                 ha="right", va="top")
    else:
        axL.text(0.5, 0.5, "representative curves unavailable",
                 ha="center", va="center", transform=axL.transAxes, fontsize=8)
    axL.set_xlabel(r"iteration $t$")
    axL.set_ylabel(r"duality gap $\mathcal{I}^\star-\mathcal{I}(f^{(t)})$")
    axL.legend(loc="lower left")

    by_eps = summary.get("asga_slope_by_eps", {})
    epss = sorted(float(e) for e in by_eps)
    xs = np.arange(len(epss))
    for off, step, color, lab in ((-0.12, "emp", jko.ORANGE, r"$1/L_{\mathrm{emp}}$"),
                                  (0.12, "cert", "0.5", r"$1/L_{\mathrm{cert}}$")):
        med, lo_, hi_ = [], [], []
        for e in epss:
            d = by_eps[_epskey(e)][step]
            med.append(d["median"]); lo_.append(d["iqr_lo"]); hi_.append(d["iqr_hi"])
        med, lo_, hi_ = np.array(med), np.array(lo_), np.array(hi_)
        axR.errorbar(xs + off, med, yerr=[med - lo_, hi_ - med], fmt="o", ms=4,
                     color=color, capsize=2, lw=1.2, label=lab)
    axR.axhline(-2.0, ls=":", color="0.7", lw=0.8)
    axR.axhline(-1.0, ls=":", color="0.7", lw=0.8)
    axR.text(xs[-1] if len(xs) else 0, -2.0, r"$-2$", color="0.5", fontsize=7,
             ha="left", va="bottom")
    axR.set_xticks(xs)
    axR.set_xticklabels([f"{e:g}" for e in epss])
    axR.set_xlabel(r"entropic $\varepsilon$")
    axR.set_ylabel("ASGA fitted slope")
    axR.legend(loc="best")

    jko.finish(fig, path)


# ============================================================ paper line
def paper_line(summary, cfg=None):
    rep = summary["cells"].get(summary["representative_cell"], {})
    ss, aa = rep.get("sga_slope", {}), rep.get("asga_slope", {})
    pa = rep.get("paired_advantage", {})
    ep = summary["exact_prox"].get("emp", {})
    epx, epp = ep.get("exactprox_slope", {}), ep.get("explicit_slope", {})

    def g(d, k):
        return d.get(k, float("nan"))
    return ("[barycenter] cells={nc} rep={rc} | "
            "SGA slope {ss:.2f} [{sl:.2f},{sh:.2f}] (O(1/T)), "
            "ASGA slope {as_:.2f} [{al:.2f},{ah:.2f}] (O(1/T^2)); "
            "paired ASGA advantage median {pm:+.2f} [{pl:+.2f},{ph:+.2f}] "
            "(Wilcoxon p={pp:.1e}, {kp}/{pn} seeds); "
            "L_cert/L_emp median {lr:.0f}x (step 1/L_emp={ee:.2e} vs 1/L_cert={ec:.2e}, "
            "both convergent); exact-prox slope {xp:.2f} vs explicit {xe:.2f} "
            "at {ii:.0f} mean inner iters.").format(
        nc=summary["n_cells"], rc=summary["representative_cell"],
        ss=g(ss, "median"), sl=g(ss, "iqr_lo"), sh=g(ss, "iqr_hi"),
        as_=g(aa, "median"), al=g(aa, "iqr_lo"), ah=g(aa, "iqr_hi"),
        pm=g(pa, "median"), pl=g(pa, "ci_lo"), ph=g(pa, "ci_hi"),
        pp=g(pa, "wilcoxon_p"), kp=pa.get("k_positive", 0), pn=pa.get("n", 0),
        lr=summary["L_ratio_overall_median"],
        ee=summary["eta_emp_overall_median"], ec=summary["eta_cert_overall_median"],
        xp=g(epx, "median"), xe=g(epp, "median"),
        ii=(ep.get("inner_iters_mean") or float("nan")))


# ============================================================ drivers
def run_full(cfg, out):
    rows, rep_sga, rep_asga, rep_iters = [], [], [], None
    for eps in cfg["eps"]:
        for n in cfg["ns"]:
            for m in cfg["ms"]:
                is_rep = abs(eps - REP[0]) < 1e-12 and n == REP[1] and m == REP[2]
                for seed in range(cfg["seeds"]):
                    r, info, curves = _run_instance(eps, n, m, seed, cfg["N"], cfg["Nref"],
                                                    want_curves=is_rep)
                    rows += r
                    if is_rep:
                        rep_sga.append(curves["SGA"]); rep_asga.append(curves["ASGA"])
                        rep_iters = np.arange(1, cfg["N"] + 1)
                    print(f"[bary] eps={eps:<6g} n={n:<3d} m={m} seed={seed:<2d} "
                          f"slope SGA={info[('SGA','emp')]:.2f} ASGA={info[('ASGA','emp')]:.2f}")

    for seed in range(cfg["ep_seeds"]):
        er = _run_exactprox(cfg["ep_eps"], seed, cfg["ep_N"], cfg["ep_ref"])
        rows += er
        emp = [x for x in er if x["stepsize"] == "emp"]
        sl = {x["method"]: x["slope"] for x in emp}
        ii = next(x["inner_iters_mean"] for x in emp if x["method"] == "ASGA-exactprox")
        print(f"[bary/exact-prox] eps={cfg['ep_eps']:g} n=30 m=2 seed={seed} "
              f"explicit={sl['ASGA']:.2f} exact-prox={sl['ASGA-exactprox']:.2f} inner={ii:.1f}")

    rep = {"iters": rep_iters.tolist() if rep_iters is not None else [],
           "sga_all": np.array(rep_sga).tolist(), "asga_all": np.array(rep_asga).tolist()}
    summary = summarize(rows, rep)
    jko.save_run("barycenter", cfg, rows, summary, outdir=os.path.join(out, "results"))
    make_figure(summary, os.path.join(out, "figures", "barycenter.pdf"))
    print(paper_line(summary, cfg))


def rebuild_from_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    d = os.path.dirname(os.path.abspath(path))
    summ_path = os.path.join(d, "summary.json")
    rep = None
    if os.path.exists(summ_path):
        rep = json.load(open(summ_path)).get("rep_curves")
    summary = summarize(rows, rep)
    json.dump(summary, open(summ_path, "w"), indent=2)
    out = os.path.dirname(os.path.dirname(d)) or "."
    make_figure(summary, os.path.join(out, "figures", "barycenter.pdf"))
    print(paper_line(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50, help="random instances per cell")
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, single small cell (eps=0.008,n=60,m=4), short horizons")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--from-csv", default=None,
                    help="rebuild figure + summary from an existing per_seed.csv")
    args = ap.parse_args()

    if args.from_csv:
        rebuild_from_csv(args.from_csv)
        return
    run_full(make_cfg(args), args.out)


if __name__ == "__main__":
    main()
