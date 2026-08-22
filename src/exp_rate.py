import argparse
import os

import numpy as np
from scipy.stats import norm

import jko

GAMMA = 0.5


#  ---- target builders --------------------------------------------------------
def _target_quantile(V, xref, u):
    logp = -V(xref)
    logp -= logp.max()
    p = np.exp(logp)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xref))])
    cdf /= cdf[-1]
    return np.interp(u, cdf, xref)


def _flatvalley_1d(R):
    v, vp = jko.flatvalley(R)
    return (lambda x: v(np.abs(x)), lambda x: vp(np.abs(x)) * np.sign(x),
            np.linspace(-(R + 8), R + 8, 200001))


def _bimodal_1d(R, s=0.7):
    def lq(x):
        return np.logaddexp(norm.logpdf(x, R, s), norm.logpdf(x, -R, s)) + np.log(0.5)

    def V(x):
        return -lq(x)

    def Vp(x):
        a, b = norm.logpdf(x, R, s), norm.logpdf(x, -R, s)
        wa = np.exp(a - np.logaddexp(a, b))
        return -(wa * (-(x - R) / s**2) + (1 - wa) * (-(x + R) / s**2))

    return V, Vp, np.linspace(-(R + 8), R + 8, 200001)


#  ---- one diffusive R-sweep run (blocks to tolerance) ------------------------
def run_quantile(V, Vp, R, seed, M=1000, n_blocks=2000, mode_tol=1e-4):
    rng = np.random.default_rng(seed)
    u = (np.arange(M) + 0.5) / M
    _, _, xref = _flatvalley_1d(R)
    Qstar = _target_quantile(V, xref, u)
    Gmin = jko.quantile_energy(Qstar, V, M)
    Q0 = rng.uniform(3, 7) + rng.uniform(2, 3) * norm.ppf(u)
    prox = lambda Q: jko.prox_quantile(Q, V, Vp, GAMMA)
    gap = lambda Q: max(jko.quantile_energy(Q, V, M) - Gmin, 0.0)
    w2 = lambda Q: float(((Q - Qstar) ** 2).mean())
    out = {}
    for mode in ("std", "acc"):
        out[mode] = jko.run(mode, prox, Q0, GAMMA, n_blocks, gap,
                            geom=jko.IndexGeom(project=jko.isotonic, measure_defect=True),
                            w2_ref=w2, tol_stop=mode_tol)
    return out


def run_gaussian(d, R, seed, n_blocks=8000, mode_tol=1e-4):
    rng = np.random.default_rng(seed)
    dSig = np.concatenate([np.ones(d - 1), [float(R) ** 2]])
    Sigma = np.diag(dSig)
    S0 = np.diag(np.exp(rng.uniform(np.log(0.3), np.log(3.0), d)))
    prox = lambda S: jko.prox_gaussian(S, Sigma, GAMMA, codiagonal=True)
    gap = lambda S: 0.5 * float(np.sum(np.diag(S) / dSig - 1 - np.log(np.diag(S) / dSig)))
    w2 = lambda S: float(np.sum((np.sqrt(np.diag(S)) - np.sqrt(dSig)) ** 2))
    out = {}
    for mode in ("std", "acc"):
        out[mode] = jko.run(mode, prox, S0, GAMMA, n_blocks, gap,
                            geom=jko.GaussGeom(), w2_ref=w2, tol_stop=mode_tol)
    return out


def run_potential_1d(seed, M=1000, n_blocks=200):
    """1-D pure-potential energy (no entropy, minimizer delta_0): reports the
    fraction of momentum steps whose quantile is non-monotone (Assumption 2 check)."""
    rng = np.random.default_rng(seed)
    u = (np.arange(M) + 0.5) / M
    V, Vp = lambda x: x**4 / 4, lambda x: x**3
    Q0 = rng.uniform(3, 7) + rng.uniform(2, 3) * norm.ppf(u)
    prox = lambda Q: jko.prox_quantile(Q, V, Vp, GAMMA, entropy=False)
    gap = lambda Q: float((Q**4 / 4).mean())
    recs = jko.run("acc", prox, Q0, GAMMA, n_blocks, gap,
                   geom=jko.IndexGeom(project=jko.isotonic))
    return float(np.mean([r["mono_violation"] for r in recs]))


#  ---- non-diffusive power-law arms (log-log slope) ---------------------------
def _slope(gaps, frac=0.25):
    g = np.asarray(gaps, float)
    n = len(g)
    lo = max(2, int(frac * n))
    t = np.arange(lo, n)
    y = g[lo:]
    m = y > 1e-12
    return float(np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0]) if m.sum() > 2 else np.nan


def run_potential(d, seed, n=2000, n_blocks=300):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n, d)) * 2.5
    prox = lambda x: jko.prox_radial(x, lambda s: s**3, GAMMA)
    gap = lambda x: float((np.linalg.norm(x, axis=1) ** 4 / 4).mean())
    out = {}
    for mode in ("std", "acc"):
        out[mode] = _slope([r["gap"] for r in jko.run(mode, prox, x0, GAMMA, n_blocks, gap)])
    return out


def run_interaction(d, seed, n=300, n_blocks=120):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n, d)) * 2.5
    wval = lambda diff: np.sum((diff**2).sum(-1) ** 2)
    wgrad = lambda diff: 4 * (diff**2).sum(-1, keepdims=True) * diff
    warm = [None]

    def prox(y):
        z = jko.prox_interaction(y, wval, wgrad, GAMMA, y=y, warm=warm[0])
        warm[0] = z
        return z

    gap = lambda x: float(0.5 * (((x[:, None] - x[None]) ** 2).sum(-1) ** 2).mean())
    out = {}
    for mode in ("std", "acc"):
        warm[0] = None
        out[mode] = _slope([r["gap"] for r in jko.run(mode, prox, x0, GAMMA, n_blocks, gap)])
    return out


#  ---- statistics -------------------------------------------------------------
def exponent_ci(R_arr, blocks, n_boot=10000, sat=6):
    """Fit blocks ~ R^a over the scaling regime with a seed-bootstrap CI.
    blocks: (n_seeds, n_R). Censored entries (-1) are treated as missing, and R
    columns that saturate (median blocks <= sat, i.e. already at the floor) are
    excluded from the fit."""
    nan = float("nan")
    R_arr = np.asarray(R_arr, float)
    blocks = np.where(np.asarray(blocks, float) > 0, np.asarray(blocks, float), np.nan)
    med = np.nanmedian(blocks, 0)
    ok = np.isfinite(med) & (med > sat)
    if ok.sum() < 2:
        return {"exponent": nan, "ci_lo": nan, "ci_hi": nan, "n_points": int(ok.sum())}
    lx = np.log(R_arr[ok])
    a = float(np.polyfit(lx, np.log(med[ok]), 1)[0])
    rng = np.random.default_rng(0)
    ns, sl = blocks.shape[0], []
    for _ in range(n_boot):
        m = np.nanmedian(blocks[rng.integers(0, ns, ns)][:, ok], 0)
        if np.all(np.isfinite(m) & (m > 0)):
            sl.append(np.polyfit(lx, np.log(m), 1)[0])
    if len(sl) < 10:
        return {"exponent": a, "ci_lo": nan, "ci_hi": nan, "n_points": int(ok.sum())}
    lo, hi = np.percentile(sl, [2.5, 97.5])
    return {"exponent": a, "ci_lo": float(lo), "ci_hi": float(hi), "n_points": int(ok.sum())}


def _measured_gap(gaps):
    """Empirical spectral gap from the exponential-decay tail of a standard run."""
    g = np.asarray(gaps, float)
    t = np.arange(len(g))
    m = (g > 1e-8) & (g < 0.5 * g[0]) & (t > 0)
    return float(-np.polyfit(t[m], np.log(g[m]), 1)[0]) if m.sum() > 2 else float("nan")


#  ---- driver -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".")
    ap.add_argument("--from-csv", default=None)
    args = ap.parse_args()
    if args.from_csv:
        return _rebuild(args.from_csv, args.out)

    n_seeds = 2 if args.quick else args.seeds
    R_diff = [4, 8, 16] if args.quick else [2, 4, 8, 16, 32]
    R_quant = [4, 8, 16] if args.quick else [2, 4, 8, 16, 32]
    dims = [2, 10] if args.quick else [2, 10, 50]
    nb = 200 if args.quick else None

    rows = []
    # diffusive convex arms (flat-valley KL) + non-log-concave control (bimodal)
    diff_arms = ([("quantile-1d", None, R_quant, "flatvalley"),
                  ("control-bimodal", None, R_quant, "bimodal")]
                 + [(f"gauss-d{d}", d, R_diff, "gauss") for d in dims])
    for name, d, Rs, kind in diff_arms:
        for R in Rs:
            for seed in range(n_seeds):
                if kind == "gauss":
                    o = run_gaussian(d, R, seed, **({"n_blocks": nb} if nb else {}))
                elif kind == "flatvalley":
                    V, Vp, _ = _flatvalley_1d(R)
                    o = run_quantile(V, Vp, R, seed, **({"n_blocks": nb} if nb else {}))
                else:
                    V, Vp, _ = _bimodal_1d(R)
                    o = run_quantile(V, Vp, R, seed, **({"n_blocks": nb} if nb else {}))
                is_quant = kind in ("flatvalley", "bimodal")
                for mode in ("std", "acc"):
                    recs = o[mode]
                    g = [r["gap"] for r in recs]
                    w2 = [r.get("w2sq_z_rho", np.nan) for r in recs]
                    tp = sum(r["t_prox"] for r in recs)
                    tm = sum(r["t_momentum"] for r in recs)
                    row = {"arm": name, "d": d or 1, "R": R, "seed": seed, "method": mode,
                           "blocks_1e-3": jko.blocks_to(g, 1e-3),
                           "final_gap": g[-1], "sup_w2sq": float(np.nanmax(w2)),
                           "mom_overhead_pct": 100 * tm / (tp + tm + 1e-12)}
                    if mode == "std":
                        row["spectral_gap"] = _measured_gap(g)
                    elif is_quant:
                        row["mono_violation"] = float(np.mean([r.get("mono_violation", 0.0) for r in recs]))
                        defs = np.array([r.get("map_defect", np.nan) for r in recs], float)
                        row["max_map_defect"] = float(np.nanmax(defs)) if np.isfinite(defs).any() else np.nan
                    rows.append(row)

    # 1-D pure-potential energy: fraction of steps with non-monotone momentum (Assumption 2)
    for seed in range(n_seeds):
        rows.append({"arm": "potential-1d-mono", "d": 1, "R": -1, "seed": seed,
                     "method": "mono1d", "mono_violation": run_potential_1d(seed)})

    # non-diffusive unconditional-d>=2 arms: objective-gap power-law slope
    for d in dims:
        for seed in range(n_seeds):
            o = run_potential(d, seed, **({"n_blocks": nb} if nb else {}))
            rows.append({"arm": f"potential-d{d}", "d": d, "R": -1, "seed": seed,
                         "method": "slope", "slope_std": o["std"], "slope_acc": o["acc"]})
    for d in ([1] if args.quick else [1, 2]):
        for seed in range(n_seeds):
            o = run_interaction(d, seed, **({"n_blocks": nb} if nb else {}))
            rows.append({"arm": f"interaction-d{d}", "d": d, "R": -1, "seed": seed,
                         "method": "slope", "slope_std": o["std"], "slope_acc": o["acc"]})

    summary = _summarize(rows)
    cfg = {"seeds": n_seeds, "gamma": GAMMA, "R_diffusive": R_diff, "R_quantile": R_quant, "dims": dims}
    jko.save_run("rate", cfg, rows, summary, outdir=os.path.join(args.out, "results"))
    _figure(rows, summary, os.path.join(args.out, "paper", "rate.pdf"))
    e = summary.get("gauss-d50", {})
    print(f"[rate] gauss-d50 exponent std={e.get('exp_std',{}).get('exponent',float('nan')):.2f} "
          f"acc={e.get('exp_acc',{}).get('exponent',float('nan')):.2f}  "
          f"(prediction: a_std ~ 2 a_acc)")


def _nanmed(v):
    v = np.asarray(v, float)
    return float(np.nanmedian(v)) if v.size and np.isfinite(v).any() else float("nan")


def _blocks_matrix(rows, arm, mode, key, Rs, seeds):
    M = np.full((len(seeds), len(Rs)), np.nan)
    for r in rows:
        if r["arm"] == arm and r["method"] == mode and r.get("R") in Rs and r["seed"] in seeds \
                and r.get(key) not in (None, "", np.nan):
            M[seeds.index(r["seed"]), Rs.index(r["R"])] = float(r[key])
    return M


def _summarize(rows):
    summary = {}
    diff_arms = sorted({r["arm"] for r in rows if r["method"] in ("std", "acc")})
    for arm in diff_arms:
        Rs = sorted({r["R"] for r in rows if r["arm"] == arm and r["method"] == "std"})
        seeds = sorted({r["seed"] for r in rows if r["arm"] == arm and r["method"] == "std"})
        blk_s = _blocks_matrix(rows, arm, "std", "blocks_1e-3", Rs, seeds)
        blk_a = _blocks_matrix(rows, arm, "acc", "blocks_1e-3", Rs, seeds)
        blk_s = np.where(blk_s > 0, blk_s, np.nan)
        blk_a = np.where(blk_a > 0, blk_a, np.nan)
        j = len(Rs) - 1
        entry = {"R": Rs, "exp_std": exponent_ci(Rs, blk_s), "exp_acc": exponent_ci(Rs, blk_a),
                 "median_blocks_std": [_nanmed(blk_s[:, k]) for k in range(len(Rs))],
                 "median_blocks_acc": [_nanmed(blk_a[:, k]) for k in range(len(Rs))],
                 "paired_blocks_diff_maxR": jko.paired_summary(blk_s[:, j] - blk_a[:, j]),
                 "spectral_gap_by_R": [_nanmed(_blocks_matrix(rows, arm, "std", "spectral_gap", Rs, seeds)[:, k])
                                       for k in range(len(Rs))]}
        mono = _blocks_matrix(rows, arm, "acc", "mono_violation", Rs, seeds)
        defect = _blocks_matrix(rows, arm, "acc", "max_map_defect", Rs, seeds)
        if np.isfinite(mono).any():
            entry["mono_violation_median"] = _nanmed(mono)
        if np.isfinite(defect).any():
            entry["max_map_defect"] = float(np.nanmax(defect))
        summary[arm] = entry
    for arm in sorted({r["arm"] for r in rows if r["method"] == "slope"}):
        ss = np.array([r["slope_std"] for r in rows if r["arm"] == arm and r["method"] == "slope"], float)
        sa = np.array([r["slope_acc"] for r in rows if r["arm"] == arm and r["method"] == "slope"], float)
        m = np.isfinite(ss) & np.isfinite(sa)
        summary[arm] = {"slope_std_median": _nanmed(ss[m]), "slope_acc_median": _nanmed(sa[m]),
                        "paired_slope_diff": jko.paired_summary(ss[m] - sa[m])}
    m1 = [r["mono_violation"] for r in rows if r["method"] == "mono1d"]
    if m1:
        summary["potential-1d-mono"] = {"mono_violation_median": float(np.median(m1))}
    return summary


def _figure(rows, summary, path):
    fig, ax = jko.figure(ncols=2, full=True)
    colors = [jko.BLUE, jko.ORANGE, "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
    diff_arms = [a for a in summary if a.startswith(("quantile", "gauss", "control"))]
    for arm, c in zip(sorted(diff_arms), colors):
        Rs = np.array(summary[arm]["R"], float)
        seeds = sorted({r["seed"] for r in rows if r["arm"] == arm})
        blk = _blocks_matrix(rows, arm, "acc", "blocks_1e-3", list(Rs), seeds)
        blk_s = _blocks_matrix(rows, arm, "std", "blocks_1e-3", list(Rs), seeds)
        blk = np.where(blk > 0, blk, np.nan)
        blk_s = np.where(blk_s > 0, blk_s, np.nan)
        jko.median_iqr(ax[0], Rs, blk_s, c, label=f"{arm} std")
        jko.median_iqr(ax[0], Rs, blk, c, ls="--")
    Rref = np.array(sorted({r["R"] for r in rows if r["arm"].startswith("gauss")}), float)
    mbs = summary.get("gauss-d10", {}).get("median_blocks_std", [])
    b0 = next((m for m in mbs if m and np.isfinite(m)), 1.0)
    ax[0].loglog(Rref, b0 * (Rref / Rref[0]) ** 2, ":", color="grey", lw=0.8)
    ax[0].loglog(Rref, b0 * (Rref / Rref[0]), ":", color="grey", lw=0.8)
    ax[0].set_xlabel("degeneracy $R$")
    ax[0].set_ylabel(r"blocks to gap $<10^{-3}$")
    ax[0].legend(fontsize=6, ncol=2)

    slope_arms = [a for a in summary if "slope_std_median" in summary[a]]
    x = np.arange(len(slope_arms))
    ms = [summary[a]["slope_std_median"] for a in slope_arms]
    ma = [summary[a]["slope_acc_median"] for a in slope_arms]
    ax[1].plot(x, ms, "o", color=jko.BLUE, label="standard")
    ax[1].plot(x, ma, "s", color=jko.ORANGE, label="accelerated")
    ax[1].axhline(-2, ls=":", color="grey", lw=0.8)
    ax[1].axhline(-4, ls=":", color="grey", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(slope_arms, rotation=30, ha="right", fontsize=6)
    ax[1].set_ylabel("log-log gap slope")
    ax[1].legend()
    jko.finish(fig, path)


def _rebuild(csv_path, out):
    import csv
    import json
    rows = []
    for r in csv.DictReader(open(csv_path)):
        for k in ("d", "R", "seed"):
            r[k] = int(r[k])
        if r.get("blocks_1e-3") not in (None, ""):
            r["blocks_1e-3"] = int(r["blocks_1e-3"])
        for k in ("final_gap", "sup_w2sq", "mom_overhead_pct", "slope_std", "slope_acc",
                  "spectral_gap", "mono_violation", "max_map_defect"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        rows.append({k: v for k, v in r.items() if v not in (None, "")})
    summary = _summarize(rows)
    json.dump(summary, open(os.path.join(os.path.dirname(csv_path), "summary.json"), "w"), indent=2)
    _figure(rows, summary, os.path.join(out, "paper", "rate.pdf"))
    print("[rate] rebuilt figure/summary from", csv_path)


if __name__ == "__main__":
    main()
