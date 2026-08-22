"""Shared core for the accelerated-JKO experiment suite: exact proximal oracles,
Bures/Gaussian geometry, the standard and accelerated schemes, paired statistics,
and figure style. Run `python jko.py` to execute the correctness checks."""
import csv
import json
import os
import subprocess
import time

import numpy as np
from numpy.linalg import inv, eigh
from scipy.optimize import minimize
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#0072B2"
ORANGE = "#D55E00"


#  ------------------------------------------------------------------ SPD / Bures
def _sym(A):
    return 0.5 * (A + A.T)


def _powm(A, p):
    w, Q = eigh(_sym(A))
    return _sym((Q * np.clip(w, 1e-15, None) ** p) @ Q.T)


def sqrtm_spd(A):
    return _powm(A, 0.5)


def invsqrtm_spd(A):
    return _powm(A, -0.5)


def bw_map(S_from, S_to):
    """Optimal (Brenier) linear map between centered Gaussians N(0,S_from)->N(0,S_to)."""
    sf, isf = sqrtm_spd(S_from), invsqrtm_spd(S_from)
    return _sym(isf @ sqrtm_spd(_sym(sf @ S_to @ sf)) @ isf)


def bw_distance(S1, S2):
    s1 = sqrtm_spd(S1)
    cross = sqrtm_spd(_sym(s1 @ S2 @ s1))
    return float(np.sqrt(max(np.trace(S1 + S2 - 2 * cross), 0.0)))


def gaussian_kl(S, Sigma):
    """KL(N(0,S) || N(0,Sigma))."""
    d = S.shape[0]
    M = inv(Sigma) @ S
    return float(0.5 * (np.trace(M) - d - np.log(max(np.linalg.det(M), 1e-300))))


def map_defect(M, A, S_base):
    """||M - A||_{L2(N(0,S_base))} = sqrt(tr((M-A) S_base (M-A)^T))."""
    D = M - A
    return float(np.sqrt(max(np.trace(D @ S_base @ D.T), 0.0)))


#  ------------------------------------------------------------------ oracles
def prox_radial(x, vprime, gamma):
    """Pointwise prox of a radial potential V(x)=v(|x|): T(x)=(s*/|x|) x, where
    s* solves v'(s)+(s-|x|)/gamma=0. Exact in any dimension. x: (n,d)."""
    r = np.linalg.norm(x, axis=1, keepdims=True)
    lo = np.zeros_like(r)
    hi = np.maximum(r, 1.0)
    for _ in range(60):
        grow = vprime(hi) + (hi - r) / gamma <= 0
        if not grow.any():
            break
        hi = np.where(grow, hi * 2.0, hi)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        pos = vprime(mid) + (mid - r) / gamma > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    s = 0.5 * (lo + hi)
    return np.where(r > 0, s / np.maximum(r, 1e-300), 1.0) * x


def prox_interaction(x, wval, wgrad, gamma, y=None, warm=None):
    """Prox for interaction energy G(mu)=0.5/n^2 sum_ij w(x_i-x_j) via L-BFGS on
    the stacked positions. wval(diff)->scalar sum, wgrad(diff)->(n,n,d)=w'(u)."""
    n, d = x.shape
    y = x if y is None else y

    def fg(zf):
        z = zf.reshape(n, d)
        diff = z[:, None, :] - z[None, :, :]
        f = 0.5 * wval(diff) / n**2 + np.sum((z - y) ** 2) / (2 * gamma * n)
        g = wgrad(diff).sum(1) / n**2 + (z - y) / (gamma * n)
        return f, g.ravel()

    z0 = (warm if warm is not None else y).ravel()
    r = minimize(fg, z0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-12})
    return r.x.reshape(n, d)


def prox_gaussian(S_prev, Sigma, gamma, codiagonal=False):
    """JKO step for KL(.||N(0,Sigma)) restricted to centered Gaussians:
    argmin_{S'} G(S') + B^2(S',S_prev)/(2 gamma)."""
    d = S_prev.shape[0]
    if codiagonal:                              # S_prev, Sigma diagonal: decouples
        lam = 1.0 / np.diag(Sigma)
        s = np.diag(S_prev)
        a = lam + 1.0 / gamma
        u = (np.sqrt(s) / gamma + np.sqrt(s / gamma**2 + 4 * a)) / (2 * a)
        return np.diag(u**2)
    Sig_inv = inv(Sigma)
    Ssq = sqrtm_spd(S_prev)

    def fg(Lf):
        L = Lf.reshape(d, d)
        Sp = _sym(L @ L.T) + 1e-12 * np.eye(d)
        C = sqrtm_spd(_sym(Ssq @ Sp @ Ssq))
        f = (0.5 * (np.trace(Sig_inv @ Sp) - np.log(max(np.linalg.det(Sp), 1e-300)))
             + (np.trace(Sp) + np.trace(S_prev) - 2 * np.trace(C)) / (2 * gamma))
        gS = 0.5 * (Sig_inv - inv(Sp)) + (np.eye(d) - Ssq @ invsqrtm_spd(_sym(Ssq @ Sp @ Ssq)) @ Ssq) / (2 * gamma)
        return f, (2 * _sym(gS) @ L).ravel()

    L0 = np.linalg.cholesky(S_prev + 1e-9 * np.eye(d))
    r = minimize(fg, L0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-9})
    L = r.x.reshape(d, d)
    return _sym(L @ L.T)


def _q_unpack(theta, M):
    e = np.exp(np.clip(theta[1:], -60, 40))
    return theta[0] + np.concatenate([[0.0], np.cumsum(e)]) / M, e


def _q_theta(Q, M):
    return np.concatenate([[Q[0]], np.log(np.clip(M * np.diff(Q), 1e-12, None))])


def quantile_energy(Q, V, M, entropy=True):
    e = V(Q).mean()
    return e - np.log(M * np.diff(Q)).sum() / M if entropy else e


def prox_quantile(Q_target, V, Vp, gamma, entropy=True):
    """Prox in 1-D quantile coordinates (monotone by construction)."""
    M = len(Q_target)

    def obj(theta):
        Q, e = _q_unpack(theta, M)
        f = V(Q).mean() + ((Q - Q_target) ** 2).mean() / (2 * gamma)
        gQ = (Vp(Q) + (Q - Q_target) / gamma) / M
        if entropy:
            f -= theta[1:].sum() / M
        suf = np.cumsum(gQ[::-1])[::-1]
        grad = np.empty_like(theta)
        grad[0] = suf[0]
        grad[1:] = e * suf[1:] / M - (1.0 / M if entropy else 0.0)
        return f, grad

    r = minimize(obj, _q_theta(Q_target, M), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-10})
    return _q_unpack(r.x, M)[0]


def isotonic(y):
    n = len(y)
    lvl = list(map(float, y))
    w = [1.0] * n
    idx = list(range(n + 1))
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] > lvl[i + 1]:
            lvl[i] = (w[i] * lvl[i] + w[i + 1] * lvl[i + 1]) / (w[i] + w[i + 1])
            w[i] += w[i + 1]
            del lvl[i + 1], w[i + 1], idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty(n)
    for k in range(len(lvl)):
        out[idx[k]:idx[k + 1]] = lvl[k]
    return out


def flatvalley(R):
    """v(s)=1/4 max(0,s-R)^4 and its derivative; V_R is convex with lambda=0."""
    return (lambda s: 0.25 * np.maximum(s - R, 0.0) ** 4,
            lambda s: np.maximum(s - R, 0.0) ** 3)


#  ------------------------------------------------------------------ geometries
class IndexGeom:
    """Index-aligned extrapolation/momentum for particle and quantile states.
    measure_defect: for 1-D monotone (quantile) states, records the L-inf defect
    between the composed map T_t o Y_t and the direct monotone map z_t -> x_{t+1}
    (numerical form of the real-line corollary)."""

    def __init__(self, project=None, measure_defect=False):
        self.project = project
        self.measure_defect = measure_defect

    def interpolate(self, x, z, alpha):
        return (1 - alpha) * x + alpha * z

    def momentum(self, x_new, x, z, alpha):
        z_new = (x_new - (1 - alpha) * x) / alpha
        mono = 0.0
        if self.project is not None:
            zp = self.project(z_new)
            mono = float(np.max(np.abs(zp - z_new)) > 1e-12)
            z_new = zp
        defect = np.nan
        if self.measure_defect:
            Qy = (1 - alpha) * x + alpha * z                 # y_t at the z_t nodes
            pmid = 0.5 * (z[:-1] + z[1:])                    # off-node test points
            comp = np.interp(np.interp(pmid, z, Qy), Qy, x_new)   # T_t o Y_t
            direct = np.interp(pmid, z, x_new)                    # monotone map z_t -> x_{t+1}
            defect = float(np.max(np.abs(comp - direct)) / max(np.abs(x_new).max(), 1e-12))
        return z_new, defect, {"mono_violation": mono}


class GaussGeom:
    """Map-based extrapolation/momentum for centered Gaussian covariance states."""

    def interpolate(self, Sx, Sz, alpha):
        self.Xt = bw_map(Sz, Sx)
        self.Yt = (1 - alpha) * self.Xt + alpha * np.eye(Sx.shape[0])
        self.Sy = _sym(self.Yt @ Sz @ self.Yt.T)
        return self.Sy

    def momentum(self, Sx_new, Sx, Sz, alpha):
        Tt = bw_map(self.Sy, Sx_new)
        C = Tt @ self.Yt
        A = bw_map(Sz, Sx_new)
        w2 = bw_distance(Sz, Sx_new)
        defect = map_defect(C, A, Sz) / max(w2, 1e-12)
        Z = (C - (1 - alpha) * self.Xt) / alpha
        lam = float(np.linalg.eigvalsh(_sym(Z))[0])
        return _sym(Z @ Sz @ Z.T), defect, {"lambda_min_symZ": lam}


#  ------------------------------------------------------------------ schemes
def run(mode, prox, x0, gamma, n_blocks, gap, *, geom=None, w2_ref=None,
        restart=None, safeguard=False, tol_stop=None):
    """One JKO trajectory. mode in {'std','acc'}. Returns a per-block record list.
    Stops early once the objective gap drops below tol_stop, if given."""
    geom = geom or IndexGeom()
    x = z = x0
    recs = []
    k = 0
    prev = gap(x0)
    for t in range(n_blocks):
        ti = 0.0
        if mode == "acc":
            alpha = 3.0 / (k + 3.0)
            t0 = time.perf_counter()
            y = geom.interpolate(x, z, alpha)
            ti = time.perf_counter() - t0
        else:
            alpha, y = 1.0, x
        t1 = time.perf_counter()
        x_new = prox(y)
        t_prox = time.perf_counter() - t1
        extra, defect, t_mom = {}, np.nan, 0.0
        if mode == "acc":
            t2 = time.perf_counter()
            z_new, defect, extra = geom.momentum(x_new, x, z, alpha)
            t_mom = time.perf_counter() - t2 + ti
        else:
            z_new = x_new
        g = gap(x_new)
        if mode == "acc" and restart == "function" and g > prev:
            k, z_new, extra["restart"] = 0, x_new, 1
        elif mode == "acc" and restart == "gradient" and float(np.sum((y - x_new) * (x_new - x))) > 0:
            k, z_new, extra["restart"] = 0, x_new, 1
        elif mode == "acc":
            k += 1
        if mode == "acc" and safeguard:
            xp = prox(x)
            gp = gap(xp)
            if gp < g:
                x_new, z_new, g, k, extra["fallback"] = xp, xp, gp, 0, 1
        rec = {"block": t + 1, "gap": g, "map_defect": defect,
               "t_prox": t_prox, "t_momentum": t_mom}
        if w2_ref is not None:
            rec["w2sq_z_rho"] = w2_ref(z_new)
        rec.update(extra)
        recs.append(rec)
        x, z, prev = x_new, z_new, g
        if tol_stop is not None and g < tol_stop:
            break
    return recs


def blocks_to(gaps, tol):
    w = np.where(np.asarray(gaps) < tol)[0]
    return int(w[0]) + 1 if len(w) else -1


#  ------------------------------------------------------------------ statistics
def paired_summary(diff, n_boot=10000):
    """Median paired difference (std - acc) with bootstrap 95% CI, IQR, Wilcoxon."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    nan = float("nan")
    if n == 0:
        return {"median": nan, "ci_lo": nan, "ci_hi": nan, "iqr_lo": nan,
                "iqr_hi": nan, "wilcoxon_p": nan, "k_positive": 0, "n": 0}
    rng = np.random.default_rng(0)
    meds = np.median(diff[rng.integers(0, n, size=(n_boot, n))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    q1, q3 = np.percentile(diff, [25, 75])
    try:
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {"median": float(np.median(diff)), "ci_lo": float(lo), "ci_hi": float(hi),
            "iqr_lo": float(q1), "iqr_hi": float(q3), "wilcoxon_p": p,
            "k_positive": int(np.sum(diff > 0)), "n": n}


def bootstrap_exponent(xvals, yvals, n_boot=10000):
    """Fit y ~ x^a in log-log with a bootstrap CI on the exponent a."""
    lx, ly = np.log(np.asarray(xvals, float)), np.log(np.asarray(yvals, float))
    n = len(lx)
    rng = np.random.default_rng(0)
    a = float(np.polyfit(lx, ly, 1)[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    sl = np.array([np.polyfit(lx[i], ly[i], 1)[0] for i in idx])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    return {"exponent": a, "ci_lo": float(lo), "ci_hi": float(hi)}


#  ------------------------------------------------------------------ io / plotting
def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_run(name, config, per_seed_rows, summary, outdir="results"):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    json.dump({**config, "git_sha": git_sha()}, open(os.path.join(d, "config.json"), "w"), indent=2)
    if per_seed_rows:
        keys = list(dict.fromkeys(k for r in per_seed_rows for k in r))
        with open(os.path.join(d, "per_seed.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(per_seed_rows)
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2)


def _apply_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42})


def figure(ncols=1, nrows=1, full=False):
    _apply_style()
    w = 6.9 if full else 3.3
    return plt.subplots(nrows, ncols, figsize=(w, 2.5 * nrows))


def median_iqr(ax, x, ys, color, label=None, ls="-"):
    ys = np.asarray(ys, float)
    med = np.median(ys, axis=0)
    lo, hi = np.percentile(ys, 25, axis=0), np.percentile(ys, 75, axis=0)
    ax.plot(x, med, ls, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def finish(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


#  ------------------------------------------------------------------ correctness checks
def _checks():
    rng = np.random.default_rng(0)
    d = 4

    # prox_radial against a per-particle scipy solve on a random convex radial v
    a = 1.7
    vp = lambda s: a * s**3
    x = rng.standard_normal((50, d)) * 2
    T = prox_radial(x, vp, 0.3)
    for i in range(50):
        r = np.linalg.norm(x[i])
        s = minimize(lambda ss: 0.25 * a * ss[0] ** 4 + (ss[0] - r) ** 2 / (2 * 0.3),
                     [r], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14}).x[0]
        assert abs(np.linalg.norm(T[i]) - s) < 1e-6, "prox_radial"
    print("ok  prox_radial vs scipy")

    # codiagonal Gaussian prox against the general L-BFGS path on a diagonal instance
    Sig = np.diag(rng.uniform(0.5, 4, d))
    S0 = np.diag(rng.uniform(0.5, 4, d))
    Sc = prox_gaussian(S0, Sig, 0.5, codiagonal=True)
    Sg = prox_gaussian(S0, Sig, 0.5, codiagonal=False)
    assert np.max(np.abs(Sc - Sg)) < 1e-6, "prox_gaussian codiagonal vs general"
    print("ok  prox_gaussian codiagonal == general (diagonal instance)")

    # map defect: 0 when the base is codiagonal with the target (commuting maps),
    # strictly positive after a rotation. interpolate/momentum are called consistently
    # (same iterate Sx and base Sz), exactly as run() drives them.
    Sigma = np.diag([1.0, 4.0, 1.0, 1.0])
    geom = GaussGeom()

    def defect_of(Sz):
        Sx = prox_gaussian(Sz, Sigma, 0.5)
        Sy = geom.interpolate(Sx, Sz, 0.6)
        Sx_new = prox_gaussian(Sy, Sigma, 0.5)
        return geom.momentum(Sx_new, Sx, Sz, 0.6)[1]

    Sz = np.diag([2.0, 0.5, 1.5, 1.0])
    defect0 = defect_of(Sz)
    th = 0.6
    Rm = np.eye(4)
    Rm[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    defect_r = defect_of(_sym(Rm @ Sz @ Rm.T))
    assert defect0 < 1e-4 < defect_r, f"map defect theta gate ({defect0:.1e}, {defect_r:.1e})"
    print(f"ok  map defect: theta=0 -> {defect0:.1e}, theta>0 -> {defect_r:.3f}")

    # accelerated with alpha=1 reduces to a standard step (interpolate/momentum identities)
    ig = IndexGeom()
    xx = rng.standard_normal((10, 3))
    assert np.allclose(ig.interpolate(xx, xx, 1.0), xx)
    zn, _, _ = ig.momentum(xx * 1.3, xx, xx, 1.0)
    assert np.allclose(zn, xx * 1.3), "index momentum alpha=1"
    assert np.allclose(GaussGeom().interpolate(Sz, Sz, 1.0), Sz), "gauss interpolate alpha=1"
    print("ok  alpha=1 reduces accelerated step to standard")

    # bures distance sanity
    assert bw_distance(Sz, Sz) < 1e-6 and bw_distance(Sz, Sigma) > 0
    print("ok  bures distance")
    print("all checks passed")


if __name__ == "__main__":
    _checks()
