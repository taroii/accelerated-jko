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
os.makedirs("images", exist_ok=True)

U = (np.arange(4000) + 0.5) / 4000


def target_logpdf(x, sig2):
    s = np.sqrt(sig2)
    return np.logaddexp(np.log(10 / 11) + norm.logpdf(x, 1.5, s),
                        np.log(1 / 11) + norm.logpdf(x, -1.5, s))


def params(theta):
    a, m1, m2, ls1, ls2 = theta
    w = min(max(1 / (1 + np.exp(-np.clip(a, -30, 30))), 1e-9), 1 - 1e-9)
    return w, m1, m2, np.exp(0.5 * ls1), np.exp(0.5 * ls2)


def rho_logpdf(x, theta):
    w, m1, m2, s1, s2 = params(theta)
    return np.logaddexp(np.log(w) + norm.logpdf(x, m1, s1),
                        np.log(1 - w) + norm.logpdf(x, m2, s2))


BOUNDS = [(-30, 30), (-3.2, 3.2), (-3.2, 3.2),
          (2 * np.log(0.008), 2 * np.log(2.5)), (2 * np.log(0.008), 2 * np.log(2.5))]
LO = np.array([b[0] for b in BOUNDS])
HI = np.array([b[1] for b in BOUNDS])


def theta_init(w, m1, m2, s1sq, s2sq):
    return np.array([np.log(w / (1 - w)), m1, m2, np.log(s1sq), np.log(s2sq)])


#  Adaptive grid: dense clusters around every model + target component,
#  so KL/W2 stay accurate however sharp the densities become.
def integ_grid(theta, sig2):
    w, m1, m2, s1, s2 = params(theta)
    st = np.sqrt(sig2)
    parts = [np.linspace(-3.6, 3.6, 1500)]
    for c, s in [(m1, s1), (m2, s2), (1.5, st), (-1.5, st)]:
        parts.append(np.linspace(c - 8 * s, c + 8 * s, 1200))
    return np.unique(np.clip(np.concatenate(parts), -3.7, 3.7))


def kl_to_target(theta, sig2):
    g = integ_grid(theta, sig2)
    lr = rho_logpdf(g, theta)
    val = np.trapezoid(np.exp(lr) * (lr - target_logpdf(g, sig2)), g)
    return float(max(val, 0.0)) if np.isfinite(val) else 1e6


def quantile(theta, sig2):
    g = integ_grid(theta, sig2)
    r = np.exp(rho_logpdf(g, theta))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(U, cdf, g)


def w2sq(theta_a, theta_b, sig2):
    return float(((quantile(theta_a, sig2) - quantile(theta_b, sig2)) ** 2).mean())


def mc_kl(theta, sig2, n=300000):
    rng = np.random.default_rng(0)
    w, m1, m2, s1, s2 = params(theta)
    k = rng.random(n) < w
    x = np.where(k, rng.normal(m1, s1, n), rng.normal(m2, s2, n))
    return float(np.mean(rho_logpdf(x, theta) - target_logpdf(x, sig2)))


def fd_grad(f, theta, eps=1e-5):
    g = np.zeros(5)
    for i in range(5):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g[i] = (f(tp) - f(tm)) / (2 * eps)
    return g


def prox(theta_prev, y, sig2, gamma):
    obj = lambda th: kl_to_target(th, sig2) + w2sq(th, y, sig2) / (2 * gamma)
    res = minimize(obj, np.clip(theta_prev, LO, HI), jac=lambda th: fd_grad(obj, th),
                   method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 500, "gtol": 1e-8})
    return res.x


def run_std(theta0, sig2, gamma, N):
    th = theta0.copy()
    traj, kl = [th.copy()], [kl_to_target(th, sig2)]
    for _ in range(N):
        th = prox(th, th, sig2, gamma)
        traj.append(th.copy())
        kl.append(kl_to_target(th, sig2))
    return np.array(traj), np.array(kl)


def run_acc(theta0, sig2, gamma, N):
    x = theta0.copy()
    z = theta0.copy()
    traj, kl = [x.copy()], [kl_to_target(x, sig2)]
    for t in range(N):
        a = 3.0 / (t + 3.0)
        y = (1 - a) * x + a * z
        xn = prox(x, y, sig2, gamma)
        z = (xn - (1 - a) * x) / a
        x = xn
        traj.append(x.copy())
        kl.append(kl_to_target(x, sig2))
    return np.array(traj), np.array(kl)


def run_gd(theta0, sig2, eta, N):
    th = theta0.copy()
    traj, kl = [th.copy()], [kl_to_target(th, sig2)]
    for _ in range(N):
        th = np.clip(th - eta * fd_grad(lambda t: kl_to_target(t, sig2), th), LO, HI)
        traj.append(th.copy())
        kl.append(kl_to_target(th, sig2))
    return np.array(traj), np.array(kl)


def iters_to_separation(traj, thresh=1.5):
    for t, th in enumerate(traj):
        if abs(th[1] - th[2]) > thresh:
            return t
    return -1


def run_variance(sig2, cfg, tag):
    gamma, N = cfg["gamma"], cfg["n_blocks"]
    inits = {1: theta_init(0.5, 0.0, 0.0, 1.0, 1.0),
             2: theta_init(0.5, 0.001, -0.001, 1.0, 1.0),
             3: theta_init(0.5, 0.001, -0.001, 0.012, 0.012)}
    results = {}
    numbers = {}
    for i, th0 in inits.items():
        gd_best = None
        for eta in cfg["gd_etas"]:
            tr, kl = run_gd(th0, sig2, eta, N)
            fk = mc_kl(tr[-1], sig2)   # MC-KL: reliable even at pathological states
            if gd_best is None or fk < gd_best[3]:
                gd_best = (tr, kl, eta, fk)
        tr_std, kl_std = run_std(th0, sig2, gamma, N)
        tr_acc, kl_acc = run_acc(th0, sig2, gamma, N)
        tr_gd, kl_gd, eta_best, fk_gd = gd_best
        results[i] = {"std": (tr_std, kl_std), "acc": (tr_acc, kl_acc),
                      "gd": (tr_gd, kl_gd, eta_best)}
        fk = {"std": mc_kl(tr_std[-1], sig2), "acc": mc_kl(tr_acc[-1], sig2), "gd": fk_gd}
        for name, tr in [("std", tr_std), ("acc", tr_acc), ("gd", tr_gd)]:
            numbers[f"final_KL_{name}_init{i}"] = fk[name]
            numbers[f"iters_to_mode_separation_{name}_init{i}"] = iters_to_separation(tr)
        numbers[f"gd_best_eta_init{i}"] = float(eta_best)
        print(f"[{tag} init{i}] final KL(MC) std={fk['std']:.3e} acc={fk['acc']:.3e} "
              f"gd={fk['gd']:.3e}(η={eta_best}) sep std/acc/gd="
              f"{iters_to_separation(tr_std)}/{iters_to_separation(tr_acc)}/{iters_to_separation(tr_gd)}")
    exp_id = f"e3_mixture_{tag}"
    rows = []
    for i in results:
        for name in ("std", "acc", "gd"):
            kl = results[i][name][1]
            for t, v in enumerate(kl):
                rows.append({"init": i, "method": name, "block": t, "kl": float(v)})
    eio.save_config(exp_id, {"sig2": sig2, "gamma": gamma, "n_blocks": N,
                             "gd_etas": cfg["gd_etas"], "weights": [10 / 11, 1 / 11],
                             "centers": [1.5, -1.5]})
    eio.save_metrics(exp_id, rows)
    eio.save_summary(exp_id, numbers)
    plot_evolution(results, sig2, N, f"images/e3_mixture_{tag}.pdf")
    return numbers


def plot_evolution(results, sig2, N, savepath):
    snap = sorted({0, N // 4, N // 2, 3 * N // 4, N})
    methods = [("Standard JKO", "std", BLUE), ("Accelerated JKO", "acc", RED),
               ("Gradient descent", "gd", GREEN)]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    xg = np.linspace(-3, 3, 4000)
    q = np.exp(target_logpdf(xg, sig2))
    grad = plt.colormaps["viridis"](np.linspace(0, 0.9, len(snap)))
    for r, i in enumerate([1, 2, 3]):
        for c, (label, key, color) in enumerate(methods):
            ax = axes[r, c]
            ax.fill_between(xg, q, color="orange", alpha=0.18, zorder=0)
            tr = results[i][key][0]
            for si, cc in zip(snap, grad):
                ax.plot(xg, np.exp(rho_logpdf(xg, tr[si])), color=cc, lw=1.3, label=f"t={si}")
            ax.set_xlim(-3, 3)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"init {i}")
            if r == 2:
                ax.set_xlabel("x")
            if r == 0 and c == 2:
                ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(rf"$\rho_\theta(t)$ vs target (asymmetric 10:1 mixture, $\sigma^2={sig2}$)")
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    np.random.seed(cfg["seed"])
    allnum = {}
    allnum["primary_sig2_0.012"] = run_variance(0.012, cfg, "sig2_0p012")
    allnum["appendix_sig2_1e-4"] = run_variance(1e-4, cfg, "sig2_1e-4")
    eio.save_summary("e3_mixture", allnum)


if __name__ == "__main__":
    main()
