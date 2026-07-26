import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.linalg import solve_banded
from scipy.optimize import minimize

import experiment_io as eio
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("SGA")
RED = color_for("ASGA")
GREEN = "#009E73"


class Barycenter:
    def __init__(self, n=60, m=4, eps=0.008, sigma=0.07, seed=0):
        self.n, self.m, self.eps = n, m, eps
        self.x = np.linspace(0, 1, n)
        self.h = self.x[1] - self.x[0]
        centres = np.linspace(0.15, 0.85, m)
        self.MU = [self._bump(c, sigma) for c in centres]
        self.logMU = [np.log(mu + 1e-20) for mu in self.MU]
        self.w = np.full(m, 1.0 / m)
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

    def A(self, v):                      # Hesudo-Laplacian: grad of 0.5||v||^2_H1
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


def _slope(gap, lo_frac=0.25):
    N = len(gap)
    lo = max(2, int(lo_frac * N))
    t = np.arange(lo, N)
    y = gap[lo:]
    mk = y > 1e-12
    return float(np.polyfit(np.log(t[mk]), np.log(y[mk]), 1)[0]) if mk.sum() > 2 else float("nan")


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


def e8_1():
    prob = Barycenter(n=30, m=2, eps=0.008)
    L = empirical_L(prob)
    eta = 1.0 / L
    # tight reference: best objective reached by long explicit and long exact-prox runs
    best = prob.I_val(_long_ref(prob, 5000))
    fs = [np.zeros(prob.n) for _ in range(prob.m)]
    zs = [f.copy() for f in fs]
    for t in range(1, 3001):
        a = 3.0 / (t + 3.0)
        ys = [(1 - a) * fs[i] + a * zs[i] for i in range(prob.m)]
        fs, _ = _exact_prox(prob, ys, eta)
        zs = [zs[i] + (1.0 / a) * (fs[i] - ys[i]) for i in range(prob.m)]
        best = max(best, prob.I_val(fs))
    Istar = best + 1e-12
    g_expl = asga(prob, 400, eta, Istar)
    g_exact, inner = asga_exact_prox(prob, 400, eta, Istar)
    s = {"slope_asga_explicit": _slope(g_expl), "slope_asga_exactprox": _slope(g_exact),
         "inner_iters_per_outer_mean": float(inner.mean()),
         "final_gap_explicit": float(g_expl[-1]), "final_gap_exactprox": float(g_exact[-1])}
    eio.save_config("e8_1_exactprox", {"n": 30, "m": 2, "eps": 0.008, "eta": eta, "N": 400})
    eio.save_metrics("e8_1_exactprox",
                     [{"iter": t + 1, "gap_explicit": g_expl[t], "gap_exactprox": g_exact[t],
                       "inner_iters": int(inner[t])} for t in range(400)])
    eio.save_summary("e8_1_exactprox", s)
    it = np.arange(1, 401)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(it, g_expl, color=RED, label=f"ASGA explicit (slope {s['slope_asga_explicit']:.2f})")
    ax.loglog(it, g_exact, color=GREEN, label=f"ASGA exact-prox (slope {s['slope_asga_exactprox']:.2f})")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"$\mathcal{I}^\star - \mathcal{I}(f^{(t)})$")
    ax.set_title("E8.1 exact-prox vs explicit (n=30, m=2)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both")
    plt.tight_layout()
    fig.savefig("images/e8_exactprox.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[e8.1] slope explicit={s['slope_asga_explicit']:.2f} exactprox={s['slope_asga_exactprox']:.2f} "
          f"inner_iters_mean={s['inner_iters_per_outer_mean']:.1f}")
    return s


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


def e8_2():
    prob = Barycenter(n=60, m=4, eps=0.008)
    Le = empirical_L(prob)
    Lc = certified_L(prob)
    s = {"L_empirical": Le, "L_certified": Lc, "L_ratio": Lc / Le}
    eio.save_config("e8_2_lipschitz", {"n": 60, "m": 4, "eps": 0.008,
                                       "note": "L_cert = rho_max/(pi^2 eps), C_P=1/pi^2 Neumann Poincare"})
    eio.save_summary("e8_2_lipschitz", s)
    print(f"[e8.2] L_empirical={Le:.2f} L_certified={Lc:.2f} ratio={Lc/Le:.2f}")
    return s


def e8_3():
    rows, numbers = [], {}
    for eps in [0.002, 0.004, 0.008, 0.016]:
        for n in [60, 120, 240]:
            for m in [2, 4, 8]:
                prob = Barycenter(n=n, m=m, eps=eps)
                L = empirical_L(prob)
                eta = 1.0 / L
                Istar = prob.I_val(_long_ref(prob, 2000))
                s_sga = _slope(sga(prob, 300, eta, Istar))
                s_asga = _slope(asga(prob, 300, eta, Istar))
                tag = f"eps{eps}_n{n}_m{m}"
                numbers[f"slope_sga_{tag}"] = s_sga
                numbers[f"slope_asga_{tag}"] = s_asga
                rows.append({"eps": eps, "n": n, "m": m, "L_empirical": L,
                             "slope_sga": round(s_sga, 3), "slope_asga": round(s_asga, 3)})
                print(f"[e8.3] eps={eps} n={n} m={m}: slope SGA={s_sga:.2f} ASGA={s_asga:.2f}")
    eio.save_metrics("e8_3_sweep", rows)
    eio.save_summary("e8_3_sweep", numbers)
    return numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["e8_1", "e8_2", "e8_3", "all"])
    args = ap.parse_args()
    if args.part in ("e8_1", "all"):
        e8_1()
    if args.part in ("e8_2", "all"):
        e8_2()
    if args.part in ("e8_3", "all"):
        e8_3()


if __name__ == "__main__":
    main()
