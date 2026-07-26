import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats

import experiment_io as eio
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("Standard JKO")
RED  = color_for("Accelerated JKO")

os.makedirs("images", exist_ok=True)

#  Target: symmetric Gaussian mixture
MU_Q  = 2.0
SIG_Q = 0.7

def log_q(x):
    lp1 = stats.norm.logpdf(x,  MU_Q, SIG_Q)
    lp2 = stats.norm.logpdf(x, -MU_Q, SIG_Q)
    return np.log(0.5) + np.logaddexp(lp1, lp2)

#  Variational family: rho(m,s) = 0.5*N(m,s^2) + 0.5*N(-m,s^2)
def log_rho(x, m, s):
    lp1 = stats.norm.logpdf(x,  m, s)
    lp2 = stats.norm.logpdf(x, -m, s)
    return np.log(0.5) + np.logaddexp(lp1, lp2)


def kl_rho_q(m, s):
    """KL( rho(m,s) || q ) via numerical integration."""
    if s <= 0 or m < 0:
        return np.inf
    center = max(m, MU_Q)

    def integrand(x):
        lr = log_rho(x, m, s)
        lq = log_q(x)
        return np.exp(lr) * (lr - lq)

    val, _ = integrate.quad(integrand, -center - 8*s, center + 8*s,
                             limit=300, points=[-m, m, -MU_Q, MU_Q])
    return max(val, 0.0)


def w2_sq(m1, s1, m2, s2):
    """W_2^2(rho(m1,s1), rho(m2,s2)) = (m1-m2)^2 + (s1-s2)^2."""
    return (m1 - m2)**2 + (s1 - s2)**2


#  Numerical proximal step
def jko_prox(m_prev, s_prev, gamma):
    """
    argmin_{m>=0, s>0}  KL(rho(m,s)||q)  +  W_2^2(rho(m,s), rho(m_prev,s_prev))/(2*gamma)
    Parametrised as (m, log_s) to enforce positivity constraints.
    """
    def objective(params):
        m, log_s = params
        s = np.exp(log_s)
        return kl_rho_q(m, s) + w2_sq(m, s, m_prev, s_prev) / (2.0 * gamma)

    def grad(params, eps=1e-5):
        g = np.zeros(2)
        for i in range(2):
            p1, p2 = params.copy(), params.copy()
            p1[i] += eps; p2[i] -= eps
            g[i] = (objective(p1) - objective(p2)) / (2 * eps)
        return g

    x0  = np.array([max(m_prev, 1e-3), np.log(max(s_prev, 1e-6))])
    res = optimize.minimize(objective, x0, jac=grad, method="L-BFGS-B",
                            bounds=[(1e-4, None), (None, None)],
                            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-8})
    return res.x[0], np.exp(res.x[1])


#  Standard JKO
def run_standard_jko(m0, s0, gamma, n_steps):
    m, s   = float(m0), float(s0)
    G_vals = [kl_rho_q(m, s)]
    for k in range(n_steps):
        m, s = jko_prox(m, s, gamma)
        G_vals.append(kl_rho_q(m, s))
    return np.array(G_vals)


#  Accelerated JKO
def run_accelerated_jko(m0, s0, gamma, n_steps):
    mx, sx = float(m0), float(s0)
    mz, sz = float(m0), float(s0)
    G_vals = [kl_rho_q(mx, sx)]

    for t in range(n_steps):
        alpha = 3.0 / (t + 3.0)

        my = (1.0 - alpha) * mx + alpha * mz
        sy = max((1.0 - alpha) * sx + alpha * sz, 1e-6)

        mx_new, sx_new = jko_prox(my, sy, gamma)

        mz_new = (mx_new - (1.0 - alpha) * mx) / alpha
        sz_new = max((sx_new - (1.0 - alpha) * sx) / alpha, 1e-6)

        mx, sx = mx_new, sx_new
        mz, sz = mz_new, sz_new
        G_vals.append(kl_rho_q(mx, sx))

    return np.array(G_vals)


if __name__ == "__main__":
    #  Run
    m0, s0  = 0.5, 2.0
    GAMMA   = 0.5
    N_STEPS = 40

    G_std = run_standard_jko(m0, s0, GAMMA, N_STEPS)
    G_acc = run_accelerated_jko(m0, s0, GAMMA, N_STEPS)
    acc_runmin = np.minimum.accumulate(G_acc)

    #  Plot  (no O(t^-2) reference: this target is not geodesically convex, lambda < 0)
    iters = np.arange(N_STEPS + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.semilogy(iters, G_std, color=BLUE, label="Standard JKO")
    ax.semilogy(iters, G_acc, color=RED,  label="Accelerated JKO")
    ax.semilogy(iters, acc_runmin, ":", color=RED, lw=1.2, label="Accelerated running min")
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$")
    ax.set_title(r"double-well ($\lambda < 0$),  init $\mathcal{N}(%.1f, %.1f^2)$" % (m0, s0))
    ax.legend(loc="upper right")
    ax.grid(True, which="both")

    plt.tight_layout()
    fig.savefig("images/figure_3.pdf", bbox_inches="tight")

    eio.save_config("e2_fig4_doublewell",
                    {"init_mean": m0, "init_std": s0, "gamma": GAMMA, "n_steps": N_STEPS,
                     "target": "0.5 N(2,0.7^2)+0.5 N(-2,0.7^2)", "lam_class": "negative"})
    eio.save_metrics("e2_fig4_doublewell",
                     [{"block": t, "kl_std": G_std[t], "kl_acc": G_acc[t],
                       "acc_runmin": acc_runmin[t]} for t in range(N_STEPS + 1)])
    eio.save_summary("e2_fig4_doublewell",
                     {"final_kl_std": float(G_std[-1]), "final_kl_acc": float(G_acc[-1]),
                      "min_kl_acc": float(acc_runmin[-1]), "init": f"N({m0},{s0}^2)"})
    print(f"[e2 fig4] final KL std={G_std[-1]:.3e} acc={G_acc[-1]:.3e} min_acc={acc_runmin[-1]:.3e}")