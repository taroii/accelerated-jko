import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import experiment_io as eio
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("Standard JKO")
RED  = color_for("Accelerated JKO")

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

#  1.  CLOSED-FORM GAUSSIAN UTILITIES
def kl_gaussian(m, s, sigma_q=1.0):
    """KL( N(m,s^2) || N(0,sigma_q^2) )."""
    r = s / sigma_q
    return 0.5 * (m**2 / sigma_q**2 + r**2 - 1.0 - 2.0 * np.log(r))


def jko_prox_kl(m, s, gamma, lam):
    """Closed-form JKO proximal step for KL( . || N(0,1/lam) )."""
    m_new = m / (1.0 + lam * gamma)
    A     = lam * gamma + 1.0
    s_new = (s + np.sqrt(s**2 + 4.0 * gamma * A)) / (2.0 * A)
    return m_new, s_new


def w2_sq_1d(m1, s1, m2=0.0, s2=1.0):
    return (m1 - m2)**2 + (s1 - s2)**2


#  2.  ALGORITHM RUNNERS
def run_standard_jko(m0, s0, gamma, lam, n_steps):
    sigma_q = 1.0 / np.sqrt(lam)
    m, s    = float(m0), float(s0)
    G_vals  = []
    for k in range(n_steps + 1):
        G_vals.append(kl_gaussian(m, s, sigma_q))
        if k < n_steps:
            m, s = jko_prox_kl(m, s, gamma, lam)
    return np.array(G_vals)


def run_accelerated_jko(m0, s0, gamma, lam, n_steps):
    sigma_q     = 1.0 / np.sqrt(lam)
    mx, sx      = float(m0), float(s0)
    mz, sz      = float(m0), float(s0)
    G_vals      = [kl_gaussian(mx, sx, sigma_q)]
    for t in range(n_steps):
        alpha   = 3.0 / (t + 3.0)
        my      = (1.0 - alpha) * mx + alpha * mz
        sy      = max((1.0 - alpha) * sx + alpha * sz, 1e-12)
        mx_new, sx_new = jko_prox_kl(my, sy, gamma, lam)
        mz_new  = (mx_new - (1.0 - alpha) * mx) / alpha
        sz_new  = max((sx_new - (1.0 - alpha) * sx) / alpha, 1e-12)
        mx, sx  = mx_new, sx_new
        mz, sz  = mz_new, sz_new
        G_vals.append(kl_gaussian(mx, sx, sigma_q))
    return np.array(G_vals)


#  3.  THEORETICAL BOUNDS
def bound_std(G0, W2sq_0, gamma, lam, n_steps):
    n   = np.arange(n_steps + 1)
    rho = 1.0 / (1.0 + gamma * lam / 2.0)
    return np.where(n == 0, G0, W2sq_0 / (2.0 * gamma) * rho**(n - 1))


def bound_acc(G0, W2sq_0, gamma, n_steps):
    t      = np.arange(n_steps + 1)
    Delta0 = W2sq_0 / (2.0 * gamma) + G0
    return np.where(t == 0, G0, 9.0 * Delta0 / (t + 2)**2)


if __name__ == "__main__":
    #  4.  SHARED EXPERIMENT PARAMETERS
    m0, s0  = 5.0, 2.5
    GAMMA   = 0.5

    #  FIGURE 1
    N_WEAK   = 2000  # long horizon to expose the late std/acc crossover
    N_STRONG = 49    # exponential needs fewer steps to converge

    lam_weak   = 0.04
    lam_strong = 1.0

    sigma_weak   = 1.0 / np.sqrt(lam_weak)
    sigma_strong = 1.0 / np.sqrt(lam_strong)

    G_std_w  = run_standard_jko(m0, s0, GAMMA, lam_weak,   N_WEAK)
    G_acc_w  = run_accelerated_jko(m0, s0, GAMMA, lam_weak,   N_WEAK)
    G_std_s  = run_standard_jko(m0, s0, GAMMA, lam_strong, N_STRONG)
    G_acc_s  = run_accelerated_jko(m0, s0, GAMMA, lam_strong, N_STRONG)

    G0_w    = kl_gaussian(m0, s0, sigma_weak)
    W2sq_w  = w2_sq_1d(m0, s0, 0.0, sigma_weak)
    G0_s    = kl_gaussian(m0, s0, sigma_strong)
    W2sq_s  = w2_sq_1d(m0, s0, 0.0, sigma_strong)

    b_std_w = bound_std(G0_w, W2sq_w, GAMMA, lam_weak,   N_WEAK)
    b_acc_w = bound_acc(G0_w, W2sq_w, GAMMA, N_WEAK)
    b_std_s = bound_std(G0_s, W2sq_s, GAMMA, lam_strong, N_STRONG)
    b_acc_s = bound_acc(G0_s, W2sq_s, GAMMA, N_STRONG)

    # Depth sweep for Panel C
    N_list      = [4, 8, 16, 32, 64, 128]
    N_arr       = np.array(N_list, dtype=float)
    G_std_final = []
    G_acc_final = []
    for N in N_list:
        G_std_final.append(run_standard_jko(m0, s0, GAMMA, lam_weak, N)[-1])
        G_acc_final.append(run_accelerated_jko(m0, s0, GAMMA, lam_weak, N)[-1])
    G_std_final = np.array(G_std_final)
    G_acc_final = np.array(G_acc_final)
    depth_ratio = G_std_final / G_acc_final

    # running minimum of the accelerated gap (what the Lyapunov functional controls)
    acc_runmin_w = np.minimum.accumulate(G_acc_w)
    # crossover: standard's exponential phase eventually overtakes accelerated
    post = np.arange(len(G_std_w))
    cross = np.where((post > 20) & (G_std_w < G_acc_w))[0]
    crossover_iter = int(cross[0]) if len(cross) else -1

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: weakly convex, log-log
    ax = axes[0]
    iters_w = np.arange(1, N_WEAK + 1)   # skip t=0 for log-log
    clip = lambda a: np.maximum(a, 1e-16)
    ax.loglog(iters_w, clip(G_std_w[1:]), color=BLUE, label="Standard JKO")
    ax.loglog(iters_w, clip(G_acc_w[1:]), color=RED,  label="Accelerated JKO")
    ax.loglog(iters_w, clip(acc_runmin_w[1:]), ":", color=RED, lw=1.2,
              label="Accelerated running min")
    ax.loglog(iters_w, b_std_w[1:], "--", color=BLUE, lw=1.2, alpha=0.5,
              label="finite-horizon reference (pre-asymptotic)")
    if crossover_iter > 0:
        ax.axvline(crossover_iter, color="gray", ls=":", lw=1.0)
        ax.annotate(f"crossover $t={crossover_iter}$", (crossover_iter, ax.get_ylim()[0]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7, rotation=90,
                    va="bottom")
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$")
    ax.set_title(r"(a) $\lambda = 0.04$,  init $\mathcal{N}(5, 2.5^2)$")
    ax.legend(loc="lower left")
    ax.grid(True, which="both")

    # Panel B: strongly convex, log-linear
    ax = axes[1]
    iters_s = np.arange(N_STRONG + 1)
    clip = lambda a: np.maximum(a, 1e-16)
    cross_s = np.where((np.arange(len(G_std_s)) > 5) & (G_std_s < G_acc_s))[0]
    crossover_iter_strong = int(cross_s[0]) if len(cross_s) else -1
    ax.semilogy(iters_s, clip(G_std_s), color=BLUE, label="Standard JKO")
    ax.semilogy(iters_s, clip(G_acc_s), color=RED,  label="Accelerated JKO")
    ax.semilogy(iters_s, clip(np.minimum.accumulate(G_acc_s)), ":", color=RED, lw=1.2,
                label="Accelerated running min")
    ax.semilogy(iters_s[1:], b_std_s[1:], "--", color=BLUE, lw=1.2, alpha=0.5,
                label=r"Std bound $\propto e^{-\gamma\lambda n/2}$")
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$")
    ax.set_title(r"(b) $\lambda = 1$,  init $\mathcal{N}(5, 2.5^2)$")
    ax.legend(loc="upper right")
    ax.grid(True, which="both")

    plt.tight_layout()
    fig.savefig("images/figure_1.pdf", bbox_inches="tight")

    eio.save_config("e2_fig2_gaussian",
                    {"init_mean": m0, "init_std": s0, "gamma": GAMMA,
                     "lam_weak": lam_weak, "lam_strong": lam_strong,
                     "N_weak": N_WEAK, "N_strong": N_STRONG})
    eio.save_metrics("e2_fig2_gaussian",
                     [{"block": t, "kl_std_lam0.04": G_std_w[t], "kl_acc_lam0.04": G_acc_w[t],
                       "acc_runmin_lam0.04": acc_runmin_w[t]} for t in range(N_WEAK + 1)])
    eio.save_summary("e2_fig2_gaussian",
                     {"crossover_iter_lambda0.04": crossover_iter,
                      "crossover_iter_lambda1": crossover_iter_strong,
                      "final_kl_std_lam0.04": float(G_std_w[-1]),
                      "final_kl_acc_lam0.04": float(G_acc_w[-1]),
                      "init": f"N({m0},{s0}^2)"})
    print(f"[e2 fig2] crossover_iter lambda=0.04 -> {crossover_iter}, "
          f"lambda=1 -> {crossover_iter_strong}")


    #  FIGURE 2
    # Panel A: depth sweep  (log-log, final KL vs N)
    # Panel B: error-floor scaling with lambda  (log-log, final KL vs lambda)
    N_FLOOR   = 200
    lam_vals  = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])

    G_std_floors = []
    G_acc_floors = []
    for lv in lam_vals:
        G_std_floors.append(run_standard_jko(m0, s0, GAMMA, lv, N_FLOOR)[-1])
        G_acc_floors.append(run_accelerated_jko(m0, s0, GAMMA, lv, N_FLOOR)[-1])

    G_std_floors = np.array(G_std_floors)
    G_acc_floors = np.array(G_acc_floors)
    floor_ratio  = G_std_floors / np.maximum(G_acc_floors, 1e-300)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: depth sweep
    ax1.loglog(N_arr, G_std_final, "o-", color=BLUE, label="Standard JKO")
    ax1.loglog(N_arr, G_acc_final, "s-", color=RED,  label="Accelerated JKO")
    ref = N_arr / N_arr[0]
    ax1.loglog(N_arr, G_std_final[0] / ref,    "--", color=BLUE, lw=1.2, alpha=0.5,
               label=r"$O(1/N)$")
    ax1.loglog(N_arr, G_acc_final[0] / ref**2, "--", color=RED,  lw=1.2, alpha=0.5,
               label=r"$O(1/N^2)$")
    ax1.set_xlabel("Number of blocks $N$")
    ax1.set_ylabel(r"$\mathrm{KL}(\rho_N \| q)$")
    ax1.set_title(r"(a) $\lambda = 0.04$, varying $N$")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both")

    # Panel B: error-floor scaling with lambda
    ax2.loglog(lam_vals, G_std_floors, "o-", color=BLUE, label="Standard JKO")
    ax2.loglog(lam_vals, G_acc_floors, "s-", color=RED,  label="Accelerated JKO")
    c_ref = G_std_floors[0] * lam_vals[0]**2
    # ax2.loglog(lam_vals, c_ref / lam_vals**2, "--", color=BLUE, lw=1.2, alpha=0.5,
    #            label=r"$O(\lambda^{-2})$")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$\mathrm{KL}(\rho_N \| q)$")
    ax2.set_title(r"(b) $N = 200$, varying $\lambda$")
    ax2.legend()
    ax2.grid(True, which="both")
    ax2.invert_xaxis()

    plt.tight_layout()
    fig2.savefig("images/figure_2.pdf", bbox_inches="tight")
