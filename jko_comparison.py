import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

#  Colour palette  (consistent across all figures)
BLUE  = "#1f77b4"   # standard JKO
RED   = "#d62728"   # accelerated JKO
GREY  = "#888888"   # reference lines

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
    N_WEAK   = 300   # enough iterations to show clear slope difference
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

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: weakly convex, log-log
    ax = axes[0]
    iters_w = np.arange(1, N_WEAK + 1)   # skip t=0 for log-log
    ax.loglog(iters_w, G_std_w[1:],  color=BLUE, lw=2,   label="Standard JKO")
    ax.loglog(iters_w, G_acc_w[1:],  color=RED,  lw=2,   label="Accelerated JKO")
    ax.loglog(iters_w, b_std_w[1:], "--", color=BLUE, lw=1.2, alpha=0.5,
              label=r"Std bound $\propto t^{-1}$")
    ax.loglog(iters_w, b_acc_w[1:], "--", color=RED,  lw=1.2, alpha=0.5,
              label=r"Acc bound $\propto t^{-2}$")
    ax.set_xlabel("Block $t$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$", fontsize=11)
    ax.set_title(r"(a) $\lambda = 0.04$", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # Panel B: strongly convex, log-linear
    ax = axes[1]
    iters_s = np.arange(N_STRONG + 1)
    ax.semilogy(iters_s, G_std_s,  color=BLUE, lw=2,   label="Standard JKO")
    ax.semilogy(iters_s, G_acc_s,  color=RED,  lw=2,   label="Accelerated JKO")
    ax.semilogy(iters_s[1:], b_std_s[1:], "--", color=BLUE, lw=1.2, alpha=0.5,
                label=r"Std bound $\propto e^{-\gamma\lambda n/2}$")
    ax.semilogy(iters_s[1:], b_acc_s[1:], "--", color=RED,  lw=1.2, alpha=0.5,
                label=r"Acc bound $\propto t^{-2}$")
    ax.set_xlabel("Block $t$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$", fontsize=11)
    ax.set_title(r"(b) $\lambda = 1$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.35)

    plt.tight_layout()
    fig.savefig("images/figure_1.png", dpi=160, bbox_inches="tight")


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
    floor_ratio  = G_std_floors / G_acc_floors

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: depth sweep
    ax1.loglog(N_arr, G_std_final, "o-", color=BLUE, lw=2, label="Standard JKO")
    ax1.loglog(N_arr, G_acc_final, "s-", color=RED,  lw=2, label="Accelerated JKO")
    ref = N_arr / N_arr[0]
    ax1.loglog(N_arr, G_std_final[0] / ref,    "--", color=BLUE, lw=1.2, alpha=0.5,
               label=r"$O(1/N)$")
    ax1.loglog(N_arr, G_acc_final[0] / ref**2, "--", color=RED,  lw=1.2, alpha=0.5,
               label=r"$O(1/N^2)$")
    ax1.set_xlabel("Number of blocks $N$", fontsize=11)
    ax1.set_ylabel(r"$\mathrm{KL}(\rho_N \| q)$", fontsize=11)
    ax1.set_title(r"(a) $\lambda = 0.04$, varying $N$", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", ls=":", alpha=0.35)

    # Panel B: error-floor scaling with lambda
    ax2.loglog(lam_vals, G_std_floors, "o-", color=BLUE, lw=2, label="Standard JKO")
    ax2.loglog(lam_vals, G_acc_floors, "s-", color=RED,  lw=2, label="Accelerated JKO")
    c_ref = G_std_floors[0] * lam_vals[0]**2
    # ax2.loglog(lam_vals, c_ref / lam_vals**2, "--", color=BLUE, lw=1.2, alpha=0.5,
    #            label=r"$O(\lambda^{-2})$")
    ax2.set_xlabel(r"$\lambda$", fontsize=12)
    ax2.set_ylabel(r"$\mathrm{KL}(\rho_N \| q)$", fontsize=11)
    ax2.set_title(r"(b) $N = 200$, varying $\lambda$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.35)
    ax2.invert_xaxis()

    plt.tight_layout()
    fig2.savefig("images/figure_2.png", dpi=160, bbox_inches="tight")
