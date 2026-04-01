"""
Empirical Convergence Comparison: Standard JKO vs Accelerated JKO
==================================================================
Setting: 1-D Gaussian distributions  N(m, s^2)  with target  q = N(0, 1/sqrtlam).

G(rho) = KL( N(m,s) || N(0, sigma_q^2) )   where  sigma_q = 1/sqrtlam

All JKO proximal steps are computed in closed form via first-order conditions.

Theoretical rates being tested
-------------------------------
Standard JKO  (Cheng et al., 2024 -- Theorem 4.3 + eq. A.18):
  W_2^2(p_n, q) <= (1 + gammalam/2)^{-n} * W_2^2(p_0, q)              [exponential]
  G(p_n)      <= W_2^2(p_0,q)/(2gamma) * (1 + gammalam/2)^{-(n-1)}

Accelerated JKO  (template.tex -- Theorem "accelerated-jko-convergence"):
  G(x_t)  <=  9/(t+2)^2 * ( W_2^2(z_0,q)/(2gamma) + G(x_0) )       [O(t^-^2)]

Key insight:
  * For strongly convex G (lam > 0): standard JKO wins with exponential rate.
  * For weakly / non-strongly convex G (lam -> 0): standard JKO slows to O(1/n),
    while accelerated JKO maintains O(t^-^2) and dramatically outperforms it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ==========================================================
#  1.  GAUSSIAN UTILITIES
# ==========================================================

def kl_gaussian(m: float, s: float, sigma_q: float = 1.0) -> float:
    """KL( N(m,s^2) || N(0,sigma_q^2) ) -- standard-deviation parameterisation."""
    if s <= 0 or sigma_q <= 0:
        return np.inf
    r = s / sigma_q
    return 0.5 * (m**2 / sigma_q**2 + r**2 - 1.0 - 2.0 * np.log(r))


def jko_prox_kl(m: float, s: float, gamma: float, lam: float):
    """
    Closed-form JKO proximal step for  G(rho) = KL( N(m,s) || N(0,sigma_q^2) ),
    sigma_q = 1/sqrtlam.

    Minimises  G(rho) + (1/2gamma) W_2^2(rho, N(m,s))  over Gaussians  N(m*,s*).

    First-order conditions:
      Mean :  m*(lam + 1/gamma) = m/gamma           ->  m* = m / (1 + lamgamma)
      Std  :  (lamgamma+1)s*^2 ? s*s* ? gamma = 0   ->  s* = [s + sqrt(s^2 + 4gamma(lamgamma+1))] / [2(lamgamma+1)]
    """
    m_new = m / (1.0 + lam * gamma)
    A = lam * gamma + 1.0
    s_new = (s + np.sqrt(s**2 + 4.0 * gamma * A)) / (2.0 * A)
    return m_new, s_new


def w2_sq_1d(m1: float, s1: float, m2: float = 0.0, s2: float = 1.0) -> float:
    """W_2^2( N(m1,s1), N(m2,s2) ) in 1-D  =  |m1?m2|^2 + (s1?s2)^2."""
    return (m1 - m2)**2 + (s1 - s2)**2


# ==========================================================
#  2.  STANDARD JKO
# ==========================================================

def run_standard_jko(m0, s0, gamma, lam, n_steps):
    """
    Standard JKO (implicit Euler / backward gradient descent in W_2).
    Returns arrays (G_vals, W2_vals) of length n_steps+1.
    """
    sigma_q = 1.0 / np.sqrt(lam)
    m, s = float(m0), float(s0)
    G_vals, W2_vals = [], []
    for k in range(n_steps + 1):
        G_vals.append(kl_gaussian(m, s, sigma_q))
        W2_vals.append(np.sqrt(w2_sq_1d(m, s, 0.0, sigma_q)))
        if k < n_steps:
            m, s = jko_prox_kl(m, s, gamma, lam)
    return np.array(G_vals), np.array(W2_vals)


# ==========================================================
#  3.  ACCELERATED JKO
# ==========================================================

def run_accelerated_jko(m0, s0, gamma, lam, n_steps):
    """
    Accelerated JKO  (template.tex, Case I, W_2 setting).

    Three-sequence recursion  alpha_t = 3/(t+3)  (so alpha_0 = 1, x_0 = z_0):

      y_t:  m_y = (1?alpha_t)*m_x + alpha_t*m_z
            s_y = (1?alpha_t)*s_x + alpha_t*s_z          <- W_2 geodesic interpolation
      x_{t+1} = JKO-Prox_{gamma}(y_t)
      z-update via momentum pushforward:
            m_{z,t+1} = ( m_{x,t+1} ? (1?alpha_t)*m_{x,t} ) / alpha_t
            s_{z,t+1} = ( s_{x,t+1} ? (1?alpha_t)*s_{x,t} ) / alpha_t

    The z-update formula is derived by computing  Z_{t+1} = I + (1/alpha_t)(X_{t+1} ? Y_t)
    in closed form for 1-D Gaussians, where X_{t+1} is the OT map z_t->x_{t+1}
    and Y_t is the OT map z_t->y_t.

    Returns (G_vals, W2_vals) of length n_steps+1.
    """
    sigma_q = 1.0 / np.sqrt(lam)
    mx, sx = float(m0), float(s0)   # x_0
    mz, sz = float(m0), float(s0)   # z_0  (= x_0 by initialisation)
    G_vals  = [kl_gaussian(mx, sx, sigma_q)]
    W2_vals = [np.sqrt(w2_sq_1d(mx, sx, 0.0, sigma_q))]

    for t in range(n_steps):
        alpha = 3.0 / (t + 3.0)          # alpha_0 = 1, alpha_1 = 3/4, alpha_t -> 0

        # -- Wasserstein geodesic interpolation y_t --
        my = (1.0 - alpha) * mx + alpha * mz
        sy = max((1.0 - alpha) * sx + alpha * sz, 1e-12)

        # -- JKO proximal step from y_t --
        mx_new, sx_new = jko_prox_kl(my, sy, gamma, lam)

        # -- Momentum update for z --
        mz_new = (mx_new - (1.0 - alpha) * mx) / alpha
        sz_new = max((sx_new - (1.0 - alpha) * sx) / alpha, 1e-12)

        mx, sx = mx_new, sx_new
        mz, sz = mz_new, sz_new

        G_vals.append(kl_gaussian(mx, sx, sigma_q))
        W2_vals.append(np.sqrt(w2_sq_1d(mx, sx, 0.0, sigma_q)))

    return np.array(G_vals), np.array(W2_vals)


# ==========================================================
#  4.  THEORETICAL BOUNDS
# ==========================================================

def bound_std_jko(G0, W2sq_0, gamma, lam, n_steps):
    """
    Upper bound for standard JKO  (Theorem 4.3 + eq. A.18, Cheng et al. 2024).
      G(p_n) <= W_2^2(p_0,q)/(2gamma) * (1+gammalam/2)^{-(n-1)}   for n >= 1
    """
    n_arr = np.arange(n_steps + 1)
    rho   = 1.0 / (1.0 + gamma * lam / 2.0)
    b = np.where(n_arr == 0, G0,
                 W2sq_0 / (2.0 * gamma) * rho ** (n_arr - 1))
    return b


def bound_acc_jko(G0, W2sq_0, gamma, n_steps):
    """
    Upper bound for accelerated JKO  (Theorem, template.tex).
    With x_0 = z_0, alpha_0 = 1  ->  lam_0 = 1/gamma:
      G(x_t) <= 9/(t+2)^2 * ( W_2^2(z_0,q)/(2gamma) + G(x_0) )
    """
    t_arr  = np.arange(n_steps + 1)
    Delta0 = W2sq_0 / (2.0 * gamma) + G0
    b = np.where(t_arr == 0, G0, 9.0 * Delta0 / (t_arr + 2) ** 2)
    return b


# ==========================================================
#  5.  EXPERIMENT RUNNERS
# ==========================================================

def experiment_fixed_lam(m0, s0, gamma, lam, n_steps):
    """Run both algorithms and collect all arrays for one (lam, gamma) setting."""
    sigma_q   = 1.0 / np.sqrt(lam)
    G0        = kl_gaussian(m0, s0, sigma_q)
    W2sq_0    = w2_sq_1d(m0, s0, 0.0, sigma_q)

    G_std,  W2_std  = run_standard_jko(m0, s0, gamma, lam, n_steps)
    G_acc,  W2_acc  = run_accelerated_jko(m0, s0, gamma, lam, n_steps)

    b_std = bound_std_jko(G0, W2sq_0, gamma, lam, n_steps)
    b_acc = bound_acc_jko(G0, W2sq_0, gamma, n_steps)

    iters = np.arange(n_steps + 1)
    return dict(iters=iters, G_std=G_std, G_acc=G_acc,
                W2_std=W2_std, W2_acc=W2_acc,
                b_std=b_std, b_acc=b_acc,
                G0=G0, W2sq_0=W2sq_0,
                gamma=gamma, lam=lam, sigma_q=sigma_q)


# ==========================================================
#  6.  PLOTTING
# ==========================================================

BLUE  = '#1f77b4'
RED   = '#d62728'
LBLUE = '#aec7e8'
LRED  = '#ffb3b3'

def _ref_line(ax, x, slope, label, style='k:', lw=1.0, alpha=0.5):
    """Plot a reference line  C*x^slope through the midpoint of x."""
    mid = len(x) // 2
    C   = ax.get_lines()[-1].get_ydata()[mid] / x[mid] ** slope if slope != 0 else 1
    ax.plot(x, C * x ** slope, style, lw=lw, alpha=alpha, label=label)


def plot_scenario(res, axs, title_prefix=''):
    """
    axs : array of 3 Axes  [log-linear KL,  log-log KL,  log-linear W_2]
    """
    iters = res['iters']
    pos   = iters[iters > 0]           # skip t=0 for log-log slopes

    # -- A: log-linear KL ----------------------------------------------
    ax = axs[0]
    ax.semilogy(iters, res['G_std'],  color=BLUE,  lw=2,   label='Standard JKO')
    ax.semilogy(iters, res['G_acc'],  color=RED,   lw=2,   label='Accelerated JKO')
    ax.semilogy(iters[1:], res['b_std'][1:], '--', color=LBLUE, lw=1.5,
                label=r'Std bound $\propto(1+\gamma\lambda/2)^{-n}$')
    ax.semilogy(iters[1:], res['b_acc'][1:], '--', color=LRED,  lw=1.5,
                label=r'Acc bound $\propto t^{-2}$')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'$G(\rho_t) = \mathrm{KL}(\rho_t \| q)$', fontsize=11)
    ax.set_title(f'{title_prefix}\nLog-linear scale', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # -- B: log-log KL -------------------------------------------------
    ax = axs[1]
    ax.loglog(pos, res['G_std'][pos],  color=BLUE, lw=2,   label='Standard JKO')
    ax.loglog(pos, res['G_acc'][pos],  color=RED,  lw=2,   label='Accelerated JKO')
    ax.loglog(pos, res['b_std'][pos], '--', color=LBLUE, lw=1.5, label='Std bound')
    ax.loglog(pos, res['b_acc'][pos], '--', color=LRED,  lw=1.5, label='Acc bound')
    # Reference slopes
    c1 = res['G_std'][5] * 5;    ax.loglog(pos, c1 / pos,    'k:',  lw=1, alpha=0.6, label=r'$O(t^{-1})$')
    c2 = res['G_acc'][5] * 25;   ax.loglog(pos, c2 / pos**2, 'k-.', lw=1, alpha=0.6, label=r'$O(t^{-2})$')
    ax.set_xlabel('Iteration $t$', fontsize=11)
    ax.set_ylabel(r'$G(\rho_t)$', fontsize=11)
    ax.set_title(f'{title_prefix}\nLog-log scale (slopes)', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower left')
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # -- C: log-linear W_2 ----------------------------------------------
    ax = axs[2]
    ax.semilogy(iters, res['W2_std'], color=BLUE, lw=2, label='Standard JKO')
    ax.semilogy(iters, res['W2_acc'], color=RED,  lw=2, label='Accelerated JKO')
    # Theoretical W_2 bound for standard JKO
    rho = 1.0 / (1.0 + res['gamma'] * res['lam'] / 2.0)
    W2_bound_std = np.sqrt(res['W2sq_0']) * rho**(iters / 2)
    ax.semilogy(iters, W2_bound_std, '--', color=LBLUE, lw=1.5,
                label=r'Std $W_2$ bound $\propto\rho^{n/2}$')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'$W_2(\rho_t, q)$', fontsize=11)
    ax.set_title(f'{title_prefix}\nWasserstein-2 distance to $q$', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(True, which='both', ls=':', alpha=0.4)


# ==========================================================
#  7.  MAIN FIGURE
# ==========================================================

m0, s0 = 5.0, 2.5          # initial distribution  N(5, 2.5^2)
gamma   = 0.5              # step size

# -- Scenario 1: strongly convex G  (lam=1,  q = N(0,1)) --
res1 = experiment_fixed_lam(m0, s0, gamma=0.5, lam=1.0,  n_steps=60)

# -- Scenario 2: weakly convex G   (lam=0.04, q = N(0,5)) --
res2 = experiment_fixed_lam(m0, s0, gamma=0.5, lam=0.04, n_steps=300)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Standard JKO vs Accelerated JKO -- Empirical Convergence\n'
    r'$G(\rho)=\mathrm{KL}(\mathcal{N}(m,s^2)\|\mathcal{N}(0,1/\lambda))$'
    r',  initialisation $\mathcal{N}(5,\ 2.5^2)$,  $\gamma=0.5$',
    fontsize=13, y=1.01
)

plot_scenario(res1, axes[0],
              title_prefix=r'Strongly convex  $(\lambda=1,\ q=\mathcal{N}(0,1))$')
plot_scenario(res2, axes[1],
              title_prefix=r'Weakly convex  $(\lambda=0.04,\ q=\mathcal{N}(0,5))$')

plt.tight_layout()
fig.savefig('/mnt/user-data/outputs/jko_convergence_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved figure 1: jko_convergence_comparison.png")


# ==========================================================
#  8.  STEP-SIZE SENSITIVITY FIGURE
# ==========================================================

gammas = [0.2, 0.5, 1.0, 2.0]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gammas)))

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(r'Effect of Step Size $\gamma$ on Convergence  ($\lambda=0.04$)',
              fontsize=13)

for gamma_i, col in zip(gammas, colors):
    res_g = experiment_fixed_lam(m0, s0, gamma=gamma_i, lam=0.04, n_steps=250)
    iters = res_g['iters']

    axes2[0].semilogy(iters, res_g['G_std'], color=col, lw=2,
                      label=f'$\\gamma={gamma_i}$')
    axes2[1].semilogy(iters, res_g['G_acc'], color=col, lw=2,
                      label=f'$\\gamma={gamma_i}$')

# Reference O(t^{-2})
t_ref = np.arange(1, 251)
C_ref = res_g['G_acc'][10] * 12**2
axes2[1].loglog(t_ref, C_ref / t_ref**2, 'k--', lw=1.5, alpha=0.7,
                label=r'$O(t^{-2})$')

for ax, title in zip(axes2, ['Standard JKO (log-linear)', 'Accelerated JKO (log-linear)']):
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'$G(\rho_t)$', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls=':', alpha=0.4)

plt.tight_layout()
fig2.savefig('/mnt/user-data/outputs/jko_step_size_sensitivity.png',
             dpi=150, bbox_inches='tight')
print("Saved figure 2: jko_step_size_sensitivity.png")


# ==========================================================
#  9.  NUMERICAL SUMMARY TABLE
# ==========================================================

print("\n" + "="*70)
print("NUMERICAL SUMMARY")
print("="*70)

for scenario, lam in [("Strongly convex (lam=1.00)", 1.00),
                      ("Weakly convex   (lam=0.04)", 0.04)]:
    res = experiment_fixed_lam(m0, s0, gamma=0.5, lam=lam, n_steps=300)
    G0  = res['G0']
    print(f"\n{scenario}")
    print(f"  Initial G(rho_0) = {G0:.4f},  W_2(rho_0,q) = {np.sqrt(res['W2sq_0']):.4f}")
    print(f"  {'Target fraction':<20} {'Std JKO iters':>15} {'Acc JKO iters':>15}")
    for frac in [0.5, 0.1, 0.01, 0.001]:
        thresh = frac * G0
        n_std = next((i for i, g in enumerate(res['G_std']) if g < thresh), ">300")
        n_acc = next((i for i, g in enumerate(res['G_acc']) if g < thresh), ">300")
        print(f"  G < {frac:.3f}*G_0 = {thresh:.4f}    {str(n_std):>15}    {str(n_acc):>15}")

print("\n" + "="*70)

plt.show()
print("\nDone.")