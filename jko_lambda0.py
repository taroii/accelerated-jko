"""
Figure 3: Standard JKO vs Accelerated JKO at lambda = 0
=========================================================
Target  : q = 0.5*N(2.0, 0.7^2) + 0.5*N(-2.0, 0.7^2)  (double-well, lambda=0 exactly).
Source  : rho(0.5, 2.0) = 0.5*N(0.5, 2^2) + 0.5*N(-0.5, 2^2).
Params  : gamma=0.5, N=40 blocks.

Variational family: symmetric mixtures rho(m,s) = 0.5*N(m,s^2) + 0.5*N(-m,s^2),
parametrised by (m >= 0, s > 0). The family contains the target exactly, so
KL(rho || q) -> 0 is achievable. W_2^2 between two such mixtures is (m1-m2)^2 + (s1-s2)^2.
KL computed by scipy.quad; each proximal step solved by scipy L-BFGS-B over (m, log s).

Output  : images/figure_3.png  (single log-linear panel)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats

os.makedirs("images", exist_ok=True)

BLUE = "#1f77b4"
RED  = "#d62728"

# ------------------------------------------------------------------
#  Target: symmetric Gaussian mixture
# ------------------------------------------------------------------
MU_Q  = 2.0
SIG_Q = 0.7

def log_q(x):
    lp1 = stats.norm.logpdf(x,  MU_Q, SIG_Q)
    lp2 = stats.norm.logpdf(x, -MU_Q, SIG_Q)
    return np.log(0.5) + np.logaddexp(lp1, lp2)

# ------------------------------------------------------------------
#  Variational family: rho(m,s) = 0.5*N(m,s^2) + 0.5*N(-m,s^2)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  Numerical proximal step
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  Standard JKO
# ------------------------------------------------------------------

def run_standard_jko(m0, s0, gamma, n_steps):
    m, s   = float(m0), float(s0)
    G_vals = [kl_rho_q(m, s)]
    for k in range(n_steps):
        m, s = jko_prox(m, s, gamma)
        G_vals.append(kl_rho_q(m, s))
        print(f"  Std  [{k+1:3d}/{n_steps}]  m={m:.4f}  s={s:.4f}  KL={G_vals[-1]:.6f}")
    return np.array(G_vals)


# ------------------------------------------------------------------
#  Accelerated JKO
# ------------------------------------------------------------------

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
        print(f"  Acc  [{t+1:3d}/{n_steps}]  m={mx:.4f}  s={sx:.4f}  KL={G_vals[-1]:.6f}")

    return np.array(G_vals)


# ------------------------------------------------------------------
#  Theoretical bound  O(t^-2)
# ------------------------------------------------------------------

def bound_acc(G0, W2sq_0, gamma, n_steps):
    t      = np.arange(n_steps + 1)
    Delta0 = W2sq_0 / (2.0 * gamma) + G0
    return np.where(t == 0, G0, 9.0 * Delta0 / (t + 2)**2)


# ------------------------------------------------------------------
#  Run
# ------------------------------------------------------------------
m0, s0  = 0.5, 2.0
GAMMA   = 0.5
N_STEPS = 40

print(f"Target : 0.5*N({MU_Q},{SIG_Q}^2) + 0.5*N(-{MU_Q},{SIG_Q}^2)  [lambda=0]")
print(f"Source : rho({m0},{s0}),  gamma={GAMMA},  blocks={N_STEPS}\n")

print("Running Standard JKO...")
G_std = run_standard_jko(m0, s0, GAMMA, N_STEPS)

print("\nRunning Accelerated JKO...")
G_acc = run_accelerated_jko(m0, s0, GAMMA, N_STEPS)

G0     = kl_rho_q(m0, s0)
W2sq_0 = w2_sq(m0, s0, MU_Q, SIG_Q)
b_acc  = bound_acc(G0, W2sq_0, GAMMA, N_STEPS)

# ------------------------------------------------------------------
#  Plot
# ------------------------------------------------------------------
iters = np.arange(N_STEPS + 1)
pos   = iters[iters > 0]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.semilogy(iters, G_std, color=BLUE, lw=2, label="Standard JKO")
ax.semilogy(iters, G_acc, color=RED,  lw=2, label="Accelerated JKO")
ax.semilogy(iters[1:], b_acc[1:], "--", color=RED, lw=1.2, alpha=0.5,
            label=r"Acc bound $\propto t^{-2}$")
ax.set_xlabel("Block $t$", fontsize=11)
ax.set_ylabel(r"$\mathrm{KL}(\rho_t \| q)$", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, which="both", ls=":", alpha=0.35)

plt.tight_layout()
fig.savefig("images/figure_3.png", dpi=160, bbox_inches="tight")
print("\nSaved: images/figure_3.png")

# ------------------------------------------------------------------
#  Console summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"lambda=0  (gamma={GAMMA}, N={N_STEPS})")
print("=" * 60)
print(f"  {'t':>5}  {'G_std':>12}  {'G_acc':>12}  {'ratio':>8}")
for t in [1, 5, 10, 20, 30, 40]:
    if t <= N_STEPS:
        r = G_std[t] / G_acc[t] if G_acc[t] > 1e-12 else float("inf")
        print(f"  {t:5d}  {G_std[t]:12.6f}  {G_acc[t]:12.6f}  {r:8.3f}")

plt.show()
print("\nDone.")