"""
ASGA Barycenter Experiment: SGA vs Accelerated SGA
====================================================
Setting : Wasserstein barycenter of m=4 Gaussian marginals on a 1-D grid.
          Marginals mu_i = N(c_i, 0.07^2) with c_i in {0.15, 0.35, 0.65, 0.85}.
          Equal weights alpha_i = 1/4.  Grid: n=60 points on [0,1].

Dual objective (entropic regularisation, eps=0.008):
  I(f) = sum_i alpha_i [ sum_j f_i(j) mu_i(j) + sum_k g_i*(k) nu(k) ]
  where g_i*(k) = -eps * logsumexp_j { log mu_i(j) + f_i(j)/eps - c(j,k)/eps }.
  Sobolev gradient: nabla I(f_i) = (-Delta)^{-1} (mu_i - pi_i^{row marginal}),
  computed via a tridiagonal solve (Neumann BC).

Step size eta = 1/L, where L is the Lipschitz constant of the Sobolev gradient,
estimated empirically via finite differences (L ~ 0.015, eta ~ 68).

Algorithms:
  SGA  -- fixed step eta, last iterate.  Rate: O(1/T)  (smooth, non-accelerated).
  ASGA -- our Algorithm 2, step eta, alpha_t=3/(t+3).  Rate: O(1/t^2).

I* approximated by running ASGA for 3000 iterations.

Output: images/figure_4.png  (single log-log panel)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.linalg import solve_banded

os.makedirs("images", exist_ok=True)

BLUE = "#1f77b4"
RED  = "#d62728"

# ------------------------------------------------------------------
#  Problem setup
# ------------------------------------------------------------------
np.random.seed(0)

n       = 60
x_grid  = np.linspace(0, 1, n)
h       = x_grid[1] - x_grid[0]
M       = 4
CENTRES = [0.15, 0.35, 0.65, 0.85]
SIGMA   = 0.07
WEIGHTS = np.full(M, 0.25)
EPS     = 0.008

def make_mu(c):
    d = np.exp(-0.5 * ((x_grid - c) / SIGMA)**2)
    return d / d.sum()

MU     = [make_mu(c) for c in CENTRES]
log_MU = [np.log(mu + 1e-20) for mu in MU]
log_nu = np.log(np.full(n, 1.0 / n))
C      = 0.5 * (x_grid[:, None] - x_grid[None, :])**2

# ------------------------------------------------------------------
#  Tridiagonal Poisson solver  (-Delta u = rhs, Neumann BC)
# ------------------------------------------------------------------
def poisson_1d(rhs):
    n_ = len(rhs)
    ab = np.zeros((3, n_))
    ab[1]      =  2.0 / h**2
    ab[0, 1:]  = -1.0 / h**2
    ab[2, :-1] = -1.0 / h**2
    ab[1, 0]   =  1.0 / h**2    # Neumann left
    ab[1, -1]  =  1.0           # pin last value to 0 (remove null space)
    ab[2, -2]  =  0.0
    r = rhs.copy(); r[-1] = 0.0
    u = solve_banded((1, 1), ab, r)
    return u - u.mean()

# ------------------------------------------------------------------
#  Dual objective and Sobolev gradient
# ------------------------------------------------------------------
def I_val(fs):
    val = 0.0
    for i in range(M):
        fi  = fs[i]
        gi  = -EPS * logsumexp(log_MU[i][:, None] + fi[:, None] / EPS
                               - C / EPS, axis=0)
        val += WEIGHTS[i] * ((fi * MU[i]).sum() + (gi / n).sum())
    return val


def I_grad(fs):
    grads = []
    for i in range(M):
        fi      = fs[i]
        gi      = -EPS * logsumexp(log_MU[i][:, None] + fi[:, None] / EPS
                                   - C / EPS, axis=0)
        log_pi  = (log_MU[i][:, None] + fi[:, None] / EPS
                   + gi[None, :] / EPS - C / EPS + log_nu[None, :])
        log_pi -= logsumexp(log_pi)
        pi      = np.exp(log_pi)
        grads.append(poisson_1d(MU[i] - pi.sum(axis=1)))
    return grads

# ------------------------------------------------------------------
#  Estimate Lipschitz constant of Sobolev gradient
# ------------------------------------------------------------------
fs0    = [np.zeros(n) for _ in range(M)]
g0     = I_grad(fs0)
norms  = []
for _ in range(30):
    delta = [np.random.randn(n) * 0.1 for _ in range(M)]
    gp    = I_grad([fs0[i] + delta[i] for i in range(M)])
    dg    = np.concatenate([gp[i] - g0[i] for i in range(M)])
    dd    = np.concatenate(delta)
    norms.append(np.linalg.norm(dg) / np.linalg.norm(dd))
L   = np.max(norms) * 2.0   # safety factor of 2
eta = 1.0 / L
print(f"Estimated L = {L:.5f},  eta = {eta:.2f}")

# ------------------------------------------------------------------
#  Approximate I* via long ASGA run
# ------------------------------------------------------------------
print("Computing I* via 3000 ASGA iterations...")
fs = [np.zeros(n) for _ in range(M)]
zs = [f.copy() for f in fs]
for t in range(1, 3001):
    alpha = 3.0 / (t + 3.0)
    ys    = [(1 - alpha) * fs[i] + alpha * zs[i] for i in range(M)]
    gs    = I_grad(ys)
    fs    = [ys[i] + eta * gs[i] for i in range(M)]
    zs    = [zs[i] + (eta / alpha) * gs[i] for i in range(M)]
I_star = I_val(fs)
print(f"I* ≈ {I_star:.6f}")

# ------------------------------------------------------------------
#  SGA  (fixed step eta, O(1/T) for smooth concave)
# ------------------------------------------------------------------
N_ITERS = 400
print(f"\nRunning SGA  ({N_ITERS} iters)...")
fs_sga  = [np.zeros(n) for _ in range(M)]
gaps_sga = []
for t in range(1, N_ITERS + 1):
    gs = I_grad(fs_sga)
    for i in range(M):
        fs_sga[i] += eta * gs[i]
    gaps_sga.append(max(I_star - I_val(fs_sga), 1e-12))
    if t % 100 == 0:
        print(f"  [{t:4d}/{N_ITERS}]  gap={gaps_sga[-1]:.6f}")

# ------------------------------------------------------------------
#  ASGA  (Nesterov momentum, O(1/t^2))
# ------------------------------------------------------------------
print(f"\nRunning ASGA ({N_ITERS} iters)...")
fs_asga = [np.zeros(n) for _ in range(M)]
zs      = [f.copy() for f in fs_asga]
gaps_asga = []
for t in range(1, N_ITERS + 1):
    alpha   = 3.0 / (t + 3.0)
    ys      = [(1 - alpha) * fs_asga[i] + alpha * zs[i] for i in range(M)]
    gs      = I_grad(ys)
    fs_asga = [ys[i] + eta * gs[i] for i in range(M)]
    zs      = [zs[i] + (eta / alpha) * gs[i] for i in range(M)]
    gaps_asga.append(max(I_star - I_val(fs_asga), 1e-12))
    if t % 100 == 0:
        print(f"  [{t:4d}/{N_ITERS}]  gap={gaps_asga[-1]:.6f}")

# ------------------------------------------------------------------
#  Plot
# ------------------------------------------------------------------
iters  = np.arange(1, N_ITERS + 1)
g_sga  = np.array(gaps_sga)
g_asga = np.array(gaps_asga)

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.loglog(iters, g_sga,  color=BLUE, lw=2, label="SGA")
ax.loglog(iters, g_asga, color=RED,  lw=2, label="ASGA")

# Reference slopes anchored at t=20
c1 = g_sga[19]  * 20
c2 = g_asga[19] * 20**2
# ax.loglog(iters, c1 / iters,       "--", color=BLUE, lw=1.2, alpha=0.5,
#           label=r"$O(t^{-1})$")
ax.loglog(iters, c2 / iters**2,    "--", color=RED,  lw=1.2, alpha=0.5,
          label=r"$O(t^{-2})$")

ax.set_xlabel("Iteration $t$", fontsize=11)
ax.set_ylabel(r"$\mathcal{I}(\tilde{f}) - \mathcal{I}(f^{(t)})$", fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, which="both", ls=":", alpha=0.35)

plt.tight_layout()
fig.savefig("images/figure_4.png", dpi=160, bbox_inches="tight")
print("\nSaved: images/figure_4.png")

# ------------------------------------------------------------------
#  Console summary
# ------------------------------------------------------------------
valid  = iters > 20
s_sga  = np.polyfit(np.log(iters[valid]), np.log(g_sga[valid]),  1)[0]
s_asga = np.polyfit(np.log(iters[valid]), np.log(g_asga[valid]), 1)[0]

print(f"\n{'='*50}")
print(f"SGA  log-log slope: {s_sga:.2f}   (theory: -1.0)")
print(f"ASGA log-log slope: {s_asga:.2f}  (theory: -2.0)")
print(f"\n  {'t':>5}  {'SGA gap':>12}  {'ASGA gap':>12}  {'ratio':>8}")
for t in [10, 30, 50, 100, 200, 400]:
    r = g_sga[t-1] / g_asga[t-1]
    print(f"  {t:5d}  {g_sga[t-1]:12.6f}  {g_asga[t-1]:12.6f}  {r:8.1f}x")

plt.show()
print("\nDone.")