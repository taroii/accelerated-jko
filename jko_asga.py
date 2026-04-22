import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.linalg import solve_banded

os.makedirs("images", exist_ok=True)

BLUE = "#1f77b4"
RED  = "#d62728"

#  Problem setup
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

#  Tridiagonal Poisson solver  (-Delta u = rhs, Neumann BC)
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

#  Dual objective and Sobolev gradient
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

if __name__ == "__main__":
    #  Estimate Lipschitz constant of Sobolev gradient
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

    #  Approximate I* via long ASGA run
    fs = [np.zeros(n) for _ in range(M)]
    zs = [f.copy() for f in fs]
    for t in range(1, 3001):
        alpha = 3.0 / (t + 3.0)
        ys    = [(1 - alpha) * fs[i] + alpha * zs[i] for i in range(M)]
        gs    = I_grad(ys)
        fs    = [ys[i] + eta * gs[i] for i in range(M)]
        zs    = [zs[i] + (eta / alpha) * gs[i] for i in range(M)]
    I_star = I_val(fs)

    #  SGA  (fixed step eta, O(1/T) for smooth concave)
    N_ITERS = 400
    fs_sga  = [np.zeros(n) for _ in range(M)]
    gaps_sga = []
    for t in range(1, N_ITERS + 1):
        gs = I_grad(fs_sga)
        for i in range(M):
            fs_sga[i] += eta * gs[i]
        gaps_sga.append(max(I_star - I_val(fs_sga), 1e-12))

    #  ASGA  (Nesterov momentum, O(1/t^2))
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

    #  Plot
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

    plt.show()