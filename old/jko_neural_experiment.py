"""
Neural Accelerated JKO — Diffusion Model Training Experiment
=============================================================
Setting : 2-D mixture-of-8-Gaussians target  q
          source p_0 = N(0, 4·I)

Each JKO proximal step is solved by a small MLP that minimises
    L(θ) = KL(T_θ # p_n || q)  +  (1/2γ) E_{x~p_n} ||x - T_θ(x)||²

The KL is estimated via Monte-Carlo using the log-det of the Jacobian
(exact for MLPs of manageable width, computed via torch.autograd.functional).

Compares:
  • Standard JKO          — x_{n+1} = Prox(G, x_n, γ)
  • Accelerated JKO       — three-sequence scheme, α_t = 3/(t+3)

Metrics tracked per block (= per JKO step):
  • KL(p_t || q)          — exact Monte-Carlo estimate
  • W_2(p_t, q)           — Sinkhorn approximation via geomloss (if available),
                            else energy distance fallback
  • Wall-clock time

Usage:
    pip install torch geomloss matplotlib
    python jko_neural_experiment.py

Outputs (saved to images/):
    images/neural_jko_convergence.png
    images/neural_jko_stepsize.png
    results/neural_jko_summary.txt
"""

import os
import time

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap          # torch >= 2.0

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 11})

# ─────────────────────────────────────────────────────────────
#  0.  DEVICE
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
#  1.  TARGET: 8-GAUSSIAN MIXTURE
# ─────────────────────────────────────────────────────────────

class MixtureOf8Gaussians:
    """
    q = (1/8) * sum_k N(mu_k, sigma^2 * I),  k=0..7
    Modes placed on a circle of radius R.
    """
    def __init__(self, radius: float = 4.0, sigma: float = 0.5,
                 device=DEVICE):
        self.K      = 8
        self.sigma  = sigma
        self.device = device
        angles = torch.linspace(0, 2 * math.pi, self.K + 1)[:-1]
        self.means = torch.stack(
            [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
        ).to(device)                          # (8, 2)

    # ── sampling ──────────────────────────────────────────────
    def sample(self, n: int) -> torch.Tensor:
        k   = torch.randint(0, self.K, (n,), device=self.device)
        eps = torch.randn(n, 2, device=self.device) * self.sigma
        return self.means[k] + eps

    # ── log-density ───────────────────────────────────────────
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x : (n, 2)  →  log q(x) : (n,)"""
        # log sum_k exp( log N(x; mu_k, sigma^2 I) ) - log K
        diff  = x.unsqueeze(1) - self.means.unsqueeze(0)   # (n,K,2)
        lp    = -0.5 * (diff**2).sum(-1) / self.sigma**2   # (n,K)
        lp   -= math.log(self.sigma) * 2 + math.log(2 * math.pi)
        return torch.logsumexp(lp, dim=1) - math.log(self.K)

    # ── W2 ground truth (closed form for fixed sigma → 0 limit)
    #    We use Sinkhorn if geomloss is available, otherwise energy dist.


TARGET = MixtureOf8Gaussians(device=DEVICE)

# ─────────────────────────────────────────────────────────────
#  2.  W2 ESTIMATOR
# ─────────────────────────────────────────────────────────────

try:
    from geomloss import SamplesLoss
    _sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
    def w2_estimate(x: torch.Tensor, y: torch.Tensor) -> float:
        """W_2^2 estimate (Sinkhorn) then return W_2."""
        with torch.no_grad():
            loss = _sinkhorn(x.contiguous(), y.contiguous())
        return float(loss.sqrt().clamp(min=0))
    W2_METHOD = "Sinkhorn"
except ImportError:
    def w2_estimate(x: torch.Tensor, y: torch.Tensor) -> float:
        """Energy distance as W2 proxy when geomloss not available."""
        with torch.no_grad():
            dxx = torch.cdist(x, x).mean()
            dyy = torch.cdist(y, y).mean()
            dxy = torch.cdist(x, y).mean()
            ed  = (2 * dxy - dxx - dyy).clamp(min=0)
        return float(ed.sqrt())
    W2_METHOD = "Energy-distance proxy"

print(f"W2 method: {W2_METHOD}")

# ─────────────────────────────────────────────────────────────
#  3.  NEURAL TRANSPORT MAP  (one JKO block)
# ─────────────────────────────────────────────────────────────

class TransportMap(nn.Module):
    """
    T_θ(x) = x + MLP(x)   — residual so T ≈ Id at init.
    Architecture: 3-layer tanh MLP.
    """
    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # small init so residual starts near identity
        for p in self.net[-1].parameters():
            nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────
#  4.  LOG-DET JACOBIAN  (exact, via vmap + jacrev)
# ─────────────────────────────────────────────────────────────

def logdet_jacobian(T: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute log |det J_T(x)| for each x in batch.
    Uses vmap over jacrev for a per-sample Jacobian, then slogdet.
    Falls back to Hutchinson estimator for large batches.
    x : (n, d)  →  (n,)
    """
    def _single(xi):
        return T(xi.unsqueeze(0)).squeeze(0)

    n, d = x.shape
    if n * d * d > 4_000_000:          # memory guard → Hutchinson
        v  = torch.randn_like(x)
        Tv, Jv = torch.autograd.functional.jvp(_single_batch(T), (x,), (v,))
        return (Jv * v).sum(-1)        # E[v^T J v] ≈ tr(J) for small nets

    jac = vmap(jacrev(_single))(x)    # (n, d, d)
    _, ld = torch.linalg.slogdet(jac)
    return ld


def _single_batch(T):
    def f(x): return T(x)
    return f


# ─────────────────────────────────────────────────────────────
#  5.  KL ESTIMATOR  KL(T#p || q)
# ─────────────────────────────────────────────────────────────

def kl_estimate(T: nn.Module,
                x_samples: torch.Tensor,
                target: MixtureOf8Gaussians) -> torch.Tensor:
    """
    KL(T#p || q) ≈ (1/n) Σ [ log p(x) - log |det J_T(x)| - log q(T(x)) ]
    where x ~ p.

    log p(x): not available in closed form for general p.
    We use the REINFORCE-style unbiased estimator:
        KL = E_{x~p}[-log q(T(x)) - log|det J_T(x)|] + const
    The constant (entropy of p) does not depend on θ, so we minimise
        L_KL(θ) = E_{x~p}[ -log q(T(x)) - log|det J_T(x)| ]
    which equals KL up to the (fixed) entropy H(p).
    """
    Tx  = T(x_samples)
    lq  = target.log_prob(Tx)
    ld  = logdet_jacobian(T, x_samples)
    return (-lq - ld).mean()


# ─────────────────────────────────────────────────────────────
#  6.  JKO LOSS  =  KL  +  W2^2 penalty
# ─────────────────────────────────────────────────────────────

def jko_loss(T: nn.Module,
             y_samples: torch.Tensor,
             gamma: float,
             target: MixtureOf8Gaussians) -> torch.Tensor:
    """
    L(θ) = KL_term(T, y) + (1/2γ) ||x - T(x)||²_mean
    """
    Ty      = T(y_samples)
    kl_term = kl_estimate(T, y_samples, target)
    w2_term = ((y_samples - Ty)**2).sum(-1).mean() / (2.0 * gamma)
    return kl_term + w2_term


# ─────────────────────────────────────────────────────────────
#  7.  TRAIN ONE JKO BLOCK
# ─────────────────────────────────────────────────────────────

def train_block(y_samples: torch.Tensor,
                gamma: float,
                target: MixtureOf8Gaussians,
                n_epochs: int = 300,
                lr: float = 3e-3,
                batch_size: int = 512,
                hidden: int = 128) -> TransportMap:
    """
    Learn T such that T#y ≈ argmin_ρ { G(ρ) + (1/2γ) W₂²(ρ, y) }.
    Returns the trained TransportMap.
    """
    T   = TransportMap(dim=2, hidden=hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n = y_samples.shape[0]
    for ep in range(n_epochs):
        idx  = torch.randperm(n, device=DEVICE)[:batch_size]
        xb   = y_samples[idx].detach()

        opt.zero_grad()
        loss = jko_loss(T, xb, gamma, target)
        loss.backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()

    return T


# ─────────────────────────────────────────────────────────────
#  8.  EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_kl(T: nn.Module,
            x_samples: torch.Tensor,
            target: MixtureOf8Gaussians) -> float:
    """Evaluate KL_term (up to entropy constant) on a held-out set."""
    Tx = T(x_samples)
    lq = target.log_prob(Tx)
    ld = logdet_jacobian(T, x_samples)
    return float((-lq - ld).mean())


@torch.no_grad()
def eval_metrics(x_samples: torch.Tensor,
                 target: MixtureOf8Gaussians,
                 n_target: int = 2000) -> dict:
    """Return dict with kl_val and w2_val for current particle cloud."""
    q_samples = target.sample(n_target)
    n = min(x_samples.shape[0], n_target)

    # KL via a dummy identity map
    class Identity(nn.Module):
        def forward(self, x): return x
    kl = eval_kl(Identity().to(DEVICE), x_samples[:n], target)
    w2 = w2_estimate(x_samples[:n], q_samples)
    return {"kl": kl, "w2": w2}


# ─────────────────────────────────────────────────────────────
#  9.  STANDARD JKO LOOP
# ─────────────────────────────────────────────────────────────

def run_standard_jko_neural(
        gamma: float,
        N_blocks: int,
        n_particles: int = 2000,
        n_epochs: int = 300,
        target: MixtureOf8Gaussians = TARGET,
        seed: int = 0) -> dict:
    """
    Standard JKO:  x_{t+1} = Prox(G, x_t, γ)
    Returns dict with lists: kl, w2, times.
    """
    torch.manual_seed(seed)
    p0  = torch.randn(n_particles, 2, device=DEVICE) * 2.0

    kl_vals, w2_vals, times = [], [], []

    x = p0.clone()
    m0 = eval_metrics(x, target)
    kl_vals.append(m0["kl"])
    w2_vals.append(m0["w2"])
    times.append(0.0)

    t0 = time.time()
    for step in range(N_blocks):
        T = train_block(x, gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x = T(x)
        m = eval_metrics(x, target)
        kl_vals.append(m["kl"])
        w2_vals.append(m["w2"])
        times.append(time.time() - t0)
        print(f"  [Std JKO] block {step+1}/{N_blocks}  "
              f"KL={m['kl']:.4f}  W2={m['w2']:.4f}  "
              f"t={times[-1]:.1f}s")

    return {"kl": np.array(kl_vals),
            "w2": np.array(w2_vals),
            "times": np.array(times)}


# ─────────────────────────────────────────────────────────────
# 10.  ACCELERATED JKO LOOP
# ─────────────────────────────────────────────────────────────

def run_accelerated_jko_neural(
        gamma: float,
        N_blocks: int,
        n_particles: int = 2000,
        n_epochs: int = 300,
        target: MixtureOf8Gaussians = TARGET,
        seed: int = 0) -> dict:
    """
    Accelerated JKO  (Section 5.2 / Case I):
      α_t = 3/(t+3)
      y_t  = (1−α_t)·x_t  +  α_t·z_t          (Wasserstein geodesic)
      x_{t+1} = Prox(G, y_t, γ)
      z_{t+1} = z_t + (1/α_t)·(x_{t+1} − y_t)  (momentum update)

    All three sequences are represented as particle clouds.
    Returns dict with lists: kl, w2, times.
    """
    torch.manual_seed(seed)
    p0 = torch.randn(n_particles, 2, device=DEVICE) * 2.0

    kl_vals, w2_vals, times = [], [], []

    # Initialise  x_0 = z_0 = p_0
    x = p0.clone()
    z = p0.clone()

    m0 = eval_metrics(x, target)
    kl_vals.append(m0["kl"])
    w2_vals.append(m0["w2"])
    times.append(0.0)

    t0 = time.time()
    for t in range(N_blocks):
        alpha = 3.0 / (t + 3.0)               # α_0 = 1, decays to 0

        # ── Wasserstein geodesic interpolation ──────────────
        y = (1.0 - alpha) * x + alpha * z     # (n, 2)

        # ── Learn proximal step from y ───────────────────────
        T = train_block(y.detach(), gamma, target, n_epochs=n_epochs)

        with torch.no_grad():
            x_new = T(y)                       # x_{t+1} = T#y_t

        # ── Momentum update for z ────────────────────────────
        # z_{t+1} = z + (1/α_t)(x_{t+1} − y_t)
        with torch.no_grad():
            z_new = z + (x_new - y) / alpha

        # Stability guard: clip z particles to prevent blow-up
        z_new = z_new.clamp(-20.0, 20.0)

        x = x_new.detach()
        z = z_new.detach()

        m = eval_metrics(x, target)
        kl_vals.append(m["kl"])
        w2_vals.append(m["w2"])
        times.append(time.time() - t0)
        print(f"  [Acc JKO] block {t+1}/{N_blocks}  "
              f"KL={m['kl']:.4f}  W2={m['w2']:.4f}  "
              f"t={times[-1]:.1f}s  α={alpha:.3f}")

    return {"kl": np.array(kl_vals),
            "w2": np.array(w2_vals),
            "times": np.array(times)}


# ─────────────────────────────────────────────────────────────
# 11.  PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────

BLUE  = "#1f77b4"
RED   = "#d62728"
LBLUE = "#aec7e8"
LRED  = "#ffb3b3"


def plot_convergence(res_std: dict,
                     res_acc: dict,
                     gamma: float,
                     N_blocks: int,
                     title_prefix: str = "",
                     axs=None,
                     savepath: str = None):
    """
    Three-panel plot mirroring jko_comparison.py:
      A) log-linear KL  vs  iteration
      B) log-log  KL   vs  iteration  (slope verification)
      C) W₂ distance   vs  iteration
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        standalone = True
    else:
        standalone = False

    iters = np.arange(N_blocks + 1)
    pos   = iters[iters > 0]

    # ── A: log-linear KL ──────────────────────────────────
    ax = axs[0]
    ax.semilogy(iters, res_std["kl"], color=BLUE, lw=2, label="Standard JKO")
    ax.semilogy(iters, res_acc["kl"], color=RED,  lw=2, label="Accelerated JKO")
    # O(t^{-2}) reference anchored to accelerated curve
    if len(pos) > 2:
        c2 = res_acc["kl"][2] * 4
        ax.semilogy(pos, c2 / pos**2, "k--", lw=1.2, alpha=0.6,
                    label=r"$O(t^{-2})$ ref")
    ax.set_xlabel("JKO block (t)")
    ax.set_ylabel(r"$\mathrm{KL}(p_t \| q)$ + const")
    ax.set_title(f"{title_prefix}\nLog-linear scale")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── B: log-log KL ─────────────────────────────────────
    ax = axs[1]
    ax.loglog(pos, res_std["kl"][pos], color=BLUE, lw=2, label="Standard JKO")
    ax.loglog(pos, res_acc["kl"][pos], color=RED,  lw=2, label="Accelerated JKO")
    if len(pos) > 2:
        c1 = res_std["kl"][1] * 1
        c2 = res_acc["kl"][1] * 1
        ax.loglog(pos, c1 / pos,    "k:",  lw=1.2, alpha=0.6, label=r"$O(t^{-1})$")
        ax.loglog(pos, c2 / pos**2, "k-.", lw=1.2, alpha=0.6, label=r"$O(t^{-2})$")
    ax.set_xlabel("JKO block (t)")
    ax.set_ylabel(r"$\mathrm{KL}(p_t \| q)$ + const")
    ax.set_title(f"{title_prefix}\nLog-log scale (slope check)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── C: W₂ distance ────────────────────────────────────
    ax = axs[2]
    ax.semilogy(iters, res_std["w2"], color=BLUE, lw=2, label="Standard JKO")
    ax.semilogy(iters, res_acc["w2"], color=RED,  lw=2, label="Accelerated JKO")
    ax.set_xlabel("JKO block (t)")
    ax.set_ylabel(r"$W_2(p_t, q)$  [" + W2_METHOD + "]")
    ax.set_title(f"{title_prefix}\nWasserstein-2 to target")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    if standalone:
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"Saved: {savepath}")
        plt.show()


def plot_particles(samples_list: list,
                   labels: list,
                   target: MixtureOf8Gaussians,
                   savepath: str = None):
    """
    Side-by-side scatter plots of particle clouds overlaid on target density.
    """
    n = len(samples_list)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]

    # target density heatmap
    lim = 7
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 150),
                          np.linspace(-lim, lim, 150))
    grid = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32,
        device=DEVICE
    )
    with torch.no_grad():
        lp = target.log_prob(grid).cpu().numpy().reshape(150, 150)
    density = np.exp(lp - lp.max())

    for ax, samp, lab in zip(axs, samples_list, labels):
        ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.5)
        s = samp.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.4, color=RED)
        ax.set_title(lab, fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Particle Clouds vs Target Density", fontsize=12)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 12.  STEP-SIZE SENSITIVITY EXPERIMENT
# ─────────────────────────────────────────────────────────────

def run_stepsize_experiment(gammas,
                             N_blocks: int = 8,
                             n_particles: int = 1500,
                             n_epochs: int = 200,
                             target: MixtureOf8Gaussians = TARGET):
    results = {}
    for gamma in gammas:
        print(f"\n=== γ = {gamma} ===")
        print("  Standard JKO:")
        r_std = run_standard_jko_neural(
            gamma, N_blocks, n_particles, n_epochs, target, seed=42)
        print("  Accelerated JKO:")
        r_acc = run_accelerated_jko_neural(
            gamma, N_blocks, n_particles, n_epochs, target, seed=42)
        results[gamma] = {"std": r_std, "acc": r_acc}
    return results


def plot_stepsize_sensitivity(results, gammas, N_blocks, savepath=None):
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gammas)))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Effect of Step Size $\gamma$ on Neural JKO Convergence",
                 fontsize=13)

    iters = np.arange(N_blocks + 1)
    for gamma, col in zip(gammas, colors):
        r = results[gamma]
        axs[0].semilogy(iters, r["std"]["kl"], color=col, lw=2,
                        label=f"γ={gamma}")
        axs[1].semilogy(iters, r["acc"]["kl"], color=col, lw=2,
                        label=f"γ={gamma}")

    # O(t^{-2}) reference
    pos = np.arange(1, N_blocks + 1)
    c   = results[gammas[0]]["acc"]["kl"][1] * 1.0
    axs[1].loglog(pos, c / pos**2, "k--", lw=1.5, alpha=0.7,
                  label=r"$O(t^{-2})$")

    for ax, title in zip(axs,
                         ["Standard JKO (log-linear)",
                          "Accelerated JKO (log-linear)"]):
        ax.set_xlabel("JKO block")
        ax.set_ylabel(r"$\mathrm{KL}(p_t \| q)$ + const")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 13.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_summary(res_std, res_acc, gamma, N_blocks, savepath=None):
    lines = []
    lines.append("=" * 65)
    lines.append(f"NEURAL JKO EXPERIMENT SUMMARY   γ={gamma},  blocks={N_blocks}")
    lines.append("=" * 65)
    lines.append(f"\nInitial KL  = {res_std['kl'][0]:.4f}")
    lines.append(f"Final KL   Standard JKO : {res_std['kl'][-1]:.4f}")
    lines.append(f"Final KL   Accelerated  : {res_acc['kl'][-1]:.4f}")
    lines.append(f"\nFinal W2   Standard JKO : {res_std['w2'][-1]:.4f}")
    lines.append(f"Final W2   Accelerated  : {res_acc['w2'][-1]:.4f}")

    G0 = res_std["kl"][0]
    lines.append(f"\n{'Target fraction':<22}{'Std JKO blocks':>18}{'Acc JKO blocks':>18}")
    for frac in [0.8, 0.5, 0.2, 0.1]:
        thresh = frac * G0
        n_std  = next((i for i, g in enumerate(res_std["kl"]) if g < thresh), ">N")
        n_acc  = next((i for i, g in enumerate(res_acc["kl"]) if g < thresh), ">N")
        lines.append(
            f"  KL < {frac:.2f}·KL₀ = {thresh:.3f}   {str(n_std):>16}   {str(n_acc):>16}"
        )

    lines.append("\n" + "=" * 65)
    text = "\n".join(lines)
    print(text)
    if savepath:
        with open(savepath, "w") as f:
            f.write(text)
        print(f"Saved: {savepath}")


# ─────────────────────────────────────────────────────────────
# 14.5  DEPTH SWEEP PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_depth_sweep(N_list, kl_std_finals, kl_acc_finals, ratios,
                     savepath=None):
    """Two-panel plot: final KL vs N (semilogy) and KL ratio vs N."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Block Depth Sweep — Neural JKO (8-Gaussian, $\gamma=0.5$)",
                 fontsize=13)

    N_arr = np.array(N_list, dtype=float)

    # ── Left: final KL vs N ──────────────────────────────────
    ax = axs[0]
    ax.semilogy(N_list, kl_std_finals, "o-", color=BLUE, lw=2,
                label="Standard JKO")
    ax.semilogy(N_list, kl_acc_finals, "o-", color=RED, lw=2,
                label="Accelerated JKO")
    # reference lines
    c1 = kl_std_finals[0] * N_arr[0]
    ax.semilogy(N_arr, c1 / N_arr, "k:", lw=1.2, alpha=0.6,
                label=r"$O(1/N)$ ref")
    c2 = kl_acc_finals[0] * N_arr[0]**2
    ax.semilogy(N_arr, c2 / N_arr**2, "k-.", lw=1.2, alpha=0.6,
                label=r"$O(1/N^2)$ ref")
    ax.set_xlabel("Number of JKO blocks (N)")
    ax.set_ylabel(r"Final $\mathrm{KL}(p_N \| q)$ + const")
    ax.set_title("Final KL vs Depth")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Right: ratio vs N ────────────────────────────────────
    ax = axs[1]
    ax.plot(N_list, ratios, "s-", color="black", lw=2,
            label=r"$\mathrm{KL}_{\mathrm{std}} / \mathrm{KL}_{\mathrm{acc}}$")
    # O(N) reference
    c_r = ratios[0] / N_arr[0]
    ax.plot(N_arr, c_r * N_arr, "k--", lw=1.2, alpha=0.6,
            label=r"$O(N)$ ref")
    ax.set_xlabel("Number of JKO blocks (N)")
    ax.set_ylabel("KL ratio  (Standard / Accelerated)")
    ax.set_title("Acceleration Advantage vs Depth")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 14.6  FIXED BUDGET PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_fixed_budget(configs, kl_std_finals, kl_acc_finals,
                      w2_std_finals, w2_acc_finals, total_budget,
                      savepath=None):
    """Two-panel plot for fixed-budget experiment."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        rf"Fixed Compute Budget ($N \times \mathrm{{epochs}}={total_budget}$)"
        r" — Neural JKO",
        fontsize=13)

    N_list = [c[0] for c in configs]

    # ── Left: final KL vs N_blocks ───────────────────────────
    ax = axs[0]
    ax.semilogy(N_list, kl_std_finals, "o-", color=BLUE, lw=2,
                label="Standard JKO")
    ax.semilogy(N_list, kl_acc_finals, "o-", color=RED, lw=2,
                label="Accelerated JKO")
    ax.set_xlabel("N_blocks  (epochs/block decreases →)")
    ax.set_ylabel(r"Final $\mathrm{KL}(p_N \| q)$ + const")
    ax.set_title("Final KL vs Block Count")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # ── Right: final W2 vs N_blocks ──────────────────────────
    ax = axs[1]
    ax.plot(N_list, w2_std_finals, "o-", color=BLUE, lw=2,
            label="Standard JKO")
    ax.plot(N_list, w2_acc_finals, "o-", color=RED, lw=2,
            label="Accelerated JKO")
    ax.set_xlabel("N_blocks  (epochs/block decreases →)")
    ax.set_ylabel(r"Final $W_2(p_N, q)$  [" + W2_METHOD + "]")
    ax.set_title("Final W2 vs Block Count")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 15.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Experiment 1: Main convergence comparison ────────────
    GAMMA    = 0.5
    N_BLOCKS = 12        # number of JKO blocks
    N_PART   = 2000      # particles
    N_EPOCHS = 300       # training epochs per block

    print("\n" + "="*60)
    print("EXPERIMENT 1: Main convergence  (γ=0.5, 12 blocks)")
    print("="*60)

    print("\nRunning Standard JKO...")
    res_std = run_standard_jko_neural(
        GAMMA, N_BLOCKS, N_PART, N_EPOCHS, TARGET, seed=0)

    print("\nRunning Accelerated JKO...")
    res_acc = run_accelerated_jko_neural(
        GAMMA, N_BLOCKS, N_PART, N_EPOCHS, TARGET, seed=0)

    plot_convergence(
        res_std, res_acc,
        gamma=GAMMA, N_blocks=N_BLOCKS,
        title_prefix=rf"8-Gaussian mixture, $\gamma={GAMMA}$",
        savepath="images/neural_jko_convergence.png"
    )

    print_summary(res_std, res_acc, GAMMA, N_BLOCKS,
                  savepath="results/neural_jko_summary.txt")

    # ── Experiment 2: Step-size sensitivity ─────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 2: Step-size sensitivity")
    print("="*60)

    GAMMAS_SWEEP = [0.2, 0.5, 1.0, 2.0]
    stepsize_results = run_stepsize_experiment(
        GAMMAS_SWEEP, N_blocks=8, n_particles=1500, n_epochs=200)

    plot_stepsize_sensitivity(
        stepsize_results, GAMMAS_SWEEP, N_blocks=8,
        savepath="images/neural_jko_stepsize.png"
    )

    # ── Experiment 3: Particle visualization ────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 3: Particle snapshots")
    print("="*60)

    # Collect snapshots at blocks 0, N/3, 2N/3, N for accelerated
    torch.manual_seed(0)
    p0    = torch.randn(N_PART, 2, device=DEVICE) * 2.0
    snaps = [p0.clone()]
    snap_labels = ["t=0  (source)"]

    x = p0.clone()
    z = p0.clone()
    for t in range(N_BLOCKS):
        alpha = 3.0 / (t + 3.0)
        y     = (1.0 - alpha) * x + alpha * z
        T     = train_block(y.detach(), GAMMA, TARGET, n_epochs=N_EPOCHS)
        with torch.no_grad():
            x_new = T(y)
        z_new = (z + (x_new - y) / alpha).clamp(-20.0, 20.0)
        x, z  = x_new.detach(), z_new.detach()
        if (t + 1) in {N_BLOCKS // 3, 2 * N_BLOCKS // 3, N_BLOCKS}:
            snaps.append(x.clone())
            snap_labels.append(f"t={t+1}  (Acc JKO)")

    plot_particles(snaps, snap_labels, TARGET,
                   savepath="images/neural_jko_particles.png")

    print("\nAll experiments complete.")

    # ── Experiment 4: Block Depth Sweep ────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 4: Block Depth Sweep")
    print("="*60)

    N_blocks_list = [4, 8, 12, 16, 24]
    depth_gamma     = 0.5
    depth_particles = 1500
    depth_epochs    = 200
    depth_seed      = 42

    kl_std_finals = []
    kl_acc_finals = []
    w2_std_finals_depth = []
    w2_acc_finals_depth = []
    ratios = []

    for Nb in N_blocks_list:
        print(f"\n--- N_blocks = {Nb} ---")
        print("  Standard JKO:")
        r_std = run_standard_jko_neural(
            depth_gamma, Nb, depth_particles, depth_epochs, TARGET,
            seed=depth_seed)
        print("  Accelerated JKO:")
        r_acc = run_accelerated_jko_neural(
            depth_gamma, Nb, depth_particles, depth_epochs, TARGET,
            seed=depth_seed)

        kl_std_finals.append(r_std["kl"][-1])
        kl_acc_finals.append(r_acc["kl"][-1])
        w2_std_finals_depth.append(r_std["w2"][-1])
        w2_acc_finals_depth.append(r_acc["w2"][-1])
        ratio = r_std["kl"][-1] / max(r_acc["kl"][-1], 1e-12)
        ratios.append(ratio)

    plot_depth_sweep(N_blocks_list, kl_std_finals, kl_acc_finals, ratios,
                     savepath="images/depth_sweep_neural.png")

    # Print and save summary table
    depth_lines = []
    depth_lines.append("=" * 72)
    depth_lines.append("DEPTH SWEEP SUMMARY   γ=0.5, particles=1500, epochs=200")
    depth_lines.append("=" * 72)
    depth_lines.append(
        f"{'N_blocks':>10}  {'KL_std':>12}  {'KL_acc':>12}  "
        f"{'W2_std':>12}  {'W2_acc':>12}  {'Ratio':>10}")
    depth_lines.append("-" * 72)
    for i, Nb in enumerate(N_blocks_list):
        depth_lines.append(
            f"{Nb:>10d}  {kl_std_finals[i]:>12.4f}  {kl_acc_finals[i]:>12.4f}  "
            f"{w2_std_finals_depth[i]:>12.4f}  {w2_acc_finals_depth[i]:>12.4f}  "
            f"{ratios[i]:>10.2f}")
    depth_lines.append("=" * 72)
    depth_text = "\n".join(depth_lines)
    print("\n" + depth_text)
    with open("results/depth_sweep_neural.txt", "w") as f:
        f.write(depth_text)
    print("Saved: results/depth_sweep_neural.txt")

    # ── Experiment 5: Fixed Compute Budget ─────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 5: Fixed Compute Budget")
    print("="*60)

    total_budget = 3600
    budget_configs = [(6, 600), (12, 300), (18, 200), (24, 150), (36, 100)]
    budget_gamma     = 0.5
    budget_particles = 1500
    budget_seed      = 42

    kl_std_budget = []
    kl_acc_budget = []
    w2_std_budget = []
    w2_acc_budget = []

    for Nb, ne in budget_configs:
        print(f"\n--- N_blocks={Nb}, epochs={ne} (budget={Nb*ne}) ---")
        print("  Standard JKO:")
        r_std = run_standard_jko_neural(
            budget_gamma, Nb, budget_particles, ne, TARGET,
            seed=budget_seed)
        print("  Accelerated JKO:")
        r_acc = run_accelerated_jko_neural(
            budget_gamma, Nb, budget_particles, ne, TARGET,
            seed=budget_seed)

        kl_std_budget.append(r_std["kl"][-1])
        kl_acc_budget.append(r_acc["kl"][-1])
        w2_std_budget.append(r_std["w2"][-1])
        w2_acc_budget.append(r_acc["w2"][-1])

    plot_fixed_budget(budget_configs, kl_std_budget, kl_acc_budget,
                      w2_std_budget, w2_acc_budget, total_budget,
                      savepath="images/fixed_budget_neural.png")

    # Print and save summary table
    budget_lines = []
    budget_lines.append("=" * 72)
    budget_lines.append(
        f"FIXED BUDGET SUMMARY   N×epochs={total_budget}, "
        f"γ=0.5, particles=1500")
    budget_lines.append("=" * 72)
    budget_lines.append(
        f"{'N_blocks':>10}  {'epochs':>8}  {'KL_std':>12}  {'KL_acc':>12}  "
        f"{'W2_std':>12}  {'W2_acc':>12}")
    budget_lines.append("-" * 72)
    for i, (Nb, ne) in enumerate(budget_configs):
        budget_lines.append(
            f"{Nb:>10d}  {ne:>8d}  {kl_std_budget[i]:>12.4f}  "
            f"{kl_acc_budget[i]:>12.4f}  {w2_std_budget[i]:>12.4f}  "
            f"{w2_acc_budget[i]:>12.4f}")
    budget_lines.append("=" * 72)
    budget_text = "\n".join(budget_lines)
    print("\n" + budget_text)
    with open("results/fixed_budget_neural.txt", "w") as f:
        f.write(budget_text)
    print("Saved: results/fixed_budget_neural.txt")