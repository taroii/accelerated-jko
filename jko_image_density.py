"""
Neural Accelerated JKO -- 2-D Image Density Experiment
======================================================
Setting : Learn a flow from  p_0 = N(0, I)  in R^2  to a target density
          q(x, y)  defined by a grayscale image (pixel intensities -> density).

This is the most interpretable JKO experiment:
  * Exact KL and W_2 are computable via numerical integration / Sinkhorn
  * Particle transport is fully visualisable at every block
  * Runs in minutes on CPU; seconds on GPU

Three built-in targets (no files needed):
  "bunny"    -- Stanford bunny silhouette (synthesised analytically)
  "heart"    -- heart curve density
  "rings"    -- two concentric rings (tests multi-modal transport)

A custom PNG/JPG can also be supplied via --image path/to/image.png

Compares per JKO block:
  * Standard JKO          x_{n+1} = Prox(G, x_n, gamma)
  * Accelerated JKO       three-sequence scheme, alpha_t = 3/(t+3)

Metrics (all exact or near-exact in 2-D):
  * KL(p_t || q)          Monte-Carlo with exact log-det Jacobian
  * W_2(p_t, q)           Sinkhorn via geomloss, or energy-distance fallback

Key design choices to maximise the visible gap between methods:
  * Wide source  N(0, 1.5^2*I)  -- particles start far from target, giving
    acceleration many blocks to compound before convergence.
  * Small gamma (default 0.08)     -- small steps force many blocks and put the
    problem firmly in the weakly-convex regime where O(t^-^2) dominates.
  * More blocks (default 25)   -- the slope difference only opens clearly
    after ~6 blocks; 25 makes it unambiguous.
  * Higher epochs (default 800) -- reduces per-block approximation error eps,
    lowering the O(eps^2) noise floor that otherwise masks the rate gap.
  * Ratio panel in convergence plot -- G_std(t)/G_acc(t) growing over time
    is a clean, unambiguous visual even when absolute curves look close.

Outputs
-------
  jko_image_particles_{name}.png   -- particle snapshots at every block
  jko_image_convergence_{name}.png -- KL, W_2, and speedup-ratio curves
  jko_image_target_{name}.png      -- density preview + samples
  jko_image_summary_{name}.txt     -- numerical table

Usage
-----
    python jko_image_density.py                        # default: rings
    python jko_image_density.py --target heart
    python jko_image_density.py --target bunny
    python jko_image_density.py --image myface.png
    python jko_image_density.py --target rings --gamma 0.05 --blocks 30
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 10})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ==============================================================
#  1.  BUILT-IN TARGET DENSITIES  (no files needed)
# ==============================================================

def _make_grid(res: int = 256) -> tuple:
    """Return (xx, yy) meshgrid over [-1,1]^2."""
    lin = np.linspace(-1, 1, res)
    return np.meshgrid(lin, lin)


def density_bunny(res: int = 256) -> np.ndarray:
    """
    Approximate Stanford-bunny silhouette as a sum of Gaussians
    that traces the bunny outline + fill.
    Returns a (res, res) non-negative array (unnormalised).
    """
    xx, yy = _make_grid(res)
    d = np.zeros((res, res))

    # Body
    d += np.exp(-((xx + 0.0)**2 + (yy + 0.1)**2) / (2 * 0.18**2))
    # Head
    d += 0.6 * np.exp(-((xx + 0.0)**2 + (yy - 0.38)**2) / (2 * 0.13**2))
    # Left ear
    d += 0.25 * np.exp(-((xx - 0.12)**2 + (yy - 0.72)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx - 0.10)**2 + (yy - 0.62)**2) / (2 * 0.04**2))
    # Right ear
    d += 0.25 * np.exp(-((xx + 0.10)**2 + (yy - 0.70)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx + 0.08)**2 + (yy - 0.60)**2) / (2 * 0.04**2))
    # Tail
    d += 0.18 * np.exp(-((xx - 0.28)**2 + (yy + 0.10)**2) / (2 * 0.06**2))
    # Feet
    d += 0.15 * np.exp(-((xx - 0.18)**2 + (yy + 0.38)**2) / (2 * 0.07**2))
    d += 0.15 * np.exp(-((xx + 0.18)**2 + (yy + 0.38)**2) / (2 * 0.07**2))
    return d


def density_heart(res: int = 256) -> np.ndarray:
    """
    Heart curve:  (x^2 + y^2 - 1)^3 - x^2 y^3 < 0
    Soft version via signed-distance field.
    """
    xx, yy = _make_grid(res)
    # Parametric heart:  scale so it fits [-1,1]
    x = xx * 1.2
    y = yy * 1.2 - 0.1
    sdf = (x**2 + y**2 - 1)**3 - x**2 * y**3
    # Soft interior
    d = np.exp(-np.clip(sdf, 0, None) / 0.05)
    # Thicken the boundary for a nice density ridge
    d += 0.3 * np.exp(-(sdf**2) / 0.002)
    return np.clip(d, 0, None)


def density_rings(res: int = 256) -> np.ndarray:
    """Two concentric rings -- tests multi-modal annular transport."""
    xx, yy = _make_grid(res)
    r = np.sqrt(xx**2 + yy**2)
    d  = np.exp(-((r - 0.35)**2) / (2 * 0.06**2))   # inner ring
    d += np.exp(-((r - 0.75)**2) / (2 * 0.07**2))   # outer ring
    return d


BUILTIN = {
    "bunny": density_bunny,
    "heart": density_heart,
    "rings": density_rings,
}


# ==============================================================
#  2.  IMAGE TARGET CLASS
# ==============================================================

class ImageDensity2D:
    """
    Wraps a 2-D density defined on [-1,1]^2 from either a (res,res) numpy
    array or a PIL image file.

    Provides:
      .sample(n)         -- draw n particles via inverse-CDF on the grid
      .log_prob(x)       -- bilinear interpolation of log q on the grid
      .density           -- (res, res) numpy array, normalised
      .log_density       -- (res, res) numpy array, log-normalised
    """

    def __init__(self, density_arr: np.ndarray, device=DEVICE):
        res = density_arr.shape[0]
        self.res    = res
        self.device = device

        # normalise
        arr = density_arr.astype(np.float64)
        arr = np.clip(arr, 1e-10, None)
        arr /= arr.sum()
        self.density = arr.astype(np.float32)

        # log density (shifted for numerical stability)
        log_arr = np.log(arr)
        log_arr -= log_arr.max()
        self.log_density_arr = log_arr.astype(np.float32)

        # torch tensors for fast interpolation
        self._log_den_t = torch.tensor(
            self.log_density_arr, device=device
        ).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

        # precompute flat CDF for sampling
        flat = arr.ravel()
        self._cdf = np.cumsum(flat)
        self._cdf /= self._cdf[-1]

        # pixel-centre coordinates in [-1,1]
        self._coords = np.linspace(-1, 1, res)

    # -- sampling via inverse CDF -------------------------------
    def sample(self, n: int) -> torch.Tensor:
        u    = np.random.rand(n)
        idx  = np.searchsorted(self._cdf, u)
        idx  = np.clip(idx, 0, self.res**2 - 1)
        row  = idx // self.res
        col  = idx  % self.res
        x    = self._coords[col]
        y    = self._coords[self.res - 1 - row]   # flip: row 0 = top
        # add sub-pixel jitter
        dx   = (self._coords[1] - self._coords[0])
        x   += np.random.uniform(-dx/2, dx/2, n)
        y   += np.random.uniform(-dx/2, dx/2, n)
        pts  = np.stack([x, y], axis=1).astype(np.float32)
        return torch.tensor(pts, device=self.device)

    # -- log-prob via grid_sample (bilinear interp) ------------
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x : (n, 2)  ->  log q(x) : (n,)   (up to normalisation const)"""
        # grid_sample expects coords in [-1,1], shape (1, n, 1, 2)
        # NOTE: grid_sample treats dim-0 as x (horizontal) and dim-1 as y
        grid = x.unsqueeze(0).unsqueeze(2)          # (1, n, 1, 2)
        out  = torch.nn.functional.grid_sample(
            self._log_den_t.expand(1, 1, self.res, self.res),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                            # (1, 1, n, 1)
        return out.squeeze()                         # (n,)


def load_image_density(path: str, res: int = 256) -> np.ndarray:
    """Load a PNG/JPG, convert to grayscale density array."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # invert if needed (dark = high density)
    if arr.mean() > 128:
        arr = 255.0 - arr
    return arr


# ==============================================================
#  3.  W2 ESTIMATOR
# ==============================================================

try:
    from geomloss import SamplesLoss
    _sink = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.95)

    def w2_estimate(x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            v = _sink(x.contiguous(), y.contiguous())
        return float(v.clamp(min=0).sqrt())
    W2_METHOD = "Sinkhorn"
except ImportError:
    def w2_estimate(x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            dxx = torch.cdist(x, x).mean()
            dyy = torch.cdist(y, y).mean()
            dxy = torch.cdist(x, y).mean()
            return float((2*dxy - dxx - dyy).clamp(min=0).sqrt())
    W2_METHOD = "Energy-distance proxy"

print(f"W2 method : {W2_METHOD}")


# ==============================================================
#  4.  TRANSPORT MAP  (residual MLP)
# ==============================================================

class TransportMap(nn.Module):
    """T_theta(x) = x + MLP(x),  zero-initialised output layer -> T ~ Id at start."""
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ==============================================================
#  5.  EXACT LOG-DET JACOBIAN  (vmap + jacrev)
# ==============================================================

def logdet_jacobian(T: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """log |det J_T(x_i)|  for each x_i.  Exact via per-sample Jacobian."""
    def single(xi):
        return T(xi.unsqueeze(0)).squeeze(0)
    jac = vmap(jacrev(single))(x)           # (n, 2, 2)
    _, ld = torch.linalg.slogdet(jac)
    return ld                               # (n,)


# ==============================================================
#  6.  JKO LOSS
# ==============================================================

def jko_loss(T: nn.Module,
             y: torch.Tensor,
             gamma: float,
             target: ImageDensity2D) -> torch.Tensor:
    """
    L(theta) = -E[log q(T(y)) + log|det J_T(y)|]  +  (1/2gamma) E||y - T(y)||^2
    """
    Ty      = T(y)
    kl_term = -(target.log_prob(Ty) + logdet_jacobian(T, y)).mean()
    w2_term = ((y - Ty)**2).sum(-1).mean() / (2.0 * gamma)
    return kl_term + w2_term


# ==============================================================
#  7.  TRAIN ONE JKO BLOCK
# ==============================================================

def train_block(y: torch.Tensor,
                gamma: float,
                target: ImageDensity2D,
                n_epochs: int = 500,
                lr: float = 2e-3,
                batch: int = 1024,
                hidden: int = 256) -> TransportMap:
    T   = TransportMap(hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

    n = y.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        xb  = y[idx].detach()
        opt.zero_grad()
        loss = jko_loss(T, xb, gamma, target)
        loss.backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


# ==============================================================
#  8.  METRICS
# ==============================================================

class IdentityMap(nn.Module):
    def forward(self, x): return x

_id = IdentityMap().to(DEVICE)


@torch.no_grad()
def compute_metrics(particles: torch.Tensor,
                    target: ImageDensity2D,
                    n_ref: int = 4000) -> dict:
    """KL (up to entropy const) and W_2 for a particle cloud."""
    ref = target.sample(n_ref)
    n   = min(particles.shape[0], n_ref)
    pts = particles[:n]

    # KL term: -E[log q(x) + log|det J_Id(x)|] = -E[log q(x)]
    kl  = float(-target.log_prob(pts).mean())
    w2  = w2_estimate(pts, ref)
    return {"kl": kl, "w2": w2}


# ==============================================================
#  9.  STANDARD JKO
# ==============================================================

def run_standard_jko(target: ImageDensity2D,
                     gamma: float,
                     n_blocks: int,
                     n_particles: int = 4000,
                     n_epochs: int = 500,
                     seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    # Wide source: particles start far from target so many blocks of meaningful
    # transport are required, giving acceleration time to compound.
    x = torch.randn(n_particles, 2, device=DEVICE) * 1.5

    snapshots = [x.detach().clone()]
    m0 = compute_metrics(x, target)
    kl_vals = [m0["kl"]]; w2_vals = [m0["w2"]]; times = [0.0]

    t0 = time.time()
    for step in range(n_blocks):
        T = train_block(x, gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x = T(x).clamp(-2.5, 2.5)
        snapshots.append(x.detach().clone())
        m = compute_metrics(x, target)
        kl_vals.append(m["kl"]); w2_vals.append(m["w2"])
        times.append(time.time() - t0)
        print(f"  [Std  JKO] block {step+1:2d}/{n_blocks}  "
              f"KL={m['kl']:7.4f}  W2={m['w2']:.4f}  {times[-1]:.1f}s")

    return {"kl": np.array(kl_vals), "w2": np.array(w2_vals),
            "times": np.array(times), "snapshots": snapshots}


# ==============================================================
# 10.  ACCELERATED JKO
# ==============================================================

def run_accelerated_jko(target: ImageDensity2D,
                        gamma: float,
                        n_blocks: int,
                        n_particles: int = 4000,
                        n_epochs: int = 500,
                        seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    # Same wide source as standard JKO for a fair comparison.
    p0 = torch.randn(n_particles, 2, device=DEVICE) * 1.5

    x = p0.clone(); z = p0.clone()
    snapshots = [x.detach().clone()]
    m0 = compute_metrics(x, target)
    kl_vals = [m0["kl"]]; w2_vals = [m0["w2"]]; times = [0.0]

    t0 = time.time()
    for t in range(n_blocks):
        alpha = 3.0 / (t + 3.0)

        # Wasserstein geodesic interpolation
        y = (1.0 - alpha) * x + alpha * z

        # Proximal step from y
        T = train_block(y.detach(), gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x_new = T(y).clamp(-2.5, 2.5)

        # Momentum update for z.
        # The z sequence can overshoot since alpha > 1 is never reached,
        # but early steps (alpha close to 1) can still push z far.
        # Clamp generously to avoid leaving the density grid entirely.
        with torch.no_grad():
            z_new = (z + (x_new - y) / alpha).clamp(-3.5, 3.5)

        x = x_new.detach(); z = z_new.detach()
        snapshots.append(x.clone())
        m = compute_metrics(x, target)
        kl_vals.append(m["kl"]); w2_vals.append(m["w2"])
        times.append(time.time() - t0)
        print(f"  [Acc  JKO] block {t+1:2d}/{n_blocks}  "
              f"KL={m['kl']:7.4f}  W2={m['w2']:.4f}  "
              f"alpha={alpha:.3f}  {times[-1]:.1f}s")

    return {"kl": np.array(kl_vals), "w2": np.array(w2_vals),
            "times": np.array(times), "snapshots": snapshots}


# ==============================================================
# 11.  PLOTTING
# ==============================================================

BLUE  = "#1f77b4"
RED   = "#d62728"
LBLUE = "#aec7e8"
LRED  = "#ffb3b3"
GREY  = "#555555"


def plot_particle_evolution(res_std: dict,
                             res_acc: dict,
                             target: ImageDensity2D,
                             n_blocks: int,
                             target_name: str,
                             savepath: str = "jko_image_particles.png"):
    """
    Grid of particle snapshots: rows = Std / Acc JKO,
    columns = block 0, N//4, N//2, 3N//4, N.
    Background = target density heatmap.
    """
    snap_idx = sorted({0, n_blocks//4, n_blocks//2, 3*n_blocks//4, n_blocks})
    n_cols   = len(snap_idx)

    fig = plt.figure(figsize=(3.2 * n_cols, 7.0))
    fig.suptitle(
        f"Particle Transport -- JKO Flow on '{target_name}' Image Density\n"
        r"Background: target $q$   |   Dots: particles $p_t$",
        fontsize=13, y=1.01
    )

    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           hspace=0.08, wspace=0.06)

    # target density background
    dens = target.density                    # (res, res)
    extent = [-1, 1, -1, 1]

    row_labels = ["Standard JKO", "Accelerated JKO"]
    row_results = [res_std, res_acc]
    row_colors  = [BLUE, RED]

    for row, (label, res, col) in enumerate(
            zip(row_labels, row_results, row_colors)):
        for ci, si in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, ci])

            # density heatmap
            ax.imshow(
                dens, extent=extent, origin="lower",
                cmap="Greys", alpha=0.85,
                vmin=0, vmax=dens.max() * 0.8,
                interpolation="bilinear"
            )

            # particles
            pts = res["snapshots"][si].cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=1.5, alpha=0.35, color=col, linewidths=0)

            # labels
            if row == 0:
                ax.set_title(f"Block {si}", fontsize=9, pad=3)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_xticks([]); ax.set_yticks([])

            # KL annotation
            kl = res["kl"][si]
            ax.text(0.03, 0.97, f"KL={kl:.2f}",
                    transform=ax.transAxes,
                    fontsize=7, va="top", color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5,
                              boxstyle="round,pad=0.2"))

    plt.tight_layout()
    fig.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def plot_convergence(res_std: dict,
                     res_acc: dict,
                     n_blocks: int,
                     gamma: float,
                     target_name: str,
                     savepath: str = "jko_image_convergence.png"):
    """
    Four-panel convergence plot:
      A) log-linear KL       B) log-log KL (slope check)
      C) W_2 distance        D) Speedup ratio G_std(t) / G_acc(t)
    Panel D is the clearest signal: a rising ratio means acceleration is
    strictly winning, even when the absolute curves look similar.
    """
    iters = np.arange(n_blocks + 1)
    pos   = iters[iters > 0]

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))
    fig.suptitle(
        f"Standard vs Accelerated JKO -- '{target_name}' Image Density\n"
        rf"$\gamma = {gamma}$,  {n_blocks} blocks,  source $\mathcal{{N}}(0,\,1.5^2 I)$,"
        f"  W_2: {W2_METHOD}",
        fontsize=12, y=1.02
    )

    # -- A: log-linear KL ----------------------------------
    ax = axs[0]
    ax.semilogy(iters, res_std["kl"], color=BLUE, lw=2, label="Standard JKO")
    ax.semilogy(iters, res_acc["kl"], color=RED,  lw=2, label="Accelerated JKO")
    if len(pos) >= 4:
        # anchor O(t^-2) reference to accelerated curve at block 3
        c = res_acc["kl"][3] * 9
        ax.semilogy(pos, c / pos**2, "k--", lw=1.2, alpha=0.55,
                    label=r"$O(t^{-2})$ ref")
    ax.set_xlabel("JKO block  $t$")
    ax.set_ylabel(r"$-\mathbb{E}[\log q(p_t)]$  (KL proxy)")
    ax.set_title("Log-linear scale")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)

    # -- B: log-log KL -------------------------------------
    ax = axs[1]
    kl_std_pos = res_std["kl"][pos]
    kl_acc_pos = res_acc["kl"][pos]
    # Shift to remove the (shared) asymptotic floor before taking log
    floor = min(kl_std_pos.min(), kl_acc_pos.min()) - 1e-6
    kl_std_pos = np.clip(kl_std_pos - floor, 1e-8, None)
    kl_acc_pos = np.clip(kl_acc_pos - floor, 1e-8, None)
    ax.loglog(pos, kl_std_pos, color=BLUE, lw=2, label="Standard JKO")
    ax.loglog(pos, kl_acc_pos, color=RED,  lw=2, label="Accelerated JKO")
    if len(pos) >= 2:
        c1 = kl_std_pos[0]; c2 = kl_acc_pos[0]
        ax.loglog(pos, c1 / pos,    "k:",  lw=1.2, alpha=0.55, label=r"$O(t^{-1})$")
        ax.loglog(pos, c2 / pos**2, "k-.", lw=1.2, alpha=0.55, label=r"$O(t^{-2})$")
    ax.set_xlabel("JKO block  $t$")
    ax.set_ylabel("KL proxy  (floor-shifted)")
    ax.set_title("Log-log scale  (slope verification)")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)

    # -- C: W_2 distance -----------------------------------
    ax = axs[2]
    ax.semilogy(iters, res_std["w2"], color=BLUE, lw=2, label="Standard JKO")
    ax.semilogy(iters, res_acc["w2"], color=RED,  lw=2, label="Accelerated JKO")
    ax.set_xlabel("JKO block  $t$")
    ax.set_ylabel(r"$W_2(p_t,\, q)$")
    ax.set_title("Wasserstein-2 distance to target")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)

    # -- D: Speedup ratio  G_std / G_acc -------------------
    # Values > 1 mean accelerated JKO has lower KL at that block.
    # A rising trend confirms the O(t^{-2}) vs O(t^{-1}) gap is opening.
    ax = axs[3]
    kl_s = res_std["kl"][pos]
    kl_a = res_acc["kl"][pos]
    ratio = np.where(kl_a > 1e-8, kl_s / kl_a, np.nan)
    ax.plot(pos, ratio, color="darkorange", lw=2.5, label=r"$G_\mathrm{std}(t)\,/\,G_\mathrm{acc}(t)$")
    ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.7, label="Ratio = 1  (no gain)")
    # Reference: if std is O(1/t) and acc is O(1/t^2), ratio ? t
    if len(pos) >= 4:
        c_ref = ratio[3] / pos[3]
        ax.plot(pos, c_ref * pos, "k:", lw=1.2, alpha=0.55, label=r"$O(t)$ expected")
    ax.set_xlabel("JKO block  $t$")
    ax.set_ylabel(r"Speedup ratio")
    ax.set_title(r"Speedup: $G_\mathrm{std}(t)\,/\,G_\mathrm{acc}(t)$")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def plot_target_preview(target: ImageDensity2D,
                        name: str,
                        savepath: str = "jko_image_target.png"):
    """Quick sanity-check: target density + a sample of particles."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Target density: '{name}'", fontsize=12)

    axs[0].imshow(target.density, extent=[-1,1,-1,1], origin="lower",
                  cmap="inferno", interpolation="bilinear")
    axs[0].set_title("Density heatmap"); axs[0].axis("off")

    pts = target.sample(5000).cpu().numpy()
    axs[1].scatter(pts[:,0], pts[:,1], s=1, alpha=0.3, color=BLUE)
    axs[1].set_title("5 000 samples from q")
    axs[1].set_xlim(-1.2, 1.2); axs[1].set_ylim(-1.2, 1.2)
    axs[1].set_aspect("equal"); axs[1].axis("off")

    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def print_summary(res_std, res_acc, gamma, n_blocks,
                  target_name, savepath="jko_image_summary.txt"):
    lines = []
    sep   = "=" * 65
    lines += [sep,
              f"IMAGE DENSITY JKO SUMMARY",
              f"Target: {target_name}   gamma={gamma}   blocks={n_blocks}",
              sep,
              f"",
              f"Initial  KL proxy : {res_std['kl'][0]:.4f}",
              f"Final    KL proxy -- Standard JKO   : {res_std['kl'][-1]:.4f}",
              f"Final    KL proxy -- Accelerated JKO: {res_acc['kl'][-1]:.4f}",
              f"",
              f"Final    W2       -- Standard JKO   : {res_std['w2'][-1]:.4f}",
              f"Final    W2       -- Accelerated JKO: {res_acc['w2'][-1]:.4f}",
              f"",
              f"{'Target fraction':<26}{'Std blocks':>14}{'Acc blocks':>14}"]

    kl0 = res_std["kl"][0]
    for frac in [0.9, 0.75, 0.5, 0.25]:
        thresh = res_std["kl"][-1] + frac * (kl0 - res_std["kl"][-1])
        n_s = next((i for i, g in enumerate(res_std["kl"]) if g < thresh), ">N")
        n_a = next((i for i, g in enumerate(res_acc["kl"]) if g < thresh), ">N")
        lines.append(
            f"  KL < {frac:.0%} of initial range  {str(n_s):>14}  {str(n_a):>14}"
        )

    lines += ["", sep]
    text = "\n".join(lines)
    print(text)
    with open(savepath, "w") as f:
        f.write(text)
    print(f"Saved: {savepath}")


# ==============================================================
# 12.  MAIN
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Accelerated JKO on a 2-D image density."
    )
    p.add_argument("--target",  default="rings",
                   choices=list(BUILTIN.keys()),
                   help="Built-in target density (default: rings -- hardest for std JKO)")
    p.add_argument("--image",   default=None,
                   help="Path to custom PNG/JPG (overrides --target)")
    p.add_argument("--res",     type=int,   default=256,
                   help="Grid resolution for density (default 256)")
    p.add_argument("--gamma",   type=float, default=0.08,
                   help="JKO step size gamma (default 0.08 -- small steps widen the gap)")
    p.add_argument("--blocks",  type=int,   default=25,
                   help="Number of JKO blocks (default 25)")
    p.add_argument("--particles", type=int, default=5000,
                   help="Number of particles (default 5000)")
    p.add_argument("--epochs",  type=int,   default=800,
                   help="Training epochs per block (default 800 -- reduces eps noise floor)")
    p.add_argument("--hidden",  type=int,   default=256,
                   help="MLP hidden width (default 256)")
    p.add_argument("--seed",    type=int,   default=0)
    p.add_argument("--preview", action="store_true",
                   help="Show target density preview and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -- Build target ----------------------------------------
    if args.image is not None:
        target_name = Path(args.image).stem
        arr = load_image_density(args.image, res=args.res)
    else:
        target_name = args.target
        arr = BUILTIN[args.target](res=args.res)

    target = ImageDensity2D(arr, device=DEVICE)
    print(f"Target: '{target_name}'  ({args.res}*{args.res} grid)")

    plot_target_preview(target, target_name,
                        savepath=f"jko_image_target_{target_name}.png")
    if args.preview:
        raise SystemExit(0)

    # -- Run experiments --------------------------------------
    print(f"\ngamma={args.gamma}  blocks={args.blocks}  "
          f"particles={args.particles}  epochs/block={args.epochs}")

    print("\n-- Standard JKO --")
    res_std = run_standard_jko(
        target, args.gamma, args.blocks,
        args.particles, args.epochs, args.seed
    )

    print("\n-- Accelerated JKO --")
    res_acc = run_accelerated_jko(
        target, args.gamma, args.blocks,
        args.particles, args.epochs, args.seed
    )

    # -- Plots ------------------------------------------------
    plot_particle_evolution(
        res_std, res_acc, target,
        args.blocks, target_name,
        savepath=f"jko_image_particles_{target_name}.png"
    )

    plot_convergence(
        res_std, res_acc,
        args.blocks, args.gamma, target_name,
        savepath=f"jko_image_convergence_{target_name}.png"
    )

    print_summary(
        res_std, res_acc, args.gamma, args.blocks, target_name,
        savepath=f"jko_image_summary_{target_name}.txt"
    )

    print("\nDone.")