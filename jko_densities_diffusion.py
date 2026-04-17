"""
OU-Scheduled JKO Particle Visualization
========================================
Instead of targeting the fixed density q at every block, each block k targets
the OU-noised version q_k, defined by running k OU steps BACKWARD from the
original target:

  q_N = original target density
  q_{N-1} = one OU step applied to q_N (slightly blurred)
  ...
  q_0 = N OU steps applied (heavily blurred, near-Gaussian)

So block k minimises KL(rho || q_k), with the target progressively sharpening.
This gives a curriculum: early blocks solve easy transport problems, later blocks
refine toward the true target.

OU noising on the image grid: each step convolves the density with a Gaussian
kernel of width sigma_step (scipy.ndimage.gaussian_filter).

Figure layout: 3 rows x 5 columns
  Row 0: intermediate target q_k at each snapshot block (density heatmap)
  Row 1: Standard JKO particles at each snapshot block
  Row 2: Accelerated JKO particles at each snapshot block

Columns: blocks 0, N//4, N//2, 3N//4, N.

Usage:
  python jko_densities_diffusion.py --target rings
  python jko_densities_diffusion.py --target bunny
  python jko_densities_diffusion.py --target rings --blocks 20 --epochs 400
"""

import argparse
import os
import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 10})
os.makedirs("images", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BLUE = "#1f77b4"
RED  = "#d62728"

# ------------------------------------------------------------------
#  Built-in target densities
# ------------------------------------------------------------------

def _grid(res=256):
    lin = np.linspace(-1, 1, res)
    return np.meshgrid(lin, lin)


def density_bunny(res=256):
    xx, yy = _grid(res)
    d = np.zeros((res, res))
    d += np.exp(-((xx)**2         + (yy + 0.10)**2) / (2 * 0.18**2))
    d += 0.60 * np.exp(-((xx)**2         + (yy - 0.38)**2) / (2 * 0.13**2))
    d += 0.25 * np.exp(-((xx - 0.12)**2  + (yy - 0.72)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx - 0.10)**2  + (yy - 0.62)**2) / (2 * 0.04**2))
    d += 0.25 * np.exp(-((xx + 0.10)**2  + (yy - 0.70)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx + 0.08)**2  + (yy - 0.60)**2) / (2 * 0.04**2))
    d += 0.18 * np.exp(-((xx - 0.28)**2  + (yy + 0.10)**2) / (2 * 0.06**2))
    d += 0.15 * np.exp(-((xx - 0.18)**2  + (yy + 0.38)**2) / (2 * 0.07**2))
    d += 0.15 * np.exp(-((xx + 0.18)**2  + (yy + 0.38)**2) / (2 * 0.07**2))
    return d


def density_rings(res=256):
    xx, yy = _grid(res)
    r = np.sqrt(xx**2 + yy**2)
    d  = np.exp(-((r - 0.35)**2) / (2 * 0.06**2))
    d += np.exp(-((r - 0.75)**2) / (2 * 0.07**2))
    return d


BUILTIN = {"bunny": density_bunny, "rings": density_rings}

# ------------------------------------------------------------------
#  OU forward process on the density grid
#  OU step: convolve with Gaussian kernel (blurring toward uniform)
#  We also mix slightly toward the uniform density to model drift.
# ------------------------------------------------------------------

def build_ou_schedule(density_arr, n_steps, sigma_per_step=1.5, mix_per_step=0.04):
    """
    Build a sequence of n_steps+1 densities:
      schedule[n_steps] = original (sharpest)
      schedule[0]       = most blurred / mixed toward uniform

    Each step applies:
      1. Gaussian blur with sigma_per_step (in pixel units)
      2. Mix (1-mix_per_step) * blurred + mix_per_step * uniform

    Returns list of length n_steps+1, index k = density at block k.
    """
    res  = density_arr.shape[0]
    flat = np.ones((res, res), dtype=np.float32) / (res * res)

    schedule = [None] * (n_steps + 1)
    schedule[n_steps] = density_arr.astype(np.float32)

    current = density_arr.astype(np.float32).copy()
    for step in range(n_steps - 1, -1, -1):
        blurred  = gaussian_filter(current, sigma=sigma_per_step)
        blurred  = np.clip(blurred, 1e-10, None)
        blurred /= blurred.sum()
        mixed    = (1.0 - mix_per_step) * blurred + mix_per_step * flat
        mixed   /= mixed.sum()
        schedule[step] = mixed.copy()
        current  = mixed

    return schedule   # schedule[k] is the target for block k


# ------------------------------------------------------------------
#  Image density wrapper (same as before)
# ------------------------------------------------------------------

class ImageDensity2D:
    def __init__(self, arr, device=DEVICE):
        res = arr.shape[0]
        self.res    = res
        self.device = device
        arr = np.clip(arr.astype(np.float64), 1e-10, None)
        arr /= arr.sum()
        self.density = arr.astype(np.float32)
        log_arr = np.log(arr); log_arr -= log_arr.max()
        self._log_den_t = torch.tensor(
            log_arr.astype(np.float32), device=device
        ).unsqueeze(0).unsqueeze(0)
        flat = arr.ravel()
        self._cdf = np.cumsum(flat); self._cdf /= self._cdf[-1]
        self._coords = np.linspace(-1, 1, res)

    def sample(self, n):
        u   = np.random.rand(n)
        idx = np.clip(np.searchsorted(self._cdf, u), 0, self.res**2 - 1)
        row = idx // self.res; col = idx % self.res
        x   = self._coords[col]
        y   = self._coords[self.res - 1 - row]
        dx  = self._coords[1] - self._coords[0]
        x  += np.random.uniform(-dx/2, dx/2, n)
        y  += np.random.uniform(-dx/2, dx/2, n)
        return torch.tensor(
            np.stack([x, y], 1).astype(np.float32), device=self.device
        )

    def log_prob(self, x):
        grid = x.unsqueeze(0).unsqueeze(2)
        out  = torch.nn.functional.grid_sample(
            self._log_den_t.expand(1, 1, self.res, self.res),
            grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return out.squeeze()


# ------------------------------------------------------------------
#  Transport map (residual MLP)
# ------------------------------------------------------------------

class TransportMap(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def logdet_jacobian(T, x):
    def single(xi): return T(xi.unsqueeze(0)).squeeze(0)
    jac = vmap(jacrev(single))(x)
    _, ld = torch.linalg.slogdet(jac)
    return ld


def jko_loss(T, y, gamma, target):
    Ty  = T(y)
    kl  = -(target.log_prob(Ty) + logdet_jacobian(T, y)).mean()
    w2  = ((y - Ty)**2).sum(-1).mean() / (2.0 * gamma)
    return kl + w2


def train_block(y, gamma, target, n_epochs=400, lr=2e-3, batch=1024, hidden=256):
    T   = TransportMap(hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)
    n   = y.shape[0]
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        opt.zero_grad()
        jko_loss(T, y[idx].detach(), gamma, target).backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step(); sch.step()
    return T


# ------------------------------------------------------------------
#  OU-scheduled standard JKO
# ------------------------------------------------------------------

def run_standard_jko_ou(schedule, gamma, n_particles, n_epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n_blocks = len(schedule) - 1
    x        = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    snaps    = [x.detach().clone()]
    t0       = time.time()
    for k in range(n_blocks):
        target = ImageDensity2D(schedule[k + 1])   # target for this block
        T      = train_block(x, gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x = T(x).clamp(-2.5, 2.5)
        snaps.append(x.detach().clone())
        print(f"  [Std  {k+1:2d}/{n_blocks}]  {time.time()-t0:.0f}s")
    return snaps


# ------------------------------------------------------------------
#  OU-scheduled accelerated JKO
# ------------------------------------------------------------------

def run_accelerated_jko_ou(schedule, gamma, n_particles, n_epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n_blocks = len(schedule) - 1
    x        = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    z        = x.clone()
    snaps    = [x.detach().clone()]
    t0       = time.time()
    for t in range(n_blocks):
        alpha  = 3.0 / (t + 3.0)
        y      = (1.0 - alpha) * x + alpha * z
        target = ImageDensity2D(schedule[t + 1])   # target for this block
        T      = train_block(y.detach(), gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x_new = T(y).clamp(-2.5, 2.5)
            z_new = (z + (x_new - y) / alpha).clamp(-3.5, 3.5)
        x = x_new.detach(); z = z_new.detach()
        snaps.append(x.clone())
        print(f"  [Acc  {t+1:2d}/{n_blocks}]  alpha={alpha:.3f}  {time.time()-t0:.0f}s")
    return snaps


# ------------------------------------------------------------------
#  Plot: 3 rows x 5 columns
#  Row 0: intermediate OU target heatmap
#  Row 1: Standard JKO particles
#  Row 2: Accelerated JKO particles
# ------------------------------------------------------------------

def plot_ou_particles(snaps_std, snaps_acc, schedule, n_blocks,
                       savepath, n_ref=10000):
    snap_idx = sorted({0, n_blocks//4, n_blocks//2, 3*n_blocks//4, n_blocks})
    n_cols   = len(snap_idx)
 
    # Pre-sample particles from each OU target
    ref_samples = {}
    for si in snap_idx:
        ref_samples[si] = ImageDensity2D(schedule[si]).sample(n_ref).cpu().numpy()
 
    fig = plt.figure(figsize=(3.0 * n_cols, 8.5))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.05, wspace=0.05)
 
    GREY_PT = "#888888"
    rows = [
        ("Forward process $q_k$", None,      GREY_PT),
        ("Standard JKO",           snaps_std, BLUE),
        ("Accelerated JKO",        snaps_acc, RED),
    ]
 
    for row_idx, (label, snaps, color) in enumerate(rows):
        for ci, si in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row_idx, ci])
            ax.set_facecolor("white")
 
            if row_idx == 0:
                pts = ref_samples[si]
            else:
                pts = snaps[si].cpu().numpy()
 
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=3, alpha=0.2, color=color, linewidths=0)
 
            if row_idx == 0:
                ax.set_title(f"Block {si}", fontsize=9, pad=3)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
            lim = 1.0 if row_idx == 0 else 1.5
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([])
 
    plt.tight_layout()
    fig.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"Saved: {savepath}")


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def run(target_name, gamma=0.08, n_blocks=20, n_particles=10000,
        n_epochs=400, sigma_per_step=1.5, mix_per_step=0.04, seed=0):

    arr      = BUILTIN[target_name]()
    arr      = arr / arr.sum()
    schedule = build_ou_schedule(arr, n_blocks,
                                  sigma_per_step=sigma_per_step,
                                  mix_per_step=mix_per_step)

    print(f"\n=== {target_name}  ({n_blocks} blocks, {n_epochs} epochs/block) ===")
    print(f"    sigma_per_step={sigma_per_step}  mix_per_step={mix_per_step}")

    print("\nStandard JKO (OU-scheduled)...")
    snaps_std = run_standard_jko_ou(schedule, gamma, n_particles, n_epochs, seed)

    print("\nAccelerated JKO (OU-scheduled)...")
    snaps_acc = run_accelerated_jko_ou(schedule, gamma, n_particles, n_epochs, seed)

    savepath  = f"images/ou_particles_{target_name}.png"
    plot_ou_particles(snaps_std, snaps_acc, schedule, n_blocks, savepath)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target",         default="rings",
                   choices=list(BUILTIN.keys()))
    p.add_argument("--gamma",          type=float, default=0.08)
    p.add_argument("--blocks",         type=int,   default=20)
    p.add_argument("--particles",      type=int,   default=15000)
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--sigma_per_step", type=float, default=1.5,
                   help="Gaussian blur sigma per OU step (pixel units)")
    p.add_argument("--mix_per_step",   type=float, default=0.05,
                   help="Mixing weight toward uniform per OU step")
    p.add_argument("--seed",           type=int,   default=0)
    args = p.parse_args()

    run(args.target, args.gamma, args.blocks, args.particles,
        args.epochs, args.sigma_per_step, args.mix_per_step, args.seed)

    print("\nDone.")