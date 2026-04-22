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
from geomloss import SamplesLoss

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 10})
os.makedirs("images", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BLUE = "#1f77b4"
RED  = "#d62728"

#  Built-in target densities
def _grid(res=256):
    lin = np.linspace(-1, 1, res)
    return np.meshgrid(lin, lin)


def density_bunny(res=256):
    xx, yy = _grid(res)
    d = np.zeros((res, res))
    d += np.exp(-((xx)**2       + (yy + 0.10)**2) / (2 * 0.18**2))
    d += 0.60 * np.exp(-((xx)**2       + (yy - 0.38)**2) / (2 * 0.13**2))
    d += 0.25 * np.exp(-((xx - 0.12)**2 + (yy - 0.72)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx - 0.10)**2 + (yy - 0.62)**2) / (2 * 0.04**2))
    d += 0.25 * np.exp(-((xx + 0.10)**2 + (yy - 0.70)**2) / (2 * 0.05**2))
    d += 0.20 * np.exp(-((xx + 0.08)**2 + (yy - 0.60)**2) / (2 * 0.04**2))
    d += 0.18 * np.exp(-((xx - 0.28)**2 + (yy + 0.10)**2) / (2 * 0.06**2))
    d += 0.15 * np.exp(-((xx - 0.18)**2 + (yy + 0.38)**2) / (2 * 0.07**2))
    d += 0.15 * np.exp(-((xx + 0.18)**2 + (yy + 0.38)**2) / (2 * 0.07**2))
    return d


def density_rings(res=256):
    xx, yy = _grid(res)
    r = np.sqrt(xx**2 + yy**2)
    d  = np.exp(-((r - 0.35)**2) / (2 * 0.06**2))
    d += np.exp(-((r - 0.75)**2) / (2 * 0.07**2))
    return d


def density_rectangle(res=256):
    xx, yy = _grid(res)
    a, b = 0.55, 0.35
    n = 4
    return np.exp(-((xx / a) ** (2 * n) + (yy / b) ** (2 * n)))


def density_disk(res=256):
    xx, yy = _grid(res)
    r = np.sqrt(xx ** 2 + yy ** 2)
    R = 0.70
    n = 4
    return np.exp(-((r / R) ** (2 * n)))


def density_outer_ring(res=256):
    xx, yy = _grid(res)
    r = np.sqrt(xx ** 2 + yy ** 2)
    return np.exp(-((r - 0.75) ** 2) / (2 * 0.07 ** 2))


BUILTIN = {
    "bunny":      density_bunny,
    "rings":      density_rings,
    "rectangle":  density_rectangle,
    "disk":       density_disk,
    "outer_ring": density_outer_ring,
}

#  Image density class
class ImageDensity2D:
    def __init__(self, arr, device=DEVICE):
        res = arr.shape[0]
        self.res = res
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
        return torch.tensor(np.stack([x, y], 1).astype(np.float32), device=self.device)

    def log_prob(self, x):
        grid = x.unsqueeze(0).unsqueeze(2)
        out  = torch.nn.functional.grid_sample(
            self._log_den_t.expand(1, 1, self.res, self.res),
            grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return out.squeeze()

#  Transport map
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
    Ty = T(y)
    kl = -(target.log_prob(Ty) + logdet_jacobian(T, y)).mean()
    w2 = ((y - Ty)**2).sum(-1).mean() / (2.0 * gamma)
    return kl + w2


def train_block(y, gamma, target, n_epochs=800, lr=2e-3, batch=1024, hidden=256):
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

#  Sinkhorn W_2 estimator (geomloss)
_SINK = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.95)


def w2_to_target(x, y_ref):
    """Sinkhorn-based W_2 between particles x and a fixed reference sample from q."""
    with torch.no_grad():
        if x.shape[0] > y_ref.shape[0]:
            idx = torch.randperm(x.shape[0], device=x.device)[:y_ref.shape[0]]
            x_ = x[idx]
        else:
            x_ = x
        val = _SINK(x_.contiguous(), y_ref.contiguous())
    return float(val.clamp(min=0).sqrt())

#  Run standard JKO
def run_standard_jko(target, gamma, n_blocks, n_particles, n_epochs, seed=0, y_ref=None):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    snaps = [x.detach().clone()]
    w2s   = [w2_to_target(x, y_ref)]
    t0 = time.time()
    for k in range(n_blocks):
        T = train_block(x, gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x = T(x).clamp(-2.5, 2.5)
        snaps.append(x.detach().clone())
        w2s.append(w2_to_target(x, y_ref))
        print(f"  [Std  {k+1:2d}/{n_blocks}]  W2={w2s[-1]:.4f}  {time.time()-t0:.0f}s")
    return snaps, w2s

#  Run accelerated JKO

def run_accelerated_jko(target, gamma, n_blocks, n_particles, n_epochs, seed=0, y_ref=None):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    z = x.clone()
    snaps = [x.detach().clone()]
    w2s   = [w2_to_target(x, y_ref)]
    t0 = time.time()
    for t in range(n_blocks):
        alpha = 3.0 / (t + 3.0)
        y     = (1.0 - alpha) * x + alpha * z
        T     = train_block(y.detach(), gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x_new = T(y).clamp(-2.5, 2.5)
            z_new = (z + (x_new - y) / alpha).clamp(-3.5, 3.5)
        x = x_new.detach(); z = z_new.detach()
        snaps.append(x.clone())
        w2s.append(w2_to_target(x, y_ref))
        print(f"  [Acc  {t+1:2d}/{n_blocks}]  alpha={alpha:.3f}  W2={w2s[-1]:.4f}  {time.time()-t0:.0f}s")
    return snaps, w2s

#  Plot
def plot_particles(snaps_std, snaps_acc, w2_std, w2_acc, target, n_blocks, savepath):
    snap_idx = sorted({0, n_blocks//4, n_blocks//2, 3*n_blocks//4, n_blocks})
    n_cols   = len(snap_idx)
    dens     = target.density

    fig = plt.figure(figsize=(3.0 * n_cols, 8.0))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig,
                            height_ratios=[1, 1, 0.55],
                            hspace=0.12, wspace=0.05)

    rows = [("Standard JKO",    snaps_std, w2_std, BLUE),
            ("Accelerated JKO", snaps_acc, w2_acc, RED)]

    for row, (label, snaps, w2s, color) in enumerate(rows):
        for ci, si in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, ci])
            ax.imshow(dens, extent=[-1, 1, -1, 1], origin="lower",
                      cmap="Greys", alpha=0.7,
                      vmin=0, vmax=dens.max() * 1.5,
                      interpolation="bilinear")
            pts = snaps[si].cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=4, alpha=0.25, color=color, linewidths=0)
            ax.text(0.97, 0.97, f"$W_2$ = {w2s[si]:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8,
                              edgecolor="none", pad=1.8))
            if row == 0:
                ax.set_title(f"Block {si}", fontsize=9, pad=3)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_xticks([]); ax.set_yticks([])

    ax_c = fig.add_subplot(gs[2, :])
    t = np.arange(n_blocks + 1)
    ax_c.semilogy(t, w2_std, color=BLUE, lw=1.8, label="Standard JKO")
    ax_c.semilogy(t, w2_acc, color=RED,  lw=1.8, label="Accelerated JKO")
    for si in snap_idx:
        ax_c.axvline(si, color="gray", ls=":", alpha=0.35, lw=0.8)
    ax_c.set_xlabel("Block $t$", fontsize=10)
    ax_c.set_ylabel(r"$W_2(\rho_t,\, q)$", fontsize=10)
    ax_c.legend(fontsize=9, loc="upper right")
    ax_c.grid(True, which="both", ls=":", alpha=0.35)
    ax_c.set_xlim(0, n_blocks)

    plt.tight_layout()
    fig.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"Saved: {savepath}")

#  Main
def run(target_name, gamma=0.08, n_blocks=25, n_particles=12000,
        n_epochs=800, seed=0):
    arr    = BUILTIN[target_name]()
    target = ImageDensity2D(arr)
    print(f"\n=== {target_name} ===")

    np.random.seed(seed + 777)
    y_ref = target.sample(4096)

    print("Standard JKO...")
    snaps_std, w2_std = run_standard_jko(
        target, gamma, n_blocks, n_particles, n_epochs, seed, y_ref=y_ref)

    print("Accelerated JKO...")
    snaps_acc, w2_acc = run_accelerated_jko(
        target, gamma, n_blocks, n_particles, n_epochs, seed, y_ref=y_ref)

    plot_particles(snaps_std, snaps_acc, w2_std, w2_acc, target, n_blocks,
                   savepath=f"images/particles_{target_name}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target",    default="all",
                   choices=["bunny", "rings", "rectangle", "disk", "outer_ring", "all"])
    p.add_argument("--gamma",     type=float, default=0.04)
    p.add_argument("--blocks",    type=int,   default=25)
    p.add_argument("--particles", type=int,   default=12000)
    p.add_argument("--epochs",    type=int,   default=800)
    p.add_argument("--seed",      type=int,   default=0)
    args = p.parse_args()

    all_targets = ["bunny", "rings", "rectangle", "disk", "outer_ring"]
    targets = all_targets if args.target == "all" else [args.target]
    for t in targets:
        run(t, args.gamma, args.blocks, args.particles, args.epochs, args.seed)

