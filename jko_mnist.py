"""
Neural Accelerated JKO -- MNIST Generative Flow  (v4 - DDPM score oracle)
==========================================================================

Uses a pretrained DDPM (MNISTDiffusion) as a frozen score oracle instead of
training a score network from scratch.  The UNet noise prediction is converted
to a score via:
    grad log p_t(x) = -eps_theta(x, t) / sqrt(1 - alpha_bar_t)

Compares per JKO block:
  * Standard JKO          x_{n+1} = Prox(G, x_n, gamma)
  * Accelerated JKO       three-sequence scheme, alpha_t = 3/(t+3)

Outputs
-------
  images/checkpoint_verification.png   -- DDPM sampler sanity check
  images/mnist_jko_samples.png         -- sample grids at key blocks
  images/mnist_jko_convergence.png     -- FID & KL proxy curves
  results/mnist_jko_summary.txt        -- numerical table

Usage
-----
    python jko_mnist.py                          # defaults
    python jko_mnist.py --gamma 0.1 --blocks 15
    python jko_mnist.py --verify-only            # check DDPM then exit
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# --------------- path setup ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MNISTDiffusion"))
from model import MNISTDiffusion

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 10})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
DIM = 784  # 28 * 28


# ==============================================================
#  1.  DATA
# ==============================================================

def load_mnist(train: bool = True) -> torch.Tensor:
    """Load MNIST as flat (N, 784) tensors scaled to [-1, 1]."""
    ds = datasets.MNIST(root="./data", train=train, download=True,
                        transform=transforms.ToTensor())
    imgs = ds.data.float() / 255.0 * 2.0 - 1.0   # [-1, 1]
    return imgs.view(-1, DIM).to(DEVICE)


# ==============================================================
#  2.  DDPM SCORE ORACLE
# ==============================================================

def load_ddpm(ckpt_path: str) -> MNISTDiffusion:
    """Load the pretrained DDPM and freeze it."""
    ddpm = MNISTDiffusion(
        image_size=28,
        in_channels=1,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=64,
        dim_mults=[2, 4],
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # Checkpoint is {"model": state_dict, "model_ema": ema_state_dict}
    ddpm.load_state_dict(ckpt["model"])
    ddpm.eval()
    for p in ddpm.parameters():
        p.requires_grad_(False)
    print(f"Loaded DDPM from {ckpt_path}")
    return ddpm


@torch.no_grad()
def score_at_t(ddpm: MNISTDiffusion, x: torch.Tensor,
               t: int) -> torch.Tensor:
    """
    Returns grad log p_t(x) using the pretrained DDPM.

    The DDPM noise prediction eps_theta(x_t, t) relates to the score as:
        grad log p_t(x_t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)

    x : (B, 1, 28, 28) images in [-1, 1]
    t : single integer timestep in [0, 999]
    returns: (B, 1, 28, 28) score estimate
    """
    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    eps_pred = ddpm.model(x, t_batch)
    sqrt_one_minus_ab = ddpm.sqrt_one_minus_alphas_cumprod[t]
    return -eps_pred / sqrt_one_minus_ab


def verify_ddpm(ddpm: MNISTDiffusion, n_samples: int = 16,
                savepath: str = "images/checkpoint_verification.png"):
    """Run the built-in DDPM sampler to verify the checkpoint works."""
    print("\nVerifying DDPM checkpoint via built-in sampler...")
    samples = ddpm.sampling(n_samples=n_samples, device=DEVICE)
    # samples: (N, 1, 28, 28) in [0, 1]
    nrow = int(n_samples ** 0.5)
    fig, axes = plt.subplots(nrow, nrow, figsize=(6, 6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(samples[i, 0].cpu(), cmap="gray")
        ax.axis("off")
    plt.suptitle("DDPM checkpoint verification\n(should show clean digits)")
    plt.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"  Saved: {savepath}")
    print("  ** Check this image before proceeding **")
    plt.show()


# ==============================================================
#  3.  TRANSPORT MAP
# ==============================================================

class TransportMap(nn.Module):
    """T(x) = x + f(x), zero-init output, LayerNorm for stability."""
    def __init__(self, dim: int = DIM, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ==============================================================
#  4.  JKO LOSS
# ==============================================================

def jko_loss(T: nn.Module,
             y: torch.Tensor,
             gamma: float,
             ddpm: MNISTDiffusion,
             t_eval: int,
             score_clip: float = 50.0) -> torch.Tensor:
    """
    L(T) = -E[ s_clip(T(y))^T T(y) ]  +  (1/2*gamma) * E[ ||y - T(y)||^2 ]

    Score is detached (DDPM is frozen). Per-sample L2 norm is clipped to
    bound the KL term and prevent off-manifold drift.

    T      : TransportMap on flat (B, 784) vectors
    y      : (B, 784) current particles, flat
    ddpm   : pretrained MNISTDiffusion (frozen)
    t_eval : DDPM timestep for score evaluation
    """
    Ty = T(y)                                           # (B, 784)

    # Reshape to (B, 1, 28, 28) for UNet, get score, flatten back
    Ty_img = Ty.view(-1, 1, 28, 28)
    score = score_at_t(ddpm, Ty_img, t_eval)            # (B, 1, 28, 28)
    score = score.view(-1, DIM).detach()                # (B, 784)

    # Clip per-sample score norm
    norm = score.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    score = score * (score_clip / norm).clamp(max=1.0)

    kl_term = -(score * Ty).sum(-1).mean()
    w2_term = ((y - Ty) ** 2).sum(-1).mean() / (2.0 * gamma)

    return kl_term + w2_term


# ==============================================================
#  5.  TRAIN ONE JKO BLOCK
# ==============================================================

def train_block(y: torch.Tensor,
                gamma: float,
                ddpm: MNISTDiffusion,
                t_eval: int,
                score_clip: float,
                n_epochs: int = 300,
                lr: float = 1e-3,
                batch: int = 512,
                hidden: int = 512) -> TransportMap:
    T = TransportMap(hidden=hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs,
                                                eta_min=1e-4)
    n = y.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        xb = y[idx].detach()
        opt.zero_grad()
        loss = jko_loss(T, xb, gamma, ddpm, t_eval, score_clip)
        loss.backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


# ==============================================================
#  6.  METRICS
# ==============================================================

def compute_fid(generated_flat: torch.Tensor,
                real_flat: torch.Tensor,
                fid_metric: FrechetInceptionDistance) -> float:
    """FID via torchmetrics (InceptionV3 features)."""
    def to_rgb_uint8(flat):
        imgs = flat.view(-1, 1, 28, 28).clamp(-1, 1)
        imgs = ((imgs + 1) / 2 * 255).byte()
        return imgs.repeat(1, 3, 1, 1)  # grayscale -> 3ch

    fid_metric.reset()
    # Process in chunks to avoid OOM
    chunk = 512
    real_rgb = to_rgb_uint8(real_flat)
    gen_rgb = to_rgb_uint8(generated_flat)
    for i in range(0, real_rgb.shape[0], chunk):
        fid_metric.update(real_rgb[i:i+chunk], real=True)
    for i in range(0, gen_rgb.shape[0], chunk):
        fid_metric.update(gen_rgb[i:i+chunk], real=False)
    return float(fid_metric.compute())


# ==============================================================
#  7.  JKO LOOPS
# ==============================================================

def run_standard_jko(ddpm, real_data, fid_metric,
                     gamma, n_blocks, t_eval, score_clip,
                     n_particles=2000, n_epochs=300, hidden=512,
                     seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.randn(n_particles, DIM, device=DEVICE)

    snapshots = [x.detach().clone()]
    fid_vals = [compute_fid(x, real_data, fid_metric)]
    times = [0.0]

    t0 = time.time()
    for step in range(n_blocks):
        T = train_block(x, gamma, ddpm, t_eval, score_clip,
                        n_epochs=n_epochs, hidden=hidden)
        with torch.no_grad():
            x = T(x).clamp(-1.5, 1.5)
        snapshots.append(x.detach().clone())
        fid_vals.append(compute_fid(x, real_data, fid_metric))
        times.append(time.time() - t0)
        print(f"  [Std  JKO] block {step+1:2d}/{n_blocks}  "
              f"FID={fid_vals[-1]:.1f}  {times[-1]:.1f}s")

    return {"fid": np.array(fid_vals), "times": np.array(times),
            "snapshots": snapshots}


def run_accelerated_jko(ddpm, real_data, fid_metric,
                        gamma, n_blocks, t_eval, score_clip,
                        n_particles=2000, n_epochs=300, hidden=512,
                        seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    p0 = torch.randn(n_particles, DIM, device=DEVICE)
    x = p0.clone(); z = p0.clone()

    snapshots = [x.detach().clone()]
    fid_vals = [compute_fid(x, real_data, fid_metric)]
    times = [0.0]

    t0 = time.time()
    for t in range(n_blocks):
        alpha = 3.0 / (t + 3.0)
        y = (1.0 - alpha) * x + alpha * z

        T = train_block(y.detach(), gamma, ddpm, t_eval, score_clip,
                        n_epochs=n_epochs, hidden=hidden)
        with torch.no_grad():
            x_new = T(y).clamp(-1.5, 1.5)
            z_new = (z + (x_new - y) / alpha).clamp(-3.0, 3.0)

        x = x_new.detach(); z = z_new.detach()
        snapshots.append(x.clone())
        fid_vals.append(compute_fid(x, real_data, fid_metric))
        times.append(time.time() - t0)
        print(f"  [Acc  JKO] block {t+1:2d}/{n_blocks}  "
              f"FID={fid_vals[-1]:.1f}  alpha={alpha:.3f}  "
              f"{times[-1]:.1f}s")

    return {"fid": np.array(fid_vals), "times": np.array(times),
            "snapshots": snapshots}


# ==============================================================
#  8.  PLOTTING
# ==============================================================

BLUE = "#1f77b4"
RED  = "#d62728"


def _make_grid(flat, nrow=4):
    n = min(flat.shape[0], nrow * nrow)
    imgs = flat[:n].detach().cpu().numpy().reshape(-1, 28, 28)
    nr = (n + nrow - 1) // nrow
    grid = np.ones((nr * 29 - 1, nrow * 29 - 1)) * -1.0
    for i in range(n):
        r, c = i // nrow, i % nrow
        grid[r*29:r*29+28, c*29:c*29+28] = imgs[i]
    return grid


def plot_sample_grids(res_std, res_acc, n_blocks,
                      savepath="images/mnist_jko_samples.png"):
    """4x4 sample grids at evenly spaced blocks for both methods."""
    n_show = min(7, n_blocks + 1)
    idxs = sorted({int(round(i * n_blocks / (n_show - 1)))
                    for i in range(n_show)})
    fig, axes = plt.subplots(2, len(idxs),
                             figsize=(3.5 * len(idxs), 8))
    fig.suptitle("Generated MNIST: Standard vs Accelerated JKO",
                 fontsize=14)
    for col, bi in enumerate(idxs):
        for row, (res, lbl) in enumerate(
                [(res_std, "Std JKO"), (res_acc, "Acc JKO")]):
            ax = axes[row, col]
            ax.imshow(_make_grid(res["snapshots"][bi]),
                      cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"{lbl}\nblock {bi}\nFID={res['fid'][bi]:.0f}",
                         fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def plot_convergence(res_std, res_acc, n_blocks, gamma,
                     savepath="images/mnist_jko_convergence.png"):
    """FID convergence curves and speedup ratio."""
    iters = np.arange(n_blocks + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf"MNIST JKO Convergence  ($\gamma={gamma}$, {n_blocks} blocks)",
        fontsize=13)

    # Panel A: FID
    ax = axes[0]
    ax.semilogy(iters, res_std["fid"], BLUE, lw=2, marker="o",
                ms=4, label="Standard JKO")
    ax.semilogy(iters, res_acc["fid"], RED, lw=2, marker="s",
                ms=4, label="Accelerated JKO")
    ax.set(xlabel="JKO block", ylabel="FID", title="FID vs Block")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)

    # Panel B: FID ratio (speedup)
    ax = axes[1]
    ratio = res_std["fid"] / np.maximum(res_acc["fid"], 1e-10)
    ax.plot(iters, ratio, color="darkorange", lw=2, marker="D", ms=4)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set(xlabel="JKO block", ylabel="FID_std / FID_acc",
           title="Speedup Ratio (>1 = acceleration wins)")
    ax.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def print_summary(res_std, res_acc, gamma, n_blocks,
                  savepath="results/mnist_jko_summary.txt"):
    sep = "=" * 55
    lines = [sep, "MNIST JKO SUMMARY  (v4 - DDPM score oracle)",
             f"gamma={gamma}  blocks={n_blocks}", sep, "",
             f"Initial FID          : {res_std['fid'][0]:.1f}",
             f"Final FID -- Std JKO : {res_std['fid'][-1]:.1f}",
             f"Final FID -- Acc JKO : {res_acc['fid'][-1]:.1f}", "",
             f"Wall-clock -- Std JKO : {res_std['times'][-1]:.1f}s",
             f"Wall-clock -- Acc JKO : {res_acc['times'][-1]:.1f}s", "",
             f"{'Block':>6}  {'FID_std':>10}  {'FID_acc':>10}"]
    for i in range(n_blocks + 1):
        lines.append(
            f"  {i:4d}  {res_std['fid'][i]:10.1f}  {res_acc['fid'][i]:10.1f}")
    lines += ["", sep]
    text = "\n".join(lines)
    print(text)
    with open(savepath, "w") as f:
        f.write(text)
    print(f"Saved: {savepath}")


# ==============================================================
#  9.  MAIN
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Accelerated JKO on MNIST using pretrained DDPM.")
    p.add_argument("--ckpt", type=str,
                   default="MNISTDiffusion/results/steps_00046900.pt",
                   help="Path to DDPM checkpoint")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="JKO step size (default 0.1)")
    p.add_argument("--blocks", type=int, default=15,
                   help="Number of JKO blocks (default 15)")
    p.add_argument("--t-eval", type=int, default=1,
                   help="DDPM timestep for score evaluation (default 1)")
    p.add_argument("--score-clip", type=float, default=50.0,
                   help="Per-sample score L2 norm clip (default 50.0)")
    p.add_argument("--particles", type=int, default=2000,
                   help="Number of particles (default 2000)")
    p.add_argument("--epochs", type=int, default=300,
                   help="Training epochs per JKO block (default 300)")
    p.add_argument("--hidden", type=int, default=512,
                   help="Transport map hidden width (default 512)")
    p.add_argument("--verify-only", action="store_true",
                   help="Run DDPM verification then exit")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -- Load DDPM -----------------------------------------------
    ddpm = load_ddpm(args.ckpt)

    # -- Verify checkpoint works ----------------------------------
    verify_ddpm(ddpm)
    if args.verify_only:
        print("--verify-only: exiting.")
        raise SystemExit(0)

    # -- Load MNIST data ------------------------------------------
    print("\nLoading MNIST...")
    train_data = load_mnist(train=True)
    test_data = load_mnist(train=False)
    print(f"Train: {train_data.shape}   Test: {test_data.shape}")

    # -- FID metric -----------------------------------------------
    fid_metric = FrechetInceptionDistance(feature=64).to(DEVICE)

    # -- Run experiments ------------------------------------------
    print(f"\ngamma={args.gamma}  blocks={args.blocks}  "
          f"t_eval={args.t_eval}  score_clip={args.score_clip}  "
          f"particles={args.particles}  epochs/block={args.epochs}")

    print("\n-- Standard JKO --")
    res_std = run_standard_jko(
        ddpm, test_data, fid_metric,
        args.gamma, args.blocks, args.t_eval, args.score_clip,
        args.particles, args.epochs, args.hidden, args.seed)

    print("\n-- Accelerated JKO --")
    res_acc = run_accelerated_jko(
        ddpm, test_data, fid_metric,
        args.gamma, args.blocks, args.t_eval, args.score_clip,
        args.particles, args.epochs, args.hidden, args.seed)

    # -- Plots ----------------------------------------------------
    plot_sample_grids(res_std, res_acc, args.blocks)
    plot_convergence(res_std, res_acc, args.blocks, args.gamma)
    print_summary(res_std, res_acc, args.gamma, args.blocks)

    print("\nAll experiments complete.")
