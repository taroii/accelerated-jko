"""
Neural Accelerated JKO -- MNIST Generative Flow
=================================================
Setting : Learn a generative flow from  p_0 = N(0, I)  in R^784  to the
          MNIST image distribution  q,  using a chain of N JKO blocks.

This is the high-dimensional counterpart of the 2-D image-density experiment.
The true density q is intractable, so a pretrained score network
s_theta(x) ~ grad log q(x)  serves as an oracle throughout the JKO loop.

Compares per JKO block:
  * Standard JKO          x_{n+1} = Prox(G, x_n, gamma)
  * Accelerated JKO       three-sequence scheme, alpha_t = 3/(t+3)

Metrics:
  * FID score             Frechet Inception Distance (generated vs real MNIST)
  * KL proxy              -E[s_theta(x)^T x]  (cheap surrogate for KL)

Outputs
-------
  images/mnist_jko_samples.png          -- sample grids at key blocks
  images/mnist_jko_convergence.png      -- FID & KL proxy curves
  results/mnist_jko_summary.txt         -- numerical table

Usage
-----
    python jko_mnist.py                           # defaults
    python jko_mnist.py --gamma 0.5 --blocks 12
    python jko_mnist.py --skip-pretrain --score-ckpt score_net.pt
"""

import argparse
import os
import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 10})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DIM = 784  # 28 * 28


# ==============================================================
#  1.  MNIST DATA
# ==============================================================

def load_mnist(train: bool = True) -> torch.Tensor:
    """Load MNIST as flat (N, 784) tensors scaled to [-1, 1]."""
    ds = datasets.MNIST(root="./data", train=train, download=True,
                        transform=transforms.ToTensor())
    imgs = ds.data.float() / 255.0          # (N, 28, 28) in [0, 1]
    imgs = imgs * 2.0 - 1.0                 # scale to [-1, 1]
    return imgs.view(-1, DIM).to(DEVICE)    # (N, 784)


# ==============================================================
#  2.  SCORE NETWORK (pretrained via denoising score matching)
# ==============================================================

class ScoreNet(nn.Module):
    """
    Simple residual MLP that estimates grad log q(x).
    Operates on flat (batch, 784) vectors.
    """
    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pretrain_score_network(data: torch.Tensor,
                           sigma: float = 0.3,
                           n_epochs: int = 50,
                           batch_size: int = 512,
                           lr: float = 1e-3,
                           hidden: int = 1024) -> ScoreNet:
    """
    Denoising score matching:
      L(theta) = E_{x~q, eps~N(0,I)} || s_theta(x + sigma*eps) + eps/sigma ||^2
    """
    print(f"\n{'='*60}")
    print(f"Pretraining score network  (sigma={sigma}, epochs={n_epochs})")
    print(f"{'='*60}")

    net = ScoreNet(hidden).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    n = data.shape[0]
    t0 = time.time()

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = data[idx]
            eps = torch.randn_like(x)
            x_noisy = x + sigma * eps
            target = -eps / sigma

            score = net(x_noisy)
            loss = ((score - target) ** 2).sum(-1).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        sch.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = epoch_loss / n_batches
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={avg:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    print(f"Score network pretrained in {time.time()-t0:.1f}s")
    return net


# ==============================================================
#  3.  TRANSPORT MAP  (residual MLP for 28x28)
# ==============================================================

class TransportMap(nn.Module):
    """
    T_n(x) = x + f_n(x),  residual parameterisation.
    Zero-initialised output layer so T ~ Id at start of each block.
    """
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ==============================================================
#  4.  JKO LOSS (score-based KL + W2 proximal)
# ==============================================================

def hutchinson_log_det(T: nn.Module, x: torch.Tensor,
                       n_probes: int = 5) -> torch.Tensor:
    """
    Hutchinson trace estimator for log|det J_T(x)|.
    Uses random projections: E[v^T J v] = tr(J), then
    log|det J| ~ tr(log J) ~ tr(J - I) for J near identity.

    For residual maps T(x) = x + f(x), J_T = I + J_f, so
    log|det(I + J_f)| ~ tr(J_f) - tr(J_f^2)/2 + ...
    We use the first-order approximation tr(J_f) which is exact
    when f is small (which it is, thanks to zero init).
    """
    batch = x.shape[0]
    total = torch.zeros(batch, device=x.device)

    for _ in range(n_probes):
        v = torch.randn_like(x)
        x_req = x.detach().requires_grad_(True)
        Tx = T(x_req)
        fx = Tx - x_req  # residual part
        vjp = torch.autograd.grad(
            fx, x_req, grad_outputs=v,
            create_graph=True, retain_graph=False
        )[0]
        # v^T J_f v estimates tr(J_f) per sample
        total = total + (v * vjp).sum(-1)

    return total / n_probes  # approximate log|det J_T|


def jko_loss(T: nn.Module,
             y: torch.Tensor,
             gamma: float,
             score_net: ScoreNet,
             n_probes: int = 5) -> torch.Tensor:
    """
    L(T) = -E[s_theta(T(y))^T T(y) - 0.5||T(y)||^2 + log|det J_T(y)|]
           + (1/2gamma) E||y - T(y)||^2

    The first term approximates KL(T#mu_y || q) using the score network.
    """
    Ty = T(y)
    score = score_net(Ty)

    # KL term (score-based): -E[s(Ty)^T Ty - 0.5||Ty||^2 + logdet]
    score_term = (score * Ty).sum(-1)                    # s(Ty)^T Ty
    norm_term = 0.5 * (Ty ** 2).sum(-1)                  # 0.5||Ty||^2
    logdet = hutchinson_log_det(T, y, n_probes=n_probes) # log|det J_T|

    kl_term = -(score_term - norm_term + logdet).mean()

    # W2 proximal term
    w2_term = ((y - Ty) ** 2).sum(-1).mean() / (2.0 * gamma)

    return kl_term + w2_term


# ==============================================================
#  5.  TRAIN ONE JKO BLOCK
# ==============================================================

def train_block(y: torch.Tensor,
                gamma: float,
                score_net: ScoreNet,
                n_epochs: int = 200,
                lr: float = 2e-3,
                batch: int = 1024,
                hidden: int = 512,
                n_probes: int = 5) -> TransportMap:
    T = TransportMap(hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

    n = y.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        xb = y[idx].detach()
        opt.zero_grad()
        loss = jko_loss(T, xb, gamma, score_net, n_probes=n_probes)
        loss.backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


# ==============================================================
#  6.  METRICS
# ==============================================================

@torch.no_grad()
def kl_proxy(particles: torch.Tensor, score_net: ScoreNet) -> float:
    """Cheap KL surrogate: -E[s_theta(x)^T x]."""
    s = score_net(particles)
    return float(-(s * particles).sum(-1).mean())


@torch.no_grad()
def compute_fid(generated: torch.Tensor, real: torch.Tensor,
                max_samples: int = 10000) -> float:
    """
    Simplified FID in pixel space (no Inception network needed for MNIST).
    Computes Frechet distance between Gaussian fits in pixel space.
    """
    g = generated[:max_samples].float()
    r = real[:max_samples].float()

    mu_g = g.mean(0)
    mu_r = r.mean(0)

    # Covariance (use low-rank approximation for efficiency)
    n_g = g.shape[0]
    n_r = r.shape[0]
    g_centered = g - mu_g
    r_centered = r - mu_r

    # Use top-k singular values for tractable FID in 784D
    k = min(128, min(n_g, n_r) - 1)

    U_g, S_g, _ = torch.svd_lowrank(g_centered, q=k)
    cov_g_vals = (S_g ** 2) / (n_g - 1)

    U_r, S_r, _ = torch.svd_lowrank(r_centered, q=k)
    cov_r_vals = (S_r ** 2) / (n_r - 1)

    # FID = ||mu_g - mu_r||^2 + Tr(C_g) + Tr(C_r) - 2*Tr(sqrt(C_g C_r))
    # With diagonal approx in SVD basis:
    diff_mu = ((mu_g - mu_r) ** 2).sum()
    tr_cg = cov_g_vals.sum()
    tr_cr = cov_r_vals.sum()
    # Approximate sqrt(C_g C_r) trace using geometric mean of eigenvalues
    tr_sqrt = (cov_g_vals.sqrt() * cov_r_vals.sqrt()).sum()

    fid = float(diff_mu + tr_cg + tr_cr - 2.0 * tr_sqrt)
    return max(fid, 0.0)


# ==============================================================
#  7.  STANDARD JKO
# ==============================================================

def run_standard_jko(score_net: ScoreNet,
                     real_data: torch.Tensor,
                     gamma: float,
                     n_blocks: int,
                     n_particles: int = 4000,
                     n_epochs: int = 200,
                     hidden: int = 512,
                     seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.randn(n_particles, DIM, device=DEVICE)

    snapshots = [x.detach().clone()]
    kl_vals = [kl_proxy(x, score_net)]
    fid_vals = [compute_fid(x, real_data)]
    times = [0.0]

    t0 = time.time()
    for step in range(n_blocks):
        T = train_block(x, gamma, score_net, n_epochs=n_epochs, hidden=hidden)
        with torch.no_grad():
            x = T(x).clamp(-1.0, 1.0)
        snapshots.append(x.detach().clone())
        kl_vals.append(kl_proxy(x, score_net))
        fid_vals.append(compute_fid(x, real_data))
        times.append(time.time() - t0)
        print(f"  [Std  JKO] block {step+1:2d}/{n_blocks}  "
              f"KL_proxy={kl_vals[-1]:.4f}  FID={fid_vals[-1]:.1f}  "
              f"{times[-1]:.1f}s")

    return {"kl": np.array(kl_vals), "fid": np.array(fid_vals),
            "times": np.array(times), "snapshots": snapshots}


# ==============================================================
#  8.  ACCELERATED JKO
# ==============================================================

def run_accelerated_jko(score_net: ScoreNet,
                        real_data: torch.Tensor,
                        gamma: float,
                        n_blocks: int,
                        n_particles: int = 4000,
                        n_epochs: int = 200,
                        hidden: int = 512,
                        seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    p0 = torch.randn(n_particles, DIM, device=DEVICE)

    x = p0.clone(); z = p0.clone()
    snapshots = [x.detach().clone()]
    kl_vals = [kl_proxy(x, score_net)]
    fid_vals = [compute_fid(x, real_data)]
    times = [0.0]

    t0 = time.time()
    for t in range(n_blocks):
        alpha = 3.0 / (t + 3.0)

        # Wasserstein geodesic interpolation
        y = (1.0 - alpha) * x + alpha * z

        # Proximal step from y
        T = train_block(y.detach(), gamma, score_net,
                        n_epochs=n_epochs, hidden=hidden)
        with torch.no_grad():
            x_new = T(y).clamp(-1.0, 1.0)

        # Momentum update for z
        with torch.no_grad():
            z_new = (z + (x_new - y) / alpha).clamp(-2.0, 2.0)

        x = x_new.detach(); z = z_new.detach()
        snapshots.append(x.clone())
        kl_vals.append(kl_proxy(x, score_net))
        fid_vals.append(compute_fid(x, real_data))
        times.append(time.time() - t0)
        print(f"  [Acc  JKO] block {t+1:2d}/{n_blocks}  "
              f"KL_proxy={kl_vals[-1]:.4f}  FID={fid_vals[-1]:.1f}  "
              f"alpha={alpha:.3f}  {times[-1]:.1f}s")

    return {"kl": np.array(kl_vals), "fid": np.array(fid_vals),
            "times": np.array(times), "snapshots": snapshots}


# ==============================================================
#  9.  PLOTTING
# ==============================================================

BLUE = "#1f77b4"
RED  = "#d62728"


def plot_sample_grids(res_std: dict, res_acc: dict, n_blocks: int,
                      savepath: str = "images/mnist_jko_samples.png"):
    """8x8 sample grids at blocks 0, N/4, N/2, 3N/4, N for both methods."""
    block_idxs = sorted(set([0, n_blocks // 4, n_blocks // 2,
                             3 * n_blocks // 4, n_blocks]))

    fig, axes = plt.subplots(2, len(block_idxs), figsize=(3 * len(block_idxs), 7))
    fig.suptitle("Generated MNIST Samples: Standard vs Accelerated JKO",
                 fontsize=13)

    for col, bi in enumerate(block_idxs):
        for row, (res, label) in enumerate(
                [(res_std, "Std JKO"), (res_acc, "Acc JKO")]):
            ax = axes[row, col]
            particles = res["snapshots"][bi][:64]  # first 64
            grid = _make_image_grid(particles, nrow=8)
            ax.imshow(grid, cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"{label}\nblock {bi}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def _make_image_grid(flat_images: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """Arrange (N, 784) flat images into a single grid image."""
    n = flat_images.shape[0]
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    imgs = flat_images.detach().cpu().numpy().reshape(-1, 28, 28)

    grid = np.ones((nrow_actual * 29 - 1, ncol * 29 - 1)) * -1.0
    for i in range(n):
        r = i // ncol
        c = i % ncol
        grid[r*29:r*29+28, c*29:c*29+28] = imgs[i]
    return grid


def plot_convergence(res_std: dict, res_acc: dict,
                     n_blocks: int, gamma: float,
                     savepath: str = "images/mnist_jko_convergence.png"):
    """FID and KL proxy convergence curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(rf"MNIST JKO Convergence  ($\gamma={gamma}$, {n_blocks} blocks)",
                 fontsize=13)

    iters = np.arange(n_blocks + 1)

    # Panel A: FID
    ax = axes[0]
    ax.semilogy(iters, res_std["fid"], color=BLUE, lw=2, marker="o",
                ms=4, label="Standard JKO")
    ax.semilogy(iters, res_acc["fid"], color=RED, lw=2, marker="s",
                ms=4, label="Accelerated JKO")
    ax.set_xlabel("JKO block")
    ax.set_ylabel("FID (pixel-space)")
    ax.set_title("FID vs Block")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Panel B: KL proxy
    ax = axes[1]
    ax.plot(iters, res_std["kl"], color=BLUE, lw=2, marker="o",
            ms=4, label="Standard JKO")
    ax.plot(iters, res_acc["kl"], color=RED, lw=2, marker="s",
            ms=4, label="Accelerated JKO")
    ax.set_xlabel("JKO block")
    ax.set_ylabel(r"KL proxy $-\mathbb{E}[s(x)^\top x]$")
    ax.set_title("KL Proxy vs Block")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Panel C: FID ratio (speedup)
    ax = axes[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = res_std["fid"] / np.maximum(res_acc["fid"], 1e-10)
    ax.plot(iters, ratio, color="green", lw=2, marker="D", ms=4)
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("JKO block")
    ax.set_ylabel("FID_std / FID_acc")
    ax.set_title("Speedup Ratio (>1 = acceleration wins)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.show()


def print_summary(res_std, res_acc, gamma, n_blocks,
                  savepath="results/mnist_jko_summary.txt"):
    lines = []
    sep = "=" * 65
    lines += [sep,
              "MNIST JKO SUMMARY",
              f"gamma={gamma}   blocks={n_blocks}",
              sep,
              "",
              f"Initial FID          : {res_std['fid'][0]:.1f}",
              f"Final FID -- Std JKO : {res_std['fid'][-1]:.1f}",
              f"Final FID -- Acc JKO : {res_acc['fid'][-1]:.1f}",
              "",
              f"Initial KL proxy          : {res_std['kl'][0]:.4f}",
              f"Final KL proxy -- Std JKO : {res_std['kl'][-1]:.4f}",
              f"Final KL proxy -- Acc JKO : {res_acc['kl'][-1]:.4f}",
              "",
              f"Wall-clock -- Std JKO : {res_std['times'][-1]:.1f}s",
              f"Wall-clock -- Acc JKO : {res_acc['times'][-1]:.1f}s",
              "",
              f"{'Block':>6}  {'FID_std':>10}  {'FID_acc':>10}  "
              f"{'KL_std':>10}  {'KL_acc':>10}"]
    for i in range(n_blocks + 1):
        lines.append(
            f"  {i:4d}  {res_std['fid'][i]:10.1f}  {res_acc['fid'][i]:10.1f}  "
            f"{res_std['kl'][i]:10.4f}  {res_acc['kl'][i]:10.4f}")
    lines += ["", sep]
    text = "\n".join(lines)
    print(text)
    with open(savepath, "w") as f:
        f.write(text)
    print(f"Saved: {savepath}")


# ==============================================================
# 10.  MAIN
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Accelerated JKO on MNIST (generative flow)."
    )
    p.add_argument("--gamma", type=float, default=0.5,
                   help="JKO step size (default 0.5)")
    p.add_argument("--blocks", type=int, default=10,
                   help="Number of JKO blocks (default 10)")
    p.add_argument("--particles", type=int, default=4000,
                   help="Number of particles (default 4000)")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per block (default 200)")
    p.add_argument("--hidden", type=int, default=512,
                   help="Transport map hidden width (default 512)")
    p.add_argument("--score-epochs", type=int, default=50,
                   help="Score network pretraining epochs (default 50)")
    p.add_argument("--score-hidden", type=int, default=1024,
                   help="Score network hidden width (default 1024)")
    p.add_argument("--score-sigma", type=float, default=0.3,
                   help="Noise level for denoising score matching (default 0.3)")
    p.add_argument("--score-ckpt", type=str, default=None,
                   help="Path to pretrained score network checkpoint")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip pretraining (requires --score-ckpt)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -- Load MNIST data -----------------------------------------
    print("Loading MNIST...")
    train_data = load_mnist(train=True)
    test_data = load_mnist(train=False)
    print(f"Train: {train_data.shape}  Test: {test_data.shape}")

    # -- Score network -------------------------------------------
    if args.skip_pretrain and args.score_ckpt:
        print(f"Loading score network from {args.score_ckpt}")
        score_net = ScoreNet(args.score_hidden).to(DEVICE)
        score_net.load_state_dict(torch.load(args.score_ckpt,
                                             map_location=DEVICE))
        score_net.eval()
        for p in score_net.parameters():
            p.requires_grad_(False)
    else:
        score_net = pretrain_score_network(
            train_data,
            sigma=args.score_sigma,
            n_epochs=args.score_epochs,
            hidden=args.score_hidden,
        )
        # Save checkpoint for reuse
        ckpt_path = "results/score_net.pt"
        torch.save(score_net.state_dict(), ckpt_path)
        print(f"Saved score network checkpoint: {ckpt_path}")

    # -- Run experiments -----------------------------------------
    print(f"\ngamma={args.gamma}  blocks={args.blocks}  "
          f"particles={args.particles}  epochs/block={args.epochs}")

    print("\n-- Standard JKO --")
    res_std = run_standard_jko(
        score_net, test_data, args.gamma, args.blocks,
        args.particles, args.epochs, args.hidden, args.seed
    )

    print("\n-- Accelerated JKO --")
    res_acc = run_accelerated_jko(
        score_net, test_data, args.gamma, args.blocks,
        args.particles, args.epochs, args.hidden, args.seed
    )

    # -- Plots ---------------------------------------------------
    plot_sample_grids(res_std, res_acc, args.blocks)
    plot_convergence(res_std, res_acc, args.blocks, args.gamma)
    print_summary(res_std, res_acc, args.gamma, args.blocks)

    print("\nAll experiments complete.")
