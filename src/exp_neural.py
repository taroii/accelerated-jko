import argparse
import csv
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from geomloss import SamplesLoss

import jko

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RING_R, RING_S = 0.75, 0.07
LIM = 1.4


#  ------------------------------------------------------------------ target
class OuterRing:
    """Analytic outer ring  q(x) prop exp(-((r-R)^2)/(2 s^2))."""

    def __init__(self, R=RING_R, s=RING_S, device=DEVICE):
        self.R, self.s, self.device = R, s, device
        res = 400
        lin = np.linspace(-LIM, LIM, res)
        xx, yy = np.meshgrid(lin, lin)
        r = np.sqrt(xx ** 2 + yy ** 2)
        self.density = np.exp(-((r - R) ** 2) / (2 * s ** 2))

    def log_prob(self, x):
        r = torch.linalg.norm(x, dim=-1)
        return -((r - self.R) ** 2) / (2 * self.s ** 2)

    def sample(self, n):
        r = np.empty(0, dtype=np.float64)
        while r.shape[0] < n:
            cand = self.R + self.s * np.random.randn(n)
            r = np.concatenate([r, cand[cand > 0]])
        r = r[:n]
        th = np.random.uniform(0, 2 * np.pi, n)
        pts = np.stack([r * np.cos(th), r * np.sin(th)], 1).astype(np.float32)
        return torch.tensor(pts, device=self.device)


#  ------------------------------------------------------------------ transport map
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
    def single(xi):
        return T(xi.unsqueeze(0)).squeeze(0)
    jac = vmap(jacrev(single))(x)
    _, ld = torch.linalg.slogdet(jac)
    return ld


def jko_loss(T, y, gamma, target):
    Ty = T(y)
    kl = -(target.log_prob(Ty) + logdet_jacobian(T, y)).mean()
    w2 = ((y - Ty) ** 2).sum(-1).mean() / (2.0 * gamma)
    return kl + w2


def train_block(y, gamma, target, n_epochs=800, lr=2e-3, batch=1024, hidden=256):
    T = TransportMap(hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)
    n = y.shape[0]
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        opt.zero_grad()
        jko_loss(T, y[idx].detach(), gamma, target).backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


#  ------------------------------------------------------------------ Sinkhorn W2
_SINK = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.95)


def w2_to_target(x, y_ref):
    with torch.no_grad():
        if x.shape[0] > y_ref.shape[0]:
            idx = torch.randperm(x.shape[0], device=x.device)[:y_ref.shape[0]]
            x = x[idx]
        val = _SINK(x.contiguous(), y_ref.contiguous())
    return float(val.clamp(min=0).sqrt())


#  ------------------------------------------------------------------ transport runs
def run_standard(target, gamma, n_blocks, n_particles, n_epochs, seed, y_ref):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    snaps, w2s = [x.detach().clone()], [w2_to_target(x, y_ref)]
    t0 = time.time()
    for k in range(n_blocks):
        T = train_block(x, gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x = T(x).clamp(-2.5, 2.5)
        snaps.append(x.detach().clone())
        w2s.append(w2_to_target(x, y_ref))
        print(f"  [std seed{seed} {k+1:2d}/{n_blocks}] W2={w2s[-1]:.4f} {time.time()-t0:.0f}s")
    return snaps, w2s


def run_accelerated(target, gamma, n_blocks, n_particles, n_epochs, seed, y_ref):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.randn(n_particles, 2, device=DEVICE) * 1.5
    z = x.clone()
    snaps, w2s = [x.detach().clone()], [w2_to_target(x, y_ref)]
    t0 = time.time()
    for t in range(n_blocks):
        alpha = 3.0 / (t + 3.0)
        y = (1.0 - alpha) * x + alpha * z
        T = train_block(y.detach(), gamma, target, n_epochs=n_epochs)
        with torch.no_grad():
            x_new = T(y).clamp(-2.5, 2.5)
            z_new = (z + (x_new - y) / alpha).clamp(-3.5, 3.5)
        x, z = x_new.detach(), z_new.detach()
        snaps.append(x.clone())
        w2s.append(w2_to_target(x, y_ref))
        print(f"  [acc seed{seed} {t+1:2d}/{n_blocks}] a={alpha:.3f} W2={w2s[-1]:.4f} {time.time()-t0:.0f}s")
    return snaps, w2s


#  ------------------------------------------------------------------ figures
def _snap_cols(n_blocks):
    return sorted({0, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks})


def plot_full(snaps_std, snaps_acc, w2_std, w2_acc,
              curves_std, curves_acc, target, n_blocks, path):
    cols = _snap_cols(n_blocks)
    nc = len(cols)
    dens = target.density

    fig, _ = jko.figure(1, 1)          # applies paper style
    fig.clf()
    fig.set_size_inches(2.6 * nc, 6.4)
    gs = fig.add_gridspec(3, nc, height_ratios=[1, 1, 0.62],
                          hspace=0.10, wspace=0.05)

    rows = [("Standard", snaps_std, w2_std, jko.BLUE),
            ("Accelerated", snaps_acc, w2_acc, jko.ORANGE)]
    for ri, (label, snaps, w2s, color) in enumerate(rows):
        for ci, si in enumerate(cols):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(dens, extent=[-LIM, LIM, -LIM, LIM], origin="lower",
                      cmap="Greys", alpha=0.7, vmin=0, vmax=dens.max() * 1.5,
                      interpolation="bilinear")
            pts = snaps[si].cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.22, color=color,
                       linewidths=0, rasterized=True)
            ax.text(0.97, 0.97, f"$W_2$={w2s[si]:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.6))
            if ri == 0:
                ax.set_title(f"Block {si}", pad=3)
            if ci == 0:
                ax.set_ylabel(label)
            ax.set_xlim(-LIM, LIM)
            ax.set_ylim(-LIM, LIM)
            ax.set_xticks([])
            ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :])
    t = np.arange(n_blocks + 1)
    ax.semilogy(t, np.median(curves_std, 0), color=jko.BLUE, label="Standard")
    ax.semilogy(t, np.median(curves_acc, 0), color=jko.ORANGE, label="Accelerated")
    for si in cols:
        ax.axvline(si, color="gray", ls=":", alpha=0.35, lw=0.8)
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$W_2(\rho_t,\, q)$")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, n_blocks)
    jko.finish(fig, path)


def plot_convergence(curves_std, curves_acc, n_blocks, path):
    fig, ax = jko.figure(1, 1)
    t = np.arange(n_blocks + 1)
    ax.semilogy(t, np.median(curves_std, 0), color=jko.BLUE, label="Standard")
    ax.semilogy(t, np.median(curves_acc, 0), color=jko.ORANGE, label="Accelerated")
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$W_2(\rho_t,\, q)$")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, n_blocks)
    jko.finish(fig, path)


#  ------------------------------------------------------------------ summary
def build_summary(finals_std, finals_acc):
    ms, ma = float(np.median(finals_std)), float(np.median(finals_acc))
    red = 100.0 * (1.0 - ma / ms) if ms > 0 else 0.0
    return {
        "note": "QUALITATIVE demonstration on a target OUTSIDE the theory; "
                "no quantitative claim (inexactness covered elsewhere).",
        "median_final_w2_std": ms,
        "median_final_w2_acc": ma,
        "median_reduction_pct": red,
        "n_seeds": len(finals_std),
    }


#  ------------------------------------------------------------------ from-csv
def from_csv(path, out):
    rows = list(csv.DictReader(open(path)))
    n_blocks = max(int(r["block"]) for r in rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    curves = {"std": [], "acc": []}
    for m in ("std", "acc"):
        for s in seeds:
            sub = {int(r["block"]): float(r["w2"]) for r in rows
                   if r["method"] == m and int(r["seed"]) == s}
            curves[m].append([sub[b] for b in range(n_blocks + 1)])
    cs, ca = np.array(curves["std"]), np.array(curves["acc"])
    summary = build_summary(cs[:, -1], ca[:, -1])
    os.makedirs(os.path.join(out, "results", "neural"), exist_ok=True)
    plot_convergence(cs, ca, n_blocks, os.path.join(out, "figures", "neural.pdf"))
    import json
    json.dump(summary, open(os.path.join(out, "results", "neural", "summary.json"), "w"), indent=2)
    print(f"[neural | from-csv | QUALITATIVE] median final W2: "
          f"std={summary['median_final_w2_std']:.4f} "
          f"acc={summary['median_final_w2_acc']:.4f} "
          f"({summary['median_reduction_pct']:.1f}% reduction), "
          f"{summary['n_seeds']} seeds")


#  ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=".")
    p.add_argument("--from-csv", default=None)
    args = p.parse_args()

    if args.from_csv:
        from_csv(args.from_csv, args.out)
        return

    gamma = 0.04
    if args.quick:
        n_seeds, n_blocks, n_particles, n_epochs = 2, 6, 2000, 150
    else:
        n_seeds, n_blocks, n_particles, n_epochs = args.seeds, 25, 12000, 800

    print(f"Device: {DEVICE}")
    target = OuterRing()
    seeds = list(range(n_seeds))
    rep = seeds[0]

    per_seed_rows = []
    curves_std, curves_acc = [], []
    rep_snaps_std = rep_snaps_acc = rep_w2_std = rep_w2_acc = None

    for s in seeds:
        np.random.seed(s + 777)
        y_ref = target.sample(4096)

        snaps_std, w2_std = run_standard(
            target, gamma, n_blocks, n_particles, n_epochs, s, y_ref)
        snaps_acc, w2_acc = run_accelerated(
            target, gamma, n_blocks, n_particles, n_epochs, s, y_ref)

        curves_std.append(w2_std)
        curves_acc.append(w2_acc)
        for b in range(n_blocks + 1):
            per_seed_rows.append({"seed": s, "method": "std", "block": b, "w2": w2_std[b]})
            per_seed_rows.append({"seed": s, "method": "acc", "block": b, "w2": w2_acc[b]})
        if s == rep:
            rep_snaps_std, rep_snaps_acc = snaps_std, snaps_acc
            rep_w2_std, rep_w2_acc = w2_std, w2_acc

    curves_std, curves_acc = np.array(curves_std), np.array(curves_acc)
    summary = build_summary(curves_std[:, -1], curves_acc[:, -1])

    config = {
        "kind": "qualitative neural JKO demonstration (target outside theory)",
        "target": "outer_ring  q ~ exp(-((r-0.75)^2)/(2*0.07^2))",
        "gamma": gamma, "n_blocks": n_blocks, "n_particles": n_particles,
        "n_epochs": n_epochs, "n_seeds": n_seeds, "rep_seed": rep,
        "init": "x0 ~ N(0, 1.5^2 I)", "clamp_x": [-2.5, 2.5], "clamp_z": [-3.5, 3.5],
        "optimizer": "Adam lr=2e-3 cosine to 1e-4, batch 1024, grad clip 5.0",
        "transport_map": "residual MLP 2->256->256->256->2 SiLU zero-init last",
        "logdet": "exact via torch.func jacrev/vmap slogdet",
        "extrapolation": "y_t=(1-a)x_t+a z_t, a=3/(t+3); z_{t+1}=z_t+(x_{t+1}-y_t)/a",
        "w2_metric": "geomloss Sinkhorn p=2 blur=0.01 scaling=0.95 (approx W2)",
        "w2_ref_samples": 4096, "device": str(DEVICE),
    }
    jko.save_run("neural", config, per_seed_rows, summary, outdir=os.path.join(args.out, "results"))
    plot_full(rep_snaps_std, rep_snaps_acc, rep_w2_std, rep_w2_acc,
              curves_std, curves_acc, target, n_blocks,
              os.path.join(args.out, "figures", "neural.pdf"))

    print(f"[neural | QUALITATIVE demo, target outside theory] median final W2: "
          f"std={summary['median_final_w2_std']:.4f} "
          f"acc={summary['median_final_w2_acc']:.4f} "
          f"({summary['median_reduction_pct']:.1f}% reduction) "
          f"over {n_seeds} seeds; snapshots use seed {rep}.")


if __name__ == "__main__":
    main()
