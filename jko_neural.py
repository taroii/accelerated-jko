import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from geomloss import SamplesLoss

import experiment_io as eio
import matplotlib.pyplot as plt
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("Standard JKO")
RED = color_for("Accelerated JKO")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  d-dimensional lambda=0 targets, sampled by radius (inverse-CDF) x uniform direction.
class RadialTarget:
    def __init__(self, kind, d, R=0.0):
        self.kind, self.d, self.R = kind, d, R
        r = np.linspace(0, 30, 400000)
        if kind == "iso_quartic":
            logdens = -r**4 / 4
        elif kind == "flatvalley":
            logdens = -np.maximum(r - R, 0.0) ** 4 / 4
        else:
            raise ValueError(kind)
        logw = (d - 1) * np.log(r + 1e-300) + logdens
        logw -= logw.max()
        w = np.exp(logw)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(r))])
        cdf /= cdf[-1]
        self._r, self._cdf = r, cdf

    def log_prob(self, x):
        rn = x.norm(dim=-1)
        if self.kind == "iso_quartic":
            return -rn**4 / 4
        return -torch.clamp(rn - self.R, min=0.0) ** 4 / 4

    def sample(self, n):
        u = np.random.rand(n)
        rr = np.interp(u, self._cdf, self._r)
        g = np.random.randn(n, self.d)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        return torch.tensor((rr[:, None] * g).astype(np.float32), device=DEVICE)


class TransportMap(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def logdet_jac(T, x):
    single = lambda xi: T(xi.unsqueeze(0)).squeeze(0)
    _, ld = torch.linalg.slogdet(vmap(jacrev(single))(x))
    return ld


def jko_loss(T, y, gamma, target):
    Ty = T(y)
    kl = -(target.log_prob(Ty) + logdet_jac(T, y)).mean()
    return kl + ((y - Ty) ** 2).sum(-1).mean() / (2 * gamma)


def train_block(y, gamma, target, d, epochs, lr=2e-3, batch=1024, hidden=256):
    T = TransportMap(d, hidden).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    n = y.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)[:batch]
        opt.zero_grad()
        jko_loss(T, y[idx].detach(), gamma, target).backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


_SINK = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)


def sinkhorn_div(x, ref):
    with torch.no_grad():
        m = min(x.shape[0], ref.shape[0])
        return float(_SINK(x[:m].contiguous(), ref[:m].contiguous()).clamp(min=0).sqrt())


def run_method(method, target, d, cfg, ref):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    x = torch.randn(cfg["particles"], d, device=DEVICE) * 1.5
    z = x.clone()
    vals, times = [sinkhorn_div(x, ref)], [0.0]
    t0 = time.time()
    for t in range(cfg["blocks"]):
        if method == "acc":
            a = 3.0 / (t + 3.0)
            y = (1 - a) * x + a * z
        else:
            y = x
        T = train_block(y.detach(), cfg["gamma"], target, d, cfg["epochs"])
        with torch.no_grad():
            xn = T(y).clamp(-8, 8)
            if method == "acc":
                z = (z + (xn - y) / a).clamp(-12, 12)
            x = xn
        vals.append(sinkhorn_div(x, ref))
        times.append(time.time() - t0)
    return np.array(vals), np.array(times)


def blocks_to(vals, thr):
    w = np.where(vals < thr)[0]
    return int(w[0]) if len(w) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="iso_quartic", choices=["iso_quartic", "flatvalley"])
    ap.add_argument("--dims", default="2,5,10,20,50")
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--blocks", type=int, default=25)
    ap.add_argument("--particles", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    cfg = dict(blocks=args.blocks, particles=args.particles, epochs=args.epochs,
               gamma=args.gamma, seed=args.seed, target=args.target, R=args.R,
               thresh=args.thresh, dims=dims)
    print(f"Device: {DEVICE}  target={args.target}  dims={dims}")

    rows, numbers = [], {}
    for d in dims:
        target = RadialTarget(args.target, d, args.R)
        np.random.seed(args.seed + 777)
        ref = target.sample(4096)
        v_std, t_std = run_method("std", target, d, cfg, ref)
        v_acc, t_acc = run_method("acc", target, d, cfg, ref)
        b_std, b_acc = blocks_to(v_std, args.thresh), blocks_to(v_acc, args.thresh)
        numbers[f"blocks_to_thresh_std_d{d}"] = b_std
        numbers[f"blocks_to_thresh_acc_d{d}"] = b_acc
        numbers[f"walltime_to_thresh_std_d{d}"] = float(t_std[b_std]) if b_std > 0 else -1
        numbers[f"walltime_to_thresh_acc_d{d}"] = float(t_acc[b_acc]) if b_acc > 0 else -1
        numbers[f"final_W2_std_d{d}"] = float(v_std[-1])
        numbers[f"final_W2_acc_d{d}"] = float(v_acc[-1])
        for t in range(len(v_std)):
            rows.append({"d": d, "block": t, "sink_std": v_std[t], "sink_acc": v_acc[t],
                         "sec_std": t_std[t], "sec_acc": t_acc[t]})
        print(f"[d={d:2d}] final W2 std={v_std[-1]:.4f} acc={v_acc[-1]:.4f} "
              f"blocks->thr std/acc={b_std}/{b_acc}")

    exp_id = f"e9_{args.target}"
    eio.save_config(exp_id, cfg)
    eio.save_metrics(exp_id, rows)
    eio.save_summary(exp_id, numbers)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(dims)))
    for d, c in zip(dims, cmap):
        vs = [r["sink_std"] for r in rows if r["d"] == d]
        va = [r["sink_acc"] for r in rows if r["d"] == d]
        ax.semilogy(vs, color=c, label=f"d={d} std")
        ax.semilogy(va, "--", color=c)
    ax.set_xlabel("Block $t$")
    ax.set_ylabel("Sinkhorn divergence to $q$")
    ax.set_title(f"E9 dimension scaling — {args.target} (solid=std, dashed=acc)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both")
    plt.tight_layout()
    fig.savefig(f"images/e9_{args.target}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
