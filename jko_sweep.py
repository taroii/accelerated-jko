import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap

import experiment_io as eio
from jko_densities import ImageDensity2D, density_rings, w2_to_target, DEVICE


class Map(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def logdet(T, x):
    single = lambda xi: T(xi.unsqueeze(0)).squeeze(0)
    _, ld = torch.linalg.slogdet(vmap(jacrev(single))(x))
    return ld


def train_block(y, gamma, target, width, depth, epochs):
    T = Map(width, depth).to(DEVICE)
    opt = optim.Adam(T.parameters(), lr=2e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    n = y.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)[:1024]
        opt.zero_grad()
        yb = y[idx].detach()
        Ty = T(yb)
        loss = -(target.log_prob(Ty) + logdet(T, yb)).mean() + ((yb - Ty) ** 2).sum(-1).mean() / (2 * gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(T.parameters(), 5.0)
        opt.step()
        sch.step()
    return T


def run(method, target, gamma, width, depth, blocks, particles, epochs, seed, ref):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.randn(particles, 2, device=DEVICE) * 1.5
    z = x.clone()
    for t in range(blocks):
        if method == "acc":
            a = 3.0 / (t + 3.0)
            y = (1 - a) * x + a * z
        else:
            y = x
        T = train_block(y.detach(), gamma, target, width, depth, epochs)
        with torch.no_grad():
            xn = T(y).clamp(-2.5, 2.5)
            if method == "acc":
                z = (z + (xn - y) / a).clamp(-3.5, 3.5)
            x = xn
    return w2_to_target(x, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0.01,0.02,0.04,0.08")
    ap.add_argument("--widths", default="128,256,512")
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--blocks", type=int, default=25)
    ap.add_argument("--particles", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gammas = [float(x) for x in args.gammas.split(",")]
    widths = [int(x) for x in args.widths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    print(f"Device: {DEVICE}")

    target = ImageDensity2D(density_rings())
    np.random.seed(args.seed + 777)
    ref = target.sample(4096)

    rows = []
    t0 = time.time()
    for g in gammas:
        for w in widths:
            for dp in depths:
                w2s = run("std", target, g, w, dp, args.blocks, args.particles, args.epochs, args.seed, ref)
                w2a = run("acc", target, g, w, dp, args.blocks, args.particles, args.epochs, args.seed, ref)
                rows.append({"gamma": g, "width": w, "depth": dp,
                             "w2_std_block25": round(w2s, 4), "w2_acc_block25": round(w2a, 4)})
                print(f"gamma={g} width={w} depth={dp}: W2 std={w2s:.4f} acc={w2a:.4f} "
                      f"({time.time()-t0:.0f}s)")

    eio.save_config("e10_sensitivity", vars(args))
    eio.save_metrics("e10_sensitivity", rows)
    with open("results/e10_sensitivity/sensitivity_table.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved results/e10_sensitivity/sensitivity_table.csv ({len(rows)} cells)")


if __name__ == "__main__":
    main()
