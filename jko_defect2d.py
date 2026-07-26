import argparse
import time

import numpy as np
import torch
import ot

import experiment_io as eio
import matplotlib.pyplot as plt
from plot_style import apply_paper_style, color_for
from jko_densities import (ImageDensity2D, density_outer_ring, train_block,
                           w2_to_target, DEVICE)

apply_paper_style()
RED = color_for("Accelerated JKO")


def ot_map_defect(z, x1, n_sub=2048, seed=0):
    """delta_t = ||index-aligned composed map - true OT map||_{L2(z)} / W2(z,x1)."""
    g = torch.Generator(device=z.device).manual_seed(seed)
    m = min(n_sub, z.shape[0])
    idx = torch.randperm(z.shape[0], generator=g, device=z.device)[:m]
    zs = z[idx].double().cpu().numpy()
    xs = x1[idx].double().cpu().numpy()          # algorithm map: z_i -> x1_i (index aligned)
    a = np.full(m, 1.0 / m)
    M = ot.dist(zs, xs, metric="sqeuclidean")
    P = ot.emd(a, a, M, numThreads="max")
    Tot = (P @ xs) * m                            # barycentric OT map z_i -> Tot_i
    delta = np.sqrt(np.mean(np.sum((xs - Tot) ** 2, axis=1)))
    w2 = np.sqrt(max(np.sum(P * M), 1e-30))
    return float(delta), float(delta / w2), w2


def run(cfg):
    target = ImageDensity2D(density_outer_ring())
    gamma, N, npart, epochs = cfg["gamma"], cfg["blocks"], cfg["particles"], cfg["epochs"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    y_ref = target.sample(4096)

    x = torch.randn(npart, 2, device=DEVICE) * 1.5
    z = x.clone()
    rows = []
    t0 = time.time()
    for t in range(N):
        alpha = 3.0 / (t + 3.0)
        y = (1.0 - alpha) * x + alpha * z
        T = train_block(y.detach(), gamma, target, n_epochs=epochs)
        with torch.no_grad():
            x_new = T(y).clamp(-2.5, 2.5)
            z_new = (z + (x_new - y) / alpha).clamp(-3.5, 3.5)
        d_raw, d_norm, w2zx = ot_map_defect(z, x_new, cfg["n_sub"], cfg["seed"])
        w2q = w2_to_target(x_new, y_ref)
        rows.append({"block": t + 1, "alpha": alpha, "delta_raw": d_raw,
                     "delta_normalized": d_norm, "w2_z_x1": w2zx, "w2_to_target": w2q})
        x, z = x_new.detach(), z_new.detach()
        print(f"  [block {t+1:2d}/{N}] delta_norm={d_norm:.4f} W2(z,x1)={w2zx:.3f} "
              f"W2->q={w2q:.3f} ({time.time()-t0:.0f}s)")

    dn = np.array([r["delta_normalized"] for r in rows])
    T_last = N
    tt = np.arange(1, N + 1)
    accum = float(((tt + 3) ** 2 * dn).sum() / (T_last + 2) ** 2)
    summary = {"delta_t_normalized_max": float(dn.max()),
               "delta_t_normalized_mean": float(dn.mean()),
               "accumulated_defect_term": accum,
               "blocks": N, "particles": npart, "n_sub": cfg["n_sub"]}
    eio.save_config("e7_defect2d", cfg)
    eio.save_metrics("e7_defect2d", rows)
    eio.save_summary("e7_defect2d", summary)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(tt, dn, "o-", color=RED)
    ax.set_xlabel("Block $t$")
    ax.set_ylabel(r"$\delta_t / W_2(z_t, x_{t+1})$")
    ax.set_title("E7: normalized 2-D map defect (outer ring)")
    ax.grid(True, which="both")
    plt.tight_layout()
    fig.savefig("images/e7_defect2d.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[e7] delta_norm max={dn.max():.4f} mean={dn.mean():.4f} accumulated={accum:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        cfg = {"gamma": 0.04, "blocks": 8, "particles": 4000, "epochs": 250,
               "seed": 0, "n_sub": 2048}
    else:
        cfg = {"gamma": 0.04, "blocks": 25, "particles": 12000, "epochs": 800,
               "seed": 0, "n_sub": 2048}
    run(cfg)


if __name__ == "__main__":
    main()
