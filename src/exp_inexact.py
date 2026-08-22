import argparse
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import jko

R = 8.0
DIM = 10
N = 2000
GAMMA = 0.5
SIGMA0 = 2.5
DELTA0 = 0.3
P_VALUES = [0.0, 2.0, 2.75, 3.0, 3.25, 4.0]

_v, _vprime = jko.flatvalley(R)


def _gap(x):
    return float(np.mean(_v(np.linalg.norm(x, axis=1))))


def _base_prox(y):
    return jko.prox_radial(y, _vprime, GAMMA)


def _noisy_prox(noise, deltas):
    state = {"t": 0}
    realized = []

    def prox(y):
        t = state["t"]
        pert = deltas[t] * noise[t]
        realized.append(float(np.sqrt(np.mean(np.sum(pert ** 2, axis=1)))))
        state["t"] += 1
        return _base_prox(y) + pert

    return prox, realized


def _rhs(w2sq0, e_real):
    nb = len(e_real)
    Tb = np.arange(1, nb + 1, dtype=float)
    cum = np.cumsum((np.arange(nb) + 3.0) ** 2 * e_real)
    return 9.0 * w2sq0 / (2.0 * GAMMA * (Tb + 2.0) ** 2) + cum / (Tb + 2.0) ** 2


def compute_rows(seeds, n_blocks):
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal((N, DIM)) * SIGMA0
        w2sq0 = float(np.mean(np.sum(x0 ** 2, axis=1)))
        t_idx = np.arange(n_blocks)
        for p in P_VALUES:
            deltas = DELTA0 * (t_idx + 1.0) ** (-p / 2.0)
            noise = rng.standard_normal((n_blocks, N, DIM))
            noise /= np.linalg.norm(noise, axis=2, keepdims=True)
            for method in ("std", "acc"):
                prox, realized = _noisy_prox(noise, deltas)
                recs = jko.run(method, prox, x0, GAMMA, n_blocks, _gap)
                gaps = np.array([r["gap"] for r in recs])
                e_real = np.array(realized)
                rhs = _rhs(w2sq0, e_real)
                for b in range(len(recs)):
                    rows.append({"seed": seed, "p": p, "method": method, "block": b + 1,
                                 "gap": float(gaps[b]), "rhs": float(rhs[b]),
                                 "e_t": float(e_real[b]), "delta_prescribed": float(deltas[b])})
    return rows


def to_matrices(rows):
    seeds = sorted({int(r["seed"]) for r in rows})
    ps = sorted({float(r["p"]) for r in rows})
    nb = max(int(r["block"]) for r in rows)
    si = {s: i for i, s in enumerate(seeds)}
    G = {(p, m): np.full((len(seeds), nb), np.nan) for p in ps for m in ("std", "acc")}
    Rh = {(p, m): np.full((len(seeds), nb), np.nan) for p in ps for m in ("std", "acc")}
    for r in rows:
        p, m, s, b = float(r["p"]), r["method"], int(r["seed"]), int(r["block"])
        G[(p, m)][si[s], b - 1] = float(r["gap"])
        Rh[(p, m)][si[s], b - 1] = float(r["rhs"])
    return seeds, ps, nb, G, Rh


def _loglog_slope(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0])


def build_summary(rows, config):
    seeds, ps, nb, G, Rh = to_matrices(rows)
    t = np.arange(1, nb + 1)
    lo = nb // 2
    per_p = {}
    max_viol = -np.inf
    for p in ps:
        diff = G[(p, "std")][:, -1] - G[(p, "acc")][:, -1]
        med_std = np.nanmedian(G[(p, "std")], axis=0)
        med_acc = np.nanmedian(G[(p, "acc")], axis=0)
        viol = float(np.nanmax(G[(p, "acc")] - Rh[(p, "acc")]))
        max_viol = max(max_viol, viol)
        per_p[f"{p:g}"] = {
            "paired_final_gap": jko.paired_summary(diff),
            "slope_std": _loglog_slope(t[lo:], med_std[lo:]),
            "slope_acc": _loglog_slope(t[lo:], med_acc[lo:]),
            "final_gap_med_std": float(np.nanmedian(G[(p, "std")][:, -1])),
            "final_gap_med_acc": float(np.nanmedian(G[(p, "acc")][:, -1])),
            "bound_violation": viol,
        }
    notes = []
    degrade_p0 = None
    if 0.0 in ps:
        degrade_p0 = per_p["0"]["final_gap_med_acc"] > per_p["0"]["final_gap_med_std"]
        if not degrade_p0:
            notes.append("accelerated did NOT degrade faster than standard at p=0")
    if max_viol > 1e-9:
        notes.append(f"bound violated: max_bound_violation={max_viol:.3e}")
        print(f"!!! WARNING: theorem RHS violated, max_bound_violation={max_viol:.3e} !!!")
    return {"config": config, "per_p": per_p,
            "max_bound_violation": float(max_viol),
            "acc_degrades_faster_p0": degrade_p0, "notes": notes}


def make_figure(rows, path):
    seeds, ps, nb, G, Rh = to_matrices(rows)
    t = np.arange(1, nb + 1)
    cmap = plt.get_cmap("viridis")
    fig, ax = jko.figure(ncols=2, full=True)

    for i, p in enumerate(ps):
        med = np.nanmedian(G[(p, "acc")], axis=0)
        m = med > 0
        ax[0].loglog(t[m], med[m], color=cmap(i / (len(ps) - 1)), label=f"$p={p:g}$")

    pmax = max(ps)
    ref_src = np.nanmedian(G[(pmax, "acc")], axis=0)
    t0 = max(2, nb // 4)
    C = ref_src[t0 - 1] * t0 ** 2 if ref_src[t0 - 1] > 0 else 1.0
    ref = C / t.astype(float) ** 2
    ax[0].loglog(t, ref, ls="--", color="grey", lw=0.8)
    ax[0].text(t[-1], ref[-1], r"$\propto t^{-2}$", color="grey", fontsize=7,
               ha="right", va="bottom")

    prep = min(ps, key=lambda q: abs(q - 3.0))
    medrhs = np.nanmedian(Rh[(prep, "acc")], axis=0)
    ax[0].loglog(t, medrhs, ls="--", color="black", lw=1.0)
    ax[0].text(t[len(t) // 3], medrhs[len(t) // 3], f"RHS $p={prep:g}$", color="black",
               fontsize=7, ha="left", va="bottom")
    ax[0].set_xlabel(r"block $t$")
    ax[0].set_ylabel(r"gap $G(x_t)-G(\rho^*)$")
    ax[0].legend(ncol=2)

    finals_std = np.column_stack([G[(p, "std")][:, -1] for p in ps])
    finals_acc = np.column_stack([G[(p, "acc")][:, -1] for p in ps])
    jko.median_iqr(ax[1], ps, finals_std, jko.BLUE, "standard")
    jko.median_iqr(ax[1], ps, finals_acc, jko.ORANGE, "accelerated")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$p$")
    ax[1].set_ylabel("final gap")
    ax[1].legend()

    jko.finish(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".")
    ap.add_argument("--from-csv", default=None)
    args = ap.parse_args()

    seeds = 2 if args.quick else args.seeds
    n_blocks = 60 if args.quick else 400

    if args.from_csv:
        with open(args.from_csv) as f:
            rows = list(csv.DictReader(f))
        config = {"rebuilt_from": os.path.abspath(args.from_csv)}
    else:
        rows = compute_rows(seeds, n_blocks)
        config = {"R": R, "d": DIM, "n": N, "gamma": GAMMA, "sigma0": SIGMA0,
                  "delta0": DELTA0, "p_values": P_VALUES, "seeds": seeds,
                  "n_blocks": n_blocks, "quick": args.quick}

    summary = build_summary(rows, config)
    jko.save_run("inexact", config, rows, summary,
                 outdir=os.path.join(args.out, "results"))
    make_figure(rows, os.path.join(args.out, "figures", "inexact.pdf"))

    per_p = summary["per_p"]
    ps = sorted(float(k) for k in per_p)
    plo, phi = per_p[f"{min(ps):g}"], per_p[f"{max(ps):g}"]
    print(f"[inexact] acc late-slope: p={min(ps):g} {plo['slope_acc']:.2f} -> "
          f"p={max(ps):g} {phi['slope_acc']:.2f} (std {phi['slope_std']:.2f}); "
          f"p=0 paired median (std-acc) {per_p.get('0', {}).get('paired_final_gap', {}).get('median', float('nan')):.3e}; "
          f"acc_degrades_faster_p0={summary['acc_degrades_faster_p0']}; "
          f"max_bound_violation={summary['max_bound_violation']:.2e}")


if __name__ == "__main__":
    main()
