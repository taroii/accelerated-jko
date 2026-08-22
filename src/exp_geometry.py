import argparse
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import jko

GAMMA = 0.5
KAPPAS = [1.0, 10.0, 100.0]
REP_KAPPA = 100.0
REP_THETA = np.pi / 4.0


#  ------------------------------------------------------------------ instances
def rotation(d, theta, rng, n_planes=3):
    R = np.eye(d)
    if d < 2 or theta == 0.0:
        return R
    for _ in range(min(n_planes, d - 1)):
        i, j = rng.choice(d, size=2, replace=False)
        c, s = np.cos(theta), np.sin(theta)
        G = np.eye(d)
        G[i, i] = G[j, j] = c
        G[i, j], G[j, i] = -s, s
        R = R @ G
    return R


def make_target(d, kappa):
    return np.eye(d) if kappa == 1.0 else np.diag(np.geomspace(1.0, kappa, d))


def make_init(d, theta, rng):
    D = np.diag(np.exp(rng.uniform(np.log(0.3), np.log(3.0), size=d)))
    R = rotation(d, theta, rng)
    M = R @ D @ R.T
    return 0.5 * (M + M.T)


def run_instance(d, kappa, theta, seed, n_blocks):
    rng = np.random.default_rng(seed)
    Sigma = make_target(d, kappa)
    S0 = make_init(d, theta, rng)
    prox = lambda S: jko.prox_gaussian(S, Sigma, GAMMA)
    gap = lambda S: jko.gaussian_kl(S, Sigma)
    rec_std = jko.run("std", prox, S0, GAMMA, n_blocks, gap)
    rec_acc = jko.run("acc", prox, S0, GAMMA, n_blocks, gap, geom=jko.GaussGeom())
    return rec_std, rec_acc


#  ------------------------------------------------------------------ correctness gate
def geometry_gate():
    d = 4
    Sigma = np.diag(np.geomspace(1.0, 10.0, d))
    prox = lambda S: jko.prox_gaussian(S, Sigma, GAMMA)
    gap = lambda S: jko.gaussian_kl(S, Sigma)

    rng = np.random.default_rng(0)
    S0 = np.diag(np.exp(rng.uniform(np.log(0.3), np.log(3.0), d)))   # codiagonal
    recs = jko.run("acc", prox, S0, GAMMA, 20, gap, geom=jko.GaussGeom())
    d_codiag = max(r["map_defect"] for r in recs)
    assert d_codiag < 1e-10, f"codiagonal defect not zero: {d_codiag:.2e}"

    S0r = make_init(d, REP_THETA, np.random.default_rng(1))          # rotated
    recs_r = jko.run("acc", prox, S0r, GAMMA, 20, gap, geom=jko.GaussGeom())
    d_rot = max(r["map_defect"] for r in recs_r)
    assert d_rot > 1e-8, f"rotated defect not positive: {d_rot:.2e}"

    print(f"ok geometry gate  (theta=0 -> {d_codiag:.1e}, theta=pi/4 -> {d_rot:.3f})")


#  ------------------------------------------------------------------ data
def collect(ds, kappas, seeds, n_blocks):
    data, rows = {}, []
    for d in ds:
        for kappa in kappas:
            delta = np.zeros((len(seeds), n_blocks))
            lam = np.zeros((len(seeds), n_blocks))
            gap_acc = np.zeros((len(seeds), n_blocks))
            gap_std = np.zeros((len(seeds), n_blocks))
            for i, seed in enumerate(seeds):
                rec_std, rec_acc = run_instance(d, kappa, REP_THETA, seed, n_blocks)
                for b in range(n_blocks):
                    ra, rs = rec_acc[b], rec_std[b]
                    delta[i, b] = ra["map_defect"]
                    lam[i, b] = ra["lambda_min_symZ"]
                    gap_acc[i, b] = ra["gap"]
                    gap_std[i, b] = rs["gap"]
                    rows.append({"seed": seed, "d": d, "kappa": kappa, "theta": REP_THETA,
                                 "block": b + 1, "delta_t": delta[i, b],
                                 "lambda_min_symZ": lam[i, b],
                                 "gap_std": gap_std[i, b], "gap_acc": gap_acc[i, b]})
            data[(d, float(kappa))] = {"delta": delta, "lam": lam,
                                       "gap_acc": gap_acc, "gap_std": gap_std}
    return data, rows


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    ds = sorted({int(r["d"]) for r in rows})
    kappas = sorted({float(r["kappa"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    n_blocks = max(int(r["block"]) for r in rows)
    sidx = {s: i for i, s in enumerate(seeds)}
    data = {}
    for d in ds:
        for kappa in kappas:
            shape = (len(seeds), n_blocks)
            arr = {k: np.full(shape, np.nan) for k in ("delta", "lam", "gap_acc", "gap_std")}
            for r in rows:
                if int(r["d"]) != d or float(r["kappa"]) != kappa:
                    continue
                i, b = sidx[int(r["seed"])], int(r["block"]) - 1
                arr["delta"][i, b] = float(r["delta_t"])
                arr["lam"][i, b] = float(r["lambda_min_symZ"])
                arr["gap_acc"][i, b] = float(r["gap_acc"])
                arr["gap_std"][i, b] = float(r["gap_std"])
            data[(d, kappa)] = arr
    return data, ds, [float(k) for k in kappas], seeds, n_blocks


#  ------------------------------------------------------------------ analysis
def accum_term(delta_2d, T):
    t = np.arange(T)
    w = (t + 3.0) ** 2
    return (delta_2d * w).sum(axis=1) / (T + 2.0) ** 2          # A_T per seed


def late_slope(gap_2d, n_blocks):
    med = np.maximum(np.nanmedian(gap_2d, axis=0), 1e-300)
    t = np.arange(1, n_blocks + 1)
    lo = max(1, n_blocks // 2)
    m = (t >= lo) & (med > 1e-16)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(t[m]), np.log(med[m]), 1)[0])


def build_summary(data, ds, kappas, n_blocks, config):
    per_dk, mean_deltas, slopes = {}, [], []
    for d in ds:
        for kappa in kappas:
            a = data[(d, float(kappa))]
            delta = a["delta"]
            A = accum_term(delta, n_blocks)
            A_stat = jko.paired_summary(A)
            slope = late_slope(a["gap_acc"], n_blocks)
            md_max = float(np.nanmedian(np.nanmax(delta, axis=1)))
            md_mean = float(np.nanmedian(np.nanmean(delta, axis=1)))
            per_dk[f"d{d}_k{int(kappa)}"] = {
                "d": d, "kappa": kappa,
                "median_delta_max": md_max,
                "median_delta_mean": md_mean,
                "A_T": {k: A_stat[k] for k in ("median", "ci_lo", "ci_hi")},
                "frac_lambda_nonpos": float(np.nanmean(a["lam"] <= 0.0)),
                "gap_slope_acc": slope,
            }
            mean_deltas.append(md_mean)
            slopes.append(slope)

    mean_deltas, slopes = np.array(mean_deltas), np.array(slopes)
    ok = np.isfinite(mean_deltas) & np.isfinite(slopes)
    r = float(np.corrcoef(mean_deltas[ok], slopes[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    survived = bool(np.nanmedian(slopes) < -1.5)
    if not np.isfinite(r) or abs(r) < 0.3:
        obs = ("No visible degradation: large map defect does NOT coincide with a "
               "shallower accelerated rate (slopes stay near -2 across d and kappa). "
               "Consistent with the assumed O(1/t^2) rate holding despite defect.")
    else:
        obs = (f"Defect correlates with rate (pearson r={r:.2f}); larger delta_t "
               f"coincides with a {'shallower' if r > 0 else 'steeper'} accelerated slope.")

    d_rep = max(ds)
    rep = per_dk[f"d{d_rep}_k{int(REP_KAPPA)}"]
    return {
        "config": config,
        "per_dk": per_dk,
        "defect_vs_rate": {"pearson_r": r, "median_gap_slope_acc": float(np.nanmedian(slopes)),
                           "rate_survived": survived, "observation": obs},
        "headline": {"d": d_rep, "kappa": REP_KAPPA,
                     "median_max_defect": rep["median_delta_max"],
                     "gap_slope_acc": rep["gap_slope_acc"], "rate_survived": survived},
    }


#  ------------------------------------------------------------------ figure
def make_figure(data, ds, n_blocks, path):
    fig, (axL, axR) = jko.figure(ncols=2, full=True)
    blocks = np.arange(1, n_blocks + 1)
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(ds)))

    for i, d in enumerate(ds):                                  # left: map defect
        jko.median_iqr(axL, blocks, data[(d, REP_KAPPA)]["delta"], colors[i], label=f"$d={d}$")
    axL.set_xlabel("block $t$")
    axL.set_ylabel(r"normalized map defect $\delta_t$")
    axL.legend(ncol=1, handlelength=1.4)

    anchor = np.median([np.nanmedian(data[(d, REP_KAPPA)]["gap_acc"][:, 0]) for d in ds])
    ref = anchor * (blocks / blocks[0]) ** -2.0
    axR.plot(blocks, ref, ls="--", lw=0.8, color="0.55")
    j = len(blocks) * 2 // 3
    axR.text(blocks[j], ref[j] * 1.6, r"$O(1/t^2)$", color="0.45", fontsize=7)
    for i, d in enumerate(ds):                                  # right: objective gap
        med = np.maximum(np.nanmedian(data[(d, REP_KAPPA)]["gap_acc"], axis=0), 1e-300)
        axR.plot(blocks, med, color=colors[i], label=f"$d={d}$")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("block $t$")
    axR.set_ylabel("objective gap (accelerated)")

    jko.finish(fig, path)


#  ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".")
    ap.add_argument("--from-csv", default=None)
    args = ap.parse_args()

    out = args.out
    fig_path = os.path.join(out, "figures", "geometry.pdf")

    if args.from_csv:
        data, ds, kappas, seeds, n_blocks = load_csv(args.from_csv)
        config = {"rebuilt_from": args.from_csv, "ds": ds, "kappas": kappas,
                  "n_seeds": len(seeds), "n_blocks": n_blocks, "gamma": GAMMA,
                  "theta": REP_THETA}
        summary = build_summary(data, ds, kappas, n_blocks, config)
        json.dump(summary, open(os.path.join(os.path.dirname(args.from_csv), "summary.json"), "w"), indent=2)
        make_figure(data, ds, n_blocks, fig_path)
        h = summary["headline"]
        print(f"[from-csv] d={h['d']} kappa={h['kappa']:g}: median max-defect "
              f"{h['median_max_defect']:.3f}, acc slope {h['gap_slope_acc']:.2f}, "
              f"rate {'survived' if h['rate_survived'] else 'DEGRADED'}")
        return

    geometry_gate()

    if args.quick:
        ds, seeds, n_blocks = [2, 10], list(range(2)), 10
    else:
        ds, seeds, n_blocks = [2, 5, 10, 20, 50], list(range(args.seeds)), 60
    kappas = KAPPAS

    data, rows = collect(ds, kappas, seeds, n_blocks)
    config = {"ds": ds, "kappas": kappas, "n_seeds": len(seeds), "n_blocks": n_blocks,
              "gamma": GAMMA, "theta": REP_THETA, "rep_kappa": REP_KAPPA, "quick": args.quick}
    summary = build_summary(data, ds, kappas, n_blocks, config)

    jko.save_run("geometry", config, rows, summary, outdir=os.path.join(out, "results"))
    make_figure(data, ds, n_blocks, fig_path)

    h = summary["headline"]
    print(f"d={h['d']}, kappa={h['kappa']:g}, theta=pi/4: median max map-defect "
          f"delta_t = {h['median_max_defect']:.3f} (non-optimal composed map); "
          f"accelerated late-window gap slope = {h['gap_slope_acc']:.2f} -> "
          f"O(1/t^2) rate {'SURVIVED' if h['rate_survived'] else 'DEGRADED'}.")


if __name__ == "__main__":
    main()
