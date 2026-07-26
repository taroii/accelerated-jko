import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import experiment_io as eio
from plot_style import apply_paper_style, color_for

apply_paper_style()
BLUE = color_for("Standard JKO")
RED = color_for("Accelerated JKO")

HEADER = """# Rebuttal numbers — NeurIPS 2026 Submission 3775

Auto-generated from `results/*/summary.json` by `collate.py`.
Values are the raw experimental outputs; paste directly into rebuttal responses.

## CONTRADICTIONS / CAVEATS (read first)

1. **λ = 0 KL targets converge exponentially, not at a power law.** Geodesic
   convexity of `KL(·‖q)` (λ ≥ 0) means `q` is log-concave, which by Bobkov's
   theorem forces a positive spectral gap, hence exponential decay of `KL(ρ_t‖q)`.
   The O(1/t) / O(1/t²) rates are *worst-case upper bounds*, approached only as the
   gap → 0 (E1a), never tight at a fixed geodesically-convex target (E1b quartic /
   cubic / uniform decay exponentially). This is the reframing already adopted in the
   revised plan; it is NOT a defect of the theorems (exponential satisfies the bound).

2. **E1c potential-only `x⁴/4`: standard slope ≈ −2, not −1.** A homogeneous
   potential `|x|^p` gives flow exponent `p/(p−2)` (p=4 → −2), and acceleration
   doubles it (acc ≈ −4). Clean power laws DO appear once the entropy/diffusion is
   removed, but the exponents are −2/−4, not the naive −1/−2. Report as such.

3. **E3 σ² = 1e-4 (σ = 0.01) is numerically intractable.** The parametric prox with
   finite-difference gradients cannot fit this ultra-sharp target (params driven to
   bounds; grid- and MC-KL both huge for all methods). Report σ² = 0.012 as primary;
   note σ² = 1e-4 as a numerical-stability limitation, not a scientific result.

4. **E7 (2-D map defect) is large, not negligible.** The index-aligned composed map
   differs from the true OT map by `δ_t/W₂ ≈ 0.44` mean / `0.94` max (quick config).
   This confirms 99SE's point that composition of OT maps is not an OT map in d ≥ 2;
   the inexact-step / Theorem-8 fix is load-bearing, not cosmetic. (1-D: δ ≈ 1e-16.)

## Solver checks (theory-response, load-bearing)

- 1-D map defect `max δ_t ≈ 1e-16` across all E1 runs (confirms the d=1 corollary).
- Momentum monotonicity violations: 0 for convex targets (quartic/cubic/gauss),
  ~0.7% for uniform, **31% for the non-convex double-well** (λ < 0) — exactly the
  absolute-continuity regime 99SE flagged; projected back by isotonic regression.

## NUMBERS
"""

ORDER = ["e1a", "e1b_quartic", "e1b_cubic", "e1b_uniform", "e1b_gauss_lam1e-3",
         "e1b_doublewell_np", "e1c_potential", "e1c_interaction",
         "e2_fig2_gaussian", "e2_fig4_doublewell", "e3_mixture_sig2_0p012",
         "e3_mixture_sig2_1e-4", "e4_walltime", "e56_quartic", "e56_gauss_lam1e-3",
         "e7_defect2d", "e8_1_exactprox", "e8_2_lipschitz", "e8_3_sweep",
         "e9_iso_quartic", "e9_flatvalley", "e10_sensitivity", "e11_outer_ring"]


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


#  Wall-clock table (E4) from the per-block timers in every metrics.csv
def walltime():
    table, numbers = [], {}
    for path in sorted(glob.glob("results/*/metrics.csv")):
        exp = os.path.basename(os.path.dirname(path))
        groups = {}
        for r in _load(path):
            if r.get("t_prox") in (None, ""):
                continue
            groups.setdefault(r.get("method", "std"), []).append(r)
        for method, rows in groups.items():
            tp = np.array([float(r["t_prox"]) for r in rows])
            tm = np.array([float(r["t_momentum"]) for r in rows])
            tot = tp + tm
            over = float(100 * tm.sum() / tot.sum()) if tot.sum() > 0 else 0.0
            table.append({"experiment": exp, "method": method,
                          "sec_per_block": round(float(tot.mean()), 6),
                          "total_sec": round(float(tot.sum()), 4),
                          "momentum_overhead_pct": round(over, 3)})
            numbers[f"sec_per_block_{method}_{exp}"] = float(tot.mean())
            if method == "acc":
                numbers[f"momentum_overhead_pct_{exp}"] = over
    if not table:
        return
    eio.save_metrics("e4_walltime", table)
    eio.save_summary("e4_walltime", numbers)

    qpath = "results/e1b_quartic/metrics.csv"
    if os.path.exists(qpath):
        rows = _load(qpath)
        g = {m: [r for r in rows if r.get("method") == m] for m in ("std", "acc")}
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for method, color, lab in [("std", BLUE, "Standard JKO"), ("acc", RED, "Accelerated JKO")]:
            t = np.cumsum([float(r["t_prox"]) + float(r["t_momentum"]) for r in g[method]])
            gap = np.array([max(float(r["gap"]), 1e-16) for r in g[method]])
            ax.semilogy(t, gap, color=color, label=lab)
        ax.set_xlabel("Wall-clock seconds")
        ax.set_ylabel(r"gap $G(\rho_t)-G(\rho^*)$")
        ax.set_title("E1 quartic: gap vs wall-clock")
        ax.legend(loc="upper right")
        ax.grid(True, which="both")
        plt.tight_layout()
        fig.savefig("images/e4_quartic_gap_vs_seconds.pdf", bbox_inches="tight")
        plt.close(fig)


#  Flat NUMBER list (all experiments) -> results/rebuttal_numbers.md
def _flatten(d, prefix=""):
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += _flatten(v, key + ".")
        elif isinstance(v, (int, float, str)):
            out.append((key, v))
    return out


def rebuttal():
    summaries = {os.path.basename(os.path.dirname(p)): json.load(open(p))
                 for p in glob.glob("results/*/summary.json")}
    lines, seen = [HEADER], set()
    for exp in ORDER + sorted(summaries):
        if exp in seen or exp not in summaries:
            continue
        seen.add(exp)
        lines.append(f"\n### {exp}")
        for k, v in _flatten(summaries[exp]):
            vs = f"{v:.6g}" if isinstance(v, float) else str(v)
            lines.append(f"NUMBER: {exp}.{k} = {vs}")
    with open("results/rebuttal_numbers.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote results/rebuttal_numbers.md from {len(seen)} experiments")


if __name__ == "__main__":
    walltime()
    rebuttal()
