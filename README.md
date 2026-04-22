# Accelerated Proximal Optimization in Wasserstein and Sobolev Spaces

This repository is the official implementation of **Accelerated Proximal Optimization in Wasserstein and Sobolev Spaces**.

We propose a unified accelerated proximal framework for optimization in Wasserstein and $\dot{\mathbb{H}}^{-1}$ spaces, and demonstrate its power across two fundamental problems in optimal transport and generative modeling. First, we develop an accelerated proximal scheme for optimization in the space of probability measures. Unlike prior work, which requires *strong* convexity of the objective functional $G$ along generalized geodesics, our analysis requires only (non-strong) geodesic convexity, yet achieves an $O(1/t^2)$ convergence rate in both the objective gap and Wasserstein distance. The key mechanism is Nesterov-style momentum applied directly in Wasserstein space via the Jordan–Kinderlehrer–Otto (JKO) scheme, which realizes proximal gradient descent in the space of probability measures. Second, we introduce an Accelerated Sobolev Gradient Ascent (ASGA) algorithm for computing Wasserstein barycenters in the dual space. Under $L$-smoothness of the Kantorovich dual functional in the $\dot{\mathbb{H}}^1$ geometry, ASGA achieves $O(1/t^2)$ convergence — improving from the $O(1/\sqrt{T})$ rate of the (non-accelerated) Sobolev Gradient Ascent of Kim (2025) — while retaining the constraint-free structure that eliminates expensive $c$-concavity projections.

This repository contains the closed-form Gaussian experiments, the non-strongly-convex ($\lambda = 0$) case, the ASGA barycenter experiment, and the neural transport-map experiments that together reproduce every figure in the paper.

## Requirements

Create a fresh environment and install the Python dependencies:

```setup
conda create -n ajko python=3.11
conda activate ajko
pip install -r requirements.txt
```

Install PyTorch separately to match your hardware. On CPU only:

```setup
pip install torch torchvision
```

On CUDA (example — substitute your CUDA version):

```setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

See https://pytorch.org/get-started/locally/ for the correct command for your system.

## Training

The neural transport experiments train a fresh residual MLP $T_\theta(x) = x + \mathrm{MLP}(x)$ at every JKO block via mini-batch Adam. To reproduce the particle-transport figures for all five 2-D targets (bunny, rings, rectangle, disk, outer_ring):

```train
python jko_densities.py --target all
```

Default hyperparameters (matching the paper): `--gamma 0.08 --blocks 25 --particles 12000 --epochs 800 --seed 0`. Select a single target with `--target rings` (or any one name). Each block solves a JKO proximal step by minimising $-\mathbb{E}[\log q(T(x)) + \log|\det \nabla T(x)|] + \|T(x) - x\|^2 / (2\gamma)$ for 800 epochs with cosine-annealed Adam.

The other three experiments use closed-form or numerical proximal steps (no training):

```train
python jko_comparison.py        # 1-D Gaussians, closed-form
python jko_lambda0.py           # double-well target, numerical L-BFGS proximal step
python jko_asga.py              # ASGA Wasserstein barycenter on a 1-D grid
```

## Evaluation

Convergence is measured directly in the scripts above: `jko_densities.py` reports per-block Sinkhorn $W_2$ between particles and a reference sample from the target (via `geomloss`); the Gaussian and mixture experiments track KL in closed form or by `scipy.integrate.quad`; the ASGA experiment tracks the duality gap $I^\star - I(f^{(t)})$, with $I^\star$ estimated by a long (3000-iteration) ASGA run.

No separate evaluation script is needed — each experiment saves its figure to `images/` on completion.

## Pre-trained Models

Not applicable. The JKO transport maps are re-trained from scratch per block and are not reused across runs, so no model weights are released. The closed-form and mixture experiments have no learned parameters.

## Results

Running the four scripts above regenerates every figure in the paper:

| Script              | Output                                                      | What it shows                                                                                                                     |
| ------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `jko_comparison.py` | `images/figure_1.png`, `images/figure_2.png`                | Closed-form 1-D Gaussians. Fig 1: $O(t^{-1})$ vs. $O(t^{-2})$ rates (weakly convex, $\lambda = 0.04$) and exponential vs. $O(t^{-2})$ rates (strongly convex, $\lambda = 1$). Fig 2: final KL vs. block count $N$ and vs. $\lambda$. |
| `jko_lambda0.py`    | `images/figure_3.png`                                       | Symmetric Gaussian-mixture target with $\lambda = 0$ exactly. Accelerated JKO matches the $O(t^{-2})$ bound; standard JKO stalls at an error floor. |
| `jko_asga.py`       | `images/figure_4.png`                                       | Wasserstein barycenter of four 1-D Gaussians. ASGA duality gap empirically tracks $O(t^{-2})$; SGA tracks $O(t^{-1})$.             |
| `jko_densities.py`  | `images/particles_{bunny,rings,rectangle,disk,outer_ring}.png` | Neural JKO on five 2-D targets. Each figure shows particle snapshots for standard vs. accelerated JKO plus a Sinkhorn $W_2$-vs-block convergence strip. |

All figures land in `images/`.

## Contributing

This code is released under the MIT License (see `LICENSE`). Issues and pull requests are welcome.
