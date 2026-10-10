Reference prototypes (NOT the final code; they use a periodic torus / FFT and predate the spec):
- pdhg_inner.py : torus prototype of the two-channel preconditioned PDHG (Gaussian target, quadratic transport prox in closed form). Shows the proxes and the preconditioner working.
- dct_check.py  : finite-difference DCT diagonalization check (the spec uses the cosine-series collocation instead; multipliers kappa_k = pi k).
- gauss_check.py: exact Gaussian machinery (OT maps, W2, closed-form JKO for Q = I, and `lin_prox` = the restricted prox over symmetric maps via BFGS).
