# Tables for the paper

## Outer scheme, exact steps (Gaussian target N(0, I/lambda))

| curvature lambda | steps to gap 1e-2: accelerated | JKO | steps to gap 1e-4: accelerated | JKO | gap after 100 steps: accelerated | JKO |
|---|---|---|---|---|---|---|
| 0.001 | **90** | >1000 | **103** | >1000 | **6.9e-4** | 1.4 |
| 0.01 | **29** | 198 | **34** | 425 | **2.3e-5** | 8.4e-2 |
| 0.1 | **9** | 17 | **12** | 40 | 1.2e-9 | **1.0e-9** |

## Inner solver: iterations to a relative gap, median over 50 random instances [95% CI]

| inner solver | to 1e-4 | to 1e-6 | to 1e-8 |
|---|---|---|---|
| scalar-step PDHG | >3000 (0/50) | >3000 (0/50) | >3000 (0/50) |
| preconditioned, (sig_L, sig_m) = (1, 1) | 154 [142, 161] | 277 [245, 305] | 425 [375, 476] |
| preconditioned, (sig_L, sig_m) = (1, 10) | 54 [53, 57] | **86** [81, 89] | **121** [111, 126] |
| preconditioned, (sig_L, sig_m) = (1, 100) | **52** [51, 56] | 89 [86, 92] | 128 [120, 135] |
| preconditioned, (sig_L, sig_m) = (10, 100) | 502 [491, 533] | 849 [819, 881] | 1229 [1160, 1285] |
| preconditioned, balanced steps | 57 [51, 61] | 98 [87, 102] | 136 [123, 148] |

## End to end: median normalised gap over 20 random instances

| inner accuracy | outer scheme | gap at t=50 | gap at t=100 | gap at t=300 | inner iterations |
|---|---|---|---|---|---|
| 50 inner iterations per step | accelerated (Alg. 1) | **2.9e-4** | **1.2e-5** | **4.7e-9** | 15000 |
| 50 inner iterations per step | non-accelerated | 2.9e-2 | 3.6e-3 | 8.2e-6 | 15000 |
| certified accuracy | accelerated (Alg. 1) | **2.9e-4** | **1.2e-5** | **4.7e-9** | 43412 |
| certified accuracy | non-accelerated | 2.9e-2 | 3.6e-3 | 8.2e-6 | 45975 |
