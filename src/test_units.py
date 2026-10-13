import sys

import numpy as np

import gaussian as ga
from grid import Gauss, Grid, Mix2, Quartic, SQ2, truncated_gaussian_weights
from pdhg import ProxProblem, certificate, h1_conj, pdhg, prox_h1

RNG = np.random.default_rng(0)
REPORT = []


def check(name, err, tol):
    REPORT.append((name, err, tol))
    assert err <= tol, f"{name}: {err:.3e} > {tol:.1e}"


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_derivatives_analytic():
    for n in (16, 64):
        g = Grid(n)
        x, y = g.X[..., 0], g.X[..., 1]
        a, b = 3 * np.pi, 2 * np.pi
        psi = np.cos(a * x) * np.cos(b * y)
        G, H = g.K2(psi), g.K1(psi)
        check(f"psi_x  analytic (n={n})", np.abs(G[..., 0] + a * np.sin(a * x) * np.cos(b * y)).max(), 1e-10)
        check(f"psi_y  analytic (n={n})", np.abs(G[..., 1] + b * np.cos(a * x) * np.sin(b * y)).max(), 1e-10)
        check(f"psi_xx analytic (n={n})", np.abs(H[..., 0] + a * a * psi).max(), 1e-10)
        check(f"psi_xy analytic (n={n})", np.abs(H[..., 1] / SQ2 - a * b * np.sin(a * x) * np.sin(b * y)).max(), 1e-10)
        check(f"psi_yy analytic (n={n})", np.abs(H[..., 2] + b * b * psi).max(), 1e-10)


def test_adjoints():
    g = Grid(32)
    psi = RNG.normal(size=(32, 32))
    m = RNG.normal(size=(32, 32, 2))
    L = RNG.normal(size=(32, 32, 3))
    for name, lhs, rhs in [("K2", (g.K2(psi) * m).sum(), (psi * g.K2T(m)).sum()),
                           ("K1", (g.K1(psi) * L).sum(), (psi * g.K1T(L)).sum()),
                           ("B ", (g.B(m) * L).sum(), (m * g.BT(L)).sum())]:
        check(f"adjoint <{name} u, w> = <u, {name}^T w>", abs(lhs - rhs) / abs(lhs), 1e-12)


def test_multipliers():
    g = Grid(32)
    psi = RNG.normal(size=(32, 32))
    c = g.coef(psi)
    check("K2^T K2 = diag(om)", _rel(g.K2T(g.K2(psi)), g.synth(g.om * c)), 1e-10)
    check("K1^T K1 = diag(om^2)", _rel(g.K1T(g.K1(psi)), g.synth(g.om ** 2 * c)), 1e-10)


def test_K1_equals_B_K2():
    g = Grid(32)
    psi = RNG.normal(size=(32, 32))
    check("K1 = B K2", _rel(g.B(g.K2(psi)), g.K1(psi)), 1e-10)
    r = RNG.normal(size=(32, 32, 2))
    P = g.proj_grad(r)
    check("proj_grad idempotent", _rel(g.proj_grad(P), P), 1e-10)
    check("K2^T (r - P r) = 0", np.abs(g.K2T(r - P)).max() / np.abs(g.K2T(r)).max(), 1e-10)


def _mat(v):
    return np.array([[v[0], v[1] / SQ2], [v[1] / SQ2, v[2]]])


def test_entropy_prox():
    M = RNG.normal(size=(50, 3)) * 3
    tw = np.exp(RNG.normal(size=50))
    P = prox_h1(M, tw)
    worst_kkt = worst_fd = 0.0
    obj = lambda p, m, t: -t * np.log(np.linalg.det(np.eye(2) + _mat(p))) + 0.5 * ((p - m) ** 2).sum()
    for p, m, t in zip(P, M, tw):
        Pm = _mat(p)
        assert np.linalg.eigvalsh(np.eye(2) + Pm).min() > 0
        worst_kkt = max(worst_kkt, np.abs(Pm - _mat(m) - t * np.linalg.inv(np.eye(2) + Pm)).max())
        f0 = obj(p, m, t)
        for _ in range(20):
            f1 = obj(p + 1e-4 * RNG.normal(size=3), m, t)
            worst_fd = max(worst_fd, f0 - f1)
    check("entropy prox: stationarity P - M - tw (I+P)^-1 = 0", worst_kkt, 1e-10)
    check("entropy prox: finite-difference minimality", worst_fd, 0.0)


def _targets():
    return [Gauss((0.5, 0.5), (np.sqrt(0.02), np.sqrt(0.05))), Quartic((0.5, 0.5), 200.0),
            Mix2(0.9, (0.3, 0.5), (0.7, 0.5), 0.06)]


def test_transport_prox():
    g = Grid(16)
    gamma = 0.05
    Y = g.X + 0.1 * RNG.normal(size=g.X.shape)
    for tgt in _targets():
        for t in (2e-4,) if not tgt.convex else (2e-4, 0.1, 3.0):
            tw = t * np.exp(0.3 * RNG.normal(size=(16, 16, 1)))
            a = g.X + 0.3 * RNG.normal(size=g.X.shape)
            u = tgt.solve(tw, 1.0, a, 0.0, Y, gamma, u0=None if tgt.convex else np.clip(a, 0, 1), steps=40) \
                if not tgt.convex else tgt.solve(tw, 1.0, a, 0.0, Y, gamma)
            F = lambda z: tw[..., 0] * (tgt.V(z) + ((z - Y) ** 2).sum(-1) / (2 * gamma)) + ((z - a) ** 2).sum(-1) / 2
            f0 = F(u)
            worst = 0.0
            for _ in range(30):
                worst = max(worst, (f0 - F(np.clip(u + 1e-4 * RNG.normal(size=u.shape), 0, 1))).max())
            check(f"transport prox minimality ({tgt.name}, t={t})", worst, 1e-13)
            assert u.min() >= 0 and u.max() <= 1


def test_normalisation():
    g = Grid(400)
    for tgt in _targets():
        check(f"q = exp(-V) integrates to 1 on the box ({tgt.name})",
              abs(np.exp(-tgt.V(g.X)).mean() - 1), 1e-5)


def test_conjugates():
    g = Grid(16)
    gamma = 0.05
    Y = g.X + 0.1 * RNG.normal(size=g.X.shape)
    for tgt in _targets():
        ut = 30 * RNG.normal(size=g.X.shape)
        gs = tgt.conj(ut, g.X, Y, gamma)
        gfun = lambda z: tgt.V(z) + ((z - Y) ** 2).sum(-1) / (2 * gamma)
        worst = -np.inf
        for _ in range(200):
            z = RNG.uniform(size=g.X.shape)
            worst = max(worst, (((z - g.X) * ut).sum(-1) - gfun(z) - gs).max())
        check(f"Fenchel-Young for g* ({tgt.name})", worst, 1e-10)
    wt = np.exp(RNG.normal(size=(8, 8)))
    P0 = 0.5 * RNG.normal(size=(8, 8, 3))
    L = -wt[..., None] * prox_h1(P0, 1.0) - wt[..., None] * np.array([1.0, 0.0, 1.0])
    hs = h1_conj(L, wt)
    worst = -np.inf
    for _ in range(200):
        Hh = RNG.normal(size=(8, 8, 3)) * 0.4
        det = (1 + Hh[..., 0]) * (1 + Hh[..., 2]) - Hh[..., 1] ** 2 / 2
        if det.min() <= 0 or (1 + Hh[..., 0]).min() <= 0:
            continue
        worst = max(worst, (Hh * L).sum() - (wt * -np.log(det)).sum() - hs)
    check("Fenchel-Young for h1*", worst, 1e-10)
    a, b, c = L[..., 0] / wt, L[..., 1] / (SQ2 * wt), L[..., 2] / wt
    det = a * c - b * b
    Hs = np.stack([-c / det - 1, SQ2 * b / det, -a / det - 1], -1)
    check("h1* attained at I + H = -(L/w)^-1", abs((Hs * L).sum() - (wt * np.log(det)).sum() - hs) / abs(hs), 1e-12)


def test_moreau_entropy_dual_in_domain():
    L = RNG.normal(size=(100, 3)) * 5
    s = 2.0
    wt = np.exp(RNG.normal(size=100))
    Ln = L - s * prox_h1(L / s, wt / s)
    assert np.isfinite(h1_conj(Ln, wt))
    P = prox_h1(L / s, wt / s)
    worst = max(np.abs(_mat(l) + w * np.linalg.inv(np.eye(2) + _mat(p))).max() for l, p, w in zip(Ln, P, wt))
    check("Moreau: prox_{s h1*}(L) = grad h1(prox_{h1/s}(L/s))", worst, 1e-10)


def _e2_field(g):
    x, y = g.X[..., 0], g.X[..., 1]
    gx = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y)
    gy = -4 * np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y) - np.pi * np.sin(2 * np.pi * y)
    return g.X + 0.08 * np.stack([gx, gy], -1)


def test_weak_duality_and_convergence():
    g = Grid(24)
    for tgt in _targets()[:2]:
        for wt in (None, truncated_gaussian_weights(g, (0.52, 0.5), 0.25)):
            prob = ProxProblem(g, tgt, 0.05, _e2_field(g), wt)
            ref = pdhg(prob, 1.0, 10.0, 6000, record=True)
            Jstar = ref["J"].min()
            run = pdhg(prob, 1.0, 10.0, 1500, record=True)
            cs = run["certs"]
            gap = cs[:, 3] - Jstar
            tag = f"{tgt.name}, {'uniform' if wt is None else 'non-uniform'} nu"
            check(f"weak duality, projected certificate >= gap ({tag})", ((gap - cs[:, 2] / prob.h2)).max(), 1e-9)
            check(f"weak duality, spec certificate >= gap ({tag})", ((gap - cs[:, 1] / prob.h2)).max(), 1e-9)
            check(f"projected certificate -> 0 ({tag})", ref["cert"], 1e-9)
            check(f"PDHG converges: J_1500 - J* ({tag})", (run["J"][-1] - Jstar) * prob.h2, 1e-9)


def test_scalar_baseline_runs():
    g = Grid(16)
    prob = ProxProblem(g, _targets()[0], 0.05, _e2_field(g))
    r = pdhg(prob, None, None, 200, scalar=True, record=True)
    s = r["sig"][0]
    check("scalar steps: tau sigma ||K||^2 = theta", abs(s * s * (g.om ** 2 + g.om).max() - 0.99), 1e-12)
    assert r["last"] is not None


def _rand_spd():
    M = RNG.normal(size=(2, 2))
    return M @ M.T + 0.2 * np.eye(2)


def test_gaussian_ot():
    worst = worst_w = 0.0
    for _ in range(50):
        A, B = _rand_spd(), _rand_spd()
        T = ga.ot_map(A, B)
        worst = max(worst, np.abs(T @ A @ T - B).max())
        worst_w = max(worst_w, abs(ga.w2sq(A, B) - ga.norm2_nu(T - np.eye(2), A)))
    check("OT map: T A T = B", worst, 1e-10)
    check("W2^2(A,B) = ||T - I||_A^2", worst_w, 1e-10)
    check("KL(S|S) = 0", abs(ga.kl(A, A)), 1e-13)
    worst = 0.0
    for _ in range(50):
        A = _rand_spd()
        M = RNG.normal(size=(2, 2))
        worst = max(worst, np.abs(ga.inv(M) - np.linalg.inv(M)).max() / np.abs(np.linalg.inv(M)).max(),
                    np.abs(ga.sqrtm_spd(A) @ ga.sqrtm_spd(A) - A).max(), np.abs(ga.chol(A) - np.linalg.cholesky(A)).max(),
                    abs(ga.min_eig(A) - np.linalg.eigvalsh(A)[0]), abs(ga.logdet(A) - np.linalg.slogdet(A)[1]))
        J3, b3 = RNG.normal(size=(3, 3)), RNG.normal(size=3)
        worst = max(worst, np.abs(ga.solve3(J3, b3) - np.linalg.solve(J3, b3)).max() / np.abs(np.linalg.solve(J3, b3)).max())
    check("closed-form 2x2 inverse / sqrt / Cholesky / eig / logdet, 3x3 solve", worst, 1e-10)


def test_gaussian_restricted_prox():
    worst = 0.0
    worst_min = -np.inf
    for _ in range(20):
        Zc, Q, Y = _rand_spd(), _rand_spd() * 3, RNG.normal(size=(2, 2)) + np.eye(2)
        gam = float(np.exp(RNG.normal()))
        S, r = ga.restricted_prox(Y, gam, Zc, Q, np.eye(2))
        worst = max(worst, r)
        f0 = ga._prox_obj(S, Y, gam, Zc, np.linalg.inv(Q))
        for _ in range(20):
            D = ga.sym(1e-4 * RNG.normal(size=(2, 2)))
            worst_min = max(worst_min, f0 - ga._prox_obj(S + D, Y, gam, Zc, np.linalg.inv(Q)))
    check("restricted prox: KKT residual", worst, 1e-10)
    check("restricted prox: minimality over symmetric perturbations", worst_min, 0.0)


def test_gaussian_jko():
    worst = worst_cf = worst_min = 0.0
    for _ in range(20):
        Sy, Q = _rand_spd(), _rand_spd()
        gam = float(np.exp(RNG.normal()))
        S, r = ga.jko_step(Sy, gam, Q)
        worst = max(worst, r)
        f0 = ga.kl(S, Q) + ga.w2sq(S, Sy) / (2 * gam)
        for _ in range(20):
            Sp = S + ga.sym(1e-4 * RNG.normal(size=(2, 2)))
            worst_min = max(worst_min, f0 - ga.kl(Sp, Q) - ga.w2sq(Sp, Sy) / (2 * gam))
        S1, _ = ga.jko_step(Sy, gam, np.eye(2))
        w, V = np.linalg.eigh(Sy)
        u = (np.sqrt(w) + np.sqrt(w + 4 * gam * (1 + gam))) / (2 * (1 + gam))
        worst_cf = max(worst_cf, np.abs(S1 - (V * u ** 2) @ V.T).max())
    check("JKO step: residual of M S M = Sy", worst, 1e-9)
    check("JKO step: minimality", worst_min, 1e-12)
    check("JKO step: closed form for Q = I", worst_cf, 1e-9)


def test_gaussian_commuting_case():
    Zc, Q, gam = np.diag([2.0, 0.7]), np.diag([1.0, 3.0]), 0.4
    Y = np.diag([1.3, 0.8])
    S, _ = ga.restricted_prox(Y, gam, Zc, Q, np.eye(2))
    Sj, _ = ga.jko_step(Y @ Zc @ Y, gam, Q)
    check("commuting case: restricted prox = JKO", np.abs(S @ Zc @ S - Sj).max(), 1e-9)


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        k0 = len(REPORT)
        try:
            t()
            status = "PASS"
        except AssertionError as e:
            status = f"FAIL  {e}"
            failed += 1
        print(f"[{status}] {t.__name__}")
        for name, err, tol in REPORT[k0:]:
            print(f"      {name:<66s} {err:10.3e}  (tol {tol:.0e})")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} tests passed")
    sys.exit(1 if failed else 0)
