import numpy as np
from scipy.optimize import least_squares, minimize

I2 = np.eye(2)


def sym(M):
    return 0.5 * (M + M.T)


def det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def inv(M):
    d = det2(M)
    if d == 0 or not np.isfinite(d):
        raise np.linalg.LinAlgError("singular 2x2 matrix")
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / d


def logdet(M):
    d = det2(M)
    if not d > 0:
        raise np.linalg.LinAlgError("non-positive determinant")
    return np.log(d)


def min_eig(S):
    return 0.5 * (S[0, 0] + S[1, 1]) - np.sqrt(0.25 * (S[0, 0] - S[1, 1]) ** 2 + S[0, 1] ** 2)


def sqrtm_spd(S):
    S = sym(S)
    r = np.sqrt(max(det2(S), 0.0))
    t = np.sqrt(max(S[0, 0] + S[1, 1] + 2 * r, 0.0))
    if t == 0:
        return np.zeros((2, 2))
    return (S + r * I2) / t


def chol(S):
    if not (S[0, 0] > 0 and det2(S) > 0):
        raise np.linalg.LinAlgError("matrix not positive definite")
    a = np.sqrt(S[0, 0])
    b = S[1, 0] / a
    return np.array([[a, 0.0], [b, np.sqrt(S[1, 1] - b * b)]])


def fro(M):
    return float(np.sqrt((M * M).sum()))


def solve3(A, b):
    c0 = np.cross(A[1], A[2])
    c1 = np.cross(A[2], A[0])
    c2 = np.cross(A[0], A[1])
    d = A[0] @ c0
    if d == 0 or not np.isfinite(d):
        raise np.linalg.LinAlgError("singular 3x3 matrix")
    return (c0 * b[0] + c1 * b[1] + c2 * b[2]) / d


def ot_map(A, B):
    r = sqrtm_spd(A)
    ri = inv(r)
    return sym(ri @ sqrtm_spd(r @ B @ r) @ ri)


def w2sq(A, B):
    r = sqrtm_spd(A)
    return float(np.trace(A) + np.trace(B) - 2 * np.trace(sqrtm_spd(r @ B @ r)))


def kl(S, Q):
    M = inv(Q) @ S
    return float(0.5 * (np.trace(M) - 2 - logdet(M)))


def norm2_nu(M, Zc):
    return float(np.trace(M @ Zc @ M.T))


def _chol_to_spd(p):
    L = np.array([[np.exp(p[0]), 0.0], [p[1], np.exp(p[2])]])
    return L @ L.T, L


def _spd_to_chol(S):
    L = chol(sym(S))
    return np.array([np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])])


def _prox_obj(S, Y, gamma, Zc, Qi):
    D = S - Y
    return 0.5 * (np.trace(Qi @ S @ Zc @ S) - 2 - logdet(Qi) - 2 * logdet(S) - logdet(Zc)) \
        + np.trace(D @ Zc @ D.T) / (2 * gamma)


def prox_kkt(S, Y, gamma, Zc, Qi):
    return sym(0.5 * (Qi @ S @ Zc + Zc @ S @ Qi) - inv(S) + (S - Y) @ Zc / gamma)


def restricted_prox(Y, gamma, Zc, Q, S_init):
    Qi = inv(Q)

    def fg(p):
        S, L = _chol_to_spd(p)
        GL = 2 * prox_kkt(S, Y, gamma, Zc, Qi) @ L
        return _prox_obj(S, Y, gamma, Zc, Qi), np.array([GL[0, 0] * L[0, 0], GL[1, 0], GL[1, 1] * L[1, 1]])

    p = minimize(fg, _spd_to_chol(S_init), jac=True, method="BFGS", options=dict(gtol=1e-13, maxiter=2000)).x
    p = minimize(lambda q: fg(q)[0], p, method="Nelder-Mead",
                 options=dict(xatol=1e-13, fatol=1e-15, maxiter=2000)).x
    S = _chol_to_spd(p)[0]
    basis = [np.array([[1.0, 0], [0, 0]]), np.array([[0, 1.0], [1.0, 0]]), np.array([[0, 0], [0, 1.0]])]
    vec = lambda M: np.array([M[0, 0], M[0, 1], M[1, 1]])
    res = fro(prox_kkt(S, Y, gamma, Zc, Qi))
    for _ in range(50):
        Si = inv(S)
        Jm = np.stack([vec(sym(0.5 * (Qi @ E @ Zc + Zc @ E @ Qi) + Si @ E @ Si + E @ Zc / gamma))
                       for E in basis], 1)
        d = solve3(Jm, -vec(prox_kkt(S, Y, gamma, Zc, Qi)))
        D = np.array([[d[0], d[1]], [d[1], d[2]]])
        al, improved = 1.0, False
        while al > 1e-4:
            Sn = S + al * D
            if min_eig(Sn) > 0:
                rn = fro(prox_kkt(Sn, Y, gamma, Zc, Qi))
                if rn < res:
                    S, res, improved = Sn, rn, True
                    break
            al *= 0.5
        if not improved:
            break
    return sym(S), float(res)


def jko_step(Sy, gamma, Q):
    Qi = inv(Q)

    def resid(p):
        S = _chol_to_spd(p)[0]
        M = I2 + gamma * (Qi - inv(S))
        R = M @ S @ M - Sy
        return np.array([R[0, 0], R[0, 1], R[1, 1]])

    tol = 1e-9 * max(1.0, np.abs(Sy).max())
    best = None
    for S0 in (Sy, Q, 0.5 * (Sy + Q)):
        sol = least_squares(resid, _spd_to_chol(S0), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        S = _chol_to_spd(sol.x)[0]
        M = I2 + gamma * (Qi - inv(S))
        r = float(np.abs(resid(sol.x)).max())
        if min_eig(sym(M)) > 0 and r < tol:
            return sym(S), r
        if best is None or r < best[1]:
            best = (S, r)
    raise RuntimeError(f"JKO step not solved (best residual {best[1]:.2e})")


_FAIL = (RuntimeError, np.linalg.LinAlgError, ValueError, FloatingPointError)


def _pad(vals, T):
    return np.array(list(vals) + [np.nan] * (T + 1 - len(vals)))


def run_fixed_reference(Zc, Q, gamma, T, accel):
    P = ot_map(Zc, Q)
    X = I2.copy()
    Z = I2.copy()
    gap = [kl(X @ Zc @ X.T, Q)]
    zdist = [np.sqrt(norm2_nu(Z - P, Zc))]
    kkt = 0.0
    for t in range(T):
        a = 2.0 / (t + 2) if accel else 1.0
        Y = (1 - a) * X + a * Z
        try:
            X1, r = restricted_prox(Y, gamma, Zc, Q, X)
        except _FAIL:
            return _pad(gap, T), _pad(zdist, T), kkt
        kkt = max(kkt, r)
        Z = Z + (X1 - Y) / a
        X = X1
        gap.append(kl(X @ Zc @ X.T, Q))
        zdist.append(np.sqrt(norm2_nu(Z - P, Zc)))
    return np.array(gap), np.array(zdist), kkt


def run_moving_reference(Zc, Q, gamma, T):
    Zt = Zc.copy()
    Sx = Zc.copy()
    gap = [kl(Sx, Q)]
    res = 0.0
    for t in range(T):
        a = 2.0 / (t + 2)
        X = ot_map(Zt, Sx)
        Y = (1 - a) * X + a * I2
        Sy = sym(Y @ Zt @ Y.T)
        try:
            Sx, r = jko_step(Sy, gamma, Q)
            X1 = ot_map(Sy, Sx) @ Y
        except _FAIL:
            return _pad(gap, T), res
        res = max(res, r)
        W = I2 + (X1 - Y) / a
        Zt = sym(W @ Zt @ W.T)
        gap.append(kl(Sx, Q))
    return np.array(gap), res


def run_jko(Zc, Q, gamma, T):
    Sx = Zc.copy()
    gap = [kl(Sx, Q)]
    res = 0.0
    for t in range(T):
        try:
            Sx, r = jko_step(Sx, gamma, Q)
        except _FAIL:
            return _pad(gap, T), res
        res = max(res, r)
        gap.append(kl(Sx, Q))
    return np.array(gap), res
