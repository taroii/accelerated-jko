"""Inner solver (spec section 0.6, Algorithm 2 of the paper): two-channel spectrally
preconditioned PDHG for one restricted proximal step on the grid, the duality certificate,
and the outer scheme (Algorithm 1) driven by it.

Units. Cell weights are stored as wt = nu(x_cell)/mean(nu) (wt = 1 for the uniform reference),
i.e. the spec's w(x) = nu(x) h^2 with the factor h^2 dropped. `J` and the dual objects are sums
over cells in these units; everything returned as a reported value (G, certificate, tolerances)
is multiplied by h^2."""
import time

import numpy as np

from grid import SQ2, admissible, hess_parts


class ProxProblem:
    """min_psi J(psi) = sum wt [ -log det(I + Hess psi) + V(x + grad psi) + |x + grad psi - Y|^2/(2 gamma) ]."""

    def __init__(self, grid, target, gamma, Y, wt=None):
        self.grid = grid
        self.target = target
        self.gamma = gamma
        self.Y = Y
        self.wt = np.ones((grid.n, grid.n)) if wt is None else wt
        self.h2 = grid.h ** 2
        # constant sum w log nu of the reference entropy (0 for the uniform reference)
        self.ent0 = float((self.wt * np.log(self.wt)).sum()) * self.h2

    def values(self, H, g):
        """(admissible, J, G) with J in internal units and G = KL(S#nu | q) in reported units."""
        U = self.grid.X + g
        if not admissible(H, U):
            return False, np.inf, np.inf
        det = hess_parts(H)[3]
        base = (self.wt * (-np.log(det) + self.target.V(U))).sum()
        J = base + (self.wt * ((U - self.Y) ** 2).sum(-1)).sum() / (2 * self.gamma)
        return True, float(J), float(base) * self.h2 + self.ent0


def prox_h1(M, tw):
    """prox_{t h1}(M), h1(H) = w (-log det(I+H)), tw = t w: minimiser of -tw log det(I+P) + |P-M|_F^2/2."""
    A = M[..., 0]
    Bm = M[..., 1] / SQ2
    C = M[..., 2]
    half = 0.5 * (A + C)
    disc = np.sqrt(0.25 * (A - C) ** 2 + Bm * Bm)
    f = lambda e: 0.5 * ((e + 1) + np.sqrt((e + 1) ** 2 + 4 * tw)) - 1
    p1 = f(half + disc)
    p2 = f(half - disc)
    th = 0.5 * np.arctan2(2 * Bm, A - C)
    c = np.cos(th)
    s = np.sin(th)
    return np.stack([p1 * c * c + p2 * s * s, SQ2 * (p1 - p2) * c * s, p1 * s * s + p2 * c * c], -1)


def h1_conj(Lam, wt):
    """h1*(Lam) = sum wt f*(Lam/wt), f*(L) = -tr L - 2 - log det(-L) for L negative definite."""
    a = Lam[..., 0] / wt
    b = Lam[..., 1] / (SQ2 * wt)
    c = Lam[..., 2] / wt
    det = a * c - b * b
    if not (a.max() < 0 and c.max() < 0 and det.min() > 0):
        return np.inf
    return float((wt * (-(a + c) - 2 - np.log(det))).sum())


def certificate(prob, J_adm, Lam, mu, R=1.0):
    """Duality certificates for an admissible psi with objective J_adm (reported units).

    Both use the lower bound J* >= -h1*(Lam) - h2*(mu) - R ||v||_1 from weak duality with
    K1 = B K2 and |grad psi| <= R. `spec` takes v = r := B^T Lam + mu (Remark 4 as written);
    `proj` takes v = P r, the projection of r onto gradient fields, which is equally valid
    because <K2 psi, r> = <K2 psi, P r>, and is the one that vanishes at a saddle point
    (there K2^T r = 0, but r itself need not be zero)."""
    g = prob.grid
    r = g.BT(Lam) + mu
    dual = -h1_conj(Lam, prob.wt) - float(
        (prob.wt * prob.target.conj(mu / prob.wt[..., None], g.X, prob.Y, prob.gamma)).sum())
    spec = (J_adm - (dual - R * np.abs(r).sum())) * prob.h2
    proj = (J_adm - (dual - R * np.abs(g.proj_grad(r)).sum())) * prob.h2
    return spec, proj


def new_state(grid):
    n = grid.n
    return dict(c=np.zeros((n, n)), Lam=np.zeros((n, n, 3)), mu=np.zeros((n, n, 2)), u=None)


def pdhg(prob, sig_L, sig_m, max_iter, theta=0.99, scalar=False, state=None, tol=None,
         cert_every=25, cert_kind="proj", record=False):
    """Run Algorithm 2 for at most `max_iter` iterations from `state` (warm start; zeros if None).

    scalar=True is the Chambolle–Pock baseline T = tau I with tau = sigma = sqrt(theta)/||K||.
    Stops early when the certificate (checked every `cert_every` iterations) is <= tol.
    Returns the last admissible iterate: its coefficients `c`, fields `H`, `g`, values `J`, `G`.
    With record=True also the per-iteration J (inf where inadmissible) and all certificates."""
    grid, tgt, wt = prob.grid, prob.target, prob.wt
    wt2 = wt[..., None]
    if scalar:
        normK2 = (grid.om ** 2 + grid.om).max()
        sig_L = sig_m = np.sqrt(theta / normK2)
        Tk = np.where(grid.om > 0, sig_L, 0.0)
    else:
        with np.errstate(divide="ignore"):
            Tk = np.where(grid.om > 0, theta / (sig_L * grid.om ** 2 + sig_m * grid.om), 0.0)
    st = new_state(grid) if state is None else state
    c, Lam, mu, u = st["c"], st["Lam"], st["mu"], st["u"]
    H, g = grid.K1c(c), grid.K2c(c)
    Hb, gb = H, g
    ok, J, G = prob.values(H, g)
    last = dict(c=c, H=H, g=g, J=J, G=G) if ok else None
    Jhist = np.full(max_iter, np.inf) if record else None
    certs = []
    cert = (np.inf, np.inf)
    t0 = time.perf_counter()
    k = 0
    for k in range(1, max_iter + 1):
        # entropy channel: Lam <- prox_{sig_L h1*}(Lam + sig_L K1 psib)
        v = Lam + sig_L * Hb
        Lam = v - sig_L * prox_h1(v / sig_L, wt2[..., 0] / sig_L)
        # transport channel: mu <- prox_{sig_m h2*}(mu + sig_m K2 psib)
        v = mu + sig_m * gb
        u = tgt.solve(wt2 / sig_m, 1.0, grid.X + v / sig_m, 0.0, prob.Y, prob.gamma, u0=u)
        mu = v - sig_m * (u - grid.X)
        # primal: psi <- psi - T (K1^T Lam + K2^T mu), mean removed (Tk = 0 on constants)
        c = c - Tk * (grid.K1T_c(Lam) + grid.K2T_c(mu))
        Hn, gn = grid.K1c(c), grid.K2c(c)
        Hb, gb = 2 * Hn - H, 2 * gn - g
        H, g = Hn, gn
        ok, J, G = prob.values(H, g)
        if ok:
            last = dict(c=c, H=H, g=g, J=J, G=G)
        if record:
            Jhist[k - 1] = J
        if cert_every and (k % cert_every == 0 or k == max_iter) and last is not None:
            cert = certificate(prob, last["J"], Lam, mu)
            if record:
                certs.append((k, cert[0], cert[1], last["J"]))
            if tol is not None and cert[cert_kind == "proj"] <= tol:
                break
    out = dict(state=dict(c=c, Lam=Lam, mu=mu, u=u), last=last, iters=k,
               cert_spec=cert[0], cert=cert[1], time=time.perf_counter() - t0,
               sig=(float(sig_L), float(sig_m)))
    if record:
        out["J"] = Jhist[:k]
        out["certs"] = np.array(certs).reshape(-1, 4)
    return out


def run_outer(grid, target, gamma, T, accel, sig, wt=None, eps0=None, cap=5000, fixed=None,
              tol_const=None, snapshots=(), log=None):
    """Algorithm 1 on the grid with X_0 = Z_0 = Id; alpha_t = 2/(t+2) (accel) or 1 (minimizing movement).

    Inner accuracy: `fixed` iterations per step, or the certified schedule eps_t = eps0 (t+1)^-4.5
    (or the constant `tol_const`), capped at `cap` inner iterations. Duals and psi are warm-started."""
    X = grid.X.copy()
    Z = grid.X.copy()
    state = new_state(grid)
    p0 = ProxProblem(grid, target, gamma, X, wt)
    G = [p0.values(grid.K1c(state["c"]), grid.K2c(state["c"]))[2]]
    inner, wall, hit, cert, fails = [], [], [], [], 0
    snaps = {}
    H = np.zeros((grid.n, grid.n, 3))
    if 0 in snapshots:
        snaps[0] = (X.copy(), p0.wt / hess_parts(H)[3])
    t0 = time.perf_counter()
    for t in range(T):
        a = 2.0 / (t + 2) if accel else 1.0
        Y = (1 - a) * X + a * Z
        prob = ProxProblem(grid, target, gamma, Y, wt)
        if fixed is not None:
            res = pdhg(prob, sig[0], sig[1], fixed, state=state, cert_every=None)
            tol = None
        else:
            tol = tol_const if tol_const is not None else eps0 * (t + 1) ** -4.5
            res = pdhg(prob, sig[0], sig[1], cap, state=state, tol=tol)
        state = res["state"]
        if res["last"] is None:          # no admissible iterate in this solve: keep the previous map
            fails += 1
            X1, Gt = X, G[-1]
        else:
            X1, Gt, H = grid.X + res["last"]["g"], res["last"]["G"], res["last"]["H"]
        Z = Z + (X1 - Y) / a
        X = X1
        G.append(Gt)
        inner.append(res["iters"])
        cert.append(res["cert"])
        hit.append(tol is not None and not res["cert"] <= tol)
        wall.append(time.perf_counter() - t0)
        if t + 1 in snapshots:
            snaps[t + 1] = (X.copy(), prob.wt / hess_parts(H)[3])
        if log:
            log(t + 1, Gt, res["iters"], res["cert"])
    return dict(G=np.array(G), inner=np.array(inner), wall=np.array(wall), cap_hit=np.array(hit),
                cert=np.array(cert), X=X, Z=Z, c=state["c"], fails=fails, snaps=snaps)
