import numpy as np
from scipy.fft import dct, dctn, dst, dstn, idct, idctn, idst, idstn
from scipy.special import erf
from scipy.integrate import quad

SQ2 = np.sqrt(2.0)
_O = dict(type=2, norm="ortho")


def _shift(c, axis):
    c = np.moveaxis(c, axis, 0)
    out = np.zeros_like(c)
    out[:-1] = c[1:]
    return np.moveaxis(out, 0, axis)


def _unshift(d, axis):
    d = np.moveaxis(d, axis, 0)
    out = np.zeros_like(d)
    out[1:] = d[:-1]
    return np.moveaxis(out, 0, axis)


def apply_mixed_inverse(coef, sine_axes):
    a = coef
    for ax in sine_axes:
        a = _shift(a, ax)
    cos_axes = tuple(ax for ax in (0, 1) if ax not in sine_axes)
    if sine_axes:
        a = idstn(a, axes=tuple(sine_axes), **_O)
    if cos_axes:
        a = idctn(a, axes=cos_axes, **_O)
    return a


def apply_mixed_forward(f, sine_axes):
    a = f
    cos_axes = tuple(ax for ax in (0, 1) if ax not in sine_axes)
    if cos_axes:
        a = dctn(a, axes=cos_axes, **_O)
    if sine_axes:
        a = dstn(a, axes=tuple(sine_axes), **_O)
    for ax in sine_axes:
        a = _unshift(a, ax)
    return a


class Grid:

    def __init__(self, n):
        self.n = n
        self.h = 1.0 / n
        x = (np.arange(n) + 0.5) / n
        self.X = np.stack(np.meshgrid(x, x, indexing="ij"), -1)
        kap = np.pi * np.arange(n)
        self.kap = kap
        self.kx = kap[:, None]
        self.ky = kap[None, :]
        self.om = self.kx ** 2 + self.ky ** 2
        self._kb = [kap[:, None], kap[None, :]]
        with np.errstate(divide="ignore"):
            self.om_pinv = np.where(self.om > 0, 1.0 / self.om, 0.0)

    def coef(self, psi):
        return dctn(psi, **_O)

    def synth(self, c):
        return idctn(c, **_O)

    def K2c(self, c):
        return np.stack([-apply_mixed_inverse(self.kx * c, (0,)),
                         -apply_mixed_inverse(self.ky * c, (1,))], -1)

    def K1c(self, c):
        return np.stack([-idctn(self.kx ** 2 * c, **_O),
                         SQ2 * apply_mixed_inverse(self.kx * self.ky * c, (0, 1)),
                         -idctn(self.ky ** 2 * c, **_O)], -1)

    def K2T_c(self, m):
        return -self.kx * apply_mixed_forward(m[..., 0], (0,)) - self.ky * apply_mixed_forward(m[..., 1], (1,))

    def K1T_c(self, L):
        return (-self.kx ** 2 * dctn(L[..., 0], **_O)
                + SQ2 * self.kx * self.ky * apply_mixed_forward(L[..., 1], (0, 1))
                - self.ky ** 2 * dctn(L[..., 2], **_O))

    def K2(self, psi):
        return self.K2c(self.coef(psi))

    def K1(self, psi):
        return self.K1c(self.coef(psi))

    def K2T(self, m):
        return self.synth(self.K2T_c(m))

    def K1T(self, L):
        return self.synth(self.K1T_c(L))

    def dC(self, f, axis):
        return -idst(_shift(self._kb[axis] * dct(f, axis=axis, **_O), axis), axis=axis, **_O)

    def dS(self, f, axis):
        return idct(self._kb[axis] * _unshift(dst(f, axis=axis, **_O), axis), axis=axis, **_O)

    def B(self, g):
        return np.stack([self.dS(g[..., 0], 0),
                         (self.dC(g[..., 0], 1) + self.dC(g[..., 1], 0)) / SQ2,
                         self.dS(g[..., 1], 1)], -1)

    def BT(self, L):
        return np.stack([-self.dC(L[..., 0], 0) - self.dS(L[..., 1], 1) / SQ2,
                         -self.dS(L[..., 1], 0) / SQ2 - self.dC(L[..., 2], 1)], -1)

    def proj_grad(self, r):
        return self.K2c(self.om_pinv * self.K2T_c(r))


def hess_parts(H):
    a1 = 1.0 + H[..., 0]
    b = H[..., 1] / SQ2
    c1 = 1.0 + H[..., 2]
    return a1, b, c1, a1 * c1 - b * b


def admissible(H, U):
    a1, _, c1, det = hess_parts(H)
    return bool(a1.min() > 0 and c1.min() > 0 and det.min() > 0 and U.min() >= 0 and U.max() <= 1)


def _newton_box(F1, F2, u0, maxit=60):
    lo = np.zeros_like(u0)
    hi = np.ones_like(u0)
    at_lo = F1(lo) >= 0
    at_hi = F1(hi) <= 0
    u = np.clip(u0, 0.0, 1.0)
    for it in range(maxit):
        f = F1(u)
        lo = np.where(f < 0, u, lo)
        hi = np.where(f > 0, u, hi)
        un = u - f / F2(u)
        mid = 0.5 * (lo + hi)
        un = np.where((un >= lo) & (un <= hi), un, mid)
        if it >= 10:
            un = np.where(np.abs(un - u) > 0.5 * (hi - lo), mid, un)
        step = np.abs(un - u).max()
        u = un
        if step <= 4e-16:
            break
    return np.where(at_lo, 0.0, np.where(at_hi, 1.0, u))


class _Separable:
    separable = True
    convex = True

    def V(self, U):
        return self.v(U).sum(-1) + self.logZ

    def solve(self, A, Bq, a, L, Y, gamma, u0=None):
        F1 = lambda u: A * (self.dv(u) + (u - Y) / gamma) + Bq * (u - a) - L
        F2 = lambda u: A * (self.d2v(u) + 1.0 / gamma) + Bq
        return _newton_box(F1, F2, Y if u0 is None else u0)

    def conj(self, ut, X, Y, gamma, u0=None):
        z = self.solve(1.0, 0.0, 0.0, ut, Y, gamma)
        return ((z - X) * ut).sum(-1) - self.V(z) - ((z - Y) ** 2).sum(-1) / (2 * gamma)


class Gauss(_Separable):

    def __init__(self, m, s):
        self.m = np.asarray(m, float)
        self.s2 = np.asarray(s, float) ** 2
        sd = np.sqrt(self.s2)
        z1 = sd * np.sqrt(np.pi / 2) * (erf((1 - self.m) / (sd * SQ2)) - erf(-self.m / (sd * SQ2)))
        self.logZ = float(np.log(z1).sum())
        self.name = "gauss"

    def v(self, U):
        return (U - self.m) ** 2 / (2 * self.s2)

    def dv(self, U):
        return (U - self.m) / self.s2

    def d2v(self, U):
        return 1.0 / self.s2 + 0.0 * U


class Quartic(_Separable):

    def __init__(self, m, c):
        self.m = np.asarray(m, float)
        self.c = float(c)
        z1 = [quad(lambda t: np.exp(-self.c * (t - mi) ** 4 / 4), 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
              for mi in self.m]
        self.logZ = float(np.log(z1).sum())
        self.name = "quartic"

    def v(self, U):
        return self.c * (U - self.m) ** 4 / 4

    def dv(self, U):
        return self.c * (U - self.m) ** 3

    def d2v(self, U):
        return 3 * self.c * (U - self.m) ** 2


def _sq(a):
    return a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1]


class Mix2:
    separable = False
    convex = False

    def __init__(self, w, m1, m2, s):
        self.w = float(w)
        self.m1 = np.asarray(m1, float)
        self.m2 = np.asarray(m2, float)
        self.s2 = float(s) ** 2
        self.logw = np.log([self.w, 1 - self.w])
        sd = float(s)
        box = lambda m: np.prod(sd * np.sqrt(np.pi / 2) * (erf((1 - m) / (sd * SQ2)) - erf(-m / (sd * SQ2))))
        self.logZ = float(np.log(self.w * box(self.m1) + (1 - self.w) * box(self.m2)))
        self.name = f"mix2_w{w}"

    def _logparts(self, U):
        e1 = self.logw[0] - _sq(U - self.m1) / (2 * self.s2)
        e2 = self.logw[1] - _sq(U - self.m2) / (2 * self.s2)
        return e1, e2

    def V(self, U):
        return -np.logaddexp(*self._logparts(U)) + self.logZ

    def _post(self, U):
        e1, e2 = self._logparts(U)
        return np.exp(e1 - np.logaddexp(e1, e2))

    def grad(self, U):
        p1 = self._post(U)[..., None]
        return (U - p1 * self.m1 - (1 - p1) * self.m2) / self.s2

    def hess(self, U):
        p1 = self._post(U)
        d = self.m1 - self.m2
        k = p1 * (1 - p1) / self.s2 ** 2
        return 1 / self.s2 - k * d[0] * d[0], -k * d[0] * d[1], 1 / self.s2 - k * d[1] * d[1]

    def solve(self, A, Bq, a, L, Y, gamma, u0=None, steps=5):
        A = np.asarray(A, float)
        A0 = A[..., 0] if A.ndim else A
        lin = np.ndim(L) > 0
        F = lambda u: A0 * (self.V(u) + _sq(u - Y) / (2 * gamma)) + (Bq * _sq(u - a) / 2 if Bq else 0.0) \
            - ((L[..., 0] * u[..., 0] + L[..., 1] * u[..., 1]) if lin else 0.0)
        u = np.clip(Y if u0 is None else u0, 0.0, 1.0)
        for _ in range(steps):
            g = A * (self.grad(u) + (u - Y) / gamma) + Bq * (u - a) - L
            h11, h12, h22 = self.hess(u)
            h11 = A0 * (h11 + 1 / gamma) + Bq
            h22 = A0 * (h22 + 1 / gamma) + Bq
            h12 = A0 * h12
            lam_min = 0.5 * (h11 + h22) - np.sqrt(0.25 * (h11 - h22) ** 2 + h12 ** 2)
            scale = 0.5 * (np.abs(h11) + np.abs(h22)) + 1e-300
            sh = np.maximum(0.0, 1e-3 * scale - lam_min)
            h11 = h11 + sh
            h22 = h22 + sh
            act = ((u <= 0) & (g > 0)) | ((u >= 1) & (g < 0))
            det = h11 * h22 - h12 * h12
            d0 = -(h22 * g[..., 0] - h12 * g[..., 1]) / det
            d1 = -(h11 * g[..., 1] - h12 * g[..., 0]) / det
            d0 = np.where(act[..., 1], -g[..., 0] / h11, d0)
            d1 = np.where(act[..., 0], -g[..., 1] / h22, d1)
            d = np.stack([np.where(act[..., 0], 0.0, d0), np.where(act[..., 1], 0.0, d1)], -1)
            f0 = F(u)
            best = u
            done = np.zeros(f0.shape, bool)
            al = 1.0
            slack = 1e-14 * (1.0 + np.abs(f0))
            for _bt in range(8):
                cand = np.clip(u + al * d, 0.0, 1.0)
                ok = (F(cand) <= f0 + slack) & ~done
                best = np.where(ok[..., None], cand, best)
                done |= ok
                if done.all():
                    break
                al *= 0.5
            u = best
        return u

    def conj(self, ut, X, Y, gamma, u0=None):
        val = None
        for mk in (self.m1, self.m2):
            z0 = np.clip((mk / self.s2 + Y / gamma + ut) / (1 / self.s2 + 1 / gamma), 0.0, 1.0)
            z = self.solve(1.0, 0.0, 0.0, ut, Y, gamma, u0=z0, steps=12)
            v = ((z - X) * ut).sum(-1) - self.V(z) - _sq(z - Y) / (2 * gamma)
            val = v if val is None else np.maximum(val, v)
        return val


def truncated_gaussian_weights(grid, mean, sd):
    d = ((grid.X - np.asarray(mean, float)) ** 2).sum(-1)
    w = np.exp(-d / (2 * sd ** 2))
    return w / w.mean()
