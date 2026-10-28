# -*- coding: utf-8 -*-
"""Euclidean O(4) bounce solvers for a scalar field with an explicit potential, in flat space and with Einstein gravity.

Potential (dimensionless; the field is measured in units of f and the potential in units of f^4):

    v(x; lam, eta, v0) = lam * [ (x^2 - 1)^2 / 8 + eta * x / 2 ] + c(lam, eta) + v0

The tilt term makes x ~ -1 the true vacuum and x ~ +1 the false vacuum. The constant c is chosen so that the TRUE minimum
sits at v = v0 (v0 = 0: decay to a Minkowski vacuum; v0 < 0: AdS; v0 > 0: de Sitter). The false-vacuum energy is then
v_f = v0 + lam * dtil(eta), with dtil the difference between the two minima in units of lam.

Gravity enters through a single dimensionless number g = f^2 / M_Pl^2 (reduced Planck mass), because the equations are scale
free once the field is in units of f. With xi the Euclidean arclength and rho the O(4) radius (both in units of 1/f):

    x'' + 3 (rho'/rho) x' = dv/dx
    rho'' = -(g rho / 3) (x'^2 + v)             rho'^2 = 1 + (g rho^2 / 3)(x'^2/2 - v)

The Euclidean action of the bounce relative to the false vacuum is

    B = S[bounce] - S[false vacuum],      S = 4 pi^2 * Integral ( rho^3 v - 3 rho / g ) d xi,    S[dS] = -24 pi^2 / (g^2 v_f).

In flat space (g -> 0) B = 2 pi^2 * Integral rho^3 (x'^2/2 + v - v_f) d rho. The thin-wall value is B = 27 pi^2 sigma^4 / (2 DeltaV^3).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ------------------------------------------------------------------------------------------------------- potential
class Potential:
    def __init__(self, lam, eta, v0=0.0):
        self.lam, self.eta = float(lam), float(eta)
        # stationary points of  p(x) = (x^2-1)^2/8 + eta x/2 :  p' = x (x^2 - 1)/2 + eta/2 = 0  ->  x^3 - x + eta = 0
        roots = np.roots([1.0, 0.0, -1.0, self.eta])
        real = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
        self.has_false_vacuum = real.size == 3
        if self.has_false_vacuum:
            self.x_t, self.x_top, self.x_f = real            # true minimum, barrier top, false minimum
            p = lambda x: (x * x - 1.0) ** 2 / 8.0 + self.eta * x / 2.0
            self._c = -p(self.x_t)
            self.dtil = p(self.x_f) - p(self.x_t)             # energy difference / lam
            self.btil = p(self.x_top) - p(self.x_f)           # barrier height above the false vacuum / lam
            self.v_t = v0
            self.v_f = v0 + self.lam * self.dtil
            self.v_top = v0 + self.lam * (self.dtil + self.btil)
            self.m2_top = self.lam * (3.0 * self.x_top ** 2 - 1.0) / 2.0     # v'' at the barrier top (negative)
            self.m2_f = self.lam * (3.0 * self.x_f ** 2 - 1.0) / 2.0
        self.v0 = v0

    def v(self, x):
        return self.lam * ((x * x - 1.0) ** 2 / 8.0 + self.eta * x / 2.0) + self._c * self.lam + self.v0

    def dv(self, x):
        return self.lam * (x * (x * x - 1.0) / 2.0 + self.eta / 2.0)

    def wall_tension(self):
        """Thin-wall tension (the symmetric-limit integral of sqrt(2 v) between the minima, lam-dependent)."""
        xs = np.linspace(self.x_t, self.x_f, 20001)
        vv = np.clip(self.lam * ((xs ** 2 - 1.0) ** 2 / 8.0), 0.0, None)
        return float(np.trapezoid(np.sqrt(2.0 * vv), xs))


def eta_crit():
    """The false minimum exists only for eta < eta_c = 2 / (3 sqrt 3)."""
    return 2.0 / (3.0 * np.sqrt(3.0))


# ------------------------------------------------------------------------------------------------ flat-space bounce
def _flat_rhs(r, y, pot):
    x, p = y
    return [p, pot.dv(x) - 3.0 * p / r]


def _flat_shoot(x0, pot, r_max=400.0, rtol=1e-11, atol=1e-13):
    """Integrate from the centre. Returns +1 for overshoot (crosses the false minimum), -1 for undershoot, and the path."""
    r0 = 1e-6
    d = pot.dv(x0)
    y0 = [x0 + d * r0 ** 2 / 8.0, d * r0 / 4.0]

    def over(r, y, pot):
        return y[0] - pot.x_f
    over.terminal, over.direction = True, 1

    def under(r, y, pot):
        return y[1]
    under.terminal, under.direction = True, -1

    sol = solve_ivp(_flat_rhs, [r0, r_max], y0, args=(pot,), events=(over, under), rtol=rtol, atol=atol, dense_output=True,
                    method="DOP853")
    if sol.t_events[0].size:
        return +1, sol
    if sol.t_events[1].size:
        return -1, sol
    return -1, sol


def flat_bounce(lam, eta, tol=1e-13, r_max=400.0, with_profile=False):
    """Flat-space O(4) bounce; returns B (action relative to the false vacuum), the central field value and the profile."""
    pot = Potential(lam, eta)
    if not pot.has_false_vacuum:
        return None
    # x_s: the point on the true-vacuum side with v(x_s) = v_f (energy-conserving undershoot limit)
    xs_grid = np.linspace(pot.x_t, pot.x_top, 4001)
    f = lambda x: pot.v(x) - pot.v_f
    sgn = np.sign(f(xs_grid))
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return None
    x_s = brentq(f, xs_grid[idx[0]], xs_grid[idx[0] + 1], xtol=1e-14)
    lo, hi = x_s, pot.x_t + 1e-9          # lo: undershoot; hi: overshoot (checked below)
    # bisect on x0 between the undershoot (x_s) and the overshoot (near x_t) ends
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        kind, sol = _flat_shoot(mid, pot, r_max)
        if kind > 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    x0 = 0.5 * (lo + hi)
    kind, sol = _flat_shoot(x0, pot, r_max)
    # action from the profile up to where the solution has settled (the integrand is negligible there)
    r_end = sol.t[-1]
    rr = np.linspace(1e-6, r_end, 200001)
    xx, pp = sol.sol(rr)
    integrand = rr ** 3 * (pp ** 2 / 2.0 + pot.v(xx) - pot.v_f)
    B = 2.0 * np.pi ** 2 * float(np.trapezoid(integrand, rr))
    out = dict(B=B, x0=x0, pot=pot, r_end=r_end)
    if with_profile:
        out["r"], out["x"], out["p"] = rr, xx, pp
    return out


def thin_wall_B(lam, eta):
    pot = Potential(lam, eta)
    sig = pot.wall_tension()
    dV = pot.lam * pot.dtil
    return 27.0 * np.pi ** 2 * sig ** 4 / (2.0 * dV ** 3)


# --------------------------------------------------------------------------------------------- bounce with gravity
def _grav_rhs(s, y, pot, g):
    x, p, rho, rp = y
    return [p, pot.dv(x) - 3.0 * rp * p / rho, rp, -(g * rho / 3.0) * (p * p + pot.v(x))]


def _grav_shoot(x0, pot, g, s_max):
    s0 = 1e-6
    v0 = pot.v(x0)
    d = pot.dv(x0)
    y0 = [x0 + d * s0 ** 2 / 8.0, d * s0 / 4.0, s0 - g * v0 * s0 ** 3 / 18.0, 1.0 - g * v0 * s0 ** 2 / 6.0]

    def over(s, y, pot, g):
        return y[0] - pot.x_f
    over.terminal, over.direction = True, 1

    def under(s, y, pot, g):
        return y[1]
    under.terminal, under.direction = True, -1

    def pole(s, y, pot, g):                    # second pole (dS false vacuum): rho returns to zero
        return y[2] - 1e-4
    pole.terminal, pole.direction = True, -1

    sol = solve_ivp(_grav_rhs, [s0, s_max], y0, args=(pot, g), events=(over, under, pole), rtol=1e-11, atol=1e-13,
                    dense_output=True, method="DOP853")
    if sol.t_events[0].size:
        return +1, sol
    return -1, sol


def grav_bounce(lam, eta, g, v0=0.0, tol=1e-13):
    """CDL bounce (false vacuum with v_f > 0, i.e. a de Sitter parent). Returns None if no oscillating bounce is found."""
    pot = Potential(lam, eta, v0)
    if not pot.has_false_vacuum or pot.v_f <= 0:
        return None
    H = np.sqrt(g * pot.v_f / 3.0)
    s_max = 4.0 * np.pi / H if g * pot.v_f > 0 else 400.0
    xs_grid = np.linspace(pot.x_t, pot.x_top, 4001)
    f = lambda x: pot.v(x) - pot.v_f
    sgn = np.sign(f(xs_grid))
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return None
    x_s = brentq(f, xs_grid[idx[0]], xs_grid[idx[0] + 1], xtol=1e-14)
    lo, hi = x_s, pot.x_t + 1e-9
    k_lo, _ = _grav_shoot(lo, pot, g, s_max)
    k_hi, _ = _grav_shoot(hi, pot, g, s_max)
    if k_lo > 0 or k_hi < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        kind, sol = _grav_shoot(mid, pot, g, s_max)
        if kind > 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    x0 = 0.5 * (lo + hi)
    kind, sol = _grav_shoot(x0, pot, g, s_max)
    return dict(x0=x0, pot=pot, sol=sol, H=H)


def grav_action(res, g, n=400001):
    """B = S[bounce] - S[dS]. The part of the bounce outside the wall is the false-vacuum dS geometry with the other pole, so
    the difference is taken analytically there (keeps the 1/g^2 pieces from cancelling in floating point)."""
    pot, sol, H = res["pot"], res["sol"], res["H"]
    # end of the wall region: where the field is within a small distance of the false vacuum
    s = np.linspace(sol.t[0], sol.t[-1], n)
    x, p, rho, rp = sol.sol(s)
    near = np.where(np.abs(x - pot.x_f) < 1e-3 * (pot.x_f - pot.x_t))[0]
    iw = near[0] if near.size else len(s) - 1
    sw, rw, rpw = s[iw], rho[iw], rp[iw]
    integrand = rho[: iw + 1] ** 3 * pot.v(x[: iw + 1]) - 3.0 * rho[: iw + 1] / g
    S_in = 4.0 * np.pi ** 2 * float(np.trapezoid(integrand, s[: iw + 1]))
    # the rest of the bounce follows rho = sin(H eta)/H towards the second pole; eta_w is the remaining arclength
    sin_w = np.clip(H * rw, 0.0, 1.0)
    th_w = np.arcsin(sin_w)
    # tail angle measured from the second pole: rho(eta) = sin(H eta)/H, rho' = -cos(H eta)
    th = th_w if rpw < 0 else np.pi - th_w          # rpw < 0 : past the equator, small remaining angle
    # remaining arclength is th/H  (angle from the second pole), full dS range is pi/H
    # S_tail(bounce) = 4 pi^2 Integral_0^{th/H} (v_f rho^3 - 3 rho/g) d eta;  S_bg = 4 pi^2 Integral_0^{pi/H} (same)
    def F(a):          # integral of (v_f sin^3(H e)/H^3 - 3 sin(H e)/(g H)) d e from 0 to a/H
        c = np.cos(a)
        i3 = (2.0 / 3.0 - c + c ** 3 / 3.0) / H ** 4          # Integral_0^{a/H} sin^3(H e) d e / H^3 ... (see below)
        i1 = (1.0 - c) / H ** 2
        return pot.v_f * i3 - 3.0 * i1 / g
    S_tail = 4.0 * np.pi ** 2 * F(th)
    S_bg = 4.0 * np.pi ** 2 * F(np.pi)
    return S_in + S_tail - S_bg


# --------------------------------------------------------------------------------------------- analytic references
def hm_action(lam, eta, g, v0=0.0):
    """Hawking-Moss action  B_HM = 24 pi^2 (1/V_f - 1/V_top) / g^2   (reduced Planck units, rescaled potential)."""
    pot = Potential(lam, eta, v0)
    return 24.0 * np.pi ** 2 * (1.0 / pot.v_f - 1.0 / pot.v_top) / g ** 2


def parke_thin_wall(sigma, dV, v_f, g, v_t=None):
    """Thin-wall CDL action with gravity (Parke 1983) for a dS or Minkowski false vacuum, rescaled units (M_Pl^2 = 1/g).
    kappa = g. B = B0 * r(x, y), B0 = 27 pi^2 sigma^4 / (2 dV^3), computed directly from the wall radius:
        1/rho^2 = 1/rho0^2 + kappa V_f/3 ... (standard); here the radius and action are evaluated from the exact thin-wall formulas
    """
    kap = g
    v_t = v_f - dV if v_t is None else v_t
    # wall radius: 1/rho^2 = kap*v_f/3 + (dV/(3 sigma) - kap sigma/4)^2
    inv_rho2 = kap * v_f / 3.0 + (dV / (3.0 * sigma) - kap * sigma / 4.0) ** 2
    rho = 1.0 / np.sqrt(inv_rho2)
    # action of a thin-wall bounce with wall radius rho between a dS/Mink false vacuum and a dS/Mink/AdS true vacuum:
    # B = 2 pi^2 sigma rho^3 + (12 pi^2 / kap^2) [ ((1 - kap rho^2 v_t/3)^{3/2} - 1)/v_t - ((1 - kap rho^2 v_f/3)^{3/2} - 1)/v_f ]
    # (signs of the square roots follow the geometry; valid while the wall lies before the equator of the false vacuum)
    def term(vv, sgn=1.0):
        if abs(vv) < 1e-14:
            return -kap * rho ** 2 / 2.0 * sgn           # limit  ((1 - a v)^{3/2} - 1)/v  ->  -3 a/2 with a = kap rho^2/3
        return ((1.0 - kap * rho ** 2 * vv / 3.0) ** 1.5 - 1.0) / vv
    return 2.0 * np.pi ** 2 * sigma * rho ** 3 + (12.0 * np.pi ** 2 / kap ** 2) * (term(v_t) - term(v_f))


# ------------------------------------------------------------------------------ flat-space bounce by boundary-value solver
def flat_bounce_bvp(lam, eta, r_pad=30.0):
    """Independent check of flat_bounce using a collocation boundary-value solver. It does not suffer from the exponential
    sensitivity of the shooting method in the thin-wall regime. Initial guess: a tanh wall at the thin-wall radius."""
    from scipy.integrate import solve_bvp
    pot = Potential(lam, eta)
    if not pot.has_false_vacuum:
        return None
    dV = pot.lam * pot.dtil
    sig = pot.wall_tension()
    Rc = 3.0 * sig / dV
    r_max = Rc * 1.0 + r_pad + 6.0 / np.sqrt(abs(pot.m2_f))
    r0 = 1e-4
    w = max(1.0 / np.sqrt(pot.lam), 0.3)
    r = np.linspace(r0, r_max, 4000)
    xc = 0.5 * (pot.x_t + pot.x_f)
    xg = xc + 0.5 * (pot.x_f - pot.x_t) * np.tanh((r - Rc) / w)
    pg = 0.5 * (pot.x_f - pot.x_t) / w / np.cosh((r - Rc) / w) ** 2

    def rhs(r, y):
        return np.vstack([y[1], pot.dv(y[0]) - 3.0 * y[1] / r])

    def bc(ya, yb):
        return np.array([ya[1] - pot.dv(ya[0]) * r0 / 4.0, yb[0] - pot.x_f])

    sol = solve_bvp(rhs, bc, r, np.vstack([xg, pg]), tol=1e-9, max_nodes=400000)
    if not sol.success:
        return dict(B=np.nan, success=False)
    rr = np.linspace(r0, r_max, 400001)
    xx, pp = sol.sol(rr)
    B = 2.0 * np.pi ** 2 * float(np.trapezoid(rr ** 3 * (pp ** 2 / 2.0 + pot.v(xx) - pot.v_f), rr))
    return dict(B=B, success=True, x0=float(sol.sol(r0)[0]), pot=pot, Rc=Rc)
