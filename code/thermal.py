# -*- coding: utf-8 -*-
"""Finite-temperature extension: does cooling change the decay of the metastable vacuum?   (Steps 1-4 of the research program)

Thermally corrected potential (leading high-temperature term, a scalar coupled to a bath with coefficient c):

        V_eff(phi, T) = V_0(phi) + (1/2) c T^2 phi^2 ,       theta  =  c T^2 / (lam f^2)  (dimensionless thermal strength)

Exact reduction. With x = phi/f and p(x) = (x^2-1)^2/8 + eta x/2 + theta x^2/2 = (x^2 - (1-2 theta))^2 / 8 + eta x / 2 + const, the substitution x = s y,
s = sqrt(1 - 2 theta), gives  p = s^4 [ (y^2-1)^2/8 + (eta/s^3) y/2 ] + const : the thermal mass is a rescaling of the zero-temperature family with

        eta_eff = eta / (1 - 2 theta)^(3/2)          (false minimum exists iff eta_eff < eta_c = 2/(3 sqrt 3), i.e. theta < theta_c(eta))

and, because the action of a bounce for V = A w(phi/F) is B = (F^4/A) b(eta) with F = f s, A = lam f^4 s^4,
        O(4) (quantum):    B_4(eta, theta) = b(eta_eff) / lam                        (b: the zero-temperature table of bounce_tables.json)
        O(3) (thermal):    S_3(eta, theta) = (f s / sqrt(lam)) * s3(eta_eff),   s3 = 4 pi Integral rho^2 [x'^2/2 + w - w_f] d rho  (zero-theta function)
so only b(eta) and s3(eta) are needed, and both are tabulated once. Bounces with an explicit thermal term are solved directly as a check (verify.py V19).

Which saddle dominates at temperature T: the O(3) (static, period 1/T) one if S_3/T < B_4, otherwise the O(4); the rate takes the exponent of the dominant saddle (the two
are not added). The thermal prefactor is  Gamma_T = D_T T^4 (S_3 / 2 pi T)^(3/2) e^{-S_3/T}  (D_T of order unity, varied as a systematic), the quantum one is that of rates.py.

Cooling: T(a) = T_0 / a for the CMB (T_0 = 2.7255 K) from the observed expansion history (survival.py), and the de Sitter horizon temperature T_dS = H_Lambda / 2 pi is kept separate
(it is 10^-30 of T_0 and is not a homogeneous bath). The survival probability uses a time-dependent rate on the past light cone,
        I = Integral dt Gamma(t) (4 pi / 3) a^3 chi^3 ,   P_surv = exp(-I).

    python thermal.py          (writes thermal_tables.json: O(3) table, validation, thermal maps)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import logsumexp

import cdl_lib as C
import constants as K
import rates as R
import survival as S

HERE = os.path.dirname(os.path.abspath(__file__))
ETA_C = C.eta_crit()
T0_MEV = K.T_CMB_GEV * 1e12                      # CMB temperature today in meV
HL_MEV = K.H_L * 1e12                            # asymptotic de Sitter rate in meV
OMEGA_L2 = K.OMEGA_L ** 2


# ----------------------------------------------------------------------------------------------- generic O(d) bounce solver (d = 3, 4)
def _shoot(x0, dv, x_f, d, r_max=400.0, rtol=1e-11, atol=1e-13):
    r0 = 1e-6
    g = dv(x0)
    y0 = [x0 + g * r0 ** 2 / (2.0 * d), g * r0 / d]

    def rhs(r, y):
        return [y[1], dv(y[0]) - (d - 1.0) * y[1] / r]

    def over(r, y):
        return y[0] - x_f
    over.terminal, over.direction = True, 1

    def under(r, y):
        return y[1]
    under.terminal, under.direction = True, -1
    sol = solve_ivp(rhs, [r0, r_max], y0, events=(over, under), rtol=rtol, atol=atol, dense_output=True, method="DOP853")
    return (+1 if sol.t_events[0].size else -1), sol


def bounce_shoot(v, dv, x_t, x_top, x_f, d):
    """O(d) bounce by overshoot/undershoot bisection on the central value. Returns the action Integral Omega_d r^(d-1) [x'^2/2 + v - v_f] dr and x0."""
    v_f = v(x_f)
    xs_grid = np.linspace(x_t, x_top, 4001)
    f = lambda x: v(x) - v_f
    sgn = np.sign(f(xs_grid))
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return None
    x_s = brentq(f, xs_grid[idx[0]], xs_grid[idx[0] + 1], xtol=1e-14)
    lo, hi = x_s, x_t + 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        kind, _ = _shoot(mid, dv, x_f, d)
        if kind > 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-13:
            break
    x0 = 0.5 * (lo + hi)
    kind, sol = _shoot(x0, dv, x_f, d)
    rr = np.linspace(1e-6, sol.t[-1], 200001)
    xx, pp = sol.sol(rr)
    om = {4: 2.0 * np.pi ** 2, 3: 4.0 * np.pi}[d]
    return om * float(np.trapezoid(rr ** (d - 1) * (pp ** 2 / 2.0 + v(xx) - v_f), rr)), x0


def bounce_bvp(v, dv, x_t, x_f, d, sigma, dV, r_pad=30.0, m_f=0.8):
    """Collocation bounce for the thin-wall regime (shooting is exponentially sensitive there). sigma, dV: thin-wall estimates for the initial guess."""
    v_f = v(x_f)
    Rc = (d - 1.0) * sigma / dV
    r_max = Rc + r_pad + 6.0 / m_f
    r0 = 1e-4
    r = np.linspace(r0, r_max, 4000)
    w = 0.5
    xg = 0.5 * (x_t + x_f) + 0.5 * (x_f - x_t) * np.tanh((r - Rc) / w)
    pg = 0.5 * (x_f - x_t) / w / np.cosh((r - Rc) / w) ** 2

    def rhs(rr, y):
        return np.vstack([y[1], dv(y[0]) - (d - 1.0) * y[1] / rr])

    def bc(ya, yb):
        return np.array([ya[1] - dv(ya[0]) * r0 / d, yb[0] - x_f])
    sol = solve_bvp(rhs, bc, r, np.vstack([xg, pg]), tol=1e-9, max_nodes=400000)
    if not sol.success:
        return None
    rr = np.linspace(r0, r_max, 400001)
    xx, pp = sol.sol(rr)
    om = {4: 2.0 * np.pi ** 2, 3: 4.0 * np.pi}[d]
    return om * float(np.trapezoid(rr ** (d - 1) * (pp ** 2 / 2.0 + v(xx) - v_f), rr))


def action_zero_theta(eta, d):
    """b(eta) (d = 4) or s3(eta) (d = 3) at lam = 1, theta = 0."""
    p = C.Potential(1.0, eta)
    if eta < 0.10:
        return bounce_bvp(p.v, p.dv, p.x_t, p.x_f, d, p.wall_tension(), p.dtil, m_f=np.sqrt(abs(p.m2_f)))
    return bounce_shoot(p.v, p.dv, p.x_t, p.x_top, p.x_f, d)[0]


def action_with_theta(eta, theta, d):
    """Direct solution with the thermal term in the potential (check of the exact reduction): lam = 1, f = 1."""
    p0 = C.Potential(1.0, eta)
    v = lambda x: p0.v(x) + 0.5 * theta * x ** 2
    dv = lambda x: p0.dv(x) + theta * x
    roots = np.roots([1.0, 0.0, -(1.0 - 2.0 * theta), eta])
    real = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
    if real.size < 3:
        return None
    x_t, x_top, x_f = real
    return bounce_shoot(v, dv, x_t, x_top, x_f, d)[0]


# ----------------------------------------------------------------------------------------------- tables and analytic scalings
_T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
_ETA = np.array(sorted(r["eta"] for r in _T["flat"]))
_DTIL = PchipInterpolator(_ETA, np.log([r["dtil"] for r in sorted(_T["flat"], key=lambda r: r["eta"])]))
_S3_PATH = os.path.join(HERE, "thermal_tables.json")
_S3 = None


def _load_s3():
    global _S3
    if _S3 is None:
        tab = json.load(open(_S3_PATH, encoding="utf-8"))["s3_table"]
        e = np.array([r["eta"] for r in tab])
        _S3 = PchipInterpolator(e, np.log([r["s3"] for r in tab]))
    return _S3


def s3_of_eta(eta):
    """O(3) action s3(eta) at lam = 1: the table for 0.03 <= eta <= 0.38, the thin-wall power law s3 ~ eta^-2 below (negative thermal mass makes eta_eff small)."""
    e = np.asarray(eta, dtype=float)
    lo = np.exp(_load_s3()(R.ETA_MIN)) * (R.ETA_MIN / np.maximum(e, 1e-6)) ** 2
    return np.where(e < R.ETA_MIN, lo, np.exp(_load_s3()(np.clip(e, R.ETA_MIN, R.ETA_MAX))))


def b_ext(eta):
    """O(4) action b(eta) at lam = 1 with the thin-wall power law b ~ eta^-3 below the table (eta < 0.03)."""
    e = np.asarray(eta, dtype=float)
    lo = float(R.b_of_eta(R.ETA_MIN)) * (R.ETA_MIN / np.maximum(e, 1e-6)) ** 3
    return np.where(e < R.ETA_MIN, lo, R.b_of_eta(np.clip(e, R.ETA_MIN, R.ETA_MAX)))


def theta_critical(eta):
    """The false minimum disappears when theta reaches theta_c(eta) = [1 - (eta/eta_c)^(2/3)] / 2."""
    return 0.5 * (1.0 - (np.asarray(eta, dtype=float) / ETA_C) ** (2.0 / 3.0))


def eta_eff(eta, theta):
    return np.asarray(eta, dtype=float) / np.maximum(1.0 - 2.0 * np.asarray(theta, dtype=float), 1e-12) ** 1.5


def f_scale_gev(lam, eta, r_L=1.0):
    """f from V_f = rho_Lambda at the present temperature: f^4 = V_f / (r_L lam dtil)."""
    return (K.V_F / (r_L * lam * np.exp(_DTIL(np.clip(eta, R.ETA_MIN, R.ETA_MAX))))) ** 0.25


def theta_of_T(c, lam, eta, T_mev, r_L=1.0):
    f_mev = f_scale_gev(lam, eta, r_L) * 1e12
    return c * (np.asarray(T_mev, dtype=float) / f_mev) ** 2 / lam


def exponents(lam, eta, c, T_mev, r_L=1.0):
    """(B_4, S_3/T, theta, exists) at temperature T for the family (lam, eta) with thermal coefficient c. B_4 = inf where no false minimum exists."""
    lam, eta, c, T = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (lam, eta, c, T_mev)])
    f_mev = f_scale_gev(lam, eta, r_L) * 1e12
    theta = c * (T / f_mev) ** 2 / lam
    ee = eta_eff(eta, theta)
    exists = ee < ETA_C
    ee_c = np.minimum(ee, R.ETA_MAX)
    s = np.sqrt(np.maximum(1.0 - 2.0 * theta, 1e-12))
    B4 = b_ext(ee_c) / lam
    S3T = (f_mev * s / np.sqrt(lam)) * s3_of_eta(ee_c) / np.maximum(T, 1e-300)
    return np.where(exists, B4, np.inf), np.where(exists, S3T, np.inf), theta, exists


def ln_rate_over_hl4(lam, eta, c, T_mev, lnD=0.0, lnDT=0.0, r_L=1.0):
    """ln(Gamma / H_Lambda^4) at temperature T, with the exponent of the dominant saddle and its prefactor. Returns (ln_rate, channel) with channel 4 or 3 (0: no false minimum)."""
    B4, S3T, theta, exists = exponents(lam, eta, c, T_mev, r_L)
    dtil = np.exp(_DTIL(np.clip(eta, R.ETA_MIN, R.ETA_MAX)))
    Kq = (np.where(np.isfinite(B4), B4, 1.0) / (2.0 * np.pi)) ** 2 * np.exp(lnD)
    lnq = np.log(9.0 * Kq * lam / (r_L * dtil)) + K.LN_M4_OVER_VF - B4
    T = np.asarray(T_mev, dtype=float)
    f_mev = f_scale_gev(lam, eta, r_L) * 1e12
    lnT = np.log(np.maximum(T, 1e-300) / HL_MEV) * 4.0 + 1.5 * np.log(np.maximum(S3T, 1e-300) / (2.0 * np.pi)) + lnDT - S3T
    thermal = exists & (S3T < B4) & (T > 0)
    ln_rate = np.where(thermal, lnT, lnq)
    return np.where(exists, ln_rate, np.inf), np.where(~exists, 0, np.where(thermal, 3, 4))


# ----------------------------------------------------------------------------------------------- the survival integral with a time-dependent rate
_GRID = {}


def light_cone_grid(z_max=1100.0, n=600):
    """Quadrature weights w_i (units H_0^-4) of the past-light-cone four-volume on a grid in s = -ln a (flat Planck 2018 history), and z_i."""
    key = (z_max, n)
    if key not in _GRID:
        from scipy.integrate import cumulative_simpson
        s_top = np.log(1.0 + z_max)
        sg = np.linspace(0.0, s_top, 8001)
        a = np.exp(-sg)
        E = S.e_of_a(a)
        # chi is measured from the present all the way to the grid end; the light cone extends beyond z_max, but chi only needs the integral up to the event
        chi = cumulative_simpson(1.0 / (a * E), x=sg, initial=0.0)
        dens = (4.0 * np.pi / 3.0) * a ** 3 * chi ** 3 / E                # dV4/ds
        s = np.linspace(0.0, s_top, n)
        d_i = np.interp(s, sg, dens)
        w = np.gradient(s) * d_i
        w[0] *= 0.5
        w[-1] *= 0.5
        _GRID[key] = (np.exp(s) - 1.0, w)
    return _GRID[key]


def ln_I(lam, eta, c, lnD=0.0, lnDT=0.0, r_L=1.0, z_max=1100.0, n=600):
    """ln of the expected number of nucleations in the past light cone, and a flag 'false minimum existed at every redshift up to z_max'. Vectorized over draws."""
    z, w = light_cone_grid(z_max, n)
    lam, eta, c, lnD, lnDT = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (lam, eta, c, lnD, lnDT)])
    acc = np.full(lam.shape, -np.inf)
    ok = np.ones(lam.shape, dtype=bool)
    for zi, wi in zip(z, w):
        lr, ch = ln_rate_over_hl4(lam, eta, c, T0_MEV * (1.0 + zi), lnD, lnDT, r_L)
        ok &= ch != 0
        acc = np.logaddexp(acc, np.where(ch != 0, lr + 2.0 * np.log(K.OMEGA_L) + np.log(wi), -np.inf))
    return acc, ok


if __name__ == "__main__":
    t_start = time.time()
    out = {}
    # ---- (1) O(3) and O(4) actions at theta = 0 on the common tilt grid
    etas = sorted(set(r["eta"] for r in _T["flat"]))
    rows = []
    for e in etas:
        s3 = action_zero_theta(e, 3)
        rows.append(dict(eta=float(e), s3=float(s3), b=float([r["b"] for r in _T["flat"] if r["eta"] == e][0])))
        print("eta %.3f  s3 %.4f   b %.3f   [%.0fs]" % (e, s3, rows[-1]["b"], time.time() - t_start), flush=True)
    out["s3_table"] = rows
    json.dump(out, open(_S3_PATH, "w"), indent=1)
    _S3 = None
    # thin-wall checks of the O(3) action: S_3 = 16 pi sigma^3 / (3 DeltaV^2)
    chk = []
    for e in (0.03, 0.05, 0.08):
        p = C.Potential(1.0, e)
        thin = 16.0 * np.pi * p.wall_tension() ** 3 / (3.0 * p.dtil ** 2)
        chk.append(dict(eta=e, s3=float(s3_of_eta(e)), thin=float(thin), ratio=float(s3_of_eta(e) / thin)))
    out["s3_thin_wall"] = chk
    # ---- exact reduction: direct bounce with the thermal term against b(eta_eff), s3(eta_eff) (theta in sensible range, lam = 1)
    red = []
    for e, th in ((0.30, 0.02), (0.30, 0.06), (0.20, 0.10), (0.25, -0.10), (0.34, 0.01)):
        s = np.sqrt(1.0 - 2.0 * th)
        ee = float(eta_eff(e, th))
        b4_dir, s3_dir = action_with_theta(e, th, 4), action_with_theta(e, th, 3)
        red.append(dict(eta=e, theta=th, eta_eff=ee, B4_direct=float(b4_dir), B4_scaling=float(R.b_of_eta(ee)), S3_direct=float(s3_dir), S3_scaling=float(s * s3_of_eta(ee)),
                        rel4=float(abs(b4_dir / R.b_of_eta(ee) - 1)), rel3=float(abs(s3_dir / (s * s3_of_eta(ee)) - 1))))
        print("reduction", red[-1], flush=True)
    out["reduction_check"] = red
    # ---- (1) vacuum structure against temperature (lam = 1): extrema, barrier, existence
    ths = np.linspace(-0.3, 0.12, 211)
    vs = []
    for eta in (0.20, 0.30, 0.34):
        for th in ths:
            roots = np.roots([1.0, 0.0, -(1.0 - 2.0 * th), eta])
            real = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
            if real.size == 3:
                x_t, x_top, x_f = real
                p = lambda x: (x * x - 1.0) ** 2 / 8.0 + eta * x / 2.0 + th * x * x / 2.0
                vs.append(dict(eta=eta, theta=float(th), x_t=float(x_t), x_top=float(x_top), x_f=float(x_f), barrier=float(p(x_top) - p(x_f)), dV=float(p(x_f) - p(x_t))))
            else:
                vs.append(dict(eta=eta, theta=float(th), x_t=None))
    out["vacuum_structure"] = vs
    out["theta_critical"] = {("%.2f" % e): float(theta_critical(e)) for e in (0.10, 0.20, 0.25, 0.30, 0.32, 0.34, 0.36, 0.38)}
    json.dump(out, open(_S3_PATH, "w"), indent=1)
    print("saved %s (%.0fs)" % (_S3_PATH, time.time() - t_start))
