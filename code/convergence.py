# -*- coding: utf-8 -*-
"""Numerical error budget of the bounce action, and the negative-mode structure of the solutions.

(A) flat-space action B(eta) at lam = 1 against: integrator tolerance, integrator (explicit Runge-Kutta 8(5,3), 5(4), LSODA, implicit Radau),
    quadrature of the action integral, tolerance of the bisection on the central field value, and truncation of the integration domain;
(B) the independent boundary-value solution against: solver tolerance, initial mesh, and the padding of the domain;
(C) the gravitational suppression coefficient kappa (B/B_flat = 1 - kappa g) against the integrator and the quadrature;
(D) the second variation about the bounce, -Laplacian_4 + V''(phi_b), in each angular sector l: the number of negative eigenvalues (a CDL bounce
    has exactly one, at l = 0), the l = 1 translational zero modes, the monotonicity (no node) of phi(r), and the Derrick virial identity.

    python convergence.py          (about 15 minutes; writes convergence.json)
"""
import json
import os
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

import cdl_lib as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
T0 = time.time()
ETAS = (0.15, 0.30, 0.36)


def stamp(msg):
    print("[%5.0fs] %s" % (time.time() - T0, msg), flush=True)


def rel(b, ref):
    return float(abs(b / ref - 1.0))


# --------------------------------------------------------------------------------------------------- (A) flat shooting
A = {}
for eta in ETAS:
    ref = C.flat_bounce(1.0, eta)["B"]                      # production settings: DOP853, rtol 1e-11, 2e5 quadrature points, bisection 1e-13
    rows = dict(reference=float(ref))
    rows["rtol"] = []
    for rt in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        B = C.flat_bounce(1.0, eta, rtol=rt, atol=rt * 1e-2)["B"]
        rows["rtol"].append(dict(rtol=rt, B=float(B), rel=rel(B, ref)))
    rows["method"] = []
    for m in ("DOP853", "RK45", "LSODA", "Radau"):
        try:
            B = C.flat_bounce(1.0, eta, method=m, rtol=1e-10 if m != "RK45" else 1e-11, atol=1e-12)["B"]
            rows["method"].append(dict(method=m, B=float(B), rel=rel(B, ref)))
        except Exception as e:                          # a solver may fail to resolve the thin wall; recorded, not hidden
            rows["method"].append(dict(method=m, B=None, rel=None, note=str(e)[:60]))
    rows["quadrature"] = []
    for nq in (20001, 200001, 2000001):
        B = C.flat_bounce(1.0, eta, n_quad=nq)["B"]
        rows["quadrature"].append(dict(n_quad=nq, B=float(B), rel=rel(B, ref)))
    rows["bisection"] = []
    for tl in (1e-6, 1e-8, 1e-10, 1e-12, 1e-13):
        B = C.flat_bounce(1.0, eta, tol=tl)["B"]
        rows["bisection"].append(dict(tol=tl, B=float(B), rel=rel(B, ref)))
    # truncation of the integration domain: the action accumulated up to R, relative to the full value
    prof = C.flat_bounce(1.0, eta, with_profile=True)
    r, x, p, pot = prof["r"], prof["x"], prof["p"], prof["pot"]
    dens = r ** 3 * (p ** 2 / 2.0 + pot.v(x) - pot.v_f)
    cum = 2.0 * np.pi ** 2 * np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(r))])
    # characteristic length: where the field has moved half way across, and the mass of the false vacuum
    i_half = int(np.argmin(np.abs(x - 0.5 * (pot.x_t + pot.x_f))))
    r_half, m_f = float(r[i_half]), float(np.sqrt(abs(pot.m2_f)))
    rows["truncation"] = []
    for k in (1.0, 2.0, 4.0, 8.0, 16.0):
        R = r_half + k / m_f
        j = int(np.searchsorted(r, R))
        if j < len(r):
            rows["truncation"].append(dict(R_over_wall_radius=float(R / r_half), R=float(R), B=float(cum[j]), rel=rel(cum[j], ref)))
    rows["wall_radius"], rows["m_false"] = r_half, m_f
    A[str(eta)] = rows
    stamp("A eta=%.2f  worst deviation: rtol %.1e, method %.1e, quad %.1e, bisect %.1e" % (
        eta, max(x_["rel"] for x_ in rows["rtol"][:-1]) if rows["rtol"] else 0, max((x_["rel"] or 0) for x_ in rows["method"]),
        max(x_["rel"] for x_ in rows["quadrature"]), max(x_["rel"] for x_ in rows["bisection"])))
OUT["A_flat_shooting"] = A

# --------------------------------------------------------------------------------------------------- (B) boundary-value solver
Bv = {}
for eta in (0.05, 0.10):
    ref = C.flat_bounce_bvp(1.0, eta, r_pad=60.0, tol=1e-10, n_init=8000)
    rows = dict(reference=float(ref["B"]))
    rows["tol"] = [dict(tol=t, B=float(C.flat_bounce_bvp(1.0, eta, tol=t)["B"]), rel=rel(C.flat_bounce_bvp(1.0, eta, tol=t)["B"], ref["B"])) for t in (1e-6, 1e-8, 1e-10)]
    rows["mesh"] = []
    for n0 in (500, 2000, 8000):
        r_ = C.flat_bounce_bvp(1.0, eta, n_init=n0)
        rows["mesh"].append(dict(n_init=n0, nodes=r_["nodes"], B=float(r_["B"]), rel=rel(r_["B"], ref["B"])))
    rows["domain_padding"] = []
    for pad in (10.0, 30.0, 60.0, 120.0):
        r_ = C.flat_bounce_bvp(1.0, eta, r_pad=pad)
        rows["domain_padding"].append(dict(r_pad=pad, B=float(r_["B"]), rel=rel(r_["B"], ref["B"])))
    if eta >= 0.10:
        rows["against_shooting"] = rel(ref["B"], C.flat_bounce(1.0, eta)["B"])
    Bv[str(eta)] = rows
    stamp("B eta=%.2f  done" % eta)
OUT["B_boundary_value"] = Bv

# --------------------------------------------------------------------------------------------------- (C) gravity coefficient
G = {}
eta = 0.30
Bf = C.flat_bounce(1.0, eta)["B"]
rows = []
for label, kw in [("DOP853 rtol 1e-9", dict(rtol=1e-9, atol=1e-11)), ("DOP853 rtol 1e-11", dict()), ("DOP853 rtol 1e-13", dict(rtol=1e-13, atol=1e-15)),
                  ("Radau rtol 1e-11", dict(method="Radau")), ("LSODA rtol 1e-11", dict(method="LSODA"))]:
    ks = []
    for g in (1e-3, 2e-3):
        try:
            res = C.grav_bounce(1.0, eta, g, **kw)
            ks.append(float(-(C.grav_action(res, g) / Bf - 1.0) / g))
        except Exception as e:
            ks.append(None)
    k0 = (2 * ks[0] - ks[1]) if None not in ks else None
    rows.append(dict(setting=label, kappa_1em3=ks[0], kappa_2em3=ks[1], kappa_g0=k0))
res = C.grav_bounce(1.0, eta, 1e-3)
quad_rows = []
for nq in (100001, 400001, 1600001):
    quad_rows.append(dict(n=nq, kappa_1em3=float(-(C.grav_action(res, 1e-3, n=nq) / Bf - 1.0) / 1e-3)))
G = dict(eta=eta, settings=rows, quadrature=quad_rows)
good = [r_["kappa_g0"] for r_ in rows if r_["kappa_g0"] is not None]
G["kappa_g0_mean"] = float(np.mean(good))
G["kappa_g0_spread"] = float(np.max(good) - np.min(good))
OUT["C_gravity_coefficient"] = G
stamp("C kappa0 = %.4f +- %.1e" % (G["kappa_g0_mean"], G["kappa_g0_spread"]))


# --------------------------------------------------------------------------------------------------- (D) negative modes
def spectrum(r, x, pot, l_max=4, n=30000, how_many=3):
    """Lowest eigenvalues of  -u'' + [(l(l+2) + 3/4)/r^2 + V''(phi_b)] u  on (0, R) with Dirichlet ends (u = r^{3/2} psi); V'' = d2v/dx2 in units of f."""
    dev = np.abs(x - pot.x_f)
    ok = np.where(dev < 1e-5 * (pot.x_f - pot.x_t))[0]
    R = r[ok[0]] if ok.size else r[-1]
    h = R / n
    rr = h * np.arange(1, n)
    xx = np.interp(rr, r, x)
    d2v = pot.lam * (3.0 * xx ** 2 - 1.0) / 2.0
    res = {}
    for l in range(l_max + 1):
        diag = 2.0 / h ** 2 + (l * (l + 2) + 0.75) / rr ** 2 + d2v
        off = -np.ones(n - 2) / h ** 2
        w = eigh_tridiagonal(diag, off, select="i", select_range=(0, how_many - 1), eigvals_only=True)
        res[l] = [float(v) for v in w]
    return R, res


D = {}
for eta in (0.05, 0.15, 0.30, 0.36):
    if eta < 0.10:
        pr = C.flat_bounce_bvp(1.0, eta, with_profile=True)
    else:
        pr = C.flat_bounce(1.0, eta, with_profile=True)
    r, x, p, pot = pr["r"], pr["x"], pr["p"], pr["pot"]
    mf2 = float(abs(pot.m2_f))
    R, sp = spectrum(r, x, pot)
    # monotonic profile: the field never turns back, so phi - phi_f has no node and phi' does not change sign
    inside = r <= R
    sign_changes = int(np.sum(np.diff(np.sign(p[inside][np.abs(p[inside]) > 1e-9 * np.max(np.abs(p))])) != 0))
    nodes_of_phi_minus_phi_f = int(np.sum(np.diff(np.sign(x[inside] - pot.x_f)) != 0))
    # Derrick identity for the O(4) bounce: 2 T + 4 U = 0 with T = int r^3 phi'^2 / 2 and U = int r^3 (V - V_f); B = T + U = T / 2
    Tk = 2.0 * np.pi ** 2 * float(np.trapezoid((r ** 3 * p ** 2 / 2.0)[inside], r[inside]))
    Up = 2.0 * np.pi ** 2 * float(np.trapezoid((r ** 3 * (pot.v(x) - pot.v_f))[inside], r[inside]))
    D[str(eta)] = dict(
        R_domain=float(R), m_false_sq=mf2,
        eigenvalues={str(l): sp[l] for l in sp},
        n_negative_l0=int(sum(v < 0 for v in sp[0])), n_negative_l1=int(sum(v < -1e-3 * mf2 for v in sp[1])),
        n_negative_l2_to_4=int(sum(sum(v < 0 for v in sp[l]) for l in (2, 3, 4))),
        l1_lowest_over_mf2=float(sp[1][0] / mf2), l0_lowest_over_mf2=float(sp[0][0] / mf2),
        phi_prime_sign_changes=sign_changes, phi_minus_phi_f_nodes=nodes_of_phi_minus_phi_f,
        virial_residual=float(abs(2 * Tk + 4 * Up) / (2 * Tk)), B_from_T=float(Tk / 2.0), B_full=float(Tk + Up))
    stamp("D eta=%.2f  l=0 eigenvalues %s  l=1 lowest/m_f^2 = %.2e  virial %.1e" % (eta, ["%.3g" % v for v in sp[0]], sp[1][0] / mf2, D[str(eta)]["virial_residual"]))
OUT["D_negative_mode"] = D

json.dump(OUT, open(os.path.join(HERE, "convergence.json"), "w"), indent=1)
stamp("saved convergence.json")
