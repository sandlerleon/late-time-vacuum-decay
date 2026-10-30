# -*- coding: utf-8 -*-
"""Coupled matter-scalar background (flat FLRW, general relativity) from today into the future: does dilution trigger a turnaround?

State y = (ln a, H, phi, pi, rho_r). With rho_* = rho_*0 a^-3 (exact), baryons rho_b0 a^-3 and an optional dissipation Upsilon (scalar -> radiation):
    d ln a/dt = H
    dH/dt     = -( 4/3 rho_r + rho_b + A(phi) rho_* + pi^2 ) / 2
    dphi/dt   = pi
    dpi/dt    = -(3H + Upsilon) pi - V_0'(phi) - rho_* A'(phi) - dV_T/dphi
    drho_r/dt = -4 H rho_r + Upsilon pi^2
Constraint (checked, not imposed):  3 H^2 = rho_r + rho_b + A rho_* + pi^2/2 + V_0 (+ V_T).
Thermal branch (matter_coupling.Model with a_T): the bath that gives the scalar its thermal mass is a test reservoir at T = T_0/a, and its free
energy as a function of the field is
    V_T(phi, T) = (c T^2 phi^2 / 2) / (1 + c phi^2 / (2 T^2))
which is the high-temperature result c T^2 phi^2/2 while the bath particles are light (sqrt(c) |phi| << T) and saturates at T^4 (the free energy of
the species) once they become heavy, as the full one-loop thermal function does. The truncated form c T^2 phi^2/2 alone is not used: it lets the
field-dependent mass exceed T, where the high-temperature expansion fails. The bath's back-reaction is of order T^4 (below 10^-4 of the total density
after recombination); it is the only term not conserved exactly and it appears in the constraint residual.
Total stress-energy is conserved: the scalar and the coupled matter exchange energy through A'(phi) (d(A rho_*)/dt = -3H A rho_* + rho_* A' pi).
Initial data today (t = 0, a = 1): phi = phi_0 (a small displacement), pi = 0, H from the constraint (H = 1 for phi_0 -> 0).

Events: turnaround H = 0 (dH/dt < 0 there in flat GR), roll (|phi| reaches v/2), and crunch (total density 10^6 V_c in contraction).

    python dilution_background.py      (writes dilution_background.json: the three controls of the reference model)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))


def rhs_factory(m, ups=0.0):
    rs0, rb0 = MC.RHO_DM0, MC.RHO_B0

    def rhs(t, y):
        la, H, p, pi, rr = y
        a3 = np.exp(-3.0 * la)
        rs, rb = rs0 * a3, rb0 * a3
        dH = -0.5 * (4.0 / 3.0 * rr + rb + m.A(p) * rs + pi * pi)
        dpi = -(3.0 * H + ups) * pi - m.dV0(p) - rs * m.dA(p) - dVT(m, p, np.exp(la))
        return [H, dH, pi, dpi, -4.0 * H * rr + ups * pi * pi]
    return rhs


def VT(m, p, a):
    """Saturating thermal free energy of the bath as a function of the field (zero without the thermal branch)."""
    if not m.c_T:
        return 0.0 * p
    T = m.T0 / a
    return 0.5 * m.c_T * T * T * p * p / (1.0 + m.c_T * p * p / (2.0 * T * T))


def dVT(m, p, a):
    if not m.c_T:
        return 0.0 * p
    T = m.T0 / a
    return m.c_T * T * T * p / (1.0 + m.c_T * p * p / (2.0 * T * T)) ** 2


def lcdm_rhs(t, y):
    la, H = y
    a3 = np.exp(-3.0 * la)
    rr = MC.RHO_R0 * a3 * np.exp(-la)
    return [H, -0.5 * (4.0 / 3.0 * rr + (MC.RHO_B0 + MC.RHO_DM0) * a3)]


def densities(m, y):
    la, H, p, pi, rr = y
    a3 = np.exp(-3.0 * la)
    rs = MC.RHO_DM0 * a3
    return dict(rho_r=rr, rho_b=MC.RHO_B0 * a3, rho_dm=m.A(p) * rs, kin=0.5 * pi * pi, V=m.V0(p) + VT(m, p, np.exp(la)))


def total_density(m, y):
    return sum(densities(m, y).values())


def constraint_residual(m, y):
    """|3H^2 - rho| relative to the larger of the summed absolute components and V_c (finite through a turnaround, where H and rho vanish;
    V_c is the scale on which the late universe and its turnaround are determined)."""
    d = densities(m, y)
    return abs(3.0 * y[1] ** 2 - sum(d.values())) / max(sum(abs(v) for v in d.values()), MC.VC)


def initial_state(m, phi0, a_i=1.0):
    rho = MC.RHO_R0 / a_i ** 4 + MC.RHO_B0 / a_i ** 3 + m.A(phi0) * MC.RHO_DM0 / a_i ** 3 + m.V0(phi0) + VT(m, phi0, a_i)
    return [np.log(a_i), np.sqrt(rho / 3.0), phi0, 0.0, MC.RHO_R0 / a_i ** 4]


def evolve(m, phi0, t_max=400.0, ups=0.0, rtol=1e-10, atol=1e-13, method="DOP853", crunch=1e6, dense=False, max_step=np.inf, a_i=1.0, stop_at_roll=False):
    """Evolve from a = a_i (a_i = 1: today). Times in the result are measured from a = 1 (t_today) when a_i < 1."""
    rhs = rhs_factory(m, ups)

    def today(t, y):
        return y[0]
    today.terminal, today.direction = False, 1

    def turn(t, y):
        return y[1]
    turn.terminal, turn.direction = False, -1

    def roll(t, y):
        return abs(y[2]) - 0.5 * m.v
    roll.terminal, roll.direction = bool(stop_at_roll), 1

    def crunch_ev(t, y):
        return (total_density(m, y) - crunch * MC.VC) if y[1] < 0 else -1.0
    crunch_ev.terminal, crunch_ev.direction = True, 1

    # the field may start 60 orders of magnitude below its vev: its absolute tolerance must follow the seed, not the background
    y0 = initial_state(m, phi0, a_i)
    fa = max(abs(phi0), 1e-300) * 1e-6
    atol_v = [atol, atol, fa, fa * max(1.0, m.mu), 1e-6 * MC.RHO_R0 * 1e-12]   # H and rho_r absolute: late, near-empty phases stay resolved
    sol = solve_ivp(rhs, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol_v, events=[turn, roll, crunch_ev, today],
                    dense_output=dense, max_step=max_step)
    t0_ = sol.t_events[3][0] if (a_i < 1.0 and sol.t_events[3].size) else 0.0
    first = lambda k: (sol.t_events[k][0] - t0_) if sol.t_events[k].size else None
    t_turn, t_roll, t_crunch = first(0), first(1), first(2)
    res = max(constraint_residual(m, sol.y[:, i]) for i in range(0, sol.y.shape[1], max(1, sol.y.shape[1] // 400)))
    out = dict(t_turn=t_turn, t_roll=t_roll, t_crunch=t_crunch, a_turn=float(np.exp(sol.y_events[0][0][0])) if t_turn is not None else None,
               a_roll=float(np.exp(sol.y_events[1][0][0])) if t_roll is not None else None, max_residual=float(res), status=sol.status,
               t_end=float(sol.t[-1] - t0_), a_max=float(np.exp(sol.y[0].max())), t_today_from_start=float(t0_),
               reached_today=bool(a_i >= 1.0 or sol.t_events[3].size > 0))
    if t_turn is not None:
        yt = sol.y_events[0][0]
        d = densities(m, yt)
        out["turn_densities"] = {k: float(v) for k, v in d.items()}
        out["D_turn"] = float(MC.RHO_DM0 * np.exp(-3 * yt[0]) / m.rho_crit)       # rho_* at turnaround in units of rho_crit
        out["phi_turn_over_v"] = float(yt[2] / m.v)
    out["sol"] = sol
    return out


def classify(m, r):
    if r["t_turn"] is None:
        return "transition, no recollapse" if (r["t_roll"] is not None) else "eternal expansion (within t_max)"
    if r["t_roll"] is None:
        return "recollapse without transition"
    return "transition and recollapse"


def lcdm_future(t_max=400.0):
    sol = solve_ivp(lcdm_rhs, (0.0, t_max), [0.0, 1.0], method="DOP853", rtol=1e-10, atol=1e-12)
    return dict(H_end=float(sol.y[1, -1]), a_end=float(np.exp(sol.y[0, -1])), turnaround=bool((sol.y[1] < 0).any()))


def w_today(m, phi0):
    """Dark-energy equation of state today, w = (K - V_0)/(K + V_0) with K = 0 at t = 0 and the first derivative ignored: -1 exactly for pi = 0;
    the relevant measure is the drift of w over the past (phi frozen at phi_0 while m_eff^2 > 0)."""
    return -1.0


if __name__ == "__main__":
    t0 = time.time()
    out = {"units": "time 1/H_0 (%.2f Gyr), densities H_0^2 M_Pl^2, fields M_Pl" % MC.H0_GYR}
    ref = dict(a_c=2.0, mu=10.0, q=1.0)
    for label, coupled in (("C_coupled", True), ("B_uncoupled", False)):
        m = MC.Model(coupled=coupled, **ref)
        r = evolve(m, 1e-6 * m.v)
        r.pop("sol")
        r["class"] = classify(m, r)
        r["model"] = m.describe()
        out[label] = r
        print(label, {k: r[k] for k in ("t_roll", "t_turn", "t_crunch", "a_roll", "a_turn", "max_residual", "class")}, "[%.0fs]" % (time.time() - t0))
    out["A_lcdm"] = lcdm_future()
    print("A_lcdm", out["A_lcdm"])
    json.dump(out, open(os.path.join(HERE, "dilution_background.json"), "w"), indent=1)
