# -*- coding: utf-8 -*-
"""Causal test of the turnaround: is dilution the cause of the recollapse?

Five models are evolved from today (a = 1, field seeded at 1e-60 v, the seed of turnaround_scan.py) over the grid of turnaround_scan.py:
  P  positive vacuum, no field (LambdaCDM): the reference that expands forever.
  N  constant negative vacuum -q V_c with matter (flat, H(today) = H0): recollapses whatever the matter does; its deceleration today is reported
     (it has no accelerated expansion, so it is excluded by the data and is not a model of our universe).
  U  the same scalar and potential V_0 without the coupling (rolls at once: the uncoupled control of turnaround_scan.py).
  F  the coupled model with the matter density seen by the field frozen at its present value (an intervention, do(rho_* = rho_*(today)), on the
     field equation only; gravity still sees the diluting matter): the vacuum stays held, m_eff^2 = mu^2 (a_c^3 - 1) > 0.
  D  the coupled model with dilution (the model of the paper).
Dilution causes the recollapse wherever F expands forever and D recollapses. The run with q < 0 (positive broken vacuum) gives the region where
the release happens but the expansion continues; the transition here is global and homogeneous (a rolling field), not a nucleated bubble (the
tunneling branch, density_bounce.py, is the local alternative and is treated separately).

    python causal_controls.py      (writes causal_controls.json)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
A_C = [1.25, 1.5, 2.0, 3.0, 5.0]
MU = [3.0, 10.0, 30.0, 100.0, 300.0]
Q = [-0.5, 0.1, 1.0, 10.0]
SEED = 1e-60
T_MAX = 150.0


def evolve(m, kind, t_max=T_MAX, dense=False, stop_at_roll=False):
    rs0, rb0, rr0 = MC.RHO_DM0, MC.RHO_B0, MC.RHO_R0

    def rhs(t, y):
        la, H, p, pi = y
        a3 = np.exp(-3.0 * la)
        rs = rs0 * a3
        rr = rr0 * a3 * np.exp(-la)
        rs_field = rs0 * max(a3, 1.0) if kind == "F" else rs
        if kind == "U":
            A, dA = 1.0, 0.0
        else:
            A, dA = m.A(p), m.dA(p)
        dH = -0.5 * (4.0 / 3.0 * rr + rb0 * a3 + A * rs + pi * pi)
        dpi = -3.0 * H * pi - m.dV0(p) - rs_field * dA
        return [H, dH, pi, dpi]

    def turn(t, y):
        return y[1]
    turn.terminal, turn.direction = True, -1

    def roll(t, y):
        return abs(y[2]) - 0.5 * m.v
    roll.direction = 1
    roll.terminal = bool(stop_at_roll)

    p0 = SEED * m.v
    rho0 = MC.RHO_R0 + MC.RHO_B0 + m.A(p0) * MC.RHO_DM0 + m.V0(p0)
    y0 = [0.0, np.sqrt(rho0 / 3.0), p0, 0.0]
    fa = abs(p0) * 1e-6
    sol = solve_ivp(rhs, (0.0, t_max), y0, method="DOP853", rtol=1e-9, atol=[1e-12, 1e-12, fa, fa * m.mu], events=[turn, roll], dense_output=dense,
                    t_eval=None if dense else [t_max])                # only the end state is stored (long oscillating runs)
    sol.t, sol.y = np.asarray(sol.t, dtype=float), np.asarray(sol.y, dtype=float).reshape(4, -1)
    t_end = float(sol.t_events[0][0]) if sol.t_events[0].size else (float(sol.t[-1]) if sol.t.size else t_max)
    out = dict(turn=bool(sol.t_events[0].size), roll=bool(sol.t_events[1].size), t_end=t_end, ln_a_end=float(sol.y[0, -1]) if sol.y.size else None)
    out["t_turn"] = float(sol.t_events[0][0]) if out["turn"] else None
    out["outcome"] = "recollapse" if out["turn"] else ("expanding at t_max (H = %.3g)" % sol.y[1, -1] if sol.y.size else "expanding at t_max")
    if dense:
        out["sol"] = sol
    return out


def constant_vacuum(q, t_max=T_MAX, dense=False):
    """P (q = None) or N: matter + radiation + constant vacuum, H(today) = 1."""
    lam = MC.VC if q is None else -q * MC.VC
    rm = 3.0 - MC.RHO_R0 - lam

    def rhs(t, y):
        la, H = y
        a3 = np.exp(-3.0 * la)
        return [H, -0.5 * (4.0 / 3.0 * MC.RHO_R0 * a3 * np.exp(-la) + rm * a3)]

    def turn(t, y):
        return y[1]
    turn.terminal, turn.direction = True, -1
    sol = solve_ivp(rhs, (0.0, t_max), [0.0, 1.0], method="DOP853", rtol=1e-10, atol=1e-12, events=[turn], dense_output=dense)
    q0 = (rm + 2 * MC.RHO_R0 - 2 * lam) / 6.0            # deceleration today, (rho + 3p)/(2 rho) with rho = 3
    return dict(turn=bool(sol.t_events[0].size), t_turn=float(sol.t_events[0][0]) if sol.t_events[0].size else None, q0_deceleration=float(q0),
                outcome="recollapse" if sol.t_events[0].size else "expanding at t_max", sol=sol if dense else None)


def grid_point(args):
    """U and D to their turnaround (or a horizon that scales with the slowest roll), F to three times D's turnaround time: the frozen model is
    analytically held (m_eff^2 = mu^2 (a_c^3 - 1) > 0) and approaches de Sitter, so a horizon well beyond the turnaround of D shows the difference."""
    a_c, mu, q = args
    m = MC.Model(a_c, mu, q)
    horizon = T_MAX if mu <= 10 else 40.0
    row = dict(a_c=a_c, mu=mu, q=q)
    # q < 0: V_0 >= V_0(v) = -q V_c > 0 for every phi, so rho > 0 and H cannot vanish: after the release no turnaround is possible (flat FRW);
    # those runs stop at the release
    neg = q < 0
    row["U"] = evolve(m, "U", t_max=horizon, stop_at_roll=neg)
    row["D"] = evolve(m, "D", t_max=horizon, stop_at_roll=neg)
    if neg:
        for k in ("U", "D"):
            if row[k]["roll"]:
                row[k]["outcome"] = "released; no turnaround possible (V0 >= -q Vc > 0)"
    tF = max(3.0 * row["D"]["t_turn"], 20.0) if row["D"]["turn"] else (20.0 if neg else horizon)
    row["F"] = evolve(m, "F", t_max=tF)
    row["F_horizon"] = tF
    row["dilution_causes_recollapse"] = bool(row["D"]["turn"] and not row["F"]["turn"])
    row["F_held"] = not row["F"]["roll"]
    return row


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    out = dict(P=constant_vacuum(None), N={str(q): constant_vacuum(q) for q in Q if q > 0}, rows=[])
    print("P", out["P"]); print("N", out["N"])
    with Pool(5) as pool:
        for row in pool.imap_unordered(grid_point, [(a_c, mu, q) for a_c in A_C for mu in MU for q in Q]):
            out["rows"].append(row)
            print("%5.2f %5.0f %5.1f  D %-12s F %-12s U %-10s [%.0fs]" % (row["a_c"], row["mu"], row["q"], row["D"]["outcome"][:12], row["F"]["outcome"][:12],
                                                                        row["U"]["outcome"][:10], time.time() - t0), flush=True)
    print("grid done [%.0fs]" % (time.time() - t0), flush=True)
    rows = out["rows"]
    pos = [r for r in rows if r["q"] > 0]
    out["summary"] = dict(
        runs=len(rows), q_positive=len(pos),
        dilution_causes_recollapse=sum(r["dilution_causes_recollapse"] for r in pos),
        frozen_held_all=all(r["F_held"] and not r["F"]["turn"] for r in rows),
        D_recollapse_q_positive=sum(r["D"]["turn"] for r in pos),
        D_recollapse_q_negative=sum(r["D"]["turn"] for r in rows if r["q"] < 0),
        D_released_q_negative=sum(r["D"]["roll"] for r in rows if r["q"] < 0),
        U_rolls_all=all(r["U"]["roll"] for r in rows),
        U_never_held=all(r["U"]["roll"] or r["U"]["turn"] for r in rows))                 # the uncoupled field leaves the false vacuum or recollapses first
    print(out["summary"], "[%.0fs]" % (time.time() - t0))
    json.dump(out, open(os.path.join(HERE, "causal_controls.json"), "w"), indent=1)
