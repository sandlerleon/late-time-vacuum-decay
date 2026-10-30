# -*- coding: utf-8 -*-
"""Cycles of the dilution-triggered model: turnaround in general relativity, contraction, an effective nonsingular bounce, re-expansion.

The low-density evolution is that of dilution_background.py. Near the bounce the effective loop-quantum-cosmology equations for a flat (k = 0)
universe are used, for which they are derived (Ashtekar, Pawlowski and Singh 2006):
    H^2 = (rho/3)(1 - rho/rho_c),       dH/dt = -(rho + p)/2 (1 - 2 rho/rho_c)
They reduce to general relativity for rho << rho_c. No curvature term is appended. rho_c (the bounce density) is a model parameter in units of
the dark-energy density; the physical value, of order the Planck density, is 10^122 V_c and cannot be integrated directly, so rho_c/V_c = 10^4-10^6
is used and the dependence on it is reported.

Dissipation. The scalar loses energy to radiation at the rate Upsilon = Upsilon_0 + C_T T^3 (T = rho_r^(1/4) in these units): a constant part and a
thermal part that is negligible in the cold expanding universe (rho_r ~ 10^-4 today) and dominant in the hot contracting one, where it converts
scalar kinetic energy into radiation as the density rises (the form used for dissipative scalar dynamics in warm inflation). Without it the
contraction is dominated by the scalar's kinetic energy (growing as a^-6, faster than radiation), and the bounce occurs at low density.

State y = (ln a, H, phi, pi, rho_r, W) with W = Integral p d(a^3) (the pressure-volume work of the cycle). The coupled dark matter keeps rho_* a^3
fixed through every cycle; radiation keeps rho_r a^4 fixed except for the dissipation Upsilon pi^2 (scalar -> radiation), which produces entropy.

Diagnostics per cycle n (between turnarounds): a_max,n, a_min,n (bounce), G_n = a_max,n+1/a_max,n, W_n, D_n = rho_*(turnaround)/rho_crit,
radiation entropy S_n = a^3 rho_r^(3/4), the scalar amplitude when the expanding universe passes a = 1, the maximum vacuum fraction V_0/rho
and the number of e-folds of accelerated expansion in the expanding phase (is there a dark-energy epoch like ours?).

    python cyclic_extension.py         (writes cyclic_extension.json and cyclic_cycles.csv)
"""
import csv
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))


def make_rhs(m, rho_c, ups, c_T=0.0):
    rs0, rb0 = MC.RHO_DM0, MC.RHO_B0

    def parts(y):
        la, H, p, pi, rr, W = y
        a3 = np.exp(-3.0 * la)
        rs, rb = rs0 * a3, rb0 * a3
        rho = rr + rb + m.A(p) * rs + 0.5 * pi * pi + m.V0(p)
        rho_p = 4.0 / 3.0 * rr + rb + m.A(p) * rs + pi * pi
        return rs, rb, rho, rho_p

    def rhs(t, y):
        la, H, p, pi, rr, W = y
        rs, rb, rho, rho_p = parts(y)
        dH = -0.5 * rho_p * (1.0 - 2.0 * rho / rho_c)
        U = ups + c_T * max(rr, 0.0) ** 0.75
        dpi = -(3.0 * H + U) * pi - m.dV0(p) - rs * m.dA(p)
        press = rho_p - rho
        return [H, dH, pi, dpi, -4.0 * H * rr + U * pi * pi, 3.0 * H * press * np.exp(3.0 * la)]
    return rhs, parts


def run(a_c=2.0, mu=30.0, q=1.0, rho_c_over_Vc=1e4, ups=0.0, n_cycles=6, seed=1e-30, rtol=1e-10, t_cycle_max=200.0, method="DOP853", c_T=0.0):
    m = MC.Model(a_c, mu, q)
    rho_c = rho_c_over_Vc * MC.VC
    rhs, parts = make_rhs(m, rho_c, ups, c_T)
    phi0 = seed * m.v
    rho0 = MC.RHO_R0 + MC.RHO_B0 + m.A(phi0) * MC.RHO_DM0 + m.V0(phi0)
    y = np.array([0.0, np.sqrt(rho0 / 3.0 * (1.0 - rho0 / rho_c)), phi0, 0.0, MC.RHO_R0, 0.0])
    fa = max(abs(phi0), 1e-300) * 1e-6

    def turn(t, y):
        return y[1] if parts(y)[2] < 0.5 * rho_c else 1.0
    turn.terminal, turn.direction = True, -1

    def bounce(t, y):
        return y[1] if parts(y)[2] > 0.5 * rho_c else -1.0
    bounce.direction = 1

    def pass1(t, y):
        return y[0]
    pass1.direction = 1

    cycles = []
    t_abs = 0.0
    status = "completed"
    max_res = 0.0
    for n in range(n_cycles + 1):
        sol = solve_ivp(rhs, (t_abs, t_abs + t_cycle_max), y, method=method, rtol=rtol, atol=[1e-12, 1e-12, fa, fa * mu, 1e-14, 1e-12],
                        events=[turn, bounce, pass1], dense_output=True)
        Y = sol.y
        la = Y[0]
        # constraint residual of the effective equations along the segment
        idx = np.linspace(0, Y.shape[1] - 1, min(400, Y.shape[1])).astype(int)
        for i in idx:
            rs, rb, rho, rho_p = parts(Y[:, i])
            comp = abs(Y[4, i]) + rb + m.A(Y[2, i]) * rs + 0.5 * Y[3, i] ** 2 + abs(m.V0(Y[2, i]))
            max_res = max(max_res, abs(3 * Y[1, i] ** 2 - rho * (1 - rho / rho_c)) / max(comp * (1 + rho / rho_c), 1e-300))
        info = dict(n=n)
        if sol.t_events[1].size:
            yb = sol.y_events[1][0]
            info["a_min"] = float(np.exp(yb[0]))
            rsb = MC.RHO_DM0 * np.exp(-3 * yb[0])
            tot = yb[4] + MC.RHO_B0 * np.exp(-3 * yb[0]) + m.A(yb[2]) * rsb + 0.5 * yb[3] ** 2 + max(m.V0(yb[2]), 0.0)
            info["bounce_fractions"] = dict(radiation=float(yb[4] / tot), matter=float((m.A(yb[2]) * rsb + MC.RHO_B0 * np.exp(-3 * yb[0])) / tot),
                                            kinetic=float(0.5 * yb[3] ** 2 / tot))
        # expanding phase: vacuum fraction and accelerated e-folds
        tt = np.linspace(sol.t[0], sol.t[-1], 20001)
        yy = sol.sol(tt)
        rhos = np.array([parts(yy[:, i])[2] for i in range(tt.size)])
        Hs = yy[1]
        V = m.V0(yy[2])
        expanding = Hs > 0
        fvac = np.where(expanding & (rhos > 0), V / np.maximum(rhos, 1e-300), -np.inf)
        acc = np.array([(-0.5 * parts(yy[:, i])[3] * (1 - 2 * parts(yy[:, i])[2] / rho_c) + yy[1, i] ** 2) > 0 for i in range(tt.size)]) & expanding
        info["max_vacuum_fraction"] = float(np.max(fvac)) if np.isfinite(np.max(fvac)) else None
        rm = np.array([MC.RHO_DM0 * np.exp(-3 * yy[0, i]) * m.A(yy[2, i]) + MC.RHO_B0 * np.exp(-3 * yy[0, i]) for i in range(tt.size)])
        info["max_matter_fraction"] = float(np.max(np.where(expanding & (rhos > 0), rm / np.maximum(rhos, 1e-300), 0.0)))
        info["max_radiation_fraction"] = float(np.max(np.where(expanding & (rhos > 0), yy[4] / np.maximum(rhos, 1e-300), 0.0)))
        held = expanding & (np.abs(yy[2]) < 0.1 * m.v)
        info["held_accelerated_efolds"] = float(np.sum(np.where((acc & held)[:-1], Hs[:-1] * np.diff(tt), 0.0)))
        k_c = np.where(expanding[1:] & (yy[0, :-1] < np.log(m.a_c)) & (yy[0, 1:] >= np.log(m.a_c)))[0]
        if k_c.size:
            info["phi_at_ac_over_v"] = float(abs(yy[2, k_c[-1]]) / m.v)
        info["accelerated_efolds"] = float(np.sum(np.where(acc[:-1], Hs[:-1] * np.diff(tt), 0.0)))
        if sol.t_events[2].size:
            yp = sol.y_events[2][-1]
            info["phi_at_a1_over_v"] = float(abs(yp[2]) / m.v)
        if sol.status == 1 and sol.t_events[0].size:
            yt = sol.y_events[0][0]
            info.update(a_max=float(np.exp(yt[0])), t_turn=float(sol.t_events[0][0]), W=float(yt[5]),
                        D=float(MC.RHO_DM0 * np.exp(-3 * yt[0]) / m.rho_crit), S_rad=float(np.exp(3 * yt[0]) * max(yt[4], 0) ** 0.75),
                        phi_turn_over_v=float(yt[2] / m.v))
            cycles.append(info)
            y = yt.copy()
            y[5] = 0.0
            y[1] = -1e-14                                          # step past the turnaround
            t_abs = sol.t_events[0][0]
            fa = max(abs(y[2]), 1e-300) * 1e-6 if abs(y[2]) < 1e-3 * m.v else 1e-14
            if info["a_max"] > 1e40:
                status = "runaway"
                break
        else:
            cycles.append(info)
            status = "no turnaround within t_cycle_max" if sol.status == 0 else "integration failed: " + str(sol.message)
            break
    for k in range(len(cycles) - 1):
        if "a_max" in cycles[k] and "a_max" in cycles[k + 1]:
            cycles[k]["G"] = cycles[k + 1]["a_max"] / cycles[k]["a_max"]
    return dict(params=dict(a_c=a_c, mu=mu, q=q, rho_c_over_Vc=rho_c_over_Vc, ups=ups, c_T=c_T, seed=seed, n_cycles=n_cycles), model=m.describe(),
                cycles=cycles, status=status, max_residual=float(max_res))


if __name__ == "__main__":
    t0 = time.time()
    out = {"runs": []}
    for kw in [dict(), dict(ups=0.1), dict(ups=1.0), dict(rho_c_over_Vc=1e5), dict(rho_c_over_Vc=1e6), dict(mu=100.0), dict(q=0.1)]:
        r = run(**kw)
        out["runs"].append(r)
        print(kw, r["status"], "cycles", len(r["cycles"]), "res %.1e" % r["max_residual"], "[%.0fs]" % (time.time() - t0), flush=True)
        for c in r["cycles"]:
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "cyclic_extension.json"), "w"), indent=1)
    rows = [dict(run=i, **r["params"], **c) for i, r in enumerate(out["runs"]) for c in r["cycles"]]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(HERE, "cyclic_cycles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
