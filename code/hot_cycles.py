# -*- coding: utf-8 -*-
"""Hot cycles: turnaround by dilution, contraction, a hot nonsingular bounce, re-thermalization, and a dark-energy epoch in every cycle.

Physics added to cyclic_extension.py:
  (1) Thermal dissipation, Upsilon = C_T T^3 (T = rho_r^(1/4) in these units): negligible in the cold expanding universe, dominant in the hot
      contracting one, so the scalar energy released by the vacuum transition is converted into radiation and the bounce is radiation dominated.
  (2) Re-thermalization and freeze-out. The dark-matter abundance of our universe is not an initial condition but is set by thermal freeze-out,
      as a fixed ratio to the entropy, Y = n_dm/s. A bounce hotter than the dark-matter mass returns the dark matter to equilibrium (it is then
      relativistic and part of the bath), and as the bath cools after the bounce, freeze-out fixes the same ratio again. This is applied when the
      expanding bath cools through T_fo = 5 T_eq (T_eq = kappa_dm + kappa_b, the equality temperature in these units): the comoving densities become
      N_dm = kappa_dm S, N_b = kappa_b S, with S = a^3 rho_r^(3/4) and kappa fixed by the present universe (N_dm0/S_0), and the energy of the
      new matter (a fraction T_eq/T_fo of the bath) is taken from the radiation, so the total density and H are continuous. Baryons are treated the
      same way (a baryogenesis that fixes n_b/s). This requires the bounce to be hotter than T_fo: rho_c > T_fo^4, i.e. rho_c/V_c > ~10^13 here.
  (3) Symmetry restoration in the hot phase. While the dark matter is relativistic, its field-dependent mass m A(phi) gives the scalar a thermal
      mass m^2 T^2 / (12 M^2) through the same coupling; it exceeds H ~ T^2/M_Pl once T < T_hold ~ m M_Pl / (6 M), i.e. well before freeze-out
      (T_fo ~ m/20). From then on the field oscillates about phi = 0 with an angular frequency 10^3-10^9 times the expansion rate, and its
      amplitude falls adiabatically as (n/m_eff)^(1/2) ~ T (radiation era): by freeze-out it is reduced by T_fo/T_hold = 20 x 6 M / M_Pl ~ 0.3 M.
      This fast oscillation is not integrated; its adiabatic result is applied at freeze-out (phi -> phi T_fo/T_hold, pi -> 0 after phase
      averaging), with T_hold capped at the bounce temperature. The dust-phase holding that follows is integrated explicitly.
Consequence: the matter-to-entropy ratio, and therefore the whole thermal history as a function of the radiation temperature, is the same in
every cycle. The vacuum is held by matter during the matter era (m_eff/H = mu a_c^(3/2) (3/rho_dm0)^(1/2) ~ 5.5 mu there, independent of a)
and released when rho_* falls to rho_crit, i.e. at a fixed temperature. The entropy produced each cycle (second law) does not accumulate in the
local state; it enlarges the next cycle (Tolman): a_max,n+1/a_max,n = (S_n+1/S_n)^(1/3).

Constraint projection. A violation of the Friedmann constraint, delta = 3H^2 - rho(1 - rho/rho_c), is conserved by the evolution equations and so
acts as a constant energy density: a relative error of 10^-10 made at rho ~ 10^13 V_c leaves an offset of 10^3 V_c, which would masquerade as a
cosmological constant. H is therefore projected onto the constraint, H -> sign(H) [rho(1 - rho/rho_c)/3]^(1/2), at every doubling (halving) of the
scale factor away from the turning points (rho > 0, rho < 0.4 rho_c, |H| above 10^-3 of its constraint value), so the residual error is set by the
tolerance at the current density; the projections are counted and their total correction reported.

Bounce: effective loop-quantum-cosmology equations for k = 0, rho_c/V_c = 10^5 or 10^6 (the physical value, 10^122, cannot be integrated; at it
the bounce temperature is ~10^18 GeV, far above any dark-matter mass, which is what (2) requires).

    python hot_cycles.py               (writes hot_cycles.json and hot_cycles.csv)
"""
import csv
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA_DM = MC.RHO_DM0 / MC.RHO_R0 ** 0.75
KAPPA_B = MC.RHO_B0 / MC.RHO_R0 ** 0.75


T_EQ = KAPPA_DM + KAPPA_B


class _Piecewise:
    """Concatenation of solve_ivp segments (chunked integration), with the interface used below."""

    def __init__(self, sols, n_ev):
        self.sols = sols
        self.t = np.concatenate([q.t if i == 0 else q.t[1:] for i, q in enumerate(sols)])
        self.y = np.concatenate([q.y if i == 0 else q.y[:, 1:] for i, q in enumerate(sols)], axis=1)
        self.status = sols[-1].status
        self.message = sols[-1].message
        self.t_events = [np.concatenate([q.t_events[k] for q in sols]) for k in range(n_ev)]
        self.y_events = [np.concatenate([q.y_events[k].reshape(-1, self.y.shape[0]) for q in sols]) for k in range(n_ev)]
        self.edges = np.array([q.t[-1] for q in sols])

    def sol(self, tt):
        tt = np.atleast_1d(tt)
        out = np.empty((self.y.shape[0], tt.size))
        for j, t_ in enumerate(tt):
            k = min(int(np.searchsorted(self.edges, t_)), len(self.sols) - 1)
            out[:, j] = self.sols[k].sol(t_)
        return out


def run(a_c=2.0, mu=30.0, q=1.0, rho_c_over_Vc=1e14, c_T=1.0, n_cycles=6, seed=1e-30, rtol=1e-10, t_max_seg=300.0, reset=True, T_fo_over_eq=5.0, diag=None):
    m = MC.Model(a_c, mu, q)
    rho_c = rho_c_over_Vc * MC.VC
    N = dict(dm=MC.RHO_DM0, b=MC.RHO_B0)

    def comps(y):
        la, H, p, pi, rr, W = y
        a3 = np.exp(-3.0 * la)
        rs, rb = N["dm"] * a3, N["b"] * a3
        return dict(rr=rr, rm=m.A(p) * rs + rb, rs=rs, K=0.5 * pi * pi, V=m.V0(p))

    def rho_of(y):
        c = comps(y)
        return c["rr"] + c["rm"] + c["K"] + c["V"]

    def rhs(t, y):
        la, H, p, pi, rr, W = y
        c = comps(y)
        rho = c["rr"] + c["rm"] + c["K"] + c["V"]
        rho_p = 4.0 / 3.0 * rr + c["rm"] + pi * pi
        dH = -0.5 * rho_p * (1.0 - 2.0 * rho / rho_c)
        # thermal dissipation, capped where it is already overdamped (the cap changes nothing physical and keeps the hot phase integrable)
        U = min(c_T * max(rr, 0.0) ** 0.75, 30.0 * (abs(H) + mu))
        dpi = -(3.0 * H + U) * pi - m.dV0(p) - c["rs"] * m.dA(p)
        return [H, dH, pi, dpi, -4.0 * H * rr + U * pi * pi, 3.0 * H * (rho_p - rho) * np.exp(3.0 * la)]

    def turn(t, y):
        return y[1] if rho_of(y) < 0.5 * rho_c else 1.0
    turn.terminal, turn.direction = True, -1

    def bounce(t, y):
        return y[1] if rho_of(y) > 0.5 * rho_c else -1.0
    bounce.terminal, bounce.direction = True, 1

    def roll(t, y):
        return abs(y[2]) - 0.5 * m.v
    roll.direction = 1

    T_fo = T_fo_over_eq * T_EQ

    def freeze(t, y):
        return max(y[4], 0.0) ** 0.25 - T_fo
    freeze.terminal, freeze.direction = True, -1

    phi0 = seed * m.v
    rho0 = MC.RHO_R0 + MC.RHO_B0 + m.A(phi0) * MC.RHO_DM0 + m.V0(phi0)
    y = np.array([0.0, np.sqrt(rho0 / 3.0 * (1.0 - rho0 / rho_c)), phi0, 0.0, MC.RHO_R0, 0.0])
    t_abs, cycles, status, max_res = 0.0, [], "completed", 0.0
    cyc = dict(n=0, start="today")
    phase = "expanding"
    fa = max(abs(phi0), 1e-300) * 1e-6
    pending_fo = False
    proj = dict(n=0, max_rel=0.0)
    while len(cycles) < n_cycles + 1:
        evs = [turn, bounce, roll] + ([freeze] if pending_fo else [])
        # chunked integration: while matter makes the scalar heavy (m_eff >> H) the step is bounded by a tenth of the local oscillation period,
        # re-evaluated every doubling of the scale factor (m_eff ~ a^-3/2 falls as the universe expands)
        sols, t_now, y_now = [], t_abs, np.array(y, dtype=float)
        while True:
            a_now = np.exp(y_now[0])
            m_eff = np.sqrt(max(N["dm"] / a_now ** 3 / m.M2 - mu ** 2, mu ** 2))
            heavy = phase == "expanding" and m_eff > 20.0 * abs(y_now[1]) and abs(y_now[1]) > 0
            # constraint projection (see the docstring)
            rho_now = rho_of(y_now)
            hc2 = rho_now * (1.0 - rho_now / rho_c) / 3.0
            if 0.0 < rho_now < 0.4 * rho_c and hc2 > 0 and abs(y_now[1]) > 1e-3 * np.sqrt(hc2):
                proj["n"] += 1
                proj["max_rel"] = max(proj["max_rel"], abs(abs(y_now[1]) / np.sqrt(hc2) - 1.0))
                y_now[1] = np.sign(y_now[1]) * np.sqrt(hc2)
            step_t = np.log(2.0) / abs(y_now[1]) if abs(y_now[1]) > 0 else t_max_seg
            t_end = min(t_abs + t_max_seg, t_now + step_t)
            seg = solve_ivp(rhs, (t_now, t_end), y_now, method="DOP853", rtol=rtol, atol=[1e-12, 1e-12, fa, fa * mu, 1e-16, 1e-12],
                          events=evs, dense_output=True, max_step=(0.6 / m_eff if heavy else np.inf))
            sols.append(seg)
            if diag is not None:
                yl = seg.y[:, -1]
                diag.append((float(np.exp(yl[0])), float(yl[1]), float(rho_of(yl)), phase, dict(N), float(yl[2]), float(yl[3]), yl.copy(), seg.t[-1]))
            if seg.status != 0 or seg.t[-1] >= t_abs + t_max_seg - 1e-12:
                break
            t_now, y_now = seg.t[-1], seg.y[:, -1]
        sol = _Piecewise(sols, len(evs))
        if pending_fo and sol.status == 1 and sol.t_events[3].size:
            # freeze-out in the expanding phase after a bounce: matter-to-entropy ratio reset, energy taken from the bath
            yf = sol.y_events[3][0].copy()
            a3 = np.exp(3 * yf[0])
            S = a3 * max(yf[4], 0) ** 0.75
            # hot-phase restoration by the relativistic dark matter (adiabatic result, item (3) of the docstring)
            # T_hold/T_fo = (m M_Pl / 6M) / (m / 20) = 20/(6M) in reduced Planck units; the physical bounce (~10^18 GeV) lies far above T_hold, so
            # the full factor applies (the integrable rho_c of the simulation truncates the hot phase and is not used to cap it)
            T_hold_over_fo = 20.0 / (6.0 * np.sqrt(m.M2))
            cyc["phi_before_restoration_over_v"] = float(abs(yf[2]) / m.v)
            cyc["restoration_factor"] = float(1.0 / max(T_hold_over_fo, 1.0))
            old = (m.A(yf[2]) * N["dm"] + N["b"]) / a3
            K_old = 0.5 * yf[3] ** 2 + m.V0(yf[2])
            yf[2] = yf[2] / max(T_hold_over_fo, 1.0)
            yf[3] = 0.0
            N["dm"], N["b"] = KAPPA_DM * S, KAPPA_B * S
            new = (m.A(yf[2]) * N["dm"] + N["b"]) / a3
            # the scalar energy released by the restoration goes to the bath (it is radiated by the oscillation)
            dK = K_old - (0.5 * yf[3] ** 2 + m.V0(yf[2]))
            cyc["freeze_out_matter_fraction"] = float((new - old) / yf[4])
            cyc["phi_at_freeze_out_over_v"] = float(abs(yf[2]) / m.v)
            yf[4] += dK - (new - old)
            y, t_abs, pending_fo = yf, sol.t_events[3][0], False
            continue
        # residual of the effective constraint
        for i in np.linspace(0, sol.y.shape[1] - 1, min(300, sol.y.shape[1])).astype(int):
            c = comps(sol.y[:, i])
            rho = c["rr"] + c["rm"] + c["K"] + c["V"]
            scale = (c["rr"] + c["rm"] + c["K"] + abs(c["V"])) * (1 + rho / rho_c)
            max_res = max(max_res, abs(3 * sol.y[1, i] ** 2 - rho * (1 - rho / rho_c)) / max(scale, MC.VC))
        tt = np.unique(np.concatenate([sol.t, np.linspace(sol.t[0], sol.t[-1], 6001)]))     # solver steps resolve the early, fast epochs
        yy = sol.sol(tt)
        if phase == "expanding":
            # thermal history of the expanding phase
            cs = [comps(yy[:, i]) for i in range(tt.size)]
            rr = np.array([c["rr"] for c in cs]); rm = np.array([c["rm"] for c in cs]); K = np.array([c["K"] for c in cs]); V = np.array([c["V"] for c in cs])
            pos = rr + rm + K + np.maximum(V, 0.0)
            H = yy[1]
            ex = H > 0
            cyc["max_matter_fraction"] = float(np.max(np.where(ex, rm / pos, 0)))
            sel = np.where(ex & (rr > 0))[0]
            if sel.size:
                pick = sel[np.unique(np.linspace(0, sel.size - 1, 500).astype(int))]
                cyc["history"] = dict(T=(rr[pick] ** 0.25).tolist(), rad=(rr[pick] / pos[pick]).tolist(), mat=(rm[pick] / pos[pick]).tolist(),
                                      vac=(np.maximum(V[pick], 0) / pos[pick]).tolist(), phi=(np.abs(yy[2, pick]) / m.v).tolist(), a=np.exp(yy[0, pick]).tolist())
            cyc["max_vacuum_fraction"] = float(np.max(np.where(ex, np.maximum(V, 0) / pos, 0)))
            eq = np.where(ex & (rm > rr))[0]
            if eq.size:
                cyc["T_equality"] = float(max(rr[eq[0]], 0) ** 0.25)
                cyc["phi_at_eq_over_v"] = float(abs(yy[2, eq[0]]) / m.v)
            rho_p = 4.0 / 3.0 * rr + rm + 2.0 * K
            rho = rr + rm + K + V
            acc = ex & ((-0.5 * rho_p * (1 - 2 * rho / rho_c) + H ** 2) > 0)
            held = np.abs(yy[2]) < 0.1 * m.v
            cyc["held_accelerated_efolds"] = float(np.sum(np.where((acc & held)[:-1], H[:-1] * np.diff(tt), 0.0)))
            cyc["accelerated_efolds"] = float(np.sum(np.where(acc[:-1], H[:-1] * np.diff(tt), 0.0)))
            if sol.t_events[2].size:
                yr = sol.y_events[2][0]
                cyc["T_release"] = float(max(yr[4], 0) ** 0.25)
                cyc["a_release"] = float(np.exp(yr[0]))
            if sol.status == 1 and sol.t_events[0].size:
                yt = sol.y_events[0][0]
                cyc.update(a_max=float(np.exp(yt[0])), S_turn=float(np.exp(3 * yt[0]) * max(yt[4], 0) ** 0.75), T_turn=float(max(yt[4], 0) ** 0.25),
                           W_expand=float(yt[5]))
                y = yt.copy(); y[1] = -1e-14; t_abs = sol.t_events[0][0]; phase = "contracting"
                fa = 1e-14
                continue
            status = "no turnaround" if sol.status == 0 else "failed: " + str(sol.message)
            cycles.append(cyc)
            break
        else:
            if sol.status == 1 and sol.t_events[1].size:
                yb = sol.y_events[1][0].copy()
                c = comps(yb)
                pos = c["rr"] + c["rm"] + c["K"] + max(c["V"], 0.0)
                cyc.update(a_min=float(np.exp(yb[0])), T_bounce=float(max(yb[4], 0) ** 0.25), bounce_radiation_fraction=float(c["rr"] / pos),
                           bounce_kinetic_fraction=float(c["K"] / pos), S_bounce=float(np.exp(3 * yb[0]) * max(yb[4], 0) ** 0.75), W_cycle=float(cyc.get("W_expand", 0) + yb[5]))
                cycles.append(cyc)
                if cyc["a_max"] > 1e60:
                    status = "runaway"
                    break
                # re-thermalization at the hot bounce; freeze-out follows in the expanding phase (event 'freeze')
                pending_fo = bool(reset) and cyc["T_bounce"] > T_fo
                T_b_prev = cyc["T_bounce"]
                y = yb; y[5] = 0.0; y[1] = 1e-14; t_abs = sol.t_events[1][0]; phase = "expanding"
                cyc = dict(n=len(cycles), start="bounce", T_bounce_prev=T_b_prev)
                fa = 1e-14
                continue
            status = "no bounce" if sol.status == 0 else "failed: " + str(sol.message)
            cycles.append(cyc)
            break
    for k in range(len(cycles) - 1):
        if "a_max" in cycles[k] and "a_max" in cycles[k + 1]:
            cycles[k]["G"] = cycles[k + 1]["a_max"] / cycles[k]["a_max"]
            if cycles[k].get("S_turn", 0) > 0 and cycles[k + 1].get("S_turn", 0) > 0:
                cycles[k]["G_from_entropy"] = (cycles[k + 1]["S_turn"] / cycles[k]["S_turn"]) ** (1.0 / 3.0)
    return dict(params=dict(a_c=a_c, mu=mu, q=q, rho_c_over_Vc=rho_c_over_Vc, c_T=c_T, n_cycles=n_cycles, seed=seed, reset=reset, T_fo=T_fo, T_eq=T_EQ),
                cycles=cycles, status=status, max_residual=float(max_res), projections=proj)


def show(r):
    print(r["params"], r["status"], "res %.1e" % r["max_residual"])
    for c in r["cycles"]:
        print("  n%d %s" % (c["n"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("n",)}))


if __name__ == "__main__":
    t0 = time.time()
    out = {"runs": []}
    for kw in [dict(), dict(reset=False), dict(c_T=0.0), dict(c_T=0.1), dict(c_T=10.0), dict(rho_c_over_Vc=1e15), dict(rho_c_over_Vc=1e16),
               dict(mu=100.0), dict(q=0.3), dict(a_c=3.0)]:
        r = run(**kw)
        out["runs"].append(r)
        show(r)
        print("[%.0fs]" % (time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, "hot_cycles.json"), "w"), indent=1)
    rows = [dict(run=i, **r["params"], **{k: v for k, v in c.items() if not isinstance(v, dict)}) for i, r in enumerate(out["runs"]) for c in r["cycles"]]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(HERE, "hot_cycles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
