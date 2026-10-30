# -*- coding: utf-8 -*-
"""Option B: can the same region oscillate after the dark energy decays? Turnaround from an anti-de Sitter vacuum, an effective bounce, repeated cycles.

The decay of the present vacuum into a true vacuum with V_t < 0 is the quantum trigger that can stop the expansion (Section 5.3: the open interior crunches). A crunch becomes a bounce
only with physics beyond general relativity; we use the effective loop-quantum-cosmology Friedmann equation with a critical density rho_c (an effective model, not established physics):

        H^2 = (rho / 3 M^2)(1 - rho / rho_c) - k / a^2 ,      dH/dt = -(rho + p)(1 - 2 rho / rho_c) / (2 M^2) + k / a^2

with k = -1 (the open interior of a bubble; k = 0 as a check), radiation, and the scalar oscillating about its AdS minimum, V = V_t + m^2 phi^2 / 2, with optional dissipation Upsilon (phi -> radiation).
Units M_Pl = 1. The hierarchy rho_c / |V_t| ~ 1e120 cannot be integrated directly, so the dynamics are simulated at rho_c / |V_t| = 1e4 ... 1e8 and the scaling of every
quantity with |V_t| and rho_c is measured and extrapolated (the period, the bounce density, the turnaround size).

Outputs: periodicity and its stability, the bounce (a_min > 0, rho_max = rho_c, finite curvature), the constraint residual, the shape of H(t) against a sine wave, the fraction of each cycle that
needs Hdot > 0 (null-energy-condition violation in the effective sense), entropy growth over cycles with dissipation, and the numbers for the physical case.

    python oscillation.py          (writes oscillation.json; about 1 minute)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
T0 = time.time()


def rhs_factory(Vt, m, rho_c, k, ups):
    def rhs(t, s):
        a, H, ph, pd, rr = s
        V = Vt + 0.5 * m * m * ph * ph
        kin = 0.5 * pd * pd
        rho = kin + V + rr
        p = kin - V + rr / 3.0
        Hd = -0.5 * (rho + p) * (1.0 - 2.0 * rho / rho_c) + k / a ** 2
        return [a * H, Hd, pd, -(3.0 * H + ups) * pd - m * m * ph, -4.0 * H * rr + ups * pd * pd]
    return rhs


def start_at_turnaround(Vt, m, rho_c, k, a_max, ph0):
    """State at a turnaround (H = 0): solve the constraint for the radiation density."""
    # rho (1 - rho / rho_c) = 3 k / a^2  ->  rho = [rho_c - sqrt(rho_c^2 - 12 rho_c k / a^2)] / 2   (the root continuous with GR)
    rho = 0.5 * (rho_c - np.sqrt(rho_c ** 2 - 12.0 * rho_c * k / a_max ** 2))
    rr = rho - Vt - 0.5 * m * m * ph0 ** 2
    if rr <= 0:
        raise ValueError("no radiation at this turnaround")
    return [a_max, 0.0, ph0, 0.0, rr]


def run(Vt, m, rho_c, k, a_max, ph0, ups=0.0, n_cycles=8, rtol=1e-10, atol=1e-14):
    rhs = rhs_factory(Vt, m, rho_c, k, ups)
    s0 = start_at_turnaround(Vt, m, rho_c, k, a_max, ph0)
    period_guess = 2.0 * np.pi * np.sqrt(3.0) / np.sqrt(abs(Vt))
    def bounce(t, s):
        return s[1]
    bounce.direction = 1
    def turn(t, s):
        return s[1]
    turn.direction = -1
    sol = solve_ivp(rhs, [0.0, (n_cycles + 0.6) * period_guess], s0, method="DOP853", rtol=rtol, atol=atol, events=(bounce, turn), dense_output=True, max_step=period_guess / 400)
    tb, tt = sol.t_events[0], sol.t_events[1]
    t = np.linspace(0, sol.t[-1], 200000)
    a, H, ph, pd, rr = sol.sol(t)
    V = Vt + 0.5 * m * m * ph ** 2
    rho = 0.5 * pd ** 2 + V + rr
    resid = (H ** 2 - (rho / 3.0) * (1 - rho / rho_c) + k / a ** 2) / np.maximum(H ** 2 + abs(rho) / 3.0, 1e-300)
    # values at events
    yb = sol.y_events[0]
    yt = sol.y_events[1]
    rho_at = lambda y: 0.5 * y[:, 3] ** 2 + Vt + 0.5 * m * m * y[:, 2] ** 2 + y[:, 4]
    p_at = lambda y: 0.5 * y[:, 3] ** 2 - (Vt + 0.5 * m * m * y[:, 2] ** 2) + y[:, 4] / 3.0
    Hd = -0.5 * (rho + (0.5 * pd ** 2 - V + rr / 3.0)) * (1.0 - 2.0 * rho / rho_c) + k / a ** 2
    return dict(sol=sol, t=t, a=a, H=H, ph=ph, pd=pd, rr=rr, rho=rho, Hd=Hd, resid=resid, t_bounce=tb, t_turn=tt, y_bounce=yb, y_turn=yt,
                rho_bounce=rho_at(yb) if len(yb) else np.array([]), a_bounce=yb[:, 0] if len(yb) else np.array([]), a_turn=yt[:, 0] if len(yt) else np.array([]),
                rr_turn=yt[:, 4] if len(yt) else np.array([]))


def sine_fundamental_fraction(t, H, t0, t1):
    """Fraction of the power of H(t) on [t0, t1) (one bounce-to-bounce cycle) in the fundamental harmonic."""
    sel = (t >= t0) & (t < t1)
    tt = t[sel]
    hh = H[sel]
    n = 4096
    tu = np.linspace(t0, t1, n, endpoint=False)
    hu = np.interp(tu, tt, hh)
    F = np.fft.rfft(hu - hu.mean())
    P = np.abs(F) ** 2
    return float(P[1] / P[1:].sum())


# ------------------------------------------------------------------------------------------ baseline: radiation + AdS + effective bounce, k = -1 (no scalar excitation)
def run_events(Vt, m, rc, k, a_max, ph0, ups=0.0, T=None, dense=False):
    rhs = rhs_factory(Vt, m, rc, k, ups)
    s0 = start_at_turnaround(Vt, m, rc, k, a_max, ph0)
    def b_ev(t, s):
        return s[1]
    b_ev.direction = 1
    def t_ev(t, s):
        return s[1]
    t_ev.direction = -1
    T = T or 12.0 * np.pi * np.sqrt(3.0) / np.sqrt(abs(Vt))
    sol = solve_ivp(rhs, [0.0, T], s0, method="DOP853", rtol=1e-11, atol=1e-16, events=(b_ev, t_ev), dense_output=dense)
    return sol


def true_turns(sol, Vt, m):
    """Turnaround states only (rho < 0): H also crosses zero near rho_c when the scalar dominates a bounce."""
    y = sol.y_events[1]
    if len(y) == 0:
        return y
    rho = 0.5 * y[:, 3] ** 2 + Vt + 0.5 * m * m * y[:, 2] ** 2 + y[:, 4]
    return y[rho < 1e-3]                                         # |rho| << rho_c = 1 (at large a it is zero to rounding)


def residual_on(sol, Vt, m, rc, k, n=200000):
    t = np.linspace(sol.t[0], sol.t[-1], n)
    a, H, ph, pd, rr = sol.sol(t)
    rho = 0.5 * pd ** 2 + Vt + 0.5 * m * m * ph ** 2 + rr
    scale = np.abs(rho) / 3.0 * np.abs(1 - rho / rc) + 1.0 / a ** 2
    return t, a, H, ph, pd, rr, rho, (H ** 2 - (rho / 3.0) * (1 - rho / rc) + k / a ** 2) / scale


base = dict(Vt=-1e-6, m=0.05, rc=1.0, k=-1.0, a_max=3000.0)
sol = run_events(base["Vt"], base["m"], base["rc"], base["k"], base["a_max"], 0.0, dense=True)
t, a, H, ph, pd, rr, rho, res = residual_on(sol, base["Vt"], base["m"], base["rc"], base["k"])
tb, tt = sol.t_events[0], sol.t_events[1]
P = np.diff(tb)
T_ads = np.pi * np.sqrt(3.0) / np.sqrt(abs(base["Vt"]))
Hd = np.gradient(H, t)
sel = (t >= tb[1]) & (t < tb[2])
rho_b = 0.5 * sol.y_events[0][:, 3] ** 2 + base["Vt"] + sol.y_events[0][:, 4]
out["periodic"] = dict(params=base, n_bounces=int(len(tb)), period_mean=float(P.mean()), period_spread=float(P.std() / P.mean()), period_over_ads_crunch=float(P.mean() / T_ads),
                       rho_bounce_over_rho_c=float(np.mean(rho_b) / base["rc"]), a_min=float(np.mean(sol.y_events[0][:, 0])), a_max=float(np.mean(sol.y_events[1][:, 0])),
                       a_max_closure=float(np.max(np.abs(sol.y_events[1][:, 0] / base["a_max"] - 1))), max_residual=float(np.max(np.abs(res))),
                       fundamental_power_fraction=sine_fundamental_fraction(t, H, tb[1], tb[2]), fraction_of_cycle_with_Hdot_positive=float(np.mean(Hd[sel] > 0)),
                       H_max=float(H.max()), H_min=float(H.min()), max_curvature_R_over_rho_c=float(np.max(np.abs(6 * (Hd + 2 * H ** 2)))) )
print("periodic:", {k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in out["periodic"].items() if k_ != "params"}, flush=True)
# dense trace over three cycles (the bounce lasts ~1 time unit out of a period of ~3000)
t3 = np.concatenate([np.linspace(0, 3.2 * P.mean(), 20000)] + [np.linspace(tb_ - 6, tb_ + 6, 2000) for tb_ in tb if tb_ < 3.2 * P.mean()])
t3 = np.unique(t3)
a3, H3 = sol.sol(t3)[0], sol.sol(t3)[1]
out["trace"] = dict(t=t3.tolist(), a=a3.tolist(), H=H3.tolist())

# scaling with |V_t| and rho_c
sc = []
for Vt, rc in ((-1e-4, 1.0), (-1e-6, 1.0), (-1e-8, 1.0), (-1e-6, 0.1), (-1e-6, 10.0)):
    s_ = run_events(Vt, 0.05, rc, -1.0, 3.0 / np.sqrt(abs(Vt)), 0.0, T=5 * np.pi * np.sqrt(3.0) / np.sqrt(abs(Vt)))
    Pm = float(np.mean(np.diff(s_.t_events[0])))
    rb = float(np.mean(s_.y_events[0][:, 4] + Vt) / rc)
    sc.append(dict(Vt=Vt, rho_c=rc, period=Pm, period_over_ads=float(Pm / (np.pi * np.sqrt(3.0) / np.sqrt(abs(Vt)))), rho_b_over_rho_c=rb, a_min=float(np.mean(s_.y_events[0][:, 0]))))
    print("scaling", sc[-1], flush=True)
slope = np.polyfit(np.log([-x["Vt"] for x in sc[:3]]), np.log([x["period"] for x in sc[:3]]), 1)[0]
out["scaling"] = dict(rows=sc, period_slope_in_Vt=float(slope))

# stability (a): radiation perturbation keeps periodic cycles (neutral); (b) a scalar perturbation grows from cycle to cycle (Floquet multiplier in energy)
stab = []
for da in (-0.01, 0.01):
    s_ = run_events(base["Vt"], base["m"], base["rc"], base["k"], base["a_max"] * (1 + da), 0.0)
    stab.append(dict(delta_a_max=da, period_mean=float(np.mean(np.diff(s_.t_events[0]))), a_max_drift=float(np.max(np.abs(s_.y_events[1][:, 0] / s_.y_events[1][0, 0] - 1)))))
floq = []
for m in (0.01, 0.03, 0.05, 0.1, 0.2):
    s_ = run_events(base["Vt"], m, base["rc"], base["k"], base["a_max"], 1e-8, T=8 * np.pi * np.sqrt(3.0) / np.sqrt(abs(base["Vt"])))
    y = true_turns(s_, base["Vt"], m)
    E = 0.5 * y[:, 3] ** 2 + 0.5 * m * m * y[:, 2] ** 2
    ratios = E[2:] / E[1:-1]
    floq.append(dict(m=m, m_over_H_ads=float(m / np.sqrt(abs(base["Vt"]) / 3.0)), m_over_H_bounce=float(m / np.sqrt(base["rc"] / 12.0)), energy_multiplier_per_cycle=float(np.median(ratios[1:5])),
                     ratios=[float(x) for x in ratios]))
    print("Floquet m=%.2f: energy x%.1f per cycle" % (m, floq[-1]["energy_multiplier_per_cycle"]), flush=True)
out["stability"] = dict(radiation=stab, scalar_floquet=floq)

# nonlinear: a finite scalar excitation -> growing cycles (hysteresis)
s_ = run_events(base["Vt"], base["m"], base["rc"], base["k"], base["a_max"], 1e-3, T=10 * np.pi * np.sqrt(3.0) / np.sqrt(abs(base["Vt"])), dense=True)
out["hysteresis"] = dict(phi0=1e-3, a_max=[float(x) for x in true_turns(s_, base["Vt"], base["m"])[:, 0]])
th = np.linspace(0, s_.t[-1], 6000)
ah, Hh = s_.sol(th)[0], s_.sol(th)[1]
out["trace_hysteresis"] = dict(t=th.tolist(), a=ah.tolist(), H=Hh.tolist())
print("hysteresis a_max", np.round(true_turns(s_, base["Vt"], base["m"])[:, 0], 0), flush=True)

# dissipation against amplification: a finite scalar excitation with friction Upsilon -> runaway (hysteresis) or stabilised periodic cycles; entropy per cycle
ent = []
P_cyc = out["periodic"]["period_mean"]
for ups in (0.0, 1e-4, 3e-4, 1e-3, 1e-2):
    s_ = run_events(base["Vt"], base["m"], base["rc"], base["k"], base["a_max"], 1e-3, ups=ups, T=10 * np.pi * np.sqrt(3.0) / np.sqrt(abs(base["Vt"])))
    y = true_turns(s_, base["Vt"], base["m"])
    S_ = y[:, 4] ** 0.75 * y[:, 0] ** 3
    ent.append(dict(upsilon=ups, upsilon_times_period=float(ups * P_cyc), a_max=[float(x) for x in y[:, 0]], S_growth=[float(x) for x in S_[1:] / S_[:-1]],
                    runaway=bool(y[-1, 0] > 10 * base["a_max"]), total_entropy_growth=float(S_[-1] / S_[0])))
    print("dissipation %.0e (x period %.2f): a_max last %.3g, runaway %s, S_total x%.4g" % (ups, ups * P_cyc, y[-1, 0], ent[-1]["runaway"], ent[-1]["total_entropy_growth"]), flush=True)
out["dissipation"] = ent
out["ln_multiplier_m005"] = float(np.log([f["energy_multiplier_per_cycle"] for f in floq if f["m"] == 0.05][0]))

# ------------------------------------------------------------------------------------------ the physical case: numbers from the scalings
rho_c_lqc = 0.41 * 64.0 * np.pi ** 2              # 0.41 rho_Pl with rho_Pl = 1/G^2 = 64 pi^2 M^4 (reduced Planck units)
gstar = 106.75
T_bounce = (30.0 * rho_c_lqc / (np.pi ** 2 * gstar)) ** 0.25 * K.M_PL           # GeV
phys = []
for kappa in (1.0, 15.7, 1e10, 1e35):
    Vt_phys = kappa * K.V_F
    period_s = np.pi * np.sqrt(3.0) * K.M_PL / np.sqrt(Vt_phys) * K.HBAR_GEV_S * float(np.mean([x["period_over_ads"] for x in sc]))
    phys.append(dict(kappa=kappa, period_s=float(period_s), period_Gyr=float(period_s / K.YEAR_S / 1e9)))
out["physical"] = dict(rho_c_over_M4=float(rho_c_lqc), T_bounce_GeV=float(T_bounce), rows=phys)
# validity of a CDL description near the bounce: bubble size against the time scale of the background at rho_c
R_b = 5.3 / (3.05e-12)                              # GeV^-1, the benchmark bubble radius (5.3 / (sqrt(lam) f), f = 3.05 meV)
H_b = np.sqrt(rho_c_lqc / 3.0) * K.M_PL            # the background rate near the bounce, GeV
out["cdl_near_bounce"] = dict(R_b_GeVinv=R_b, H_bounce_GeV=float(H_b), R_b_times_H=float(R_b * H_b))
# sine-wave law H = H0 sin(w t) in flat GR: rho + p = -2 M^2 Hdot < 0 whenever cos(w t) > 0, i.e. half of every period
out["sine_wave_law"] = dict(fraction_of_period_violating_NEC=0.5)
print("physical", out["physical"], "CDL near bounce R_b H =", "%.1e" % (R_b * H_b))
json.dump(out, open(os.path.join(HERE, "oscillation.json"), "w"), indent=1)
print("saved oscillation.json [%.0fs]" % (time.time() - T0))
