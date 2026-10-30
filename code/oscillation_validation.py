# -*- coding: utf-8 -*-
"""Option B, validation and long-term behaviour: is the cycle-to-cycle amplification real, what law does it obey, when do cycles stabilise, and what do vacuum transitions do?

Model (as oscillation.py; Planck units M_Pl = 1, open interior k = -1): radiation, a scalar about an AdS minimum V = V_t + m^2 phi^2 / 2 with dissipation Upsilon, and an effective bounce
        H^2 = (rho/3)(1 - (rho/rho_c)^n) + 1/a^2 ,   dH/dt = -(rho + p)(1 - (n+1)(rho/rho_c)^n)/2 - 1/a^2
with n = 1 (loop quantum cosmology) and n = 2 (a second prescription, to show that the findings do not depend on one choice). Two extra state variables accumulate the pressure-volume work
W = Integral p dV and W_phi = Integral p_phi dV with V = a^3 (a comoving region of unit comoving volume), so that the hysteresis loop of each cycle is W_n = W(turnaround n+1) - W(turnaround n).

Energy balance. The continuity equation d(rho a^3) = -p d(a^3) holds for both prescriptions, and at a turnaround rho (1 - (rho/rho_c)^n) = -3/a^2, so rho a^3 = -3 a_max (rho << rho_c there).
Hence the hysteresis law   Delta a_max,n = W_n / 3   (tested below to the integration accuracy).

Parts
  1 numerical validation: two integrators (explicit DOP853, implicit Radau, and LSODA), tolerances, maximum step; continuity across each bounce; constraint and energy-transfer residuals
  2 hysteresis: W_n, Delta a_max, G_n = a_max,n+1 / a_max,n for every cycle
  3 long-term behaviour: 100 cycles over a grid in (m, Upsilon P, phi_0), selected cases to 1000 cycles, classification and phase diagram, the second bounce prescription, and rho_c, |V_t|
  4 vacuum transitions: Model A (none), Model B (a CDL transition to a deeper AdS vacuum with a rate, only where the semiclassical description holds: R_b |H| < 0.1), Model C1 (the released
    energy thermalised) and C2 (the field left displaced from the new minimum); their effect on the cycles

    python oscillation_validation.py          (writes oscillation_validation.json, oscillation_cycles.csv; about 10 minutes)
"""
import csv
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
out = {}
CFG = dict(Vt=-1e-6, m=0.05, rho_c=1.0, k=-1.0, a_max=3000.0, n=1)
json.dump(dict(baseline=CFG, units="Planck units, M_Pl = 1", integrator="DOP853 rtol 1e-11 atol 1e-16, max_step = P/400 unless stated"),
          open(os.path.join(HERE, "oscillation_config.json"), "w"), indent=1)


def make_rhs(Vt, m, rho_c, k, ups, n):
    def rhs(t, s):
        a, H, ph, pd, rr, W, Wp = s
        Vf = Vt + 0.5 * m * m * ph * ph
        kin = 0.5 * pd * pd
        rho = kin + Vf + rr
        p = kin - Vf + rr / 3.0
        x = rho / rho_c
        Hd = -0.5 * (rho + p) * (1.0 - (n + 1) * x ** n) + k / a ** 2
        dV = 3.0 * a ** 3 * H                                  # d(a^3)/dt
        return [a * H, Hd, pd, -(3.0 * H + ups) * pd - m * m * ph, -4.0 * H * rr + ups * pd * pd, p * dV, (kin - (0.5 * m * m * ph * ph)) * dV]
    return rhs


def turnaround_state(Vt, m, rho_c, k, a_max, ph0, n):
    # rho (1 - (rho/rho_c)^n) = 3 k / a^2: Newton from the GR root
    rho = 3.0 * k / a_max ** 2
    for _ in range(50):
        f = rho * (1 - (rho / rho_c) ** n) - 3.0 * k / a_max ** 2
        fp = 1 - (n + 1) * (rho / rho_c) ** n
        rho -= f / fp
    rr = rho - Vt - 0.5 * m * m * ph0 ** 2
    if rr <= 0:
        raise ValueError("no radiation at this turnaround")
    return [a_max, 0.0, ph0, 0.0, rr, 0.0, 0.0]


def evolve(cfg, ph0, ups=0.0, n_cycles=100, method="DOP853", rtol=1e-11, atol=1e-16, step_div=400, a_stop=1e30, state0=None, t_span=None):
    Vt, m, rc, k, n = cfg["Vt"], cfg["m"], cfg["rho_c"], cfg["k"], cfg["n"]
    rhs = make_rhs(Vt, m, rc, k, ups, n)
    s0 = state0 if state0 is not None else turnaround_state(Vt, m, rc, k, cfg["a_max"], ph0, n)
    P = np.pi * np.sqrt(3.0) / np.sqrt(abs(Vt))
    def turn(t, s):
        return s[1]
    turn.direction = -1
    def bnc(t, s):
        return s[1]
    bnc.direction = 1
    def blow(t, s):
        return s[0] - a_stop
    blow.terminal = True
    T = t_span if t_span is not None else (n_cycles + 0.5) * P * 1.0
    kw = dict(max_step=P / step_div) if method != "LSODA" else dict(max_step=P / step_div)
    # run in chunks so that runaway growth (longer cycles) still reaches the requested number of cycles
    t_all, turns, bounces = 0.0, [], []
    sol = solve_ivp(rhs, [0.0, T], s0, method=method, rtol=rtol, atol=atol, events=(turn, bnc, blow), **kw)
    rho_of = lambda y: 0.5 * y[3] ** 2 + Vt + 0.5 * m * m * y[2] ** 2 + y[4]
    # a turnaround has rho = -3/a^2 (|rho| << rho_c; at large a it is zero to rounding); a bounce has rho ~ rho_c. Zero crossings of H elsewhere
    # (a scalar-dominated bounce can make H cross zero more than once near rho_c) are not cycle events.
    turns = [(t_, y_) for t_, y_ in zip(sol.t_events[0], sol.y_events[0]) if rho_of(y_) < 1e-3 * rc]
    bounces = [(t_, y_) for t_, y_ in zip(sol.t_events[1], sol.y_events[1]) if rho_of(y_) > 0.25 * rc]
    return dict(sol=sol, turns=turns, bounces=bounces, P=P, stopped=bool(sol.t_events[2].size), status=sol.status)


def cycle_table(res, cfg):
    """Per-cycle quantities from successive turnarounds: a_max, W_n, Delta a_max, G_n, scalar energy, and the energy-balance check -Delta(rho a^3) = W."""
    rows = []
    T = res["turns"]
    Vt, m = cfg["Vt"], cfg["m"]
    for i in range(len(T) - 1):
        (t1, y1), (t2, y2) = T[i], T[i + 1]
        rho = lambda y: 0.5 * y[3] ** 2 + Vt + 0.5 * m * m * y[2] ** 2 + y[4]
        W = y2[5] - y1[5]
        dE = rho(y2) * y2[0] ** 3 - rho(y1) * y1[0] ** 3
        Ephi = 0.5 * y1[3] ** 2 + 0.5 * m * m * y1[2] ** 2
        rows.append(dict(n=i, t=t1, a_max=y1[0], a_max_next=y2[0], G=y2[0] / y1[0], delta_a_max=y2[0] - y1[0], W=W, W_phi=y2[6] - y1[6],
                         energy_balance=(dE + W) / max(abs(W), abs(dE), 1e-300 + abs(rho(y1) * y1[0] ** 3) * 1e-12), law=(y2[0] - y1[0]) - W / 3.0,
                         E_phi_over_rho_r=Ephi / y1[4], period=t2 - t1))
    return rows


# ------------------------------------------------------------------------------------------------------- Part 1: validation
val = {}
ref = None
for label, kw in [("DOP853 1e-11 P/400", dict()), ("DOP853 1e-12 P/800", dict(rtol=1e-12, step_div=800)), ("DOP853 1e-9 P/200", dict(rtol=1e-9, step_div=200)),
                  ("DOP853 1e-11 P/1600", dict(step_div=1600)), ("Radau 1e-10 P/400", dict(method="Radau", rtol=1e-10, atol=1e-16)), ("LSODA 1e-11 P/400", dict(method="LSODA"))]:
    t1 = time.time()
    lin = evolve(CFG, 1e-8, n_cycles=6, **kw)
    ct = cycle_table(lin, CFG)
    mult = [ct[i + 1]["E_phi_over_rho_r"] / ct[i]["E_phi_over_rho_r"] for i in range(1, 4)]
    non = evolve(CFG, 1e-3, n_cycles=6, **kw)
    amax = [float(y[0]) for _, y in non["turns"]][:6]
    val[label] = dict(multiplier=float(np.median(mult)), a_max_nonlinear=amax, seconds=time.time() - t1)
    print("validation %-22s multiplier %.4f  a_max %s  [%.0fs]" % (label, np.median(mult), ["%.4g" % x for x in amax], time.time() - T0), flush=True)
refm = val["DOP853 1e-11 P/400"]["multiplier"]
refa = val["DOP853 1e-11 P/400"]["a_max_nonlinear"]
for k_, v in val.items():
    v["multiplier_rel_dev"] = float(abs(v["multiplier"] / refm - 1))
    n_ = min(len(v["a_max_nonlinear"]), len(refa))
    v["a_max_rel_dev"] = float(max(abs(np.array(v["a_max_nonlinear"][:n_]) / np.array(refa[:n_]) - 1)))
out["validation_integrators"] = val

# continuity across each bounce and residuals (dense output, nonlinear run)
cont = evolve(CFG, 1e-3, n_cycles=6)
sol = cont["sol"]
Vt, m, rc = CFG["Vt"], CFG["m"], CFG["rho_c"]
jumps, cres, eres = [], [], []
dense = solve_ivp(make_rhs(Vt, m, rc, CFG["k"], 0.0, 1), [0, sol.t[-1]], turnaround_state(Vt, m, rc, CFG["k"], CFG["a_max"], 1e-3, 1), method="DOP853", rtol=1e-11, atol=1e-16,
                  dense_output=True, max_step=cont["P"] / 400)
for tb, yb in cont["bounces"]:
    tt = np.linspace(tb - 5.0, tb + 5.0, 20001)
    a, H, ph, pd, rr, W, Wp = dense.sol(tt)
    rho = 0.5 * pd ** 2 + Vt + 0.5 * m * m * ph ** 2 + rr
    p = 0.5 * pd ** 2 - (Vt + 0.5 * m * m * ph ** 2) + rr / 3.0
    # largest step-to-step change relative to the local scale: no jumps
    jumps.append(float(max(np.max(np.abs(np.diff(rho))) / np.max(np.abs(rho)), np.max(np.abs(np.diff(pd))) / max(np.max(np.abs(pd)), 1e-300))))
    scale = np.abs(rho) / 3.0 + 1.0 / a ** 2
    cres.append(float(np.max(np.abs(H ** 2 - (rho / 3.0) * (1 - rho / rc) - 1.0 / a ** 2) / scale)))
    drho = np.gradient(rho, tt)
    eres.append(float(np.max(np.abs(drho + 3 * H * (rho + p))) / np.max(np.abs(3 * H * (rho + p)))))
out["bounce_continuity"] = dict(max_relative_step_change=max(jumps), constraint_residual_at_bounces=max(cres), continuity_residual_at_bounces=max(eres), n_bounces=len(jumps))
print("continuity: step change %.1e, constraint %.1e, continuity %.1e over %d bounces" % (max(jumps), max(cres), max(eres), len(jumps)), flush=True)

# ------------------------------------------------------------------------------------------------------- Part 2: hysteresis law
hyst = {}
for label, ph0, ups in (("phi0=1e-3, no dissipation", 1e-3, 0.0), ("phi0=1e-3, Upsilon P=0.92", 1e-3, 3e-4), ("phi0=1e-8, no dissipation", 1e-8, 0.0)):
    r = evolve(CFG, ph0, ups=ups, n_cycles=14)
    ct = cycle_table(r, CFG)
    hyst[label] = ct
    resolved = [c for c in ct if abs(c["delta_a_max"]) > 1e-3 * c["a_max"] and c["a_max_next"] < 1e6]
    law = max([abs(c["law"]) / max(abs(c["delta_a_max"]), abs(c["W"]) / 3.0, 1e-300) for c in resolved] or [0.0])
    bal = max(abs(c["energy_balance"]) for c in ct)
    corr = float(np.corrcoef([c["W"] for c in ct], [c["delta_a_max"] for c in ct])[0, 1]) if len(ct) > 2 else None
    print("hysteresis %-28s cycles %d  law rel. error %.1e  energy balance %.1e  corr(W, da) %.6f  W_phi/W %s" % (label, len(ct), law, bal, corr if corr is not None else 0,
          ["%.2f" % (c["W_phi"] / c["W"]) for c in ct[:4] if c["W"] != 0]), flush=True)
out["hysteresis"] = {k_: [{kk: float(vv) for kk, vv in c.items()} for c in v] for k_, v in hyst.items()}

# ------------------------------------------------------------------------------------------------------- Part 3: long-term behaviour and phase diagram
def classify(res, ct, n_req):
    if res["stopped"]:
        return "growing (runaway)"
    if len(ct) < n_req - 2:
        return "terminating (turnaround lost)" if len(res["turns"]) < 3 else "growing (runaway)"
    G = np.array([c["G"] for c in ct[-20:]])
    if np.all(np.abs(G - 1) < 1e-3):
        return "stable"
    if np.all(G > 1):
        return "growing"
    if np.all(G < 1):
        return "shrinking"
    return "irregular"


grid, csv_rows = [], []
P0 = np.pi * np.sqrt(3.0) / np.sqrt(abs(CFG["Vt"]))
Pcyc = 0.5641 * P0
for mm in (0.01, 0.03, 0.05, 0.1, 0.2):
    for UP in (0.0, 0.3, 1.0, 3.0, 10.0):
        for ph0 in (1e-8, 1e-3):
            cfg = dict(CFG, m=mm)
            ups = UP / Pcyc
            try:
                r = evolve(cfg, ph0, ups=ups, n_cycles=100, a_stop=1e30)
            except ValueError:
                continue
            ct = cycle_table(r, cfg)
            cls = classify(r, ct, 100)
            G = [c["G"] for c in ct]
            grid.append(dict(m=mm, upsilon_P=UP, phi0=ph0, cycles=len(ct), cls=cls, G_last=float(G[-1]) if G else None, a_max_last=float(r["turns"][-1][1][0]) if r["turns"] else None,
                             E_phi_last=float(ct[-1]["E_phi_over_rho_r"]) if ct else None))
            for c in ct:
                csv_rows.append(dict(m=mm, upsilon_P=UP, phi0=ph0, **{kk: float(vv) for kk, vv in c.items()}))
            print("grid m=%.2f UP=%4.1f phi0=%.0e: %3d cycles  %-22s G_last %s [%.0fs]" % (mm, UP, ph0, len(ct), cls, "%.4f" % G[-1] if G else "-", time.time() - T0), flush=True)
out["phase_diagram"] = grid
with open(os.path.join(HERE, "oscillation_cycles.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)

# 1000 cycles for selected stable cases
long = []
for mm, UP in ((0.05, 3.0), (0.2, 10.0)):
    cfg = dict(CFG, m=mm)
    r = evolve(cfg, 1e-3, ups=UP / Pcyc, n_cycles=1000)
    ct = cycle_table(r, cfg)
    G = np.array([c["G"] for c in ct])
    long.append(dict(m=mm, upsilon_P=UP, cycles=len(ct), max_abs_G_minus_1_last_500=float(np.max(np.abs(G[-500:] - 1))), cls=classify(r, ct, 1000),
                     period_spread_last_500=float(np.std([c["period"] for c in ct[-500:]]) / np.mean([c["period"] for c in ct[-500:]]))))
    print("1000 cycles m=%.2f UP=%.0f: %d cycles, %s, max|G-1| %.1e [%.0fs]" % (mm, UP, len(ct), long[-1]["cls"], long[-1]["max_abs_G_minus_1_last_500"], time.time() - T0), flush=True)
# radiation-only control to 1000 cycles
r = evolve(CFG, 0.0, n_cycles=1000)
ct = cycle_table(r, CFG)
long.append(dict(m=CFG["m"], upsilon_P=0.0, phi0=0.0, cycles=len(ct), max_abs_G_minus_1_last_500=float(np.max(np.abs(np.array([c["G"] for c in ct])[-500:] - 1))), cls=classify(r, ct, 1000)))
print("radiation-only 1000 cycles: %d cycles, %s, max|G-1| %.1e" % (len(ct), long[-1]["cls"], long[-1]["max_abs_G_minus_1_last_500"]), flush=True)
out["long_runs"] = long

# second bounce prescription (n = 2), and other rho_c, |V_t|
alt = []
for label, cfg in (("n=2", dict(CFG, n=2)), ("rho_c=0.1", dict(CFG, rho_c=0.1)), ("rho_c=10", dict(CFG, rho_c=10.0)), ("Vt=-1e-4", dict(CFG, Vt=-1e-4, a_max=300.0)), ("Vt=-1e-8", dict(CFG, Vt=-1e-8, a_max=30000.0))):
    lin = evolve(cfg, 1e-8, n_cycles=6)
    ctl = cycle_table(lin, cfg)
    mult = float(np.median([ctl[i + 1]["E_phi_over_rho_r"] / ctl[i]["E_phi_over_rho_r"] for i in range(1, 4)]))
    non = evolve(cfg, 1e-3, n_cycles=10)
    ctn = cycle_table(non, cfg)
    law = max([abs(c["law"]) / max(abs(c["delta_a_max"]), abs(c["W"]) / 3.0, 1e-300) for c in ctn if abs(c["delta_a_max"]) > 1e-3 * c["a_max"] and c["a_max_next"] < 1e6] or [0.0])
    rad = evolve(cfg, 0.0, n_cycles=6)
    ctr = cycle_table(rad, cfg)
    alt.append(dict(label=label, multiplier=mult, law_rel_error=float(law), radiation_only_max_abs_G_minus_1=float(max(abs(c["G"] - 1) for c in ctr)), growth_10_cycles=float(non["turns"][-1][1][0] / non["turns"][0][1][0])))
    print("alt %-10s multiplier %.2f  law %.1e  radiation-only |G-1| %.1e  growth %.3g" % (label, mult, law, alt[-1]["radiation_only_max_abs_G_minus_1"], alt[-1]["growth_10_cycles"]), flush=True)
out["robustness"] = alt

# ------------------------------------------------------------------------------------------------------- Part 4: vacuum transitions
# Model A: baseline (stable, Upsilon P = 3). Model C1/C2: at the 3rd turnaround (H = 0, a large: the semiclassical regime) the vacuum drops by dV = 0.5 |V_t|:
#   C1: the released energy is thermalised (radiation += dV, field at the new minimum)  C2: the field is left displaced, (1/2) m^2 dphi^2 = dV.
def with_transition(model, dV_frac=0.5, UP=3.0):
    cfg = dict(CFG)
    ups = UP / Pcyc
    r1 = evolve(cfg, 1e-3, ups=ups, n_cycles=3, t_span=None)
    tt, y = r1["turns"][2]
    y = np.array(y, dtype=float)
    dV = dV_frac * abs(cfg["Vt"])
    cfg2 = dict(cfg, Vt=cfg["Vt"] - dV)
    if model == "C1":
        y[4] += dV
    elif model == "C2":
        y[2] = np.sqrt(2.0 * dV) / cfg["m"]
    r2 = evolve(cfg2, None, ups=ups, n_cycles=30, state0=list(y))
    return cycle_table(r1, cfg) + cycle_table(r2, cfg2), cfg2


trans = {}
for model in ("A", "C1", "C2"):
    if model == "A":
        r = evolve(CFG, 1e-3, ups=3.0 / Pcyc, n_cycles=33)
        ct = cycle_table(r, CFG)
    else:
        ct, _ = with_transition(model)
    G = np.array([c["G"] for c in ct])
    trans[model] = dict(cycles=len(ct), a_max=[float(c["a_max"]) for c in ct], G=[float(x) for x in G], period=[float(c["period"]) for c in ct],
                        final_cls="stable" if np.all(np.abs(G[-10:] - 1) < 1e-3) else ("growing" if np.all(G[-5:] > 1) else "other"))
    print("transition model %-3s: %d cycles, G last %s, class %s" % (model, len(ct), np.round(G[-3:], 4), trans[model]["final_cls"]), flush=True)
out["transitions"] = trans
# Model C2 without enough dissipation: does the displaced field destabilise the cycles?
ct, _ = with_transition("C2", UP=1.0)
out["transitions"]["C2_UP1"] = dict(G=[float(c["G"]) for c in ct], a_max=[float(c["a_max"]) for c in ct])
print("C2 with Upsilon P = 1: a_max", np.round([c["a_max"] for c in ct[:12]], 0), flush=True)
# Model B: where is the semiclassical description valid, and what is the transition probability per cycle for a rate Gamma per unit four-volume?
r = evolve(CFG, 0.0, n_cycles=3)
tb = r["bounces"]
t1, t2 = r["turns"][0][0], r["turns"][1][0]
dd = solve_ivp(make_rhs(CFG["Vt"], CFG["m"], CFG["rho_c"], CFG["k"], 0.0, 1), [t1, t2], r["turns"][0][1], method="DOP853", rtol=1e-11, atol=1e-16, dense_output=True, max_step=Pcyc / 400)
tt = np.linspace(t1, t2, 400001)
aa, HH = dd.sol(tt)[0], dd.sol(tt)[1]
vol = np.trapezoid(aa ** 3, tt)                                             # four-volume per unit comoving volume over one cycle
modelB = []
for Rb in (1.0, 10.0, 100.0):
    valid = Rb * np.abs(HH) < 0.1
    modelB.append(dict(R_b=Rb, valid_fraction_of_time=float(valid.mean()), valid_fraction_of_four_volume=float(np.trapezoid(aa ** 3 * valid, tt) / vol)))
out["model_B"] = dict(four_volume_per_cycle=float(vol), rows=modelB, note="P(transition per cycle and comoving volume) = 1 - exp(-Gamma x valid four-volume)")
print("model B", out["model_B"], flush=True)
json.dump(out, open(os.path.join(HERE, "oscillation_validation.json"), "w"), indent=1, default=float)
print("saved oscillation_validation.json, oscillation_cycles.csv [%.0fs]" % (time.time() - T0))
