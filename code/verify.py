# -*- coding: utf-8 -*-
"""Executable checks of the numerical machinery behind the paper. Writes verification.json; the manuscript builder refuses to run
unless every check passes.

    python verify.py          (about 3 minutes; needs bounce_tables.json, monte_carlo.json, interior_evolution.json)

 V1  flat-space bounce action tends to the thin-wall value as the tilt vanishes
 V2  shooting and boundary-value solutions agree
 V3  scaling B(lam, eta) = b(eta) / lam
 V4  the de Sitter background action is -24 pi^2 / (g^2 v_f)
 V5  the gravitational bounce tends to the flat bounce, with the tabulated linear coefficient
 V6  the gravitational suppression agrees with the analytic thin-wall (Parke) suppression
 V7  Hawking-Moss consistency: no CDL bounce where |V''_top| < 4 H^2, and B_CDL < B_HM wherever one exists
 V8  interpolation of b(eta) off the grid
 V9  the critical action solves Gamma/H^4 = 1
 V10 the constraint rho'^2 = 1 + (g rho^2 / 3)(x'^2/2 - v) holds along the numerical bounce
 V11 closed-form interior radiation density against the full numerical integration
 V12 physical constants and the reheating bound
 V13 Monte Carlo: regime fractions, reproducibility, convergence
"""
import json
import os

import numpy as np

import cdl_lib as C
import constants as K
import rates as R
import interior_evolution as I
import monte_carlo as M

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
MC = json.load(open(os.path.join(HERE, "monte_carlo.json"), encoding="utf-8"))
res = {}


def rec(name, ok, **kw):
    res[name] = dict(passed=bool(ok), **kw)
    print(("PASS " if ok else "FAIL ") + name, kw)


flat = {r["eta"]: r for r in T["flat"]}
# V1
ratios = [flat[e]["ratio"] for e in (0.03, 0.04, 0.05, 0.06)]
rec("V1_thin_wall_limit", all(np.diff(ratios) < 0) and ratios[0] > 0.99, ratio_at_eta_0p03=round(ratios[0], 4), ratio_at_eta_0p06=round(ratios[-1], 4))
# V2
d = [abs(flat[e]["b"] / flat[e]["bvp_check"] - 1) for e in flat if flat[e]["bvp_check"]]
rec("V2_shooting_vs_bvp", max(d) < 1e-6, cases=len(d), max_relative_difference=float(max(d)))
# V3
sc = max(c["rel"] for c in T["scaling_check"])
rec("V3_lambda_scaling", sc < 1e-10, max_relative_difference=float(sc))
# V4
g, eta = 0.1, 0.3
pot = C.Potential(1.0, eta)
H = np.sqrt(g * pot.v_f / 3.0)


def F(a):
    c = np.cos(a)
    return pot.v_f * (2.0 / 3.0 - c + c ** 3 / 3.0) / H ** 4 - 3.0 * (1.0 - c) / (g * H ** 2)


S_bg = 4 * np.pi ** 2 * F(np.pi)
rec("V4_de_sitter_action", abs(S_bg / (-24 * np.pi ** 2 / (g ** 2 * pot.v_f)) - 1) < 1e-12, rel=float(abs(S_bg / (-24 * np.pi ** 2 / (g ** 2 * pot.v_f)) - 1)))
# V5
B0 = flat[0.3]["b"]
rows = [r for r in T["grid"] if r["eta"] == 0.3 and r["found"]]
rows.sort(key=lambda r: r["g"])
mono = all(np.diff([r["ratio"] for r in rows]) < 0)
sl = [s for s in T["slopes"] if s["eta"] == 0.3][0]
r5 = C.grav_action(C.grav_bounce(1.0, 0.3, 1e-4), 1e-4) / B0 - 1.0
rec("V5_flat_limit_of_gravity", mono and abs(r5) < 1e-3 and abs(r5 + sl["kappa"] * 1e-4) < 3e-4 and abs(sl["kappa"] / sl["kappa2"] - 1) < 0.15,
    ratio_minus_one_at_g_1em4=float(r5), predicted_from_kappa=float(-sl["kappa"] * 1e-4), kappa_g1em3=float(sl["kappa"]), kappa_g2em3=float(sl["kappa2"]), monotone=bool(mono))
# V6
ok6, vals = True, []
for eta6 in (0.12, 0.2):
    p6 = C.Potential(1.0, eta6)
    sig, dV = p6.wall_tension(), p6.dtil
    for g6 in (1e-3, 1e-2):
        Bn = C.grav_action(C.grav_bounce(1.0, eta6, g6), g6)
        Bp = C.parke_thin_wall(sig, dV, p6.v_f, g6, v_t=0.0)
        rn, rp = Bn / flat[eta6]["b"], Bp / C.thin_wall_B(1.0, eta6)
        vals.append((eta6, g6, round(float(rn), 4), round(float(rp), 4)))
        ok6 &= abs(rn - rp) < 0.015
rec("V6_parke_thin_wall_suppression", ok6, numerical_vs_analytic_ratio=vals)
# V7
bad_exist = [r for r in T["grid"] if r["found"] and r["hm_ratio"] < 1.0]
bad_hm = [r for r in T["grid"] if r["found"] and not r["B"] < r["B_hm"]]
rec("V7_hawking_moss_consistency", not bad_exist and not bad_hm, grid_points=len(T["grid"]), bounces_found=sum(r["found"] for r in T["grid"]),
    found_with_criterion_violated=len(bad_exist), found_with_B_above_HM=len(bad_hm))
# V8
errs = []
for e in (0.1235, 0.2231, 0.3174, 0.3615):
    errs.append(abs(C.flat_bounce(1.0, e)["B"] / float(R.b_of_eta(e)) - 1))
rec("V8_interpolation_off_grid", max(errs) < 1e-3, max_relative_error=float(max(errs)), errors=[float(x) for x in errs])
# V9
ok9 = True
for lam, eta9, lnD in ((1.0, 0.3, 0.0), (5.0, 0.2, 1.0), (0.3, 0.37, -2.0)):
    Bc = float(R.b_critical(lam, eta9, lnD))
    Bk = (Bc / (2 * np.pi)) ** 2 * np.exp(lnD)
    lg = np.log(9.0 * Bk * lam / R.dtil_of_eta(eta9)) + K.LN_M4_OVER_VF - Bc
    ok9 &= abs(lg) < 1e-9
rec("V9_critical_action", ok9)
# V10
res10 = C.grav_bounce(1.0, 0.3, 0.1)
sol, p10 = res10["sol"], res10["pot"]
s = np.linspace(sol.t[0] * 3, sol.t[-1] * 0.98, 4000)
x, p, rho, rp = sol.sol(s)
con = rp ** 2 - 1.0 - (0.1 * rho ** 2 / 3.0) * (p ** 2 / 2.0 - p10.v(x))
rec("V10_friedmann_constraint", np.max(np.abs(con)) < 1e-6, max_residual=float(np.max(np.abs(con))))
# V11
tau0, gamma, eps = 1e-3, 500.0, 1.0
tau, al, rr, da = I.integrate_full(eps, gamma, tau0, 5.0, n=600)
cf = I.rho_r_milne(tau, eps, gamma, tau0)
sel = tau > 2 * tau0
rec("V11_interior_closed_form", np.max(np.abs(rr[sel] / cf[sel] - 1)) < 1e-5, max_relative_difference=float(np.max(np.abs(rr[sel] / cf[sel] - 1))))
# V12
Tmax = K.t_reh_max(K.V_F)
rec("V12_constants", abs(K.V_F / 2.51e-47 - 1) < 0.03 and abs(Tmax * 1e12 - 2.19) < 0.02 and 270 < K.LN_M4_OVER_VF < 285,
    rho_lambda_gev4=float(K.V_F), T_reh_max_meV=float(Tmax * 1e12), ln_M4_over_Vf=float(K.LN_M4_OVER_VF))
# V13
b = MC["baseline"]
tot = b["frac_inconsistent_with_survival"] + b["frac_decay_by"]["1e+100"] + b["frac_never_within_1e100"]
lam, eta13, lnD, rl = M.draw(10 ** 5, MC["seed"] + 7 * 10 ** 5)
s13 = M.summarize(*M.evaluate(lam, eta13, lnD, rl))
conv = MC["convergence"]
se = MC["baseline_batch"]["never"]["se"]
rec("V13_monte_carlo", abs(tot - 1.0) < 1e-9 and abs(s13["frac_never_within_1e100"] - conv[2]["never"]) < 1e-12 and abs(conv[2]["never"] - conv[3]["never"]) < 5 * max(se, 0.005),
    fractions_sum=float(tot), reproduces_stored_100k_run=bool(abs(s13["frac_never_within_1e100"] - conv[2]["never"]) < 1e-12))
json.dump(res, open(os.path.join(HERE, "verification.json"), "w", encoding="utf-8"), indent=1)
print("\nALL PASS" if all(v["passed"] for v in res.values()) else "\nSOME FAILED")
