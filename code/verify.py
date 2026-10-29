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
 V14 past-light-cone four-volume: Einstein-de Sitter analytic value, two independent quadratures, cutoff convergence, survival threshold
 V15 numerical error budget of the bounce action (integrator, tolerance, quadrature, bisection, truncation, boundary-value mesh and domain)
 V16 negative-mode structure: one negative eigenvalue (l = 0), l = 1 zero modes, none for l >= 2, no node, Derrick identity
 V17 scope of the reheating bound: matter-dilution envelope, curvature parameter, e-folds, AdS crunch, reservoir
 V18 matter-coupling estimate: internal consistency
 V19 finite temperature: exact rescaling of the thermal potential against direct O(3)/O(4) bounces, thin-wall limit of the O(3) action, critical thermal strength
 V20 time-dependent survival integral: light-cone quadrature, monotone cooling of the rate, de Sitter horizon temperature
 V21 thermal prior maps: the thermal channel only tightens survival (c >= 0) and closes the late-time decay window
 V22 interior simulation, single field: closed form reproduced, constraint residual of the gravitational scan, flat-like epoch only for g >~ 1e2
 V23 two-field reservoir: crunch time equals the AdS value, temperature at 1 s far below 1 MeV for every depth, envelope
 V24 hilltop flattening, repeated transitions (sum rule) and entropy
"""
import json
import os

import numpy as np

import cdl_lib as C
import constants as K
import rates as R
import interior_evolution as I
import monte_carlo as M
import survival as S

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
# V14
SV = json.load(open(os.path.join(HERE, "survival.json"), encoding="utf-8"))
eds = SV["check_EdS"]
v4_a, v4_b = S.v4_past_light_cone(), S.v4_past_light_cone(method="ode")
cut = [x["V4"] for x in SV["convergence"]["a_min"]]
grd = [x["V4"] for x in SV["convergence"]["grid"]]
ok14 = (abs(eds["numeric"] / eds["analytic"] - 1) < 1e-9 and abs(v4_a / v4_b - 1) < 1e-9 and abs(cut[-1] / cut[-2] - 1) < 1e-9 and abs(grd[-1] / grd[-2] - 1) < 1e-9
        and abs(SV["gamma_over_hl4_max_95"] * SV["V4_HL4"] - SV["N_95"]) < 1e-9 and abs(MC["survival"]["x_max"] - SV["gamma_over_hl4_max_95"]) < 1e-12)
rec("V14_past_light_cone", ok14, V4_HL4=SV["V4_HL4"], EdS_rel=float(abs(eds["numeric"] / eds["analytic"] - 1)), grid_vs_ode=float(abs(v4_a / v4_b - 1)), euclid_over_lorentzian=SV["ratio_euclid_to_lorentzian"])
# V15
CV = json.load(open(os.path.join(HERE, "convergence.json"), encoding="utf-8"))
dev, trunc = [], []
for eta15, r in CV["A_flat_shooting"].items():
    dev += [x["rel"] for x in r["rtol"] if x["rtol"] <= 1e-10] + [x["rel"] for x in r["method"] if x["rel"] is not None] + [x["rel"] for x in r["quadrature"]]
    dev += [x["rel"] for x in r["bisection"] if x["tol"] <= 1e-10]
    trunc.append([x["rel"] for x in r["truncation"]])
for eta15, r in CV["B_boundary_value"].items():
    dev += [x["rel"] for k in ("tol", "mesh", "domain_padding") for x in r[k]]
kap_rel = CV["C_gravity_coefficient"]["kappa_g0_spread"] / CV["C_gravity_coefficient"]["kappa_g0_mean"]
ok_trunc = all(t[-1] < 1e-5 and all(np.diff(t) < 0) for t in trunc)
rec("V15_error_budget", max(dev) < 3e-8 and kap_rel < 1e-4 and ok_trunc, worst_relative_deviation=float(max(dev)), truncation_at_largest_radius=[float(t[-1]) for t in trunc], kappa_g0=CV["C_gravity_coefficient"]["kappa_g0_mean"], kappa_spread_rel=float(kap_rel))
# V16
neg = CV["D_negative_mode"]
ok16 = all(v["n_negative_l0"] == 1 and v["n_negative_l1"] == 0 and v["n_negative_l2_to_4"] == 0 and abs(v["l1_lowest_over_mf2"]) < 1e-5
           and v["phi_prime_sign_changes"] == 0 and v["phi_minus_phi_f_nodes"] == 0 and v["virial_residual"] < 1e-6 for v in neg.values())
rec("V16_negative_mode", ok16, tilts=sorted(neg), l1_zero_mode_over_mf2=[float(neg[k]["l1_lowest_over_mf2"]) for k in sorted(neg)], virial=[float(neg[k]["virial_residual"]) for k in sorted(neg)])
# V17
SC = json.load(open(os.path.join(HERE, "scope.json"), encoding="utf-8"))
ok17 = (SC["envelope_check_max_ratio"] <= 1.0 + 1e-9 and max(e["N"] for e in SC["e_folds"]) < 1e-50 and SC["curvature"][0]["Y0_at_Vf"] < 1e-50
        and SC["needed_parent_for_bbn_at_1s"]["kappa_m_min"]["orders_above_Planck"] > 40 and SC["ads_for_1MeV"]["t_crunch_s"] < 10.0)
rec("V17_scope_of_reheating_bound", ok17, envelope_max_ratio=SC["envelope_check_max_ratio"], max_e_folds=max(e["N"] for e in SC["e_folds"]), Y0=SC["curvature"][0]["Y0_at_Vf"],
    orders_above_Planck_min_over_prior=SC["needed_parent_for_bbn_at_1s"]["kappa_m_min"]["orders_above_Planck"], crunch_time_for_1MeV_s=SC["ads_for_1MeV"]["t_crunch_s"])
# V18
CP = json.load(open(os.path.join(HERE, "coupling.json"), encoding="utf-8"))
ok18 = True
for row in CP["rows"]:
    for c in row["cases"]:
        ok18 &= abs(c["beta_linear"] * row["delta_phi_meV"] * 1e-12 / K.M_PL / c["delta_ln_m"] - 1) < 1e-9 and abs(c["delta_ln_m"] * CP["rho_m_over_Vf"] / c["dV_over_DV"] - 1) < 1e-9
rec("V18_coupling_estimate", ok18, rows=len(CP["rows"]))
# V19
TT = json.load(open(os.path.join(HERE, "thermal_tables.json"), encoding="utf-8"))
TRs = json.load(open(os.path.join(HERE, "thermal_results.json"), encoding="utf-8"))
red = TT["reduction_check"]
thin = [r["ratio"] for r in TT["s3_thin_wall"]]
ec = C.eta_crit()
tc_ok = True
for e19 in (0.1, 0.2, 0.3, 0.36):
    thc = 0.5 * (1.0 - (e19 / ec) ** (2.0 / 3.0))
    rts = np.sort(np.roots([1.0, 0.0, -(1.0 - 2.0 * thc), e19]).real)
    tc_ok &= abs(rts[2] - rts[1]) < 1e-3
rec("V19_finite_temperature_reduction", max(max(r["rel4"], r["rel3"]) for r in red) < 5e-4 and all(np.diff(thin) < 0) and thin[0] > 0.99 and tc_ok,
    max_rel_reduction=float(max(max(r["rel4"], r["rel3"]) for r in red)), thin_wall_ratio_O3=[round(x, 4) for x in thin], critical_theta_ok=bool(tc_ok))
# V20
import thermal as TH
z_, w_ = TH.light_cone_grid(1100.0, 600)
v4g = float(np.sum(w_) * K.OMEGA_L ** 2)
mono = True
for lab in ("c=0", "c=0.1"):
    lr = np.array([np.nan if x is None else x for x in TRs["rate_vs_z"]["ln_rate"][lab]])
    zz = np.array(TRs["rate_vs_z"]["z"])
    good = np.isfinite(lr)
    mono &= bool(np.all(np.diff(lr[good]) > 0))                    # the rate falls as the universe cools (z decreases)
q0 = TRs["rate_vs_z"]["ln_rate"]["decoupled"][0]
rec("V20_time_dependent_survival", abs(v4g / SV["V4_HL4"] - 1) < 1e-5 and mono and TRs["rate_vs_z"]["ln_rate"]["c=0"][0] > q0 and TRs["S3_over_T_dS"] > 1e30 and TRs["T_dS_over_T_cmb"] < 1e-29,
    V4_grid_over_survival=float(v4g / SV["V4_HL4"]), monotone_cooling=bool(mono), ln_rate_today_thermal=TRs["rate_vs_z"]["ln_rate"]["c=0"][0], ln_rate_zero_T=q0, S3_over_T_dS=TRs["S3_over_T_dS"])
# V21
pm = TRs["prior_maps"]
eqs = [v for k, v in pm.items() if isinstance(v, dict) and v.get("c") is not None and v["c"] >= 0]
tight = all(v["allowed_but_decoupled_excluded"] == 0.0 and v["excluded"] + v["inconsistent"] > pm["decoupled"]["excluded"] for v in eqs)
closed = all(v["decay_by_1e20"] == 0.0 for v in eqs)
rec("V21_thermal_prior_maps", tight and closed, scenarios=len(eqs), decoupled_excluded=pm["decoupled"]["excluded"], min_excluded_equilibrated=min(v["excluded"] + v["inconsistent"] for v in eqs))
# V22
SIMJ = json.load(open(os.path.join(HERE, "interior_sim.json"), encoding="utf-8"))
rowsA = SIMJ["A_single_field"]["rows"]
scanA = {r["log10_g"]: r for r in SIMJ["A_gravity_scan"]}
ok22 = (all(abs(r["ratio_sim_over_closed"] - 1) < 0.08 for r in rowsA) and max(r["max_residual"] for r in scanA.values() if r["log10_g"] >= -3) < 1e-7
        and not scanA[1]["flat_like_radiation_epoch"] and scanA[3]["flat_like_radiation_epoch"] and scanA[-60]["Y_max"] < 1e-60)
rec("V22_interior_simulation_single_field", ok22, ratios=[round(r["ratio_sim_over_closed"], 3) for r in rowsA], max_residual_g_ge_1e3=float(max(r["max_residual"] for r in scanA.values() if r["log10_g"] >= -3)),
    Y_max_at_g_1e2=scanA[2]["Y_max"])
# V23
rowsB = SIMJ["B_reservoir_ads"]["rows"]
Ts1 = [r["T_at_1s_GeV"] for r in rowsB]
ok23 = (all(abs(r["crunch_ratio"] - 1) < 1e-3 for r in rowsB) and max(Ts1) < 1e-20 and max(r["T_max_MeV"] for r in rowsB) < 1.0
        and max(r["max_residual"] for r in rowsB) < 1e-6 and max(abs(r["Tt_product_sim"] / rowsB[0]["Tt_product_sim"] - 1) for r in rowsB) < 1e-9)
rec("V23_two_field_reservoir_ads", ok23, crunch_ratio_range=[min(r["crunch_ratio"] for r in rowsB), max(r["crunch_ratio"] for r in rowsB)], T_at_1s_max_GeV=float(max(Ts1)), Tt=rowsB[0]["Tt_product_sim"])
# V24
rowsC = SIMJ["C_hilltop"]["rows"]
Nf = [r["N_flatness"] for r in rowsC]
ch = SIMJ["D_chain"]
ent = SIMJ["D_entropy"]
ok24 = (all(np.diff(Nf) > 0) and [r["w_over_M"] for r in rowsC if r["N_flatness"] >= 60] == [5.0, 6.0, 8.0] and Nf[3] > 55 and max(r["T_max_meV"] for r in rowsC) < 2.2 * 1.0
        and max(r["max_residual"] for r in rowsC) < 1e-7 and all(abs(c["total_released_over_rho"] - 1.0) < 1e-12 and c["T_max_total_meV"] < 2.2 for c in ch)
        and ent["ratio_to_bound"] < 1e-60 and ent["S_constant_after_reheating"])
rec("V24_hilltop_chain_entropy", ok24, N_flat=[round(x, 1) for x in Nf], T_hilltop_max_meV=max(r["T_max_meV"] for r in rowsC), entropy_ratio_to_bound=ent["ratio_to_bound"])
json.dump(res, open(os.path.join(HERE, "verification.json"), "w", encoding="utf-8"), indent=1)
print("\nALL PASS" if all(v["passed"] for v in res.values()) else "\nSOME FAILED")
