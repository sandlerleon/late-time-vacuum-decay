# -*- coding: utf-8 -*-
"""Supplementary information and cover letter for the EPJ C submission.   python build_supporting.py   (after build_paper.py)"""
import datetime
import json
import os
import platform

import numpy as np
import scipy
from docx import Document
from docx.shared import Inches, Pt

import constants as K
import render as RN


def tx(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "$%.*f\\times10^{%d}$" % (d, m, e)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "manuscript")
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
T, F, MC, IN, V = J("bounce_tables.json"), J("falsification_map.json"), J("monte_carlo.json"), J("interior_evolution.json"), J("verification.json")
SV, CV, SC, CP = J("survival.json"), J("convergence.json"), J("scope.json"), J("coupling.json")
TR, TT = J("thermal_results.json"), J("thermal_tables.json")
SM = J("interior_sim.json")
DVJ, SCJ, HCJ = J("dilution_validation.json"), J("turnaround_scan.json"), J("hot_cycles.json")
OSC, OSV = J("oscillation.json"), J("oscillation_validation.json")
CRJ, CSCN, CAUJ, CSTJ, BVJ, RLJ, PRJ = (J("continuous_cycles.json"), J("cycle_scan.json"), J("causal_controls.json"), J("contraction_stability.json"), J("bounce_validation.json"), J("compact_relics.json"), J("predictions.json"))
import cyc_blocks as CYB
assert all(v["passed"] for v in V.values())
GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
DOI_PAPER, DOI_CODE = "10.5281/zenodo.23717617", "10.5281/zenodo.23717619"
if os.path.exists(os.path.join(HERE, "dois.json")):
    _d = J("dois.json")
    DOI_PAPER, DOI_CODE = _d["paper"], _d["code"]
TITLE = ("Matter-Dependent Vacuum Interfaces and a Conditional Cyclic Hot Big Bang: Dilution-Triggered Vacuum Release, Bubble Nucleation and "
         "Nonsingular Cycles without Resets")
BASE = MC["baseline"]
A_, B_, G_, D_ = CV["A_flat_shooting"], CV["B_boundary_value"], CV["C_gravity_coefficient"], CV["D_negative_mode"]

# ------------------------------------------------------------------------------------------------------ supplement
DESC = {
    "V1_thin_wall_limit": "Flat-space action tends to the thin-wall value as the tilt vanishes",
    "V2_shooting_vs_bvp": "Shooting and boundary-value solutions agree",
    "V3_lambda_scaling": "Scaling B(lambda, eta) = b(eta)/lambda",
    "V4_de_sitter_action": "The de Sitter background action equals -24 pi^2 / (g^2 v_f)",
    "V5_flat_limit_of_gravity": "The gravitational bounce tends to the flat bounce; monotone suppression; tabulated slope",
    "V6_parke_thin_wall_suppression": "Gravitational suppression against the analytic thin-wall (Parke) suppression",
    "V7_hawking_moss_consistency": "No CDL bounce where |V''_top| < 4H^2, and B_CDL < B_HM wherever one exists",
    "V8_interpolation_off_grid": "Interpolation of b(eta) at four off-grid values of eta",
    "V9_critical_action": "The critical action solves Gamma/H^4 = 1",
    "V10_friedmann_constraint": "rho'^2 = 1 + (g rho^2/3)(x'^2/2 - v) along the numerical bounce",
    "V11_interior_closed_form": "Closed-form interior radiation density against the full numerical integration",
    "V12_constants": "Dark-energy density, bound on the reheating temperature, ln(M^4/V_f)",
    "V13_monte_carlo": "Regime fractions sum to one; the stored run is reproduced from its seed; convergence",
    "V14_past_light_cone": "Past-light-cone four-volume: Einstein-de Sitter analytic value, two quadratures, cutoff and grid convergence, survival threshold",
    "V15_error_budget": "Numerical error budget of the bounce action (integrator, tolerance, quadrature, bisection, truncation, boundary-value mesh and domain)",
    "V16_negative_mode": "One negative eigenvalue (l = 0), l = 1 zero modes, none for l >= 2, no node, Derrick identity",
    "V17_scope_of_reheating_bound": "Matter-dilution envelope, curvature parameter, e-folds, AdS crunch time, extra reservoir",
    "V18_coupling_estimate": "Internal consistency of the matter-coupling estimate",
    "V19_finite_temperature_reduction": "Exact rescaling of the thermal potential against direct O(3)/O(4) bounces; thin-wall limit of the O(3) action; critical thermal strength",
    "V20_time_dependent_survival": "Light-cone quadrature, monotone cooling of the rate, de Sitter horizon temperature",
    "V21_thermal_prior_maps": "Thermal channel only tightens survival (c >= 0) and closes the late-time decay window",
    "V22_interior_simulation_single_field": "Lorentzian interior: closed form reproduced, constraint residual in the gravity scan, flat-like epoch only for g >~ 1e2",
    "V23_two_field_reservoir_ads": "Two-field reservoir: crunch time equals the AdS value, T(1 s) far below 1 MeV for every depth",
    "V24_hilltop_chain_entropy": "Hilltop flattening needs a super-Planckian vev; repeated transitions obey the sum rule; entropy",
    "V25_option_B_cycles": "Option B: periodic radiation-only cycles, regular bounce, period scaling, constraint, not a sinusoid",
    "V26_option_B_validation": "Option B: integrator agreement, bounce continuity, hysteresis law, 1000-cycle stability, effect of a transition",
    "V27_matter_held_vacuum": "Matter-held vacuum: ten validation checks, LambdaCDM past for every coupled model, control rolls before today at mu = 300 H0",
    "V28_hot_cycles": "Hot cycles: self-similar bounce and release temperatures and held e-folds, growth = entropy^(1/3), constraint; both controls fail",
    "V29_continuous_cycles": "Continuous cycles without resets: >= 100 cycles each with dark matter regenerated, a matter era and a held epoch; growth = entropy^(1/3); the cold bounce fails",
    "V30_causal_test": "Causal test: the frozen-matter intervention never recollapses; the diluting model recollapses for every q > 0",
    "V31_stability_bounce_relics": "Shear scaling and ekpyrotic gain, bounce invariants and continuity, black-hole volume fraction below percolation",
    "V32_cycle_fixed_point": "The cycle as an attracting fixed point of the bounce-to-bounce map, closed-form held epoch, results unchanged by the solver tolerance, the overdamped threshold and the projection frequency",
    "V33_vacuum_phase": "Phase structure: the symmetron release is spinodal (occupied minimum never metastable); tunneling-branch degeneracy at eta/2; coherence of the release; exact vs logarithmic coupling",
    "V34_wall_transport_gravity": "Wall tension and thin-wall limit; transmission validated against Poeschl-Teller and unitarity; walls run away and filter a negligible dark-matter fraction; junction flat limit, wall energy = deficit",
    "V35_daughter_cycles": "Daughter regions: constraint, bounce at rho_c and continuous H at every event; entropy non-decreasing; self-similar cycles; C = B, C* fails when cold, D window narrow; all interface checks pass",
    "V36_handoff_viability": "Hand-off: percolation completes within a small fraction of a Hubble time; the cycle continues for w = 1/3 and w = 0 collision products and for every bounce variant; isolated bubbles are trapped before their bounce; independent integration agrees; viability fractions computed",
}
S = []
add = S.append
add(("title", "Supplementary Information: " + TITLE))
add(("author", "Leon Sandler\nIndependent Researcher, Northbrook, Illinois, USA"))
add(("h1", "S1. Status"))
add(("p", "This supplement documents the numerical methods, the error budget, the executable verification suite, the survival calculation, the Monte Carlo specification, the closed-form interior solution, "
          "the scope and coupling estimates, and the reproduction commands for the paper. All numbers quoted in the paper are read from the JSON files produced by the scripts described here. The dimensionless "
          "toy Monte Carlo of an earlier draft, and the Euclidean four-sphere survival estimate that it used, are superseded and are not part of the evidence."))
add(("h1", "S2. Numerical methods"))
add(("h2", "S2.1 Potential and scaling"))
add(("p", "The potential is $V=V_t+f^4\\lambda[(x^2-1)^2/8+\\eta x/2-p_{\\min}]$ with $x=\\varphi/f$. The stationary points solve $x^3-x+\\eta=0$, which has three real roots "
          "$x_t<x_{\\rm top}<x_f$ for $\\eta<\\eta_c=2/(3\\sqrt3)$. In units of $f$ for fields and $(\\sqrt\\lambda f)^{-1}$ for lengths the flat-space action is $B=b(\\eta)/\\lambda$, "
          "verified numerically to a relative $10^{-12}$ (check V3). With gravity the only additional parameter is $g=f^2/M_{\\rm Pl}^2$."))
add(("h2", "S2.2 Flat-space bounce"))
add(("p", "The equation $x''+3x'/r=v'(x)$ is integrated from a small radius ($10^{-6}$) with the series start $x=x_0+v'(x_0)r^2/8$ using an eighth-order Runge--Kutta method "
          "(DOP853) with relative tolerance $10^{-11}$ and absolute tolerance $10^{-13}$. A trajectory is an overshoot if $x$ crosses $x_f$ with positive velocity and an undershoot if $x'$ changes sign first. "
          "The central value $x_0$ is bracketed between the undershoot limit $x_s$ (the point on the true-vacuum side with $v(x_s)=v_f$) and a point $10^{-9}$ from $x_t$, and "
          "bisected to $10^{-13}$. The action is the trapezoid sum of $2\\pi^2r^3[x'^2/2+v-v_f]$ over $2\\times10^5$ points up to the radius at which the trajectory leaves. In the thin-wall regime ($\\eta<0.1$) the sensitivity of the shooting to $x_0$ is exponential in the wall radius, so the bounce is instead computed by "
          "collocation on the boundary-value problem with a tanh initial guess (SciPy solve_bvp, tolerance $10^{-9}$, 4000 initial nodes, domain padding 30). Where both methods apply ($0.10\\le\\eta\\le0.17$) they agree to "
          "%s (check V2)." % tx(V["V2_shooting_vs_bvp"]["max_relative_difference"])))
add(("h2", "S2.3 Bounce with gravity"))
add(("p", "The four-dimensional system $(x,x',\\rho,\\rho')$ is integrated in the arclength $\\xi$ with $\\rho''=-(g\\rho/3)(x'^2+v)$ and the same overshoot/undershoot "
          "classification; the constraint $\\rho'^2=1+(g\\rho^2/3)(x'^2/2-v)$ is monitored and holds to %s (check V10). The action is "
          "$B=4\\pi^2\\int(\\rho^3v-3\\rho/g)d\\xi-S_{\\rm dS}$. The part of the bounce outside the wall follows the false-vacuum geometry $\\rho=\\sin(H\\eta)/H$, where $\\eta$ is the "
          "arclength from the second pole; the tail integral and the de Sitter background are evaluated analytically, "
          "$\\int_0^{a/H}(v_f\\sin^3H\\eta/H^3-3\\sin H\\eta/(gH))d\\eta=[v_f(2/3-\\cos a+\\cos^3a/3)/H^4-3(1-\\cos a)/(gH^2)]$, so the terms of order $1/g^2$ cancel exactly and the action keeps a precision of "
          "about $10^{-4}$ in $B/B_{\\rm flat}$ down to $g=10^{-4}$. The background action reproduces $-24\\pi^2/(g^2v_f)$ to machine precision (check V4). "
          "At the physical $g\\sim10^{-60}$ no gravitational bounce is computed: the correction is the linear response $-\\kappa g$ with $\\kappa(g\\to0)=%.4f$ from $g=10^{-3}$ and $2\\times10^{-3}$ and linear extrapolation."
          % (tx(V["V10_friedmann_constraint"]["max_residual"]), G_["kappa_g0_mean"])))
add(("h2", "S2.4 Interpolation and rates"))
add(("p", "$\\ln b(\\eta)$ is tabulated at %d values of $\\eta$ between 0.03 and 0.38 and interpolated with a monotone cubic Hermite spline (PCHIP); the interpolation error at four off-grid "
          "tilts is below %s (check V8). The rate, the critical action and the decay time follow Eq. (7)--(8) of the paper and are evaluated in the logarithmic domain because "
          "$\\Gamma/H^4$ spans several hundred orders of magnitude." % (len(T["flat"]), tx(V["V8_interpolation_off_grid"]["max_relative_error"]))))
add(("h1", "S3. Error budget and negative modes"))
add(("p", "convergence.py varies each numerical choice and records the relative change of $B$ against the production setting. Table S1 gives the flat-space shooting results for three tilts, Table S2 the "
          "boundary-value solver, and Table S3 the gravity coefficient. The negative-mode analysis is in Table S4."))
rows = []
for eta, r in A_.items():
    for k, lab, key in (("rtol", "integrator tolerance", "rtol"), ("method", "integrator", "method"), ("quadrature", "quadrature points", "n_quad"), ("bisection", "bisection tolerance", "tol")):
        for x in r[k]:
            rows.append(["%s" % eta, lab, str(x[key]), ("%.1e" % x["rel"]) if x["rel"] is not None else "n/a"])
add(("table", "Table S1. Flat-space shooting: relative change of $B$ ($\\lambda=1$; production: DOP853, $10^{-11}$, $2\\times10^5$ points, bisection $10^{-13}$).", ["$\\eta$", "Setting varied", "Value", "Relative change"], rows, None))
rows = []
for eta, r in B_.items():
    for k, lab, key in (("tol", "solver tolerance", "tol"), ("mesh", "initial nodes", "n_init"), ("domain_padding", "domain padding", "r_pad")):
        for x in r[k]:
            rows.append([eta, lab, str(x[key]), "%.1e" % x["rel"]])
add(("table", "Table S2. Boundary-value solution: relative change of $B$ (reference: tolerance $10^{-10}$, 8000 nodes, padding 60).", ["$\\eta$", "Setting varied", "Value", "Relative change"], rows, None))
rows = [[x["setting"], "%.4f" % x["kappa_1em3"], "%.4f" % x["kappa_2em3"], "%.4f" % x["kappa_g0"]] for x in G_["settings"]]
add(("table", "Table S3. Gravity coefficient at $\\eta=0.30$, $B/B_{\\rm flat}=1-\\kappa g$: $\\kappa$ at $g=10^{-3}$, $2\\times10^{-3}$ and the linear extrapolation $\\kappa(g\\to0)=2\\kappa_1-\\kappa_2$.", ["Setting", "$\\kappa(10^{-3})$", "$\\kappa(2\\times10^{-3})$", "$\\kappa(g\\to0)$"], rows, None))
rows = []
for eta, r in D_.items():
    rows.append([eta, "%.4g" % r["l0_lowest_over_mf2"], str(r["n_negative_l0"]), "%.1e" % r["l1_lowest_over_mf2"], str(r["n_negative_l2_to_4"]), str(r["phi_minus_phi_f_nodes"]), "%.1e" % r["virial_residual"]])
add(("table", "Table S4. Spectrum of $-\\nabla_4^2+V''(\\varphi_b)$ on the bounce: lowest $l=0$ eigenvalue over $m_f^2$, number of negative $l=0$ eigenvalues, lowest $l=1$ eigenvalue over $m_f^2$ (translational zero modes), negative eigenvalues for $l=2$--4, nodes of $\\varphi-\\varphi_f$, Derrick residual.",
     ["$\\eta$", "$\\lambda_0/m_f^2$", "neg. $l=0$", "$\\lambda_1/m_f^2$", "neg. $l\\ge2$", "nodes", "Derrick"], rows, None))
add(("h1", "S4. Executable verification suite"))
add(("p", "verify.py recomputes the following checks from the stored tables and by direct calls to the solvers. The manuscript builder refuses to run unless every one passes."))
rows = []
for k, v in V.items():
    det = {kk: (("%.3g" % vv) if isinstance(vv, float) else str(vv)) for kk, vv in v.items() if kk != "passed"}
    rows.append([k.split("_")[0], DESC.get(k, k), "pass" if v["passed"] else "FAIL", "; ".join("%s = %s" % (a, b) for a, b in list(det.items())[:3])])
add(("table", "Table S5. Verification checks and their outcomes.", ["No.", "What is checked", "Outcome", "Key figures"], rows, None))
add(("h1", "S5. Survival on the past light cone"))
add(("p", "The probability that no bubble has nucleated in the past light cone of the present observer is $\\exp(-\\Gamma V_4)$ with $V_4=(4\\pi/3)\\int_0^{t_0}a^3\\chi^3dt$ and $\\chi(t)=\\int_t^{t_0}dt'/a$. "
          "With $s=-\\ln a$ and $E(a)=H/H_0$ the system $d\\chi/ds=1/(aE)$, $dV_4/ds=(4\\pi/3)a^3\\chi^3/E$ is integrated in two ways (cumulative Simpson rule with the trapezoid rule on $4\\times10^4$ points, and DOP853 with relative tolerance $10^{-12}$), "
          "from $a=1$ to $a=10^{-12}$. Planck 2018: $H_0=%.2f$, $\\Omega_\\Lambda=%.4f$, $\\Omega_r=%s$ ($T_{\\rm CMB}=2.7255$ K, $N_{\\rm eff}=3.046$), $\\Omega_m=%.4f$. Result: $V_4=%.5f\\,H_0^{-4}=%.5f\\,H_\\Lambda^{-4}$ with $H_\\Lambda^2=\\Omega_\\Lambda H_0^2$. "
          "Checks: Einstein--de Sitter analytic value $(4\\pi/3)\\,81\\,B(9,4)\\,t_0^4$ (relative difference %s), grid refinement from $2\\times10^3$ to $1.6\\times10^5$ points ($<10^{-8}$ already at $10^4$), cutoff from $a=10^{-6}$ to $10^{-15}$ ($<10^{-13}$). "
          "Variation with $\\Omega_\\Lambda\\pm0.0056$: $V_4=%.4f$ to $%.4f\\,H_\\Lambda^{-4}$. The Euclidean volume is $8\\pi^2/3=%.2f$. The 95%% exclusion is $\\Gamma V_4>-\\ln0.05=%.3f$, i.e. $\\Gamma/H_\\Lambda^4>%.2f$, which "
          "corresponds to a mean decay time $t_{\\rm dec}\\gtrsim%s$ yr."
          % (K.H0_KMS_MPC, K.OMEGA_L, tx(SV["omega_r"], 2), SV["omega_m"], SV["V4_over_H0m4"], SV["V4_HL4"], tx(abs(SV["check_EdS"]["numeric"] / SV["check_EdS"]["analytic"] - 1), 1),
             SV["parameter_variation"][0]["V4_HL4"], SV["parameter_variation"][2]["V4_HL4"], SV["euclid_HL4"], SV["N_95"], SV["gamma_over_hl4_max_95"], tx(SV["t_dec_min_yr"], 1))))
add(("h2", "S5.1 Finite temperature: thermal rescaling, the O(3) action and the time-dependent rate"))
add(("p", "Thermally corrected potential $V_{\\rm eff}=V_0+cT^2\\varphi^2/2$, $\\theta=cT^2/(\\lambda f^2)$. With $x=sy$, $s=\\sqrt{1-2\\theta}$, $(x^2-1)^2/8+\\eta x/2+\\theta x^2/2=s^4[(y^2-1)^2/8+(\\eta/s^3)y/2]$+const, so "
          "the family is closed under the thermal term with $\\eta_{\\rm eff}=\\eta/s^3$. For a potential $V=A\\,w(\\varphi/F)$ the O(4) action is $(F^4/A)\\,b(\\eta)$ and the O(3) action is $(F^3/\\sqrt A)\\,s_3(\\eta)$; with $F=fs$, $A=\\lambda f^4s^4$ "
          "this gives $B_4=b(\\eta_{\\rm eff})/\\lambda$ and $S_3=(fs/\\sqrt\\lambda)s_3(\\eta_{\\rm eff})$. $s_3(\\eta)=4\\pi\\int\\rho^2[x'^2/2+w-w_f]d\\rho$ is tabulated by thermal.py (shooting for $\\eta\\ge0.1$, collocation below; "
          "thin-wall check $16\\pi\\sigma^3/3\\Delta V^2$, ratio %.4f at $\\eta=0.03$). Outside the table, $b\\propto\\eta^{-3}$ and $s_3\\propto\\eta^{-2}$ (thin-wall power laws) are used for $\\eta_{\\rm eff}<0.03$ (negative $c$), and $b(0.38)$, $s_3(0.38)$ "
          "are held fixed between $0.38$ and $\\eta_c$. Validation (check V19): direct bounces with the thermal term in the potential agree with the scalings to %.1e (Table S6). The dominant saddle is the O(3) one if $S_3/T<B_4$; its prefactor is "
          "$D_TT^4(S_3/2\\pi T)^{3/2}$ with $\\ln D_T\\sim N(0,(1.5\\ln10)^2)$. The survival integral $\\ln I=\\ln\\sum_i\\Gamma(T_i)w_i$ uses 300--600 nodes in $s=-\\ln a$ with the weights of Section S5 (their sum reproduces $V_4$ to %.1e). "
          "thermal_analysis.py draws $2\\times10^5$ parameter sets with the seed and priors of Section S6 (plus $\\ln D_T$), and classifies each as above the survival bound ($\\ln I>\\ln3.0$), inconsistent (no false minimum at some redshift of the "
          "equilibrium epoch) or allowed; the future decay time is the zero-temperature O(4) value because the thermal contribution switches off as $T\\propto e^{-H_\\Lambda t}$."
          % (TT["s3_thin_wall"][0]["ratio"], max(max(r["rel4"], r["rel3"]) for r in TT["reduction_check"]), abs(TR["light_cone_check"]["V4_HL4_grid"] / TR["light_cone_check"]["V4_HL4_survival_json"] - 1))))
rows = [["%.2f" % r["eta"], "%.2f" % r["theta"], "%.4f" % r["eta_eff"], "%.4f" % r["B4_direct"], "%.4f" % r["B4_scaling"], "%.4f" % r["S3_direct"], "%.4f" % r["S3_scaling"]] for r in TT["reduction_check"]]
add(("table", "Table S6. Direct bounce with the thermal term in the potential against the exact rescaling ($\\lambda=1$): $B_4$ and $S_3$.", ["$\\eta$", "$\\theta$", "$\\eta_{\\rm eff}$", "$B_4$ direct", "$B_4$ scaling", "$S_3$ direct", "$S_3$ scaling"], rows, None))
rows = [["%.3f" % r["eta"], "%.4g" % r["s3"], "%.4g" % r["b"]] for r in TT["s3_table"] if r["eta"] in (0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.38)]
add(("table", "Table S7. The O(3) action $s_3(\\eta)$ and the O(4) action $b(\\eta)$ at $\\lambda=1$, $\\theta=0$ (selected tilts; the full table is in thermal_tables.json).", ["$\\eta$", "$s_3$", "$b$"], rows, None))
add(("h2", "S5.2 Lorentzian simulations of the daughter region (Steps 6--7)"))
add(("p", "interior_sim.py integrates $(a,\\dot a,x,\\dot x,y,\\dot y,\\rho_r)$ with DOP853 (relative tolerance $10^{-11}$, absolute $10^{-14}$) from $t_0=10^{-4}$ with the series start $a=t_0$, $\\dot a=1$, $x=x_0-V_x t_0^2/8$, $\\dot x=-V_xt_0/4$ "
          "($x_0$ the centre of the flat-space bounce, from bounce_tables.json). The constraint residual is $(\\dot a^2-1-ga^2\\rho/3)/(1+\\dot a^2)$. Part A: $\\lambda=1$, $\\eta=0.30$, $V_t=0$, dissipation $\\Upsilon=\\gamma$ with $\\gamma/m=1,10^{-1},10^{-2},10^{-3}$, "
          "and a scan of $g$ from $10^{-60}$ to $10^3$ (Table S8). Part B: $V=V_\\varphi+\\tfrac{\\lambda_y}{4}(y^2-w_y^2)^2+\\tfrac12g_c^2(x-x_t)^2y^2+V_{\\rm off}$ with $\\lambda_y=1$, $w_y=1.5$, $g_c^2=1.6\\lambda_yw_y^2/(x_f-x_t)^2$, $V_{\\rm off}=-(\\Delta V+\\lambda_yw_y^4/4)$ "
          "(false-vacuum energy zero in these units), $y(t_0)=10^{-6}$, $\\Upsilon_x=\\Upsilon_y=m/2$, run for $\\kappa=|V_t|/\\rho_\\Lambda=1,10^5,\\dots,10^{35}$, which sets $f=(\\kappa\\rho_\\Lambda/|V_t^{\\rm sim}|)^{1/4}$ and $g=f^2/M_{\\rm Pl}^2$. After the fields have decayed the evolution is the exact solution of "
          "$\\dot a^2=1-ba^2+c/a^2$ with $b=g|V_t|/3$, $c=g\\rho_ra^4/3$: with $s=a^2$ the time is $t=[\\arcsin((2s-s_++s_-)/(s_++s_-))]/(2\\sqrt b)$ between turning points $s_\\pm$, and the crunch time follows from the half-period. "
          "Part C: $H_\\Lambda=M_{\\rm Pl}=1$, $V=\\tfrac{\\lambda_c}{4}(\\chi^2-w^2)^2$ with $\\lambda_cw^4/4=3$, $\\chi(t_0)=10^{-5}w$, $\\Upsilon=0.7\\sqrt{2\\lambda_c}\\,w$, $t_{\\rm end}=4000$. Part D: the chain uses the rate of Eq. (7) with the parent energy of each generation "
          "($\\ln(M_{\\rm Pl}^4/V_{\\rm parent})$, $r_L=V_{\\rm parent}/\\Delta V$, $H_2^2=\\Delta V_2/3M_{\\rm Pl}^2$); the entropy per comoving curvature volume is $(2\\pi^2/45)g_*(T/f)^3a^3$ and the bound for the wall volume is $8\\pi^2M_{\\rm Pl}^2R_b^2$."))
rows = [["%.0e" % r["gamma_over_m"], "%.4f" % r["T_max_meV"], "%.4f" % r["T_max_closed_meV"], "%.3f" % r["ratio_sim_over_closed"], "%.2f" % r["t_peak"], "%.2f" % r["envelope_ratio"], "%.2f" % r["T_tau_final"]] for r in SM["A_single_field"]["rows"]]
add(("table", "Table S8. Single field (benchmark): maximum temperature in the simulation and in the closed form, their ratio, time of the maximum (units of $1/f$), the envelope ratio $\\max\\rho(mt)^3/E_{\\rm rel}$ and the product $Tt$ at the end.",
     ["$\\gamma/m$", "$T_{\\max}$ sim (meV)", "closed form (meV)", "ratio", "$t_{\\rm peak}$", "envelope", "$Tt$"], rows, None))
rows = [["%d" % r["log10_g"], "%.2e" % r["Y_max"], "%s" % r["flat_like_radiation_epoch"], "%.1e" % r["max_residual"]] for r in SM["A_gravity_scan"]]
add(("table", "Table S9. Gravity scan of the single-field interior: maximum curvature parameter $Y=ga^2\\rho/3$, whether a flat-like radiation epoch occurs, and the constraint residual.", ["$\\log_{10}g$", "$Y_{\\max}$", "flat-like epoch", "residual"], rows, None))
rows = [["%.1f" % r["w_y"], "%.2f" % r["E_res"], "%.2f" % r["T_max_over_f"], "%.2f" % r["Tt_product_sim"]] for r in SM["B_reservoir_scan"]]
add(("table", "Table S10. Reservoir size: stored energy $E_\\chi=\\lambda_yw_y^4/4$ (units $\\lambda f^4$), maximum temperature over $f$, and $Tt$ at the end of the run.", ["$w_y$", "$E_\\chi$", "$T_{\\max}/f$", "$Tt$"], rows, None))
add(("h2", "S5.3 Option B: oscillating cycles"))
add(("p", "oscillation.py and oscillation_validation.py integrate $(a,H,\\varphi,\\dot\\varphi,\\rho_r,W,W_\\varphi)$ in Planck units with $\\dot H=-(\\rho+p)(1-(n+1)(\\rho/\\rho_c)^n)/2+k/a^2$ ($n=1$ loop quantum cosmology, $n=2$ check), $k=-1$, "
          "$V=V_t+m^2\\varphi^2/2$, dissipation $\\Upsilon$, and $W=\\int p\\,d(a^3)$, $W_\\varphi=\\int p_\\varphi\\,d(a^3)$. Initial data at a turnaround ($H=0$) with the radiation density from the constraint; baseline $V_t=-10^{-6}$, $\\rho_c=1$, $m=0.05$, $a_{\\max}=3000$ "
          "(oscillation_config.json). Turnarounds are zeros of $H$ with $\\rho\\ll\\rho_c$, bounces zeros with $\\rho\\sim\\rho_c$ (a scalar-dominated bounce can make $H$ cross zero more than once near $\\rho_c$; those crossings are not cycle events). "
          "Per-cycle results for the phase-diagram grid are in oscillation_cycles.csv. Table S11 lists the integrator comparison."))
rows = [[k, "%.4f" % v["multiplier"], "%.1e" % v["multiplier_rel_dev"], "%.1e" % v["a_max_rel_dev"]] for k, v in OSV["validation_integrators"].items()]
add(("table", "Table S11. Option B: scalar energy multiplier per cycle (linear regime) and the turnaround radii of a nonlinear run for each integrator setting, with their deviation from the reference (DOP853, $10^{-11}$, $P/400$).",
     ["Integrator", "multiplier", "deviation", "$a_{\\max}$ deviation"], rows, None))
rows = [["%.2f" % g["m"], "%.1f" % g["upsilon_P"], "%.0e" % g["phi0"], str(g["cycles"]), g["cls"]] for g in OSV["phase_diagram"]]
add(("table", "Table S12. Option B phase diagram: classification after 100 or more cycles (runs stop when $a$ exceeds $10^{30}$).", ["$m$", "$\\Upsilon P$", "$\\varphi_0$", "cycles", "class"], rows, None))
add(("h2", "S5.4 The matter-held vacuum and the hot cycles"))
add(("p", "matter_coupling.py defines $V_0$, $A$, $\\rho_{\\rm crit}$ and $a_c$; dilution_background.py integrates the coupled background (flat, general relativity) from $z=1100$ with "
          "the field at $10^{-60}v$, with the absolute tolerance of the field set relative to its seed (an absolute tolerance on the background scale would swamp the seed); "
          "turnaround_scan.py runs the grid of Table 16 for the coupled model and the control; dilution_thermal.py the thermal branch; density_bounce.py the tunneling branch; "
          "cyclic_extension.py the cold cycles; hot_cycles.py the hot cycles with thermal dissipation, freeze-out and hot-phase restoration, chunked integration (steps bounded by the "
          "oscillation period while the field is heavy) and constraint projection at every doubling of the scale factor; dilution_validation.py runs the checks of Table S13."))
rows = [[k.split("_")[0], k.split("_", 1)[1].replace("_", " "), "pass" if v["passed"] else "FAIL"] for k, v in DVJ.items()]
add(("table", "Table S13. Validation of the dilution-trigger code (dilution_validation.py).", ["Check", "Content", "Result"], rows, None))
rows = [["%g" % r["a_c"], "%g" % r["mu"], "%g" % r["q"], "coupled" if r["coupled"] else "control", r["cls"], ("%.3g" % r["a_turn"]) if r["a_turn"] else "--", "%.0e" % r["max_residual"]]
        for r in SCJ["rows"] if r["q"] in (-0.5, 1.0) and (r["coupled"] or r["a_c"] == 2.0)]
add(("table", "Table S14. Release and turnaround over the grid ($q=-0.5$ and 1; the control does not depend on $a_c$). The full grid, including $q=0.1$ and 10, is in turnaround_scan.csv.",
     ["$a_c$", "$\\mu/H_0$", "$q$", "model", "outcome", "turnaround $a$", "residual"], rows, None))
rows = []
for r in HCJ["runs"]:
    pp = r["params"]
    lab = "a_c=%g mu=%g q=%g rho_c=1e%d C_T=%g%s" % (pp["a_c"], pp["mu"], pp["q"], round(np.log10(pp["rho_c_over_Vc"])), pp["c_T"], "" if pp["reset"] else " no reset")
    for c in r["cycles"][:4]:
        rows.append([lab if c["n"] == 0 else "", str(c["n"]), "%.4g" % c.get("T_bounce", 0), "%.3f" % c.get("max_matter_fraction", 0), "%.3f" % c.get("held_accelerated_efolds", 0),
                     ("%.4g" % c["T_release"]) if c.get("T_release") else "--", ("%.2f" % c["G"]) if c.get("G") and pp["reset"] and pp["c_T"] > 0 else "--"])
add(("table", "Table S15. Hot cycles, first four cycles of every run (cycle 0 starts today). Full per-cycle data in hot_cycles.csv.",
     ["Run", "cycle", "$T_{\\rm bounce}$", "peak matter", "held e-folds", "$T_{\\rm rel}$", "$G_n$"], rows, None))
add(("h2", "S5.5 Continuous cycles, controls, stability, bounce and relics"))
add(("p", "continuous_cycles.py integrates the cycle of Section 6.8 with state (ln a, H, phi, phi', T, N, integral of rho_r dt, pressure-volume work), in chunks of one doubling or halving "
          "of the scale factor with the clock restarted in each chunk, absolute tolerances relative to each variable, DOP853 first with LSODA and Radau as fall-backs, the overdamped "
          "reduction where Upsilon > 1e3 max(m_eff, H, mu) (%d chunks of the baseline run), and constraint projection of H away from the turning points (largest correction %s). "
          "cycle_scan.py maps the outcome over dark-matter mass and bounce density; causal_controls.py, contraction_stability.py, bounce_validation.py, compact_relics.py and "
          "predictions.py give Sections 6.9-6.14. numerics_admissibility.py repeats three baseline cycles with rtol 1e-10, an overdamped threshold of 1e4, projection at "
          "every fourth chunk, and no projection; the first three agree with the baseline within the scatter between cycles, while the run without projection fails "
          "(constraint residual growing without bound, evaluation budget exhausted), which is why projection is kept." % (CRJ.get("overdamped_chunks", 0), tx(CRJ["projections"]["max_rel"], 0))))
rows = []
for c in CRJ["cycles"]:
    if c["n"] < 1 or "held_accelerated_efolds" not in c or (c["n"] > 20 and c["n"] % 10):
        continue
    rows.append([str(c["n"]), "%.4f" % c.get("phi_bounce_over_v", 0), "%.3f" % c.get("Y_fo_over_kappa", 0), "%.3f" % c.get("Y_held_over_kappa", 0), "%.3f" % c.get("max_matter_fraction", 0),
                 "%.3f" % c["held_accelerated_efolds"], "%.4f" % c["lnG"] if c.get("lnG") else "--", "%.4f" % c["lnG_entropy"] if c.get("lnG_entropy") else "--"])
add(("table", "Table S16. Continuous cycles (baseline), cycles 1-20 and every tenth cycle thereafter; all cycles in continuous_cycles.csv.",
     ["cycle", "phi_b/v (end)", "Y_fo/kappa", "Y/kappa held", "peak matter", "held e-folds", "ln G", "(1/3) ln S ratio"], rows, None))
rows = [[CYB.lab(r["params"]), r["classification"], ", ".join("%.2f" % x for x in (r.get("held") or [])[:5] if x is not None),
         ", ".join("%.2g" % x for x in (r.get("Y") or [])[:5] if x is not None)] for r in sorted(CSCN["rows"], key=lambda r: str(r["params"]))]
add(("table", "Table S17. Outcome of five continuous cycles per run (cycle_scan.py): held e-folds and Y/kappa of cycles 1-5.", ["Run", "Outcome", "held e-folds", "Y/kappa"], rows, None))
rows = [["%g" % r["f0"], "%g" % r["rho_end_over_rho_c"], tx(max(r["s_max"], 1e-300), 1)] for r in CSTJ["onset_exit"]]
add(("table", "Table S18. Ekpyrotic phase (c = 5): largest controlled initial anisotropy against the ekpyrotic fraction at the turnaround and the exit density.",
     ["f0", "rho_end/rho_c", "s_max"], rows, None))
rows = [[k, "%.3g" % v["T_MeV"]] for k, v in RLJ["percolation"].items()] + [["bounce", "%.3g" % RLJ["T_bounce_MeV"]]]
add(("table", "Table S19. Percolation temperature of black-hole populations in the contraction (MeV), including the entropy growth of the cycle, and the bounce temperature of the baseline.",
     ["Population", "T (MeV)"], rows, None))
add(("h2", "S5.6 Density-dependent vacuum interfaces (Section 7)"))
import int_blocks as INT
add(("p", "vacuum_phase_map.py maps the stationary points of both matter-coupled potentials over density and tilt and evaluates the coherence of the symmetron release "
          "(the restored field value per cycle from cycle_map.json against $H_\\Lambda/2\\pi$; linear $\\sigma_R$ from a BBKS transfer function normalized to $\\sigma_8$). "
          "vacuum_wall.py solves the O(4) bounce of the tunneling branch with its profile (shooting; collocation where the central value is within $10^{-5}$ of the true "
          "minimum), the thin-wall tension with the energy difference removed, the static wall at the degeneracy density, and the nucleation history with the actual "
          "Hubble rate. interface_transport.py integrates the mode equation across the planar wall with complex amplitudes (DOP853, rtol $10^{-10}$) and computes the "
          "reflected momentum flux of a Bose bath with the transverse integral in closed form. bubble_gravity.py evaluates the Israel junction and evolves daughter regions "
          "by root finding in $\\ln a$ with $H$ from the constraint and proper time by quadrature (analytic end pieces at turning points). bubble_to_cycle.py runs the "
          "controls and the viability map; bubble_entropy.py the energy and entropy closure; interface_validation.py the %d checks of Table 21."
          % INT.VJ["n"]))
rows = [["%.2f" % r["gh"], "%.2f" % r["T"], INT.texsci(max(r["P_static_over_dV"], 1e-300), 1).join("$$"), INT.texsci(max(r["P_BM_over_dV"], 1e-300), 1).join("$$")] for r in INT.TJ["pressure"]]
add(("table", "Table S20. Pressure of a thermal test-scalar bath on the wall at the trigger relative to the driving pressure: static wall (computed reflection) and "
              "ultra-relativistic limit $\\Delta m^2T^2/24$. Temperatures in units of $\\sqrt\\lambda f$ (4.4 meV); the CMB at the trigger is %.3f." % INT.TJ["T_cmb_at_trigger_units"],
     ["$g/\\sqrt\\lambda$", "T", "static", "relativistic limit"], rows, None))
add(("p", "bubble_handoff.py computes the false-vacuum fraction with the Guth--Tye integral on the baseline rate history, the bounds of Table 20, the "
          "isolated-bubble test (the wall as a null ray in the open interior, the exterior false-vacuum universe at the same cosmic time; the crossing times "
          "are insensitive to that identification because the exterior density changes by less than 1%% over the interval), the bounce and collision-product "
          "variants, and the independent initial-value integration. viability_mc.py samples the stated ranges (200,000 draws for the held vacuum, %d for the "
          "tunneling branch, whose past-light-cone integral is evaluated for every draw)." % INT.VM["tunneling"]["n"]))
add(("table", "Table S21. Validation of the interface calculations (interface_validation.py).", ["Check", "Result"], INT.validation_rows(), None))
add(("h1", "S6. Monte Carlo specification"))
add(("p", "Draws: $N=%d$, NumPy PCG64 generator, seed %d. $\\log_{10}\\lambda\\sim U(-2,\\log_{10}4\\pi)$; $\\eta\\sim U(0.03,0.38)$; $\\ln D\\sim N(0,(1.5\\ln10)^2)$; $r_L=1$. Each draw is converted to "
          "$\\ln\\Gamma/H^4$ with Eq. (7), to a decay time, and classified: above the survival bound if $\\Gamma/H_\\Lambda^4\\ge%.2f$ (Section S5); otherwise decaying by $10^{12}$, $10^{20}$, $10^{50}$ or $10^{100}$ yr, or "
          "showing no decay within $10^{100}$ yr. Standard errors are batch-means over 20 batches. Convergence: the fractions are recomputed for $10^3$, $10^4$, $10^5$ and $10^6$ independent draws (Fig. 5c of the paper). "
          "Sensitivity runs change one ingredient at a time: the prefactor spread (0, 3 and 6 decades), $r_L\\in[1,10]$, $\\lambda\\le1$, $\\eta\\ge0.10$, and $\\log_{10}\\lambda\\in[-4,2]$ (Table 5 of the paper). "
          "Percolating-and-surviving draws are those with $x_{\\rm perc}\\le\\Gamma/H^4<%.1f$, for $x_{\\rm perc}=0.1$, 1 and 10 (%.3f%%, %.3f%%, %.3f%% of the prior). The first-draft Euclidean criterion would have excluded %.2f%% instead of %.2f%%. "
          "The regimes `above the survival bound' and `no decay within $10^{100}$ yr' are the two failure regions of the reset hypothesis. These fractions describe the sample space of the stated priors and are not probabilities for our universe."
          % (MC["N"], MC["seed"], MC["survival"]["x_max"], MC["survival"]["x_max"], 100 * BASE["frac_percolating_and_surviving"]["0.1"], 100 * BASE["frac_percolating_and_surviving"]["1"],
             100 * BASE["frac_percolating_and_surviving"]["10"], 100 * BASE["frac_excluded_by_old_euclidean_criterion"], 100 * BASE["frac_inconsistent_with_survival"])))
add(("h1", "S7. Interior solution, scope and coupling estimates"))
add(("p", "In the open slicing, with $\\alpha=H_\\Lambda a$ and $\\tau=H_\\Lambda t$, $(d\\alpha/d\\tau)^2=1+\\alpha^2\\rho/V_f$. For a condensate that starts to oscillate at $\\tau_0=H_\\Lambda/m$ with energy "
          "density $\\epsilon V_f$, scales as matter and decays to radiation at the rate $\\gamma$, the comoving quantities $\\Phi=\\rho_\\varphi\\alpha^3$ and $Y=\\rho_r\\alpha^4$ obey $\\Phi=\\Phi_0e^{-\\gamma(\\tau-\\tau_0)}$ "
          "and $Y'=\\gamma\\Phi\\alpha$. In the curvature-dominated regime ($\\alpha^2\\rho\\le%.1e$ here) $\\alpha=\\tau$ and $\\rho_r=\\epsilon\\tau_0^3\\gamma\\tau^{-4}\\int_{\\tau_0}^\\tau se^{-\\gamma(s-\\tau_0)}ds$, "
          "evaluated without cancellation. It agrees with the full integration to %.1e (check V11). After the maximum, $T\\propto1/\\tau$, so $T\\tau$ is constant and equals %.2e GeV; "
          "$T=1$ MeV would require $t=%.1e$ s." % (max(r["alpha2_rho_max"] for r in IN["runs"]), V["V11_interior_closed_form"]["max_relative_difference"], IN["T_times_tau"], IN["t_for_bbn_temperature_seconds"])))
add(("p", "Scope (scope.py). With the curvature parameter $Y=a^2\\rho/(3M_{\\rm Pl}^2)$, the envelope $\\rho\\le\\epsilon\\Delta V(mt)^{-3}$ is verified against the closed form on a grid of five decay rates (maximum ratio %.12f). "
          "The mass at the true minimum is $m=\\kappa_m\\Delta V^{1/4}$ with $\\kappa_m=\\lambda^{1/4}[(3x_t^2-1)/2]^{1/2}\\delta^{-1/4}$ ($%.3f\\le\\kappa_m\\le%.3f$ over the prior). The number of e-folds is $N=g\\int_{x_t}^{x_0}v/|v'|dx$ "
          "(at most %s over $\\eta=0.15$, 0.20, 0.30, 0.36). Anti-de Sitter: $t_c=\\pi\\sqrt3M_{\\rm Pl}/\\sqrt{|V_t|}$; $T_{\\max}$ is evaluated with $g_*=3.36$ and $\\rho_{\\rm BBN}$ with $g_*=10.75$. "
          "Coupling (coupling.py): $\\delta(\\Delta V)=\\rho_m\\Delta\\ln m$, $\\Delta\\eta/\\eta=2\\delta(\\eta)\\varepsilon_m/[\\eta(x_f-x_t)]$, $\\Delta B=B|d\\ln b/d\\ln\\eta|\\Delta\\eta/\\eta$, with $d\\ln b/d\\ln\\eta$ by a centred difference of the interpolated table; "
          "linear $\\beta=\\Delta\\ln m/(\\Delta\\varphi/M_{\\rm Pl})$ and quadratic $\\Lambda_q=f[|x_f^2-x_t^2|/(2\\Delta\\ln m)]^{1/2}$."
          % (SC["envelope_check_max_ratio"], SC["kappa_m_min"], SC["kappa_m_max"], tx(max(e["N"] for e in SC["e_folds"]), 1))))
add(("h1", "S8. Reproduction"))
add(("p", "Environment of the published run: Python %s, NumPy %s, SciPy %s (Windows). Requirements: NumPy, SciPy, Matplotlib, python-docx (requirements.txt). From the code directory, in this order: "
          "bounce_tables.py (about 3 minutes), survival.py (seconds), monte_carlo.py (about 20 seconds), falsification_map.py, interior_evolution.py, convergence.py (about 4 minutes), scope.py, coupling.py, "
          "thermal.py (about 40 seconds), thermal_analysis.py (about 6 minutes), interior_sim.py (about 25 seconds), oscillation.py (about 40 seconds), oscillation_validation.py (about 25 minutes), density_bounce.py (about 1 minute), turnaround_scan.py (about 5 minutes), dilution_thermal.py, cyclic_extension.py and hot_cycles.py (about 5 minutes together), dilution_validation.py (about 2 minutes), verify.py (about 25 seconds), make_figures.py (calls make_thermal_figures.py, make_sim_figures.py, make_osc_figures.py and make_dil_figures.py), build_paper.py, build_supporting.py. The scripts write the JSON files that the "
          "manuscript builder reads; monte_carlo.py needs survival.json and falsification_map.py needs survival.json. A rerun in a clean clone of the repository reproduced the stored JSON files exactly in the earlier release. Code: %s, archived at https://doi.org/%s."
          % (platform.python_version(), np.__version__, scipy.__version__, GITHUB, DOI_CODE)))
add(("p", "Section 7 (after the continuous-cycle scripts): vacuum_phase_map.py (about 40 seconds), vacuum_wall.py (about 2 minutes), interface_transport.py (about 25 minutes), "
          "bubble_gravity.py (about 30 seconds), bubble_to_cycle.py (about 3 minutes), bubble_handoff.py (about 3 minutes), viability_mc.py (about 10 minutes), "
          "bubble_entropy.py, interface_validation.py (about 1 minute) and make_int_figures.py."))
add(("h1", "S9. Known limitations of the numerics"))
add(("bullets", [
    "Shooting is unreliable for $\\eta<0.1$ (exponential sensitivity); the boundary-value solver is used there, and below $\\eta=0.03$ only the thin-wall asymptote is available, which is irrelevant because $B$ exceeds $10^5$.",
    "For $g<10^{-4}$ the cancellation of the $1/g^2$ terms limits the precision of $B/B_{\\rm flat}$ to about $10^{-4}$; the gravitational corrections used in the paper are measured at $g\\ge10^{-3}$ and extrapolated linearly.",
    "The interpolation error of $b(\\eta)$ rises to $\\sim10^{-3}$ for $\\eta>0.36$ where $b$ falls steeply; this changes $B$ by less than 0.1 and $B_c$ by nothing.",
    "The one-loop prefactor is a systematic uncertainty. The results depend on it only through $B_c$, which changes by $\\pm7$ for a change of the prefactor by three decades and $\\pm14$ for six.",
    "The negative-mode analysis is for flat space; the gravitational negative-mode problem is not addressed because $g\\sim10^{-60}$.",
]))
supp = os.path.join(OUT, "EPJC_Late_Time_Vacuum_Decay_Supplement.docx")
import copyedit as CE
S = [CE.block(b) for b in S]
RN.render_docx(S, supp, os.path.join(REPO, "figures"))

# --------------------------------------------------------------------------------------------------------- cover letter
d = Document()
st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11); st.paragraph_format.space_after = Pt(8)
for s in d.sections:
    s.left_margin = s.right_margin = Inches(1.1); s.top_margin = s.bottom_margin = Inches(1.0)


def para(text, bold=False, after=None):
    text = CE.ce(text)
    p = d.add_paragraph(); r = p.add_run(text); r.bold = bold
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


para("Leon Sandler", True, 0)
for t in ("Independent Researcher", "Northbrook, Illinois, USA", "sandler.leon@gmail.com", "ORCID 0009-0007-4584-808X"):
    para(t, after=0)
para("")
para(datetime.date.today().strftime("%d %B %Y").lstrip("0"))
para("The Editors", after=0); para("The European Physical Journal C", after=0); para("Springer Nature")
para("Dear Editors,")
title_plain = TITLE.replace("--", "–")
para("I submit the manuscript “%s” as a Regular Article (Theoretical Physics) for consideration in The European Physical Journal C." % title_plain)
HB = HCJ["runs"][0]["cycles"][1]
GS = [c["G_from_entropy"] for c in HCJ["runs"][0]["cycles"][1:] if c.get("G_from_entropy")]
para("The paper asks whether the present dark energy can end the cosmological expansion and start a new hot big bang. The direct route, tunneling of a metastable dark-energy vacuum, fails "
     "for reasons that are quantified below. A vacuum held by dark matter and released by dilution does end the expansion: the turnaround follows an exact LambdaCDM past, and an intervention that "
     "freezes the matter density seen by the field removes it, so dilution is its cause. This revision then removes every prescribed step from the cycle. With an effective bounce, thermal "
     "dissipation and dark matter that obeys its own Boltzmann equation, %d consecutive cycles were integrated with no reset of any variable; each has a radiation-dominated bounce, a freeze-out "
     "to the observed dark-matter ratio, a matter era, a dark-energy epoch and a turnaround, and the entropy produced enlarges the next cycle by (S_n+1/S_n)^(1/3). The revision also reports what "
     "the earlier prescribed resets had hidden: a bounce too cold to re-thermalize the dark matter gives an empty next cycle. It adds the conditions on anisotropy (with and without an ekpyrotic "
     "phase), an independent bounce mechanism, the black holes of the previous cycle (which bound the bounce temperature from above), a failure-mode map, and the observational consequences, "
     "including the negative one that the cycle does not produce the observed primordial perturbations." % CYB.N_CYC)
para("In plain terms, the paper reverses a common intuition about the far future. The dilution of matter is usually the end state of cosmic evolution, the "
     "approach to heat death; here it is the trigger. If the dark energy is the energy of a vacuum held in place by dark matter, the emptying of the universe "
     "destabilizes it, and expansion can turn into contraction, a bounce and a new hot phase. What the paper contributes is not any single ingredient (cyclic "
     "cosmology, vacuum decay, matter-coupled scalars and effective bounces are established and cited) but their combination into a quantitative cycle that is "
     "followed without prescribed steps, with control experiments that remove each ingredient and an explicit map of the conditions under which the cycle "
     "fails. Three limits are stated throughout: the bounce is an effective prescription, the interface results hold only for a completed transition, and the "
     "model does not produce the observed primordial perturbations, so its claims concern the background cosmology.")
para("Following a further review, this version adds a section on density-dependent vacuum interfaces (Section 7). It maps where tunneling is meaningful: the "
     "symmetron release of the main model is a classical roll and needs a coherent seed or a tilt to avoid domain walls, while a coupling centred on the false vacuum makes "
     "the held vacuum metastable below a degeneracy density. For that branch the wall tension, the density-dependent action, the transmission of fields through the wall, "
     "the Israel junction and the evolution of the enclosed region are computed, with energy and entropy closure and controls that remove the matter coupling or the "
     "selectivity. The wall runs away and becomes transparent to the dark matter within a millimetre, so its mass-selective transmission has no cosmological effect. "
     "The hand-off from the walls to the cycle is checked explicitly: a completed transition is homogeneous above the collision scale, and the cycle continues "
     "whether the collision products are relativistic or not, whereas an isolated bubble is trapped behind a horizon before its bounce. Within these effective "
     "approximations the interface is presented as a conditional extension of the dilution trigger, not as a second demonstrated mechanism. A Monte Carlo over "
     "stated parameter ranges reports how often all the conditions hold at once. The title now reflects this content and states that the cycle is conditional.")
para("For the direct route the answer is negative. A scalar potential with an explicit metastable minimum is specified, the Coleman–De Luccia bounce is solved numerically in flat space and with Einstein gravity, and the action is converted to a "
     "nucleation rate with the measured dark-energy density as the false-vacuum energy. With all scales set by the dark energy the gravitational parameter is of order 10⁻⁶⁰, so gravity and the Hawking–Moss "
     "channel play no role and the decay is controlled by a single action, which must lie near %d for a transition on cosmological time scales. The survival of the present vacuum is evaluated on the Lorentzian past light "
     "cone of the observed expansion history (%.3f Hubble four-volumes, %d times smaller than the Euclidean four-sphere often used as a shortcut), which makes the survival condition a weak one. "
     "The nucleated region reheats to at most about %.1f meV, and the reason is not only energy conservation: the open interior stays curvature dominated, so that a hot epoch at one second would require a parent "
     "energy scale at least %d orders of magnitude above the Planck scale. A trigger by matter dilution requires a change of the matter mass between the vacua of at least about 0.1%%." % (
         round(MC["B_critical_quantiles"]["0.5"]), SV["V4_HL4"], round(SV["ratio_euclid_to_lorentzian"]), F["reheat"]["T_max_meV_at_Vf"], int(np.floor(SC["needed_parent_for_bbn_at_1s"]["kappa_m_min"]["orders_above_Planck"]))))
para("Cosmological cooling is treated explicitly. A scalar coupled to the CMB acquires a thermal mass, which rescales the tilted quartic potential exactly; the quantum (O(4)) and thermal (O(3)) exponents are compared, "
     "and the survival probability is computed with the time-dependent rate on the past light cone. Cooling lowers the thermal channel monotonically and cannot raise the late-time probability of a transition: a scalar decoupled from the CMB is unaffected, "
     "and one in equilibrium with it is more strongly constrained (%s%% to %s%% of the prior excluded against %s%%) with no late-time decay window left. The conclusion is therefore about daughter regions of the present dark energy, "
     "which are cold and empty, and not about a cyclic cosmology." % (str(round(100 * TR["prior_maps"]["c=0 (equilibrated), z<=1"]["excluded"])), str(round(100 * TR["prior_maps"]["c=0, z<=10"]["excluded"])), str(round(100 * TR["prior_maps"]["decoupled"]["excluded"]))))
para("The interior of the nucleated region is simulated, not only bounded: Lorentzian evolution of one and two fields with dissipation and gravity (constraint residual at most 1e-8), a reservoir hidden by a negative vacuum offset, hilltop flattening at the dark-energy scale, "
     "a chain of repeated transitions with the energy sum rule, and entropy accounting. They give the criteria of Table 13 for a viable daughter universe, which no scenario with the observed dark energy as the parent satisfies.")
para("The same region oscillating (Option B) is tested as well, with an effective loop-quantum-cosmology bounce and the turnaround supplied by the decay to anti-de Sitter: radiation-only cycles are periodic over a thousand bounces, "
     "but a scalar excitation is amplified at every bounce, the turnaround radius follows the energy-balance law Delta a_max = W/3 with W the pressure-volume loop, the cycles are stable only when dissipation per cycle exceeds the logarithm of the amplification, "
     "and a vacuum transition during the cycles destabilises them. Cosmological hysteresis itself is known; the stability boundary, the law in this setting and the effect of a transition are what the paper adds.")
para("I want to be clear about novelty here as well. The symmetron-type coupling, coupled dark matter, warm-inflation dissipation, thermal freeze-out, the effective bounce and Tolman's growing cycles are "
     "established and cited. What the paper adds is their combination into a cycle that contains a hot big bang and an epoch like ours, followed over 100 cycles without any reset, the exact energy relation that fixes how much the dark-matter mass must "
     "grow at the release, the comparison with an uncoupled control and with cooling as an alternative trigger, the tunneling branch in which dilution opens the survival window, and the identification of the "
     "ingredients without which the cycles fail. The result is a demonstration within a model, with its limits stated (effective bounce, a bounce density far below the Planck value, homogeneous dynamics), "
     "not a claim that the universe is cyclic.")
para("I want to be clear about novelty. The bounce, Hawking–Moss and thin-wall results, the past-light-cone survival method, open-universe interiors and screened couplings are established, and are cited as such; Table 1 of the "
     "manuscript lists, for each ingredient, what is known and what is added. What is new is the quantitative combination for the dark-energy vacuum: an explicit map from potential parameters to decay times with a "
     "numerical error budget and negative-mode check; the survival bound on the past light cone and the corrected width of the transition window; a statement of which energy sources the reheating bound excludes by geometry and which only by "
     "energy conservation; a coupling-independent requirement on any matter trigger; and a separation of internal consistency conditions from observational tests. The paper does not claim to establish a cyclic cosmology; it shows "
     "that tunneling of the present dark energy is not a viable reset in the class studied, and it states what would have to change.")
para("The analysis is fully reproducible. The code (Python, MIT licence) and the data behind every figure and table are at %s and archived at https://doi.org/%s; the manuscript is archived at "
     "https://doi.org/%s. The repository contains %d executable checks (against the thin-wall limit, an independent boundary-value solver, the exact de Sitter action, "
     "the analytic gravitational suppression, the Hawking–Moss criterion, the negative-mode structure, the past-light-cone volume and the scope estimates), and the manuscript is built from the stored results, so the text cannot diverge from the computation." % (GITHUB, DOI_CODE, DOI_PAPER, len(V)))
para("The work uses no observational data other than the measured dark-energy density and the Planck cosmological parameters, and it does "
     "not claim to predict the fate of our universe. The Monte Carlo describes the sample space of a stated family of potentials and priors, not a probability of a transition. "
     "A generative AI tool (Claude, Anthropic) was used to assist with the design of the numerical analysis, the code, literature checking and drafting; I reviewed and checked all of it, "
     "and I take full responsibility for the content. This is disclosed in the manuscript.")
para("The manuscript is original, has not been published elsewhere and is not under consideration at any other journal. I am the sole author, I have no competing interests, and the work "
     "received no external funding.")
para("Thank you for considering this submission.")
para("Sincerely,")
para("Leon Sandler", True)
d.save(os.path.join(OUT, "EPJC_Cover_Letter.docx"))
print("saved supplement and cover letter")
