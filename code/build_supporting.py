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
assert all(v["passed"] for v in V.values())
GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
DOI_PAPER, DOI_CODE = "10.5281/zenodo.23717617", "10.5281/zenodo.23717619"
if os.path.exists(os.path.join(HERE, "dois.json")):
    _d = J("dois.json")
    DOI_PAPER, DOI_CODE = _d["paper"], _d["code"]
TITLE = "Late-Time Vacuum Decay as a Cosmological Reset: Coleman--De Luccia Tunneling, Reheating Constraints, and Consistency Boundaries"
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
          "collocation on the boundary-value problem with a tanh initial guess (SciPy solve\\_bvp, tolerance $10^{-9}$, 4000 initial nodes, domain padding 30). Where both methods apply ($0.10\\le\\eta\\le0.17$) they agree to "
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
          "verify.py (about 15 seconds), make_figures.py, build_paper.py, build_supporting.py. The scripts write the JSON files that the "
          "manuscript builder reads; monte\\_carlo.py needs survival.json and falsification\\_map.py needs survival.json. A rerun in a clean clone of the repository reproduced the stored JSON files exactly in the earlier release. Code: %s, archived at https://doi.org/%s."
          % (platform.python_version(), np.__version__, scipy.__version__, GITHUB, DOI_CODE)))
add(("h1", "S9. Known limitations of the numerics"))
add(("bullets", [
    "Shooting is unreliable for $\\eta<0.1$ (exponential sensitivity); the boundary-value solver is used there, and below $\\eta=0.03$ only the thin-wall asymptote is available, which is irrelevant because $B$ exceeds $10^5$.",
    "For $g<10^{-4}$ the cancellation of the $1/g^2$ terms limits the precision of $B/B_{\\rm flat}$ to about $10^{-4}$; the gravitational corrections used in the paper are measured at $g\\ge10^{-3}$ and extrapolated linearly.",
    "The interpolation error of $b(\\eta)$ rises to $\\sim10^{-3}$ for $\\eta>0.36$ where $b$ falls steeply; this changes $B$ by less than 0.1 and $B_c$ by nothing.",
    "The one-loop prefactor is a systematic uncertainty. The results depend on it only through $B_c$, which changes by $\\pm7$ for a change of the prefactor by three decades and $\\pm14$ for six.",
    "The negative-mode analysis is for flat space; the gravitational negative-mode problem is not addressed because $g\\sim10^{-60}$.",
]))
supp = os.path.join(OUT, "EPJC_Late_Time_Vacuum_Decay_Supplement.docx")
RN.render_docx(S, supp, os.path.join(REPO, "figures"))

# --------------------------------------------------------------------------------------------------------- cover letter
d = Document()
st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11); st.paragraph_format.space_after = Pt(8)
for s in d.sections:
    s.left_margin = s.right_margin = Inches(1.1); s.top_margin = s.bottom_margin = Inches(1.0)


def para(text, bold=False, after=None):
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
para("The paper tests a proposal that is often stated informally: that the decay of a metastable late-time vacuum could reset the cosmological history. It is treated as a calculation, and the answer is "
     "largely negative. A scalar potential with an explicit metastable minimum is specified, the Coleman–De Luccia bounce is solved numerically in flat space and with Einstein gravity, and the action is converted to a "
     "nucleation rate with the measured dark-energy density as the false-vacuum energy. With all scales set by the dark energy the gravitational parameter is of order 10⁻⁶⁰, so gravity and the Hawking–Moss "
     "channel play no role and the decay is controlled by a single action, which must lie near %d for a transition on cosmological time scales. The survival of the present vacuum is evaluated on the Lorentzian past light "
     "cone of the observed expansion history (%.3f Hubble four-volumes, %d times smaller than the Euclidean four-sphere often used as a shortcut), which makes the survival condition a weak one. "
     "The nucleated region reheats to at most about %.1f meV, and the reason is not only energy conservation: the open interior stays curvature dominated, so that a hot epoch at one second would require a parent "
     "energy scale at least %d orders of magnitude above the Planck scale. A trigger by matter dilution requires a change of the matter mass between the vacua of at least about 0.1%%." % (
         round(MC["B_critical_quantiles"]["0.5"]), SV["V4_HL4"], round(SV["ratio_euclid_to_lorentzian"]), F["reheat"]["T_max_meV_at_Vf"], int(np.floor(SC["needed_parent_for_bbn_at_1s"]["kappa_m_min"]["orders_above_Planck"]))))
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
