# -*- coding: utf-8 -*-
"""Supplementary information and cover letter for the EPJ C submission.   python build_supporting.py   (after build_paper.py)"""
import json
import os

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

import constants as K
import render as RN


def tx(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    return "$%.*f\times10^{%d}$" % (d, x / 10 ** e, e)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "manuscript")
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
T, F, MC, IN, V = J("bounce_tables.json"), J("falsification_map.json"), J("monte_carlo.json"), J("interior_evolution.json"), J("verification.json")
assert all(v["passed"] for v in V.values())
GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
DOI_PAPER, DOI_CODE = "10.5281/zenodo.23717617", "10.5281/zenodo.23717619"
TITLE = "Late-Time Vacuum Decay and Cosmological Reheating: Gravitational Tunneling, Stochastic Sensitivity, and Falsification Boundaries"

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
}
S = []
add = S.append
add(("title", "Supplementary Information: " + TITLE))
add(("author", "Leon Sandler\nIndependent Researcher, Northbrook, Illinois, USA"))
add(("h1", "S1. Status"))
add(("p", "This supplement documents the numerical methods, the executable verification suite, the Monte Carlo specification, the closed-form interior solution and the "
          "reproduction commands for the paper. All numbers quoted in the paper are read from the JSON files produced by the scripts described here. The dimensionless "
          "toy Monte Carlo of an earlier draft (a flat-space thin-wall action with uniformly sampled dimensionless parameters and no connection to the dark-energy "
          "density) is superseded and is not part of the evidence."))
add(("h1", "S2. Numerical methods"))
add(("h2", "S2.1 Potential and scaling"))
add(("p", "The potential is $V=V_t+f^4\\lambda[(x^2-1)^2/8+\\eta x/2-p_{\\min}]$ with $x=\\varphi/f$. The stationary points solve $x^3-x+\\eta=0$, which has three real roots "
          "$x_t<x_{\\rm top}<x_f$ for $\\eta<\\eta_c=2/(3\\sqrt3)$. In units of $f$ for fields and $(\\sqrt\\lambda f)^{-1}$ for lengths the flat-space action is $B=b(\\eta)/\\lambda$, "
          "verified numerically to a relative $10^{-12}$ (check V3). With gravity the only additional parameter is $g=f^2/M_{\\rm Pl}^2$."))
add(("h2", "S2.2 Flat-space bounce"))
add(("p", "The equation $x''+3x'/r=v'(x)$ is integrated from a small radius ($10^{-6}$) with the series start $x=x_0+v'(x_0)r^2/8$ using an eighth-order Runge--Kutta method "
          "(DOP853) with relative tolerance $10^{-11}$. A trajectory is an overshoot if $x$ crosses $x_f$ with positive velocity and an undershoot if $x'$ changes sign first. "
          "The central value $x_0$ is bracketed between the undershoot limit $x_s$ (the point on the true-vacuum side with $v(x_s)=v_f$) and a point $10^{-9}$ from $x_t$, and "
          "bisected to $10^{-13}$. In the thin-wall regime ($\\eta<0.1$) the sensitivity of the shooting to $x_0$ is exponential in the wall radius, so the bounce is instead computed by "
          "collocation on the boundary-value problem with a tanh initial guess (SciPy solve\\_bvp, tolerance $10^{-9}$). Where both methods apply ($0.10\\le\\eta\\le0.17$) they agree to "
          "%s (check V2)." % tx(V["V2_shooting_vs_bvp"]["max_relative_difference"])))
add(("h2", "S2.3 Bounce with gravity"))
add(("p", "The four-dimensional system $(x,x',\\rho,\\rho')$ is integrated in the arclength $\\xi$ with $\\rho''=-(g\\rho/3)(x'^2+v)$ and the same overshoot/undershoot "
          "classification; the constraint $\\rho'^2=1+(g\\rho^2/3)(x'^2/2-v)$ is monitored and holds to %s (check V10). The action is "
          "$B=4\\pi^2\\int(\\rho^3v-3\\rho/g)d\\xi-S_{\\rm dS}$. The part of the bounce outside the wall follows the false-vacuum geometry $\\rho=\\sin(H\\eta)/H$, where $\\eta$ is the "
          "arclength from the second pole; the tail integral and the de Sitter background are evaluated analytically, "
          "$\\int_0^{a/H}(v_f\\sin^3H\\eta/H^3-3\\sin H\\eta/(gH))d\\eta=[v_f(2/3-\\cos a+\\cos^3a/3)/H^4-3(1-\\cos a)/(gH^2)]$, so the terms of order $1/g^2$ cancel exactly and the action keeps a precision of "
          "about $10^{-4}$ in $B/B_{\\rm flat}$ down to $g=10^{-4}$. The background action reproduces $-24\\pi^2/(g^2v_f)$ to machine precision (check V4)."
          % tx(V["V10_friedmann_constraint"]["max_residual"])))
add(("h2", "S2.4 Interpolation and rates"))
add(("p", "$\\ln b(\\eta)$ is tabulated at %d values of $\\eta$ between 0.03 and 0.38 and interpolated with a monotone cubic Hermite spline (PCHIP); the interpolation error at four off-grid "
          "tilts is below %s (check V8). The rate, the critical action and the decay time follow Eq. (7)--(8) of the paper and are evaluated in the logarithmic domain because "
          "$\\Gamma/H^4$ spans several hundred orders of magnitude." % (len(T["flat"]), tx(V["V8_interpolation_off_grid"]["max_relative_error"]))))
add(("h1", "S3. Executable verification suite"))
add(("p", "verify.py recomputes the following checks from the stored tables and by direct calls to the solvers. The manuscript builder refuses to run unless every one passes."))
rows = []
for k, v in V.items():
    det = {kk: (("%.3g" % vv) if isinstance(vv, float) else str(vv)) for kk, vv in v.items() if kk != "passed"}
    rows.append([k.split("_")[0], DESC.get(k, k), "pass" if v["passed"] else "FAIL", "; ".join("%s = %s" % (a, b) for a, b in list(det.items())[:3])])
add(("table", "Table S1. Verification checks and their outcomes.", ["No.", "What is checked", "Outcome", "Key figures"], rows, None))
add(("h1", "S4. Monte Carlo specification"))
add(("p", "Draws: $N=%d$, NumPy PCG64 generator, seed %d. $\\log_{10}\\lambda\\sim U(-2,\\log_{10}4\\pi)$; $\\eta\\sim U(0.03,0.38)$; $\\ln D\\sim N(0,(1.5\\ln10)^2)$; $r_L=1$. Each draw is converted to "
          "$\\ln\\Gamma/H^4$ with Eq. (7), to a decay time, and classified: survival-excluded if $\\Gamma/H^4\\ge3/(8\\pi^2)$; otherwise decaying by $10^{12}$, $10^{20}$, $10^{50}$ or $10^{100}$ yr, or "
          "never. Standard errors are batch-means over 20 batches. Convergence: the fractions are recomputed for $10^3$, $10^4$, $10^5$ and $10^6$ independent draws (Fig. 4c of the paper). "
          "Sensitivity runs change one ingredient at a time: the prefactor spread (0 and 3 decades), $r_L\\in[1,10]$, $\\lambda\\le1$, $\\eta\\ge0.10$, and $\\log_{10}\\lambda\\in[-4,2]$ (Table 3 of the paper). "
          "Failure regions are explicit: draws with no false minimum cannot occur by construction; the regimes `survival-excluded' and `never' are the two failure regions of the reset hypothesis."
          % (MC["N"], MC["seed"])))
add(("h1", "S5. Interior solution"))
add(("p", "In the open slicing, with $\\alpha=H_\\Lambda a$ and $\\tau=H_\\Lambda t$, $(d\\alpha/d\\tau)^2=1+\\alpha^2\\rho/V_f$. For a condensate that starts to oscillate at $\\tau_0=H_\\Lambda/m$ with energy "
          "density $\\epsilon V_f$, scales as matter and decays to radiation at the rate $\\gamma$, the comoving quantities $\\Phi=\\rho_\\varphi\\alpha^3$ and $Y=\\rho_r\\alpha^4$ obey $\\Phi=\\Phi_0e^{-\\gamma(\\tau-\\tau_0)}$ "
          "and $Y'=\\gamma\\Phi\\alpha$. In the curvature-dominated regime ($\\alpha^2\\rho\\le%.1e$ here) $\\alpha=\\tau$ and $\\rho_r=\\epsilon\\tau_0^3\\gamma\\tau^{-4}\\int_{\\tau_0}^\\tau se^{-\\gamma(s-\\tau_0)}ds$, "
          "evaluated without cancellation. It agrees with the full integration to %.1e (check V11). After the maximum, $T\\propto1/\\tau$, so $T\\tau$ is constant and equals %.2e GeV; "
          "$T=1$ MeV would require $t=%.1e$ s." % (max(r["alpha2_rho_max"] for r in IN["runs"]), V["V11_interior_closed_form"]["max_relative_difference"], IN["T_times_tau"], IN["t_for_bbn_temperature_seconds"])))
add(("h1", "S6. Reproduction"))
add(("p", "Requirements: Python 3.11, NumPy, SciPy, Matplotlib, python-docx. From the code directory: bounce_tables.py (about 4 minutes), monte_carlo.py (about 1 minute), "
          "falsification_map.py, interior_evolution.py, verify.py (about 3 minutes), make_figures.py, build_paper.py, build_supporting.py. The scripts write the JSON files that the "
          "manuscript builder reads. Code: %s, archived at https://doi.org/%s." % (GITHUB, DOI_CODE)))
add(("h1", "S7. Known limitations of the numerics"))
add(("bullets", [
    "Shooting is unreliable for $\\eta<0.1$ (exponential sensitivity); the boundary-value solver is used there, and below $\\eta=0.03$ only the thin-wall asymptote is available, which is irrelevant because $B$ exceeds $10^5$.",
    "For $g<10^{-4}$ the cancellation of the $1/g^2$ terms limits the precision of $B/B_{\\rm flat}$ to about $10^{-4}$; the gravitational corrections used in the paper are measured at $g\\ge10^{-3}$ and extrapolated linearly.",
    "The interpolation error of $b(\\eta)$ rises to $\\sim10^{-3}$ for $\\eta>0.36$ where $b$ falls steeply; this changes $B$ by less than 0.1 and $B_c$ by nothing.",
    "The one-loop prefactor is an estimate. The results depend on it only through $B_c$, which changes by $\\pm7$ for a change of the prefactor by three decades.",
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
para("28 October 2026")
para("The Editors", after=0); para("The European Physical Journal C", after=0); para("Springer Nature")
para("Dear Editors,")
para("I submit the manuscript “%s” as a Regular Article (Theoretical Physics) for consideration in The European Physical Journal C." % TITLE)
para("The paper asks a question that is often posed informally: can the decay of a metastable late-time vacuum seed a new cosmological cycle? It treats the "
     "question as a calculation. A scalar potential with an explicit metastable minimum is specified, the Coleman–De Luccia bounce is solved numerically in flat space and with "
     "Einstein gravity, and the action is converted to a nucleation rate per Hubble four-volume with the measured dark-energy density as the false-vacuum energy. "
     "The result is a set of quantitative boundaries rather than a prediction: with all scales set by the dark energy, gravity and the Hawking–Moss channel play no role "
     "(the gravitational parameter is of order 10⁻⁶⁰) and the decay is a flat-space process controlled by a single action, which must lie within a narrow window near 287 for the "
     "transition to be neither excluded by the survival of the present vacuum nor effectively absent. Energy conservation then bounds the reheating of the nucleated region "
     "to about 2 meV, some nine orders of magnitude below nucleosynthesis, so that a hot big bang cannot be regenerated by the decay of the present dark energy; and "
     "a trigger by the dilution of matter would require a scalar–matter coupling some 28 to 30 orders of magnitude larger than allowed.")
para("The conclusions are largely negative for the cyclic-reset hypothesis in its simplest form, and the paper says so. Its contribution is the explicit, reproducible calculation "
     "chain, with the failure regions stated and a prior-sensitivity Monte Carlo with convergence tests, and a list of falsification conditions that can be recomputed for any other "
     "potential. I believe this fits the scope of EPJ C in cosmology, quantum field theory in curved space and gravitation.")
para("The analysis is fully reproducible. The code (Python, MIT licence) and the data behind every figure and table are at %s and archived at https://doi.org/%s; the manuscript is archived at "
     "https://doi.org/%s. The repository contains 13 executable checks of the numerical machinery (against the thin-wall limit, an independent boundary-value solver, the exact de Sitter action, "
     "the analytic gravitational suppression and the Hawking–Moss criterion) and the manuscript is built from the stored results, so the text cannot diverge from the computation." % (GITHUB, DOI_CODE, DOI_PAPER))
para("I want to be explicit about the status of the work. It uses no observational data other than the measured dark-energy density and the Planck cosmological parameters, and it does "
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
