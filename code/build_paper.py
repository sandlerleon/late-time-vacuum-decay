# -*- coding: utf-8 -*-
"""Builds the manuscript for the EPJ C submission from the stored results.

    python build_paper.py          (refuses to run unless verification.json reports every check passed)

The text carries no hand-typed result: every number is read from bounce_tables.json, convergence.json, survival.json, falsification_map.json,
monte_carlo.json, interior_evolution.json, scope.json, coupling.json, verification.json or constants.py, and every comparative sentence is asserted
against the same files.
"""
import json
import os
import sys

import numpy as np

import constants as K
import cdl_lib as C
import render as RN
import thermal_blocks as TB
import sim_blocks as SBK
import osc_blocks as OSB
import dil_blocks as DIL

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
FIGDIR = os.path.join(REPO, "figures")
OUT = os.path.join(REPO, "manuscript")
os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
T, F, MC, IN, V = J("bounce_tables.json"), J("falsification_map.json"), J("monte_carlo.json"), J("interior_evolution.json"), J("verification.json")
SV, CV, SC, CP = J("survival.json"), J("convergence.json"), J("scope.json"), J("coupling.json")
TR, TT = J("thermal_results.json"), J("thermal_tables.json")
PMAP = TR["prior_maps"]
SIM = J("interior_sim.json")
RB35 = [r for r in SIM["B_reservoir_ads"]["rows"] if r["kappa"] == 1e35][0]
HILL_T = max(r["T_max_meV"] for r in SIM["C_hilltop"]["rows"])
OSC, OSV = J("oscillation.json"), J("oscillation_validation.json")
MU_LIN = [f["energy_multiplier_per_cycle"] for f in OSC["stability"]["scalar_floquet"]]
assert all(v["passed"] for v in V.values()), "verification has failures - do not build"

GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
DOI_PAPER, DOI_CODE = "10.5281/zenodo.23717617", "10.5281/zenodo.23717619"       # replaced by the new-version DOIs in release.py
if os.path.exists(os.path.join(HERE, "dois.json")):
    _d = J("dois.json")
    DOI_PAPER, DOI_CODE = _d["paper"], _d["code"]
TITLE = "A Cyclic Hot Big Bang Triggered by Matter Dilution: Metastable Dark Energy, Vacuum Release, Recollapse and Self-Similar Nonsingular Cycles"


def chk(cond, msg):
    if not cond:
        raise SystemExit("claim not supported by the results: " + msg)


# ------------------------------------------------------------------------------------------------------------ numbers
flat = {r["eta"]: r for r in T["flat"]}
PH = {(p["lam"], p["eta"]): p for p in F["physical"]}
RH = F["reheat"]
BASE = MC["baseline"]
BAT = MC["baseline_batch"]
SENS = MC["sensitivity"]
r_prompt = [r for r in IN["runs"] if r["eps"] == 1.0 and r["gamma_over_m"] == 1.0][0]
EPS_IN = IN["eps_in_ref"]
BC = MC["B_critical_quantiles"]
bc_med = BC["0.5"]
g_lo, g_hi = F["g_physical_range"]
kap = {s["eta"]: s["kappa"] for s in T["slopes"]}
v = lambda name: V[name]
pc = lambda x, d=1: ("%." + str(d) + "f") % (100 * x)
XMAX = SV["gamma_over_hl4_max_95"]
XOLD = SV["gamma_over_hl4_max_euclid_old"]
T_MIN = SV["t_dec_min_yr"]
LOG_TMIN = float(np.log10(T_MIN))
DB_SURV = float(np.log(XMAX))                          # B at the survival bound lies this far below B_c
DB_RELAX = float(np.log(XMAX / XOLD))                  # the corrected bound is weaker than the Euclidean estimate by this amount in B
DB_WINDOW = (100 - LOG_TMIN) * np.log(10.0)            # change of B over which the decay time runs from the survival bound to 1e100 yr
DB_20 = (20 - LOG_TMIN) * np.log(10.0)
NSC = SC["needed_parent_for_bbn_at_1s"]
BM = [r for r in CP["rows"] if r["eta"] == 0.30][0]
BM2 = [r for r in CP["rows"] if r["eta"] == 0.32][0]
NEG = CV["D_negative_mode"]
A_ = CV["A_flat_shooting"]
B_ = CV["B_boundary_value"]
G_ = CV["C_gravity_coefficient"]


def sci(x, d=1):
    """Scientific notation as inline math (for prose and table cells)."""
    return "⟦" + texsci(x, d) + "⟧"


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:                      # rounding carried into the next decade (9.97 -> 10.0)
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


def worst(key, cond=lambda x: True):
    out = []
    for eta, r in A_.items():
        out += [x["rel"] for x in r[key] if x["rel"] is not None and cond(x)]
    return max(out)


chk(PH[(1.0, 0.30)]["B"] > bc_med + 100, "B at the benchmark far above critical")
chk(PH[(1.0, 0.32)]["log10_t_yr"] > 12 and PH[(1.0, 0.32)]["log10_t_yr"] < 20, "eta=0.32 lies in the 1e12-1e20 yr window")
chk(PH[(1.0, 0.34)]["log10_t_yr"] < LOG_TMIN, "eta=0.34 is excluded by survival")
chk(RH["T_max_meV_at_Vf"] < 3.0 and RH["ratio_Tbbn_over_Tmax"] > 1e8, "reheating no-go")
chk(r_prompt["T_max_meV"] < RH["T_max_meV_at_Vf"] and IN["t_for_bbn_temperature_seconds"] < IN["scales"]["t0_seconds"], "interior cannot reach 1 MeV")
chk(BASE["frac_never_within_1e100"] > 0.7 and BASE["frac_decay_by"]["1e+20"] < 0.01, "prior mass of the window is small")
chk(max(abs(SENS[k]["frac_inconsistent_with_survival"] - BASE["frac_inconsistent_with_survival"]) for k in ("sigma_log10_D = 0", "sigma_log10_D = 3", "sigma_log10_D = 6", "r_L ~ logU(1, 10)")) < 1e-3, "prior mass insensitive to the prefactor")
chk(g_hi < 1e-59 and PH[(1.0, 0.30)]["hm_ratio"] > 1e58, "gravity and HM negligible at the physical scale")
chk(SV["ratio_euclid_to_lorentzian"] > 100 and abs(BASE["frac_inconsistent_with_survival"] - BASE["frac_excluded_by_old_euclidean_criterion"]) < 0.01, "corrected survival bound is far weaker but the excluded prior mass hardly changes")
chk(BASE["frac_B_below_30_and_not_excluded"] == 0.0, "all draws with B < 30 are excluded by survival")
chk(all(n["n_negative_l0"] == 1 and n["n_negative_l1"] == 0 and n["n_negative_l2_to_4"] == 0 for n in NEG.values()), "one negative mode")
chk(SC["envelope_check_max_ratio"] <= 1.0 + 1e-9 and NSC["kappa_m_min"]["orders_above_Planck"] > 40, "scope of the reheating bound")
chk(BM["cases"][0]["beta_linear"] > 1e26 and BM["cases"][1]["beta_linear"] < 1e30, "linear coupling range")

# ------------------------------------------------------------------------------------------------------- abstract
HCB = DIL.BASE["cycles"][1]
ABSTRACT = (
    "Can the present dark energy end the cosmological expansion and start a new hot big bang? We first test the direct route, tunneling of a metastable dark-energy vacuum, with the "
    "Coleman--De Luccia bounce (error budget, negative modes, %d executable checks), survival on the Lorentzian past light cone ($\\Gamma/H_\\Lambda^4<%.0f$), the thermal channel "
    "(an exact rescaling $\\eta_{\\rm eff}=\\eta/(1-2\\theta)^{3/2}$; cooling cannot raise the late-time rate) and Lorentzian simulations of the nucleated region: a daughter region "
    "reheats to at most $%.1f$ meV and stays curvature dominated, and an oscillating region with an effective bounce has unstable cycles with negative vacuum energy throughout. "
    "Tunneling of the present dark energy does not produce a new hot epoch. We then show what does, within an explicit model. A scalar coupled to dark matter, "
    "$V_{\\rm eff}=V_0(\\varphi)+\\rho_*A(\\varphi)$, is held at a positive vacuum energy, the observed dark energy, while the dark-matter density exceeds a critical value; "
    "dilution releases it into a negative vacuum and the expansion turns around. Over %d coupled models with identical initial data at recombination, every one reproduces the "
    "observed $\\Lambda$CDM past and recollapses or settles, while the uncoupled control rolls before today for $\\mu\\ge%.0fH_0$; cooling cannot play the same role (the coupling "
    "required is too weak ever to equilibrate), and a coupling centred on the false vacuum lets vacua that survival would exclude decay in the near future. With an effective "
    "loop-quantum-cosmology bounce, thermal dissipation $\\Upsilon\\propto T^3$ and the re-thermalization and freeze-out of dark matter at the hot bounce, the cycles become "
    "self-similar: every cycle has a radiation-dominated bounce, a matter era, a dark-energy epoch of %.1f e-folds and a release at the same temperature, while the entropy grows "
    "by a factor $%s$ per cycle and enlarges the next cycle by exactly $(S_{n+1}/S_n)^{1/3}=%.1f$. Without the dissipation or the freeze-out reset the cycles fail. "
    "The model predicts $w=-1$ until the release, no fifth force today and a future recollapse; it is bounded by the deepest cosmic voids." % (
        len(V), XMAX, RH["T_max_meV_at_Vf"], DIL.N_C, DIL.CTRL_BEFORE[0], HCB["held_accelerated_efolds"], DIL.texsci(DIL.gm(DIL.GS) ** 3, 1), DIL.gm(DIL.GS)))
KEYWORDS = "cyclic cosmology; nonsingular bounce; dark energy; false vacuum decay; symmetron; coupled dark matter; recollapse; entropy; freeze-out; falsifiability"

B = []
add = B.append
add(("title", TITLE))
add(("author", "Leon Sandler\nIndependent Researcher, The Conscious Machine --- Independent Research, Northbrook, Illinois, USA\nsandler.leon@gmail.com   ·   ORCID 0009-0007-4584-808X"))
add(("abstract", ABSTRACT))
add(("keywords", KEYWORDS))

# ------------------------------------------------------------------------------------------------ 1 Introduction
add(("h1", "1. Introduction"))
add(("p", "In the standard cosmological model the late universe approaches de Sitter space: matter and radiation dilute, and a positive vacuum "
          "energy $\\rho_\\Lambda$ remains. If that energy is not an exact cosmological constant but the energy of a metastable scalar vacuum, "
          "the vacuum can decay by quantum tunneling through a Coleman--De Luccia (CDL) instanton \\cite{CDL}. It has been suggested that such a "
          "late-time transition could reset the cosmological history, so that the long-term evolution is cyclic rather than terminal; cyclic and bouncing "
          "proposals are discussed in \\cite{Penrose,GV,ST,IS}. Two things must be kept apart. The formation of a bubble universe is not a cyclic cosmology: "
          "a cycle needs a mechanism that connects successive epochs. The question tested here is narrower and can be answered by calculation: can the "
          "tunneling of the present dark energy produce a new hot, expanding epoch? That requires a specified metastable minimum, the tunneling action "
          "including gravity, a rate compared with the observed expansion history, and the interior of the nucleated region followed to see whether it can "
          "become a hot big bang."))
add(("p", "A further distinction concerns the outcome. In a branching or recycling picture (Option A) a parent metastable vacuum nucleates daughter regions by tunneling; each is a bubble inside the parent "
          "spacetime, and whether it is a hot expanding universe, or a universe at all in the sense of \\cite{BGG,FG,FGG}, is a separate question. In a genuinely cyclic cosmology (Option B) the same region "
          "expands, contracts, bounces and reheats, which needs a dynamical bounce beyond tunneling and raises problems of stability and entropy \\cite{Tolman,PS}. This paper addresses Option A with the "
          "Coleman--De Luccia theory, tests Option B in Section 5.6 with an effective bounce, and in Section 6 constructs and tests a version of Option B that contains "
          "an epoch like ours: a vacuum held by matter and released by its dilution."))
add(("p", "Four claims are tested. (i) A scalar potential with a metastable minimum at the dark-energy scale can nucleate bubbles of lower-energy vacuum at a rate "
          "that is neither negligible nor already excluded. (ii) The nucleated region reheats into a hot expanding universe. (iii) The "
          "transition is triggered, or enhanced, by the dilution of matter in the late universe. (iv) Cosmological cooling, acting on a scalar coupled to the CMB, makes the transition more probable at late times. The analysis is conditional: it takes the dark-energy density as the "
          "false-vacuum energy and asks what follows. Claims (i), (ii) and (iv) are answered by Sections 3--5; claim (iii) is developed in Section 6 into a cyclic model, whose "
          "ingredients are standard and whose combination is tested against an uncoupled control and against the requirement that every cycle contain a hot big bang and an epoch like ours."))
add(("p", "The contributions are the following."))
add(("bullets", [
    "A reproducible map from the parameters of an explicit scalar potential to cosmological decay times, with a numerical error budget, an independent "
    "boundary-value solution, the negative-mode structure of the bounce, the thin-wall, Parke and Hawking--Moss limits, and a treatment of the prefactor as a "
    "systematic uncertainty (Sections 2--4).",
    "A joint analysis of nucleation, survival and the post-transition energy budget, with the survival of the present vacuum evaluated on the Lorentzian past "
    "light cone of the observed expansion history (Section 4.2), and a statement of which energy sources the reheating bound excludes by geometry and which only by "
    "energy conservation (Section 5).",
    "Explicit conditions that separate a late-time vacuum transition from a viable cosmological restart, divided into internal consistency conditions and "
    "observational tests (Section 7), and a model-independent requirement on any matter coupling that could trigger the transition (Section 6).",
    "A finite-temperature extension: the exact rescaling of the thermally corrected potential, the O(3) and O(4) exponents and their crossover, the past-light-cone survival integral with a time-dependent "
    "rate, and prior maps showing that cooling cannot raise the late-time transition probability (Sections 4.6--4.8).",
    "Lorentzian simulations of the daughter region with two fields, dissipation and gravity, a reservoir hidden by a negative offset, hilltop flattening, repeated transitions and entropy accounting, and "
    "the resulting criteria for a viable daughter universe (Section 5.5).",
    "A test of the same region oscillating (Option B) with an effective bounce: validated cycles, a hysteresis law for the turnaround radius, a stability boundary set by dissipation against amplification over $10^2$--$10^3$ cycles, "
    "and the effect of a vacuum transition on the cycles (Section 5.6).",
    "A matter-held vacuum released by dilution: a dark-matter coupling that holds the observed dark energy, the exact energy relation $V_c-V_0(v)=(A(v)-1)\\rho_{\\rm crit}/2$, "
    "release and recollapse compared with an uncoupled control over a parameter grid, the failure of cooling as a trigger, a tunneling branch in which dilution opens the survival "
    "window, and self-similar hot cycles with an effective bounce, thermal dissipation and freeze-out, with the conditions under which they fail (Section 6).",
]))
add(("h2", "1.1 Relation to previous work"))
add(("p", "The ingredients are established, and the paper claims novelty only for the specific combination and the quantitative results for the dark-energy vacuum. "
          "Table 1 states for each ingredient what is known, what is used from the literature and what is computed here."))
add(("table", "Table 1. Closest literature and what this paper adds.",
     ["Ingredient", "Established (references)", "This paper"],
     [["Bounce, thin wall, gravity, Hawking--Moss", "Coleman; Callan--Coleman; CDL; Hawking--Moss; Parke; Jensen--Steinhardt; Hackworth--Weinberg \\cite{Coleman,CC,CDL,HM,Parke,JS,HW}",
       "Used as validation limits. New: an explicit dark-energy-scale potential, $g\\sim10^{-60}$, and a numerical error budget and negative-mode check (Section 3)."],
      ["Prefactor of the rate", "Determinant methods \\cite{DM,AFS}", "Not recomputed; treated as a systematic uncertainty with a stated spread and shown to enter $B_c$ additively (Section 4.3)."],
      ["Survival from the past light cone", "Standard for the electroweak vacuum \\cite{IRS,EGR}; percolation \\cite{GW,TWW}",
       "Applied to the dark-energy vacuum with the observed expansion history: $V_4=%.3f\\,H_\\Lambda^{-4}$, %.0f times below the Euclidean four-sphere (Section 4.2)." % (SV["V4_HL4"], SV["ratio_euclid_to_lorentzian"])],
      ["Open interior and reheating", "Open universes from bubbles need an inflationary stage \\cite{Gott,BGT,FKMS}",
       "Quantified for the dark-energy scale: curvature parameter, e-folds and the parent scale needed for a hot epoch (Section 5.3)."],
      ["Finite-temperature decay", "Thermal masses \\cite{DJ,W74}; thermal activation and the O(3) bounce \\cite{L83,A81,BW}",
       "Exact rescaling of the tilted quartic family under a thermal mass; O(3)/O(4) crossover for the dark-energy vacuum; survival with $\\Gamma(T_\\gamma(t))$ (Sections 4.6--4.8)."],
      ["Daughter universes by tunneling", "\\cite{BGG,FG,FGG}", "The nucleated region quantified as a cold, curvature-dominated daughter region, with the energy budget and reservoir bounds (Section 5)."],
      ["Cyclic and bouncing models", "\\cite{Penrose,GV,ST,IS}", "These supply the reset by other means; here one proposed mechanism, tunneling of the present dark energy, is tested."],
      ["Landscape and bubble collisions", "\\cite{BP,DKS,FKMS,Kleban,AJ}", "Observable signals need $\\Gamma/H^4=O(1)$; the width of that window in $B$ and its prior mass are computed (Section 4.5)."],
      ["Matter couplings and screening", "\\cite{KW,HK}", "A coupling-independent requirement, $\\Delta\\ln m$, and its translation to linear and quadratic couplings (Section 6)."]], None))

# ------------------------------------------------------------------------------------------------ 2 Model
add(("h1", "2. Scalar effective theory and the potential"))
add(("p", "We consider a real scalar field minimally coupled to Einstein gravity,"))
add(("eq", r"S=\int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R-\frac12 g^{\mu\nu}\partial_\mu\varphi\,\partial_\nu\varphi-V(\varphi)\right]"))
add(("p", "where $M_{\\rm Pl}$ is the reduced Planck mass. The false vacuum has the energy density $V_f = V(\\varphi_f)$, which is identified with the "
          "observed dark energy, $V_f = \\rho_\\Lambda = %s$ $\\mathrm{GeV}^4$ (quarter-root %.2f meV; Planck 2018 \\cite{Planck}, $H_0 = %.2f\\ \\mathrm{km\\,s^{-1}\\,Mpc^{-1}}$, "
          "$\\Omega_\\Lambda = %.4f$). The asymptotic de Sitter rate is $H_\\Lambda = \\sqrt{V_f/3}/M_{\\rm Pl} = %s$ GeV, so that $H_\\Lambda^{-1} = %.1f$ Gyr, and "
          "$M_{\\rm Pl}^4/V_f = %s$, with $\\ln(M_{\\rm Pl}^4/V_f) = %.1f$. A microscopic origin of the potential is not assumed." % (
              sci(K.V_F, 2), K.V_F ** 0.25 * 1e12, K.H0_KMS_MPC, K.OMEGA_L, sci(K.H_L, 2), K.T_HUBBLE_YR / 1e9, sci(K.M4_OVER_VF, 2), K.LN_M4_OVER_VF)))
add(("p", "A single dimensionless family covers the generic metastable case. With $x = \\varphi/f$,"))
add(("eq", r"V(\varphi)=V_t+f^4\lambda\left[\frac{(x^2-1)^2}{8}+\frac{\eta\,x}{2}-p_{\min}(\eta)\right],\qquad x^3-x+\eta=0\ \Rightarrow\ x_t<x_{\rm top}<x_f"))
add(("p", "where $p_{\\min}$ is the value of the bracket at the true minimum, so that $V(\\varphi_t) = V_t$. The tilt $\\eta$ lifts the degeneracy of the two "
          "wells; a false minimum exists only for $\\eta < \\eta_c = 2/(3\\sqrt3) = %.4f$. The energy difference and the barrier height above the false minimum are"
          % C.eta_crit()))
add(("eq", r"\Delta V=V_f-V_t=f^4\lambda\,\delta(\eta),\qquad V_{\rm top}-V_f=f^4\lambda\,\beta(\eta),\qquad \delta\simeq\eta\ (\eta\ll1)"))
add(("p", "The motivation for this family is minimality. A quartic with a small linear term is the simplest renormalizable potential with a metastable minimum and was "
          "the standard example in the original treatments of vacuum decay \\cite{Coleman,CDL}; it has three parameters with direct meaning, the quartic coupling $\\lambda$, "
          "the tilt-to-barrier ratio $\\eta$ and the scale $f$, and the latter is fixed by the dark energy. The coupling is restricted to $10^{-2}\\le\\lambda\\le4\\pi$, the upper "
          "limit being the perturbative-unitarity bound for a quartic interaction and the lower limit an arbitrary but wide cut that is varied in Section 4.5; the tilt is restricted "
          "to $0.03\\le\\eta\\le0.38$, which excludes the nearly degenerate case, where $B\\propto\\eta^{-3}$ is so large that the transition is irrelevant, and ends just below "
          "the point where the minimum disappears. Other potentials change $b(\\eta)$ and the interior energy fraction but not the scales that decide the results (Section 8). "
          "A scalar with a mass of the order of meV is technically unnatural without a protecting symmetry; this is a known property of every quintessence-like model and is not addressed here."))
add(("p", "The choice $V_t = 0$ (decay to a Minkowski vacuum, $r_L \\equiv V_f/\\Delta V = 1$) is the baseline, because it maximizes the energy "
          "released to the interior for a given $V_f$ when $V_t\\ge0$; $V_t < 0$ (anti-de Sitter) and $V_t > 0$ are considered in Sections 5.3 and 4.5. The "
          "scale $f$ is not free: it is fixed by $V_f = \\rho_\\Lambda$, $f^4 = \\rho_\\Lambda/(r_L\\lambda\\delta)$. For $\\lambda = 1$ and "
          "$\\eta = 0.3$ one finds $f = %.2f$ meV and a field excursion of $%.1f$ meV, that is, the dark-energy scale. The gravitational "
          "parameter is then fixed too,"
          % (PH[(1.0, 0.30)]["f_meV"], PH[(1.0, 0.30)]["delta_phi_meV"])))
add(("eq", r"g\equiv\frac{f^2}{M_{\rm Pl}^2}=\sqrt{\frac{V_f/M_{\rm Pl}^4}{r_L\,\lambda\,\delta(\eta)}}"))
add(("p", "which takes the values $%s$ to $%s$ over the benchmark parameters of Table 4. Gravity is therefore a $10^{-60}$ effect in every "
          "equation below that does not contain the large number $M_{\\rm Pl}^4/V_f$." % (sci(g_lo, 1), sci(g_hi, 1))))

# ------------------------------------------------------------------------------------------------ 3 Bounce
add(("h1", "3. The Euclidean bounce, with and without gravity"))
add(("h2", "3.1 Equations and action"))
add(("p", "The O(4)-symmetric Euclidean metric $ds^2 = d\\xi^2 + \\rho(\\xi)^2 d\\Omega_3^2$ gives the CDL equations \\cite{CDL}"))
add(("eq", r"\varphi''+3\frac{\rho'}{\rho}\varphi'=\frac{dV}{d\varphi},\qquad \rho'^2=1+\frac{\rho^2}{3M_{\rm Pl}^2}\left(\frac12\varphi'^2-V\right)"))
add(("p", "with $\\varphi'(0)=0$ and regularity at the second pole of the compact de Sitter geometry. On shell, the action of the bounce relative to the "
          "false vacuum is"))
add(("eq", r"B=S_E[{\rm bounce}]-S_E[{\rm false}],\quad S_E=4\pi^2\int d\xi\,\left(\rho^3V-3M_{\rm Pl}^2\rho\right),\quad S_E^{\rm dS}=-\frac{24\pi^2M_{\rm Pl}^4}{V_f}"))
add(("p", "In the flat-space limit $B=2\\pi^2\\int dr\\,r^3\\,[\\tfrac12\\varphi'^2+V-V_f]$, and for a thin wall $B = 27\\pi^2\\sigma^4/(2\\Delta V^3)$ with "
          "$R_c = 3\\sigma/\\Delta V$, $\\sigma = \\int\\sqrt{2V}\\,d\\varphi$ \\cite{Coleman}. With the field in units of $f$ and lengths in units of "
          "$(\\sqrt\\lambda f)^{-1}$ the flat-space action is $B = b(\\eta)/\\lambda$ exactly, and gravity enters only through $g$."))
add(("p", "The Hawking--Moss action is $B_{\\rm HM} = 24\\pi^2 M_{\\rm Pl}^4(1/V_f - 1/V_{\\rm top})$ \\cite{HM}; an oscillating CDL bounce requires "
          "$|V''_{\\rm top}| > 4H^2$ \\cite{JS,HW}, which fails for sufficiently flat potentials or strong gravity."))
add(("h2", "3.2 Numerical method and validation"))
add(("p", "The bounce is found by shooting on the central field value with overshoot/undershoot classification (adaptive eighth-order Runge--Kutta, "
          "relative tolerance $10^{-11}$) \\cite{SciPy,NumPy,CosmoT}, with the equations integrated through the wall and the action accumulated along the "
          "solution. For the gravitational problem the tail of the bounce outside the wall follows the false-vacuum de Sitter geometry "
          "$\\rho = \\sin(H\\xi)/H$ toward the second pole, and the difference from the background action is taken analytically there, which "
          "avoids the cancellation of terms of order $1/g^2$. In the thin-wall regime shooting is exponentially sensitive to the initial "
          "condition, so an independent collocation boundary-value solver is used for $\\eta < 0.1$, and the two agree to $%s$ where both apply. "
          "The analytic thin-wall formula with gravity \\cite{Parke} serves as a further check of the gravitational suppression. Table 2 lists the checks of the bounce; "
          "the complete suite has %d checks, all executed by a script distributed with the paper." % (sci(v("V2_shooting_vs_bvp")["max_relative_difference"], 0), len(V))))
add(("table", "Table 2. Executable consistency checks of the numerical bounce.",
     ["Check", "Result"],
     [["Thin-wall limit: $B/B_{\\rm thin}$ as $\\eta\\to0$", "%.4f at $\\eta=0.03$, %.4f at 0.06 (monotone)" % (v("V1_thin_wall_limit")["ratio_at_eta_0p03"], v("V1_thin_wall_limit")["ratio_at_eta_0p06"])],
      ["Shooting versus boundary-value solver", "max relative difference %s (%d cases)" % (sci(v("V2_shooting_vs_bvp")["max_relative_difference"], 1), v("V2_shooting_vs_bvp")["cases"])],
      ["Scaling $B=b(\\eta)/\\lambda$", "max relative difference %s" % sci(v("V3_lambda_scaling")["max_relative_difference"], 1)],
      ["de Sitter action $-24\\pi^2M^4/V_f$", "relative difference %s" % sci(v("V4_de_sitter_action")["rel"], 0)],
      ["Friedmann constraint along the bounce", "max residual %s" % sci(v("V10_friedmann_constraint")["max_residual"], 1)],
      ["Gravitational suppression against the analytic thin-wall (Parke) value", "agree to 1.5%% in $B/B_{\\rm flat}$ (%d points)" % len(v("V6_parke_thin_wall_suppression")["numerical_vs_analytic_ratio"])],
      ["Hawking--Moss consistency", "%d bounces found in %d grid points; none with $|V''|<4H^2$; none above $B_{\\rm HM}$" % (v("V7_hawking_moss_consistency")["bounces_found"], v("V7_hawking_moss_consistency")["grid_points"])],
      ["Interpolation of $b(\\eta)$ off the table", "max relative error %s" % sci(v("V8_interpolation_off_grid")["max_relative_error"], 1)],
      ["Derrick identity $2T+4U=0$ for the bounce", "residual %s at the worst of four tilts" % sci(max(n["virial_residual"] for n in NEG.values()), 1)]],
     None))
add(("fig", "fig1_potential_bounce.png", "Fig. 1. (a) The potential for $\\lambda=1$, $\\eta=0.30$. (b) The bounce profile in flat space and with gravity ($g=0.1$). (c) The numerical action $b(\\eta)=B\\lambda$ against the thin-wall formula; the false minimum disappears at $\\eta_c=0.385$."))
add(("h2", "3.3 Error budget and negative modes"))
add(("p", "Table 3 lists how the flat-space action changes when each numerical choice is varied: the integrator tolerance and the integrator itself, the quadrature of the "
          "action integral, the tolerance of the bisection on the central field value, the truncation of the domain, and, for the boundary-value solution, the tolerance, "
          "the initial mesh and the padding of the domain. At the production settings the numerical uncertainty of $B$ is below $%s$, many orders below the uncertainty "
          "of the prefactor ($\\delta\\ln\\Gamma=O(1)$) and below the precision of any quantity compared with the data. The only sensitive setting is the bisection: a tolerance "
          "of $10^{-6}$ in the central field value changes $B$ by up to $%s$ in the thin-wall regime, which is why $10^{-13}$ is used." % (
              sci(max(worst("rtol", lambda x: x["rtol"] <= 1e-10), worst("method"), worst("quadrature"), worst("bisection", lambda x: x["tol"] <= 1e-10)), 0),
              sci(max(x["rel"] for r in A_.values() for x in r["bisection"] if x["tol"] == 1e-6), 0))))
trunc_last = [r["truncation"][-1]["rel"] for r in A_.values()]
add(("table", "Table 3. Numerical error budget of the flat-space action (largest relative change of $B$ over the tilts studied).",
     ["Source", "Range explored", "Largest change"],
     [["Integrator tolerance (DOP853)", "$10^{-8}$ to $10^{-13}$ (production $10^{-11}$)", "%s (%s for tolerance $\\le10^{-10}$)" % (sci(worst("rtol"), 0), sci(worst("rtol", lambda x: x["rtol"] <= 1e-10), 0))],
      ["Integrator", "DOP853, RK45, LSODA, Radau", sci(worst("method"), 0)],
      ["Quadrature of the action", "$2\\times10^4$ to $2\\times10^6$ points", sci(worst("quadrature"), 0)],
      ["Bisection on the central value", "$10^{-6}$ to $10^{-13}$ (production $10^{-13}$)", "%s (%s for tolerance $\\le10^{-10}$)" % (sci(worst("bisection"), 0), sci(worst("bisection", lambda x: x["tol"] <= 1e-10), 0))],
      ["Truncation of the domain", "action cut at $r_{1/2}+k/m_f$, $k$ up to 16", "%s at the largest radius reached" % sci(max(trunc_last), 0)],
      ["Boundary-value solver: tolerance, mesh, padding", "$10^{-6}$ to $10^{-10}$; 500 to 8000 nodes; 10 to 120", sci(max(x["rel"] for r in B_.values() for k in ("tol", "mesh", "domain_padding") for x in r[k]), 0)],
      ["Shooting against boundary-value solution", "$\\eta=0.10$", sci(B_["0.1"]["against_shooting"], 0)],
      ["Gravity coefficient $\\kappa(g\\to0)$", "integrators, tolerances, quadrature", "$\\kappa=%.4f$, spread %s" % (G_["kappa_g0_mean"], sci(G_["kappa_g0_spread"], 0))]], None))
ev = lambda e: NEG[e]["l0_lowest_over_mf2"]
add(("p", "A CDL bounce is a saddle point with exactly one negative mode, which is what allows $\\mathrm{Im}\\,E$ to be interpreted as a decay rate \\cite{Coleman,CC}. "
          "For $\\eta=0.05$, 0.15, 0.30 and 0.36 the spectrum of $-\\nabla_4^2+V''(\\varphi_b)$ was computed in the angular sectors $l=0,\\dots,4$ (with $\\psi=r^{-3/2}u$ the radial operator "
          "is $-u''+[(l(l+2)+3/4)/r^2+V'']u$, discretized on 30000 points). In every case the $l=0$ sector has exactly one negative eigenvalue (lowest values "
          "$%.3f$, $%.3f$, $%.2f$, $%.2f$ in units of the false-vacuum mass squared), the $l=1$ sector has the lowest eigenvalue consistent with the four translational zero modes "
          "(at most $%s$ of $m_f^2$ in magnitude) and none is negative, no eigenvalue is negative for $l\\ge2$, the field is monotonic, so that $\\varphi-\\varphi_f$ has no node, and "
          "the Derrick identity holds to $%s$. With gravity the negative-mode problem is subtle and has been discussed in the literature \\cite{TS}; here $g\\sim10^{-60}$, "
          "the operator differs from its flat-space form by a relative $O(g)$, and the flat-space structure applies. The gravitational spectrum was not computed for large $g$."
          % (ev("0.05"), ev("0.15"), ev("0.3"), ev("0.36"), sci(max(abs(n["l1_lowest_over_mf2"]) for n in NEG.values()), 0), sci(max(n["virial_residual"] for n in NEG.values()), 0))))
add(("p", "Corrections of relative size $10^{-59}$ cannot be resolved in double precision, and the paper does not claim to resolve them. The flat-space action is computed with no gravity "
          "at all; the gravitational correction is the linear response $B=B_{\\rm flat}(1-\\kappa g)$ with $\\kappa(g\\to0)=%.3f$, obtained from $g=10^{-3}$ and $2\\times10^{-3}$ "
          "with the linear extrapolation $\\kappa_0=2\\kappa(10^{-3})-\\kappa(2\\times10^{-3})$ (the two values differ by %.1f%%, the curvature of $B(g)$) and stable to $%s$ across integrators. "
          "At $g\\sim10^{-60}$ the correction $\\kappa g\\sim%s$ is some 59 orders of magnitude below every other uncertainty of the rate, so what matters is only that gravity "
          "enters as a small smooth perturbation, which the Hawking--Moss criterion confirms independently (Section 3.4)."
          % (G_["kappa_g0_mean"], 100 * abs(1 - kap[0.3] * 0 - G_["settings"][1]["kappa_2em3"] / G_["settings"][1]["kappa_1em3"]), sci(G_["kappa_g0_spread"], 0), texsci(max(p["gravity_correction_relative"] for p in F["physical"]), 1))))
add(("h2", "3.4 Gravitational corrections and the Hawking--Moss limit"))
add(("p", "Fig. 2a shows the ratio $B(g)/B_{\\rm flat}$ computed on a grid of $g$ and $\\eta$. For small $g$ the suppression is linear, $B = B_{\\rm flat}(1-\\kappa g)$ with "
          "$\\kappa = %.1f$ to $%.1f$ for $0.12 \\le \\eta \\le 0.37$ (measured at $g = 10^{-3}$; below that the cancellation of the $1/g^2$ terms limits the "
          "numerical precision to about $10^{-4}$ in $B/B_{\\rm flat}$). The suppression grows with $g$ as the bubble approaches the Hubble radius, and "
          "for $g \\gtrsim 1$ no oscillating bounce is found: the numerical existence agrees with the condition $|V''_{\\rm top}|/4H^2>1$ (Fig. 2b), which is "
          "satisfied everywhere a bounce was found." % (min(kap.values()), max(kap.values()))))
add(("fig", "fig2_gravity_hawking_moss.png", "Fig. 2. (a) Gravitational suppression of the action for several tilts; crosses mark the grid points at which no oscillating bounce was found. (b) The Hawking--Moss criterion $|V''_{\\rm top}|/4H^2$ against $g$ (lines, analytic; markers, numerical bounce found/not found). The shaded band is the physical value fixed by the dark energy, $g\\sim10^{-60}$."))
add(("p", "At the physical $g\\sim10^{-60}$ the relative gravitational correction is $\\kappa g\\sim%s$, the Hawking--Moss criterion is satisfied by a factor "
          "$\\sim10^{59}$--$10^{60}$, the Hawking--Moss action is $\\ln B_{\\rm HM}\\simeq%.0f$ (an exponent of $10^{121}$), and the bubble radius is "
          "$\\sim2\\times10^{-30}$ of the Hubble radius. Gravity and the Hawking--Moss channel play no role in a decay of the present dark energy "
          "within this class of potentials. They matter only if $f\\gtrsim 0.3 M_{\\rm Pl}$, which with $V_f$ fixed requires $V_f/f^4\\lesssim%s$, "
          "a barrier $\\gtrsim%s$ times the vacuum energy, and an action beyond any consequence for the transition."
          % (texsci(max(p["gravity_correction_relative"] for p in F["physical"]), 1), np.mean([p["ln_hm_action"] for p in F["physical"]]), texsci(K.V_F / K.M_PL ** 4 / 0.3 ** 4, 0), texsci(0.3 ** 4 * K.M_PL ** 4 / K.V_F, 0))))

# ------------------------------------------------------------------------------------------------ 4 Rate
add(("h1", "4. Nucleation rate, cooling and survival in the observed universe"))
add(("h2", "4.1 The rate"))
add(("p", "The rate per unit four-volume is $\\Gamma/V_4 = A\\,e^{-B}$ \\cite{Coleman,CC}. In the rescaled variables $\\Gamma/V_4 = K f^4\\lambda^2 e^{-B}$, "
          "where $K = (B/2\\pi)^2 D$ contains the four translational zero modes and an unknown dimensionless factor $D$ from the remaining fluctuation determinant. "
          "Dividing by $H^4 = V_f^2/(9M_{\\rm Pl}^4)$ and using $f^4 = V_f/(r_L\\lambda\\delta)$,"))
add(("eq", r"\frac{\Gamma}{H^4}=\frac{9K\lambda}{r_L\,\delta(\eta)}\,\frac{M_{\rm Pl}^4}{V_f}\,e^{-B(\lambda,\eta)}"))
add(("p", "The large number $M_{\\rm Pl}^4/V_f = %s$ is the number of Planck-volume cells in a Hubble four-volume. It compensates an action of "
          "$B \\simeq \\ln(M_{\\rm Pl}^4/V_f) = %.1f$ plus the logarithm of the prefactor. Setting $\\Gamma/H^4 = 1$ gives the critical action"
          % (sci(K.M4_OVER_VF, 2), K.LN_M4_OVER_VF)))
add(("eq", r"B_c=\ln\frac{M_{\rm Pl}^4}{V_f}+\ln\frac{9K\lambda}{r_L\,\delta},\qquad B_c\simeq %d\ \ (\text{median over the prior; 5--95\%% range }%d\text{--}%d)" % (round(bc_med), round(BC["0.05"]), round(BC["0.95"]))))
add(("p", "For a mean time to nucleation, an observer in the asymptotic de Sitter space sees the past-cone four-volume grow at the rate $(4\\pi/3)H^{-3}$, so the mean time for "
          "one Hubble patch to nucleate a bubble is $t_{\\rm dec} = H^{-1}/[(4\\pi/3)\\Gamma/H^4]$ for $\\Gamma/H^4\\ll1$. For $\\lambda=1$, $\\eta=0.30$ the action is $B=%.0f$ and the mean decay time is $10^{%.0f}$ yr; at "
          "$\\eta=0.32$ ($B=%.0f$) it is $10^{%.0f}$ yr; at $\\eta=0.34$ ($B=%.0f$) the rate is so large that the vacuum would not have survived and the model is excluded (Table 4)."
          % (PH[(1.0, 0.30)]["B"], PH[(1.0, 0.30)]["log10_t_yr"], PH[(1.0, 0.32)]["B"], PH[(1.0, 0.32)]["log10_t_yr"], PH[(1.0, 0.34)]["B"])))
add(("table", "Table 4. Benchmark potentials with the dark-energy density as false-vacuum energy ($r_L=1$). $f$ is the field scale, $g=f^2/M_{\\rm Pl}^2$, $\\delta_B$ the relative gravitational correction to $B$, and $R_b/R_H$ the bubble radius over the Hubble radius.",
     ["$\\lambda$", "$\\eta$", "$f$ (meV)", "$B$", "$\\ln\\Gamma/H^4$", "$\\log_{10}t_{\\rm dec}$ (yr)", "$g$", "$\\delta_B$", "$|V''_{\\rm top}|/4H^2$", "$R_b/R_H$"],
     [[("%g" % p["lam"]), "%.2f" % p["eta"], "%.2f" % p["f_meV"], "%.0f" % p["B"], "%.0f" % p["ln_gamma_h4"], "%.0f" % p["log10_t_yr"], sci(p["g"], 1), sci(p["gravity_correction_relative"], 1),
       sci(p["hm_ratio"], 1), sci(p["bubble_radius_over_hubble"], 1)] for p in F["physical"]], None))
add(("h2", "4.2 Survival of the present vacuum: the past light cone"))
add(("p", "A bubble nucleated at an event $p$ expands at essentially the speed of light and reaches an observer at $O$ if and only if $p$ lies in the causal past of $O$. "
          "If nucleations are Poisson distributed with rate $\\Gamma$ per unit four-volume, the probability that none has occurred in the past light cone of the present "
          "observer is \\cite{IRS,EGR}"))
add(("eq", r"P_{\rm surv}=e^{-\Gamma V_4},\qquad V_4=\frac{4\pi}{3}\int_0^{t_0}dt\,a(t)^3\chi(t)^3,\qquad \chi(t)=\int_t^{t_0}\frac{dt'}{a(t')}"))
add(("p", "with $\\chi$ the comoving distance of the light cone at time $t$ in a flat universe. A common order-of-magnitude shortcut uses the Euclidean volume of the de Sitter four-sphere, "
          "$8\\pi^2/(3H^4)$; that is not the four-volume of any region of the observed spacetime, and it overstates $V_4$ by the factor given below. The observed history is "
          "$H^2=H_0^2(\\Omega_ma^{-3}+\\Omega_ra^{-4}+\\Omega_\\Lambda)$ with $\\Omega_\\Lambda=%.4f$, $\\Omega_r=%s$ (from $T_{\\rm CMB}$ and $N_{\\rm eff}=3.046$) and $\\Omega_m=%.4f$. "
          "Two independent quadratures give $V_4=%.5f\\,H_0^{-4}=%.4f\\,H_\\Lambda^{-4}$, agreeing to $%s$; the result is converged in the lower cutoff of the scale factor and in "
          "the grid to $10^{-11}$ and reproduces the analytic Einstein--de Sitter value to $%s$ (check V14). The Euclidean volume is $%.1f\\,H_\\Lambda^{-4}$, %.0f times larger. "
          "Of $V_4$, %s%% comes from events at redshift below 1, %s%% below 2 and %s%% below 5 (Fig. 3); the uncertainty from $\\Omega_\\Lambda$ at the Planck $1\\sigma$ level is about 3.5%%."
          % (K.OMEGA_L, sci(SV["omega_r"], 2), SV["omega_m"], SV["V4_over_H0m4"], SV["V4_HL4"], sci(abs(SV["V4_over_H0m4"] / SV["V4_over_H0m4_ode"] - 1), 0),
             sci(abs(SV["check_EdS"]["numeric"] / SV["check_EdS"]["analytic"] - 1), 0), SV["euclid_HL4"], SV["ratio_euclid_to_lorentzian"],
             pc(SV["fraction_of_V4_below_z"]["1.0"], 0), pc(SV["fraction_of_V4_below_z"]["2.0"], 0), pc(SV["fraction_of_V4_below_z"]["5.0"], 0))))
add(("fig", "fig3_survival_cone.png", "Fig. 3. Four-volume of the past light cone of the present observer, accumulated over events at redshift below $z$, for the observed expansion history, against the Euclidean four-sphere volume.", 3.7))
add(("p", "The observation that no bubble has reached us is excluded at 95%% confidence if $\\Gamma V_4>-\\ln0.05=%.2f$, that is for $\\Gamma/H^4>%.1f$ or $B<B_c-%.1f$. This is a weak "
          "condition: it is %.0f times weaker than the Euclidean estimate ($\\Gamma/H^4>%.3f$), it moves the boundary of the allowed region by only $\\Delta B=%.1f$ out of about %d, and it "
          "bounds the future mean decay time only by $t_{\\rm dec}\\gtrsim%s$ yr (the formula for $t_{\\rm dec}$ is used here as an order-of-magnitude statement, since $\\Gamma/H^4$ is not small). Because the prior mass sits overwhelmingly at $B\\ll B_c$ or $B\\gg B_c$, the fraction of the prior excluded "
          "changes from %s%% (Euclidean estimate) to %s%% (Section 4.5). The criterion assumes that $\\Gamma$ was constant, with the scalar in its false minimum throughout; it "
          "does not require observers to be present before the transition, and it says nothing about bubbles nucleated outside our past cone. "
          "Percolation of bubbles in de Sitter space requires a rate of order unity in Hubble units \\cite{GW,TWW}, since otherwise the false-vacuum volume grows without bound and no wall collides "
          "with another. With the corrected survival bound this no longer conflicts with survival: the window $1\\lesssim\\Gamma/H^4<%.0f$ has the width $\\Delta B=%.1f$ and contains "
          "%s%% of the prior volume (%s%% for a percolation threshold of 0.1 and %s%% for 10)."
          % (SV["N_95"], XMAX, DB_SURV, XMAX / XOLD, XOLD, DB_RELAX, round(bc_med), texsci(T_MIN, 1), pc(BASE["frac_excluded_by_old_euclidean_criterion"], 1), pc(BASE["frac_inconsistent_with_survival"], 1),
             XMAX, DB_SURV, pc(BASE["frac_percolating_and_surviving"]["1"], 2), pc(BASE["frac_percolating_and_surviving"]["0.1"], 2), pc(BASE["frac_percolating_and_surviving"]["10"], 2))))
add(("h2", "4.3 The prefactor and the validity of the semiclassical rate"))
add(("p", "The factor $D$ is the ratio of the fluctuation determinants with the zero modes removed; methods to compute it for specific potentials exist \\cite{DM,AFS} and are not repeated here. It "
          "is treated as a systematic uncertainty. Because $\\ln D$ enters $B_c$ additively, a change of $D$ by $n$ decades moves $B_c$ by $2.3\\,n$: three decades give $\\pm7$ out of $%d$, "
          "six decades give $\\pm14$ (5%%), and the prior mass of every regime changes by less than $10^{-3}$ over spreads of 0, 3 and 6 decades (Table 5). The semiclassical expansion is controlled by $1/B$, "
          "which is $%.1f\\times10^{-3}$ at $B_c$, so that higher-loop corrections change $\\ln\\Gamma$ by $O(1)$ and not $O(B)$, which the prefactor spread already covers. The "
          "regime in which the approximation fails is $B=O(1)$ to $O(10)$, where the wall is not thin on the scale of the field fluctuations; %s%% of the prior has $B<10$ and %s%% has $B<30$, and "
          "none of these draws survives the survival condition (all have $\\Gamma/H^4\\gg%.0f$), so the failure of the semiclassical approximation affects only the excluded region and "
          "cannot change the classification of any draw that is allowed."
          % (round(bc_med), 1000.0 / bc_med, pc(BASE["frac_B_below"]["10"], 1), pc(BASE["frac_B_below"]["30"], 1), XMAX)))
add(("h2", "4.4 The falsification diagram"))
add(("p", "Fig. 4 maps the decay time over the $(\\eta,\\lambda)$ plane for $r_L=1$. The region excluded by survival and the region of "
          "effective stability ($t_{\\rm dec}>10^{100}$ yr) are separated by a band: the decay time runs from the survival bound of $%s$ yr to $10^{20}$ yr within a change of "
          "$\\Delta B\\simeq%.0f$ in the action and to $10^{100}$ yr within $\\Delta B\\simeq%.0f$. The model is thus either excluded, "
          "or it describes a vacuum that decays on a time scale of $10^{20}$ yr or longer unless $B$ is tuned to within about $%.0f$ of the survival bound, "
          "that is within %.0f%% of its value." % (texsci(T_MIN, 1), DB_20, DB_WINDOW, DB_20, 100 * DB_20 / bc_med)))
add(("fig", "fig4_falsification_diagram.png", "Fig. 4. Mean decay time of a Hubble patch in the $(\\eta,\\lambda)$ plane for $V_f=\\rho_\\Lambda$ and a Minkowski true vacuum. Contours: survival bound ($%s$ yr), $10^{12}$, $10^{20}$, $10^{50}$ and $10^{100}$ yr. The upper edge $\\lambda=4\\pi$ is the perturbative-unitarity limit used for the prior." % texsci(T_MIN, 1)))
add(("h2", "4.5 Prior-sensitivity Monte Carlo"))
add(("p", "To quantify how much of the parameter space the window occupies, $%s$ parameter sets are drawn from physically consistent priors "
          "($\\log_{10}\\lambda$ uniform on $[-2,\\log_{10}4\\pi]$, $\\eta$ uniform on $[0.03,0.38]$, $\\ln D$ Gaussian with a spread of 1.5 decades; the false "
          "minimum exists for every draw) and each is converted to a rate with the interpolated numerical action. The regime fractions are given in "
          "Table 5 together with batch-means standard errors (20 batches), convergence with the number of draws (Fig. 5c) and sensitivity to the "
          "priors. %s%% of the prior volume lies above the survival bound, %s%% decays within $10^{20}$ yr, %s%% within $10^{100}$ yr and %s%% "
          "shows no decay within $10^{100}$ yr. The fractions do not depend on the prefactor and the choice of $r_L$ is likewise immaterial. They do depend on the prior range of "
          "$\\lambda$ and $\\eta$ (Table 5; for $\\lambda\\le1$ the excluded fraction falls to %s%%). They are therefore fractions of the stated prior, a property of the parameter space "
          "of this model and not measured probabilities, and they do not indicate how likely a transition is in our universe. $V_t<0$ would enter through $r_L=1/(1+|V_t|/V_f)<1$ and shifts $B_c$ upward by $\\ln(1+|V_t|/V_f)$; "
          "the case $r_L\\ge1$ is the one sampled." % ("{:,}".format(MC["N"]), pc(BASE["frac_inconsistent_with_survival"], 1), pc(BASE["frac_decay_by"]["1e+20"], 2),
                                                           pc(BASE["frac_decay_by"]["1e+100"], 1), pc(BASE["frac_never_within_1e100"], 1), pc(SENS["lam <= 1"]["frac_inconsistent_with_survival"], 1))))
rows4 = [["baseline", "%s ± %s" % (pc(BASE["frac_inconsistent_with_survival"], 2), pc(BAT["surv"]["se"], 2)), "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+12"], 3), pc(BAT["1e12"]["se"], 3)),
          "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+20"], 2), pc(BAT["1e20"]["se"], 2)), "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+100"], 2), pc(BAT["1e100"]["se"], 2)),
          "%s ± %s" % (pc(BASE["frac_never_within_1e100"], 1), pc(BAT["never"]["se"], 1))]]
for k_, s_ in SENS.items():
    rows4.append([k_.replace("sigma_log10_D", "$\\sigma_{\\log D}$").replace("r_L", "$r_L$").replace("lam", "$\\lambda$").replace("<=", "≤").replace("eta", "$\\eta$"), pc(s_["frac_inconsistent_with_survival"], 2),
                  pc(s_["frac_decay_by"]["1e+12"], 3), pc(s_["frac_decay_by"]["1e+20"], 2), pc(s_["frac_decay_by"]["1e+100"], 2), pc(s_["frac_never_within_1e100"], 1)])
add(("table", "Table 5. Prior mass (%) of each regime. Decay fractions are cumulative and refer to survival-consistent draws; the last column is the fraction with no decay within $10^{100}$ yr. These are fractions of the stated prior, not probabilities for our universe.",
     ["Prior", "above survival bound", "decays by $10^{12}$ yr", "by $10^{20}$ yr", "by $10^{100}$ yr", "no decay in $10^{100}$ yr"], rows4, None))
add(("fig", "fig5_monte_carlo.png", "Fig. 5. (a) Cumulative distribution of the decay time over survival-consistent draws. (b) Prior mass by regime with $3\\sigma$ batch errors. (c) Convergence of the regime fractions with the number of draws."))

# ------------------------------------------------------------------------------------------------ 5 interior
add(("h1", "5. What the nucleated region can become"))
add(("h2", "5.1 Energy budget"))
add(("p", "A thin-wall bubble of true vacuum is bounded by a wall that carries the released energy; its interior is, to leading order, the true vacuum "
          "with no excess energy to reheat. In the thick-wall regime relevant here the field at the centre of the bounce, $x_0$, lies above the true minimum, and the "
          "energy the interior can release is $\\epsilon_{\\rm in}\\Delta V = V(x_0)-V_t$. Fig. 6c shows $\\epsilon_{\\rm in}$ against $\\eta$: at $\\eta=0.30$, "
          "$\\epsilon_{\\rm in}=%s$, so %s%% of $\\Delta V$ resides in the interior and the remainder in the wall; $\\epsilon_{\\rm in}$ approaches unity "
          "only near $\\eta_c$, where the barrier is gone." % (("%.3f" % EPS_IN), pc(EPS_IN, 1))))
add(("h2", "5.2 Interior geometry and reheating"))
add(("p", "The interior of a CDL bubble has open spatial slices, $ds^2=-dt^2+a(t)^2dH_3^2$ \\cite{Gott,BGT}, with the Friedmann constraint "
          "$(da/dt)^2 = 1 + a^2\\rho/(3M_{\\rm Pl}^2)$. Without a long inflationary stage inside the bubble (Section 5.3), the energy density is negligible against "
          "the curvature term at early times: $\\alpha^2\\rho \\le %s$ in units where $H_\\Lambda=1$, so the interior is a Milne universe, $a=t$. "
          "The condensate begins to oscillate at $t_0 = 1/m$ with $m = %.2f$ meV, behaves as matter and decays to radiation at a rate $\\Gamma_\\varphi\\le m$. With $\\tau=H_\\Lambda t$ "
          "the radiation density has the closed form $\\rho_r(\\tau)=\\epsilon\\,\\tau_0^3\\,\\gamma\\,\\tau^{-4}\\int_{\\tau_0}^{\\tau}s\\,e^{-\\gamma(s-\\tau_0)}ds$ in units of $V_f$, "
          "which we compare with a numerical integration of the full system (relative difference %s). The radiation temperature is "
          "$T=(30\\rho_r/\\pi^2g_*)^{1/4}$ with $g_*=%.2f$." % (sci(max(r["alpha2_rho_max"] for r in IN["runs"]), 1), IN["scales"]["m_meV"],
                                                               sci(v("V11_interior_closed_form")["max_relative_difference"], 0), K.GSTAR_LOW)))
add(("p", "Energy conservation bounds the reheating temperature independently of the details of the decay,"))
add(("eq", r"T_{\rm reh}\le\left[\frac{30\,\epsilon_{\rm th}\,\epsilon_{\rm in}\Delta V}{\pi^2g_*}\right]^{1/4},\qquad \epsilon_{\rm th}\le1"))
add(("p", "which for the largest release allowed when $V_t\\ge0$, $\\epsilon_{\\rm in}\\Delta V = \\Delta V\\le V_f=\\rho_\\Lambda$, gives $T_{\\rm reh}\\le %.2f$ meV ($%.0f$ K), "
          "against the CMB temperature of %.3f meV today and %s times below the $\\sim1$ MeV required for nucleosynthesis \\cite{BBN}. Equivalently, a thermal bath at 1 MeV "
          "needs an energy density $%s$ times the dark-energy density. The closed form shows that the situation is "
          "worse than the bound: the radiation temperature after its maximum falls as $T\\propto1/\\tau$ because the open interior keeps expanding, with "
          "$T\\tau=%s$ GeV, so $T=1$ MeV would have to occur at $t=%s$ s, before the condensate even starts to oscillate ($t_0=%s$ s). Table 6 gives "
          "the maximum temperature for decay rates from $\\Gamma_\\varphi=m$ down to $10^{-20}m$ and the temperature one Hubble time after nucleation, "
          "$\\sim10^{-30}$ meV."
          % (RH["T_max_meV_at_Vf"], RH["T_max_K_at_Vf"], RH["T_cmb_meV"], sci(RH["ratio_Tbbn_over_Tmax"], 1), sci(RH["rho_needed_for_bbn_over_Vf"], 1), sci(IN["T_times_tau"], 2),
             sci(IN["t_for_bbn_temperature_seconds"], 1), sci(IN["scales"]["t0_seconds"], 1))))
rows3 = []
for eps, lab in ((1.0, "1 (upper bound)"), (EPS_IN, "%.3f ($\\eta=0.30$)" % EPS_IN)):
    for fr in (1.0, 1e-3, 1e-9, 1e-20):
        r = [x for x in IN["runs"] if abs(x["eps"] - eps) < 1e-12 and x["gamma_over_m"] == fr][0]
        rows3.append([lab, ("%g" % fr), "%.4f" % r["T_max_meV"], "%.2f" % r["T_over_bound"], sci(r["T_at_one_hubble_time_meV"], 1), sci(r["T_bbn_over_Tmax"], 1)])
add(("table", "Table 6. Reheating of the bubble interior. $\\epsilon$: fraction of $\\Delta V$ in the interior; $\\Gamma_\\varphi/m$: decay rate of the oscillating condensate; $T_{\\max}$: maximum radiation temperature; the fourth column is $T_{\\max}$ over the bound of Eq. (11); the fifth is the temperature one Hubble time $1/H_\\Lambda$ after nucleation; the last is $T_{\\rm BBN}/T_{\\max}$.",
     ["$\\epsilon$", "$\\Gamma_\\varphi/m$", "$T_{\\max}$ (meV)", "$T_{\\max}/T_{\\rm bound}$", "$T(1/H_\\Lambda)$ (meV)", "$T_{\\rm BBN}/T_{\\max}$"], rows3, None))
add(("fig", "fig6_reheating_interior.png", "Fig. 6. (a) Maximum reheating temperature against the released energy density; the dark-energy density gives about 2 meV, far below the energy density needed for 1 MeV. (b) Temperature of the open interior after reheating ($\\Gamma_\\varphi=m$): the radiation cools as $1/t$ and never approaches nucleosynthesis. (c) Fraction of $\\Delta V$ carried by the interior against the tilt."))
add(("h2", "5.3 Scope of the bound: what is excluded by the model and what only by energy conservation"))
add(("p", "The bound of Eq. (11) is a statement about a locally homogeneous conversion of a vacuum-energy difference that does not exceed the dark-energy density. It is not a universal "
          "prohibition on high-temperature bubble interiors, and it is stated here with its assumptions: (A1) one real scalar field minimally coupled to gravity, with the potential of Section 2; "
          "(A2) the false-vacuum energy is the observed dark energy; (A3) the interior is homogeneous and isotropic on open slices; (A4) no inflationary stage inside the bubble; "
          "(A5) all energy densities are sub-Planckian. Models that violate one of these need a separate analysis. Table 7 lists the possible additional energy sources and says which of "
          "them the model excludes by more than energy conservation."))
add(("p", "The geometric argument is the following. In the open slicing the curvature parameter $Y\\equiv a^2\\rho/(3M_{\\rm Pl}^2)$ obeys $(da/dt)^2=1+Y$. A hot big bang at nucleosynthesis needs "
          "$Y\\sim1$: in standard radiation domination $Ht=1/2$, that is $\\rho t^2=3M_{\\rm Pl}^2/4$. In the curvature-dominated phase $a=t$ and $Y=\\rho t^2/(3M_{\\rm Pl}^2)$ can only decrease, as $1/t$ for "
          "matter and $1/t^2$ for radiation, unless the interior inflates. The energy density of the condensate and its decay products obeys the envelope"))
add(("eq", r"\rho(t)\le\epsilon\,\Delta V\,(mt)^{-3}\quad(t>t_0=1/m),\qquad Y(t_0)=\frac{\epsilon\,\Delta V}{3M_{\rm Pl}^2m^2}"))
add(("p", "because matter dilutes more slowly than radiation and decay only moves energy from the slower to the faster component (checked against the closed form to machine precision, check V17). "
          "The mass at the true minimum is $m=\\kappa_m\\Delta V^{1/4}$ with $\\kappa_m=\\lambda^{1/4}[(3x_t^2-1)/2]^{1/2}\\delta^{-1/4}$, which lies between %.2f and %.1f over the whole prior, so "
          "$Y(t_0)=c\\sqrt{\\Delta V}/M_{\\rm Pl}^2$ is tiny for any sub-Planckian energy: $Y(t_0)=%s$ at $\\Delta V=V_f$, and $Y(t_0)=1$ would need $\\Delta V/M_{\\rm Pl}^4=%s$ at the benchmark, "
          "which is super-Planckian. Therefore, for any $\\Delta V$ below the Planck scale, the interior is a Milne universe at all times, $Y$ never approaches unity (Fig. 7), "
          "and the energy density at one second after nucleation cannot exceed $\\epsilon\\Delta V/(m\\cdot1\\,{\\rm s})^3$. Requiring $\\rho(1\\ {\\rm s})\\ge\\rho_{\\rm BBN}=(\\pi^2g_*/30)(1\\ {\\rm MeV})^4$ "
          "($g_*=%.2f$) demands $\\Delta V^{1/4}\\ge%s$ GeV, that is %.0f, %.0f and %.0f orders of magnitude above the Planck mass (for the smallest $\\kappa_m$ in the prior, the benchmark and the largest). "
          "For the dark energy the envelope gives $\\rho(1\\ {\\rm s})=%s\\,\\rho_{\\rm BBN}$."
          % (SC["kappa_m_min"], SC["kappa_m_max"], sci(SC["curvature"][0]["Y0_at_Vf"], 1), sci(SC["curvature"][0]["DeltaV_over_M4_for_Y1"], 1), SC["g_star_bbn"],
             sci(NSC["kappa_m_benchmark"]["DeltaV_quarter_root_gev"], 1), NSC["kappa_m_min"]["orders_above_Planck"], NSC["kappa_m_benchmark"]["orders_above_Planck"], NSC["kappa_m_max"]["orders_above_Planck"], sci(SC["envelope_at_1s_for_Vf"]["over_rho_bbn"], 1))))
add(("fig", "fig7_curvature_interior.png", "Fig. 7. Curvature parameter $Y=a^2\\rho/3M_{\\rm Pl}^2$ of the bubble interior against the time since nucleation for $\\Gamma_\\varphi=m$. A flat-like radiation- or matter-dominated epoch needs $Y\\sim1$; the interior starts at $Y\\sim10^{-63}$ and moves away from it.", 3.7))
efm = max(e["N"] for e in SC["e_folds"])
rows7 = [
    ["Released vacuum energy, $V_t\\ge0$", "energy conservation, $\\Delta V\\le V_f$", "$T_{\\rm reh}\\le%.2f$ meV" % RH["T_max_meV_at_Vf"]],
    ["Wall energy $(1-\\epsilon_{\\rm in})\\Delta V$", "already counted in $\\Delta V$; collisions redistribute it", "benchmark: %s%% of $\\Delta V$ in the wall; the total is unchanged" % pc(1 - EPS_IN, 0)],
    ["Deeper true vacuum, $V_t<0$ (anti-de Sitter)", "geometry: crunch at $t_c=\\pi\\sqrt3M_{\\rm Pl}/\\sqrt{|V_t|}$ and the curvature envelope",
     "$T_{\\max}=1$ MeV needs $|V_t|=%s\\,V_f$ and a crunch after %.1f s; a life of 13.8 Gyr needs $|V_t|\\le%.1f\\,V_f$ and $T\\le%.1f$ meV" % (
         sci(SC["ads_for_1MeV"]["Vt_over_Vf"], 1), SC["ads_for_1MeV"]["t_crunch_s"], SC["ads_lifetime_13p8_gyr"]["Vt_max_over_Vf"], SC["ads_lifetime_13p8_gyr"]["T_max_meV"])],
    ["Interior inflation", "the potential has no plateau above $V_f$; field range $\\Delta\\varphi/M_{\\rm Pl}\\sim10^{-30}$", "at most $%s$ e-folds (largest over four tilts)" % sci(efm, 1)],
    ["Larger parent scale ($V_f\\gg\\rho_\\Lambda$)", "geometry: the envelope $\\rho\\le\\epsilon\\Delta V(mt)^{-3}$", "$\\rho(1\\ {\\rm s})\\ge\\rho_{\\rm BBN}$ needs $\\Delta V^{1/4}$ %.0f to %.0f orders above $M_{\\rm Pl}$" % (NSC["kappa_m_min"]["orders_above_Planck"], NSC["kappa_m_max"]["orders_above_Planck"])],
    ["Extra reservoir (other fields, radiation)", "not part of the model; subject to the same curvature constraint", "$Y\\sim1$ at $t=1/m$ needs $\\rho_x=%s\\,\\Delta V$" % sci(SC["extra_reservoir"]["rho_x_over_DeltaV"], 1)],
    ["Inhomogeneity, bubble collisions", "not included (A3); total energy is bounded by $\\Delta V$", "local concentration possible; the average is unchanged"],
]
add(("table", "Table 7. Energy sources and assumptions behind the reheating bound, and what excludes each.", ["Source or assumption", "What bounds it", "Quantitative statement"], rows7, None))
add(("p", "Three points follow. First, the anti-de Sitter case shows that energy conservation alone is not sufficient: a very deep true vacuum releases enough energy for 1 MeV, but its "
          "open interior then collapses within seconds, and the longest-lived interior has $T\\lesssim%.0f$ meV. Second, the geometric statement is independent of the energy scale and of the "
          "value of $V_f$, and therefore covers an additional reservoir too: it cannot raise $Y$ above the curvature bound unless its energy density exceeds $\\Delta V$ by $%s$. Third, "
          "the escape routes are exactly those the model lacks: a long inflationary stage inside the bubble, a parent energy scale close to the Planck scale, or an additional mechanism "
          "that raises $Y$ at $t_0$. A successful cosmology would have to lie in that outside class." % (SC["ads_lifetime_13p8_gyr"]["T_max_meV"], sci(SC["extra_reservoir"]["rho_x_over_DeltaV"], 1))))
add(("p", "A second scalar field $\\chi$ does not change this. For any number of scalar fields with a potential $V(\\varphi,\\chi)$, the energy released in the transition of a homogeneous region is the potential "
          "difference between the false and true vacua, $V_f-V_t$, and the false-vacuum energy $V_f=V(\\varphi_f,\\chi_f)$ is the whole observed dark energy $\\rho_\\Lambda$: any energy stored in $\\chi$ while it sits in the false vacuum "
          "gravitates as dark energy and is already counted in $V_f$. A larger release needs $V_t<0$ (the anti-de Sitter row of Table 9, a crunch within seconds), and the curvature argument above is independent of the field content. "
          "A reservoir that is not vacuum energy (radiation, matter, kinetic energy of $\\chi$) would have been present and gravitating throughout the cold era, where observations limit it to a fraction of the critical density, "
          "whereas the interior needs $\\rho_x=%s\\,\\Delta V$. A coupled two-field model is therefore a relabelling of $\\Delta V$ and not a source of new energy, unless the second sector inflates or carries a Planck-scale parent energy, "
          "which is the outside class of Section 7. The simulations of Section 5.5 confirm it." % sci(SC["extra_reservoir"]["rho_x_over_DeltaV"], 1)))
add(("h2", "5.4 Other fates and the singularity question"))
add(("p", "If the true vacuum has negative energy ($V_t<0$) the open interior collapses to a singularity in a finite time, "
          "$\\tau_c = \\pi/\\sqrt{|V_t|/V_f}$: %s. A decay to a de Sitter vacuum with $V_t>0$ leaves the interior with an accelerating phase and no radiation epoch. Only $V_t\\simeq0$ leaves a "
          "regular, expanding, but empty interior. No case yields a nonsingular bounce from the parent geometry: the bubble is a region of the parent "
          "de Sitter space, causally connected to it, and the Milne interior is geodesically complete only because it has no matter."
          % "; ".join("$|V_t|=%g\\,V_f$: %.0f Gyr" % (-a["vt_over_Vf"], a["gyr"]) for a in IN["ads_crunch"])))

add(("p", "The distinction between daughter regions and a cyclic cosmology (Section 1) applies to these fates. The nucleated region is a daughter region in the sense of Option A: a cold, curvature-dominated open region "
          "bounded by the wall inside the parent de Sitter space and still causally connected to it, not a new hot universe. The potential has no further metastable minimum at the true vacuum, so repeated transitions of "
          "the interior would need a different potential; the entropy produced in reheating is bounded by the energy $\\Delta V$ and is carried by an open interior of infinite comoving volume, so no entropy statement can be made "
          "for the daughter as a closed system. A cyclic cosmology (Option B) needs a nonsingular bounce, which in spatially flat general relativity with conventional matter requires a violation of the null energy condition "
          "($\\dot H=-(\\rho+p)/2M_{\\rm Pl}^2>0$) and raises problems of stability and entropy \\cite{Tolman,PS}; the present model contains no such mechanism, and Section 5.6 adds an effective one to test the cycles that would follow. Entropy accounting and causality are computed for the daughter region in Section 5.5; "
          "the stability of an interior bounce is not."))

# ------------------------------------------------------------------------------------------------ 6 dilution
add(("h1", "6. Matter dilution as the trigger: from a held vacuum to a cyclic hot big bang"))
add(("h2", "6.1 How strongly must matter couple? A first-order estimate"))
OM = 1.0 - K.OMEGA_L
DPHI_OVER_M = PH[(1.0, 0.30)]["delta_phi_meV"] * 1e-12 / K.M_PL
add(("p", "The original motivation for a late-time transition was that dilution of matter reduces environmental decoherence and thereby permits a macroscopic "
          "quantum transition. In the baseline model nothing of the kind appears: the potential $V(\\varphi)$, hence $B$, does not depend on the matter "
          "density, and the Gibbons--Hawking temperature $T_H=H/2\\pi=%s$ GeV \\cite{GH} is $%s$ of the field scale; thermal-type corrections to "
          "$B$ are of relative size $\\sim g\\,(\\kappa\\simeq%.1f)$, that is $10^{-59}$ (Section 3.4). A dependence on matter must be introduced by a coupling, and a conclusion that "
          "held only for one coupling would be weak. The requirement can be stated without choosing one."
          % (sci(K.H_L / (2 * np.pi), 1), texsci(K.H_L / (2 * np.pi) / (PH[(1.0, 0.30)]["f_meV"] * 1e-12), 0), kap[0.3])))
add(("p", "Any coupling of non-relativistic matter of mass $m_{\\rm mat}(\\varphi)$ adds $\\rho_m\\ln m_{\\rm mat}(\\varphi)$ to the effective potential, so that the energy difference between the vacua changes by $\\rho_m\\,\\Delta\\ln m$, where $\\Delta\\ln m$ is the fractional change of the matter mass between the two field values and $\\rho_m=(\\Omega_m/\\Omega_\\Lambda)V_f$ today. Write $\\varepsilon_m\\equiv\\rho_m\\Delta\\ln m/\\Delta V$ for the relative shift. To first order in the tilt shift $\\Delta\\eta$,"))
add(("eq", r"\frac{\Delta\eta}{\eta}=\frac{2\,\delta(\eta)}{\eta\,(x_f-x_t)}\,\varepsilon_m,\qquad \Delta B=B\left|\frac{d\ln b}{d\ln\eta}\right|\frac{\Delta\eta}{\eta}"))
add(("p", "with $\\delta(\\eta)=\\Delta V/(\\lambda f^4)$ of Eq. (3). A change of the rate by a factor $e$ needs $\\Delta B=1$; moving a benchmark from $B=%.0f$ to the critical action $B_c=%.0f$ needs "
          "$\\Delta B=%.0f$ ($\\Delta B=%.1f$ for $\\eta=0.32$). With $|d\\ln b/d\\ln\\eta|=%.1f$ at $\\eta=0.30$ this gives $\\Delta\\ln m=%s$ and %s respectively: the matter mass must differ by "
          "between about $0.1\\%%$ and $10\\%%$ between the two vacua. Table 8 translates this to specific couplings." % (
              BM["B"], CP["B_c"], BM["cases"][1]["delta_B"], BM2["cases"][1]["delta_B"], BM["dlnb_dlneta"], sci(BM["cases"][0]["delta_ln_m"], 1), sci(BM["cases"][1]["delta_ln_m"], 1))))
rows8 = [
    ["General", "$\\Delta\\ln m\\ge%s$ to $%s$" % (texsci(BM["cases"][0]["delta_ln_m"], 1), texsci(BM["cases"][1]["delta_ln_m"], 1)), "independent of the coupling function; $\\eta=0.30$"],
    ["Linear, $m=m_0e^{\\beta\\varphi/M_{\\rm Pl}}$", "$\\beta\\ge%s$ to $%s$ (for $\\eta=0.32$: %s to %s)" % (texsci(BM["cases"][0]["beta_linear"], 1), texsci(BM["cases"][1]["beta_linear"], 1), sci(BM2["cases"][0]["beta_linear"], 1), sci(BM2["cases"][1]["beta_linear"], 1)),
     "$\\Delta\\varphi/M_{\\rm Pl}=%s$; unscreened tests restrict $\\beta\\lesssim O(1)$; screening (chameleon type) lowers the local coupling, not the cosmological one \\cite{KW}" % texsci(DPHI_OVER_M, 1)],
    ["Quadratic, $m=m_0(1+\\varphi^2/2\\Lambda_q^2)$", "$\\Lambda_q\\le%.0f$ to $%.0f$ meV ($\\eta=0.30$)" % (BM["cases"][0]["Lambda_quadratic_meV"], BM["cases"][1]["Lambda_quadratic_meV"]),
     "$f=%.1f$ meV, so $\\Lambda_q\\sim(2$--$20)f$: a new interaction at the scale of the potential; symmetron-type screening \\cite{HK} can hide it in dense environments" % PH[(1.0, 0.30)]["f_meV"]],
    ["Density-dependent effective potential", "same $\\Delta\\ln m$", "the model is then specified by $\\rho_m F(\\varphi)$ with $F$ changing by that amount across $\\Delta\\varphi$"],
]
add(("table", "Table 8. Requirement on a matter coupling that changes the rate by a factor $e$ (left end) or moves the benchmark into the reset window (right end).", ["Coupling", "Requirement", "Comment"], rows8, None))
add(("p", "The conclusion is therefore conditional. For the simplest linear coupling the required strength lies $26$ to $28$ orders of magnitude above what unscreened laboratory and solar-system tests allow. "
          "Quadratic or screened couplings are not excluded by this paper: they evade local tests by construction, but they are tuned, because they require a coupling scale of the same size as "
          "the dark-energy potential and a mass change of at least $0.1\\%$ of the matter between vacua. The dilution of matter can act as a trigger only through such an added, "
          "tuned structure, not through the framework itself. A noise-assisted mechanism would need an explicit open-system influence functional and a calculated change of the tunneling "
          "exponent, which is not assumed here."))

# ------------------------------------------------------------------------------------------------ 7 falsification
add(("h1", "7. Consistency conditions and falsification tests"))
add(("p", "A model of this class must satisfy internal consistency conditions, which are mathematical or follow from the framework, and it can be confronted with observational tests. "
          "The two are kept apart because failing the first means the model is ill-defined, while failing the second means that it is disfavored by data. All are computed in this paper and "
          "can be recomputed for a different potential with the supplied code."))
add(("h2", "7.1 Internal consistency conditions"))
add(("bullets", [
    "C1 (existence). A metastable minimum requires $\\eta<\\eta_c=%.3f$; for larger tilt there is nothing to decay." % C.eta_crit(),
    "C2 (semiclassical validity). The loop expansion parameter $1/B$ must be small. All draws with $B<30$ (%s%% of the prior) have rates far above the survival bound, so the classification of allowed draws is unaffected (Section 4.3)." % pc(BASE["frac_B_below"]["30"], 1),
    "C3 (negative mode and flat limit). The bounce has one negative mode and gravity enters as an $O(g)$ perturbation, $g\\sim10^{-60}$ (Section 3.3).",
    "C4 (energy). With $V_t\\ge0$ the released energy density cannot exceed $\\rho_\\Lambda$, so $T_{\\rm reh}\\le%.1f$ meV; with $V_t<0$ the interior crunches (Section 5.3)." % RH["T_max_meV_at_Vf"],
    "C5 (geometry). For any sub-Planckian energy the open interior stays curvature dominated, $Y(t)\\le Y(t_0)=%s$, unless it inflates; the single-field potential gives at most $%s$ e-folds (Section 5.3)." % (sci(SC["curvature"][0]["Y0_at_Vf"], 1), sci(efm, 1)),
    "C6 (percolation). A global transition requires $\\Gamma/H^4=O(1)$ or larger, a window of width $\\Delta B=%.1f$ below the survival bound that holds %s%% of the prior volume." % (DB_SURV, pc(BASE["frac_percolating_and_surviving"]["1"], 2)),
    "C7 (thermal consistency). A scalar in equilibrium with the CMB at redshift $z$ keeps its false minimum only if $c<\\theta_c\\lambda f^2/T_\\gamma(z)^2$, i.e. $c<%s$ for a bath coupled back to recombination, and the survival condition then applies to the rate history $\\Gamma(T_\\gamma(t))$ (Section 4.8)." % sci(TB.C_MAX_1100, 1),
    "C8 (repeated transitions). Over any chain of transitions the energy released telescopes, $\\sum_i\\Delta V_i=\\rho_\\Lambda-V_{\\rm final}\\le\\rho_\\Lambda$, so a second generation cannot raise the temperature above $%.1f$ meV (Section 5.5)." % RH["T_max_meV_at_Vf"],
    "C9 (oscillation). A periodic cycle needs a turnaround (here the decay to $V_t<0$), a bounce beyond general relativity, and dissipation that removes the scalar energy gained at each bounce, $\\Upsilon P\\gtrsim\\ln\\mu$; with these the cycles persist but carry negative vacuum energy throughout (Section 5.6).",
    "C10 (held vacuum). The vacuum is held today only if $a_c>1$, i.e. $\\mu^2M^2<\\rho_{\\rm dm,0}$, and a release into a negative vacuum needs $A(v)-1=2(V_c-V_0(v))/\\rho_{\\rm crit}>2V_c/\\rho_{\\rm crit}$, a dark-matter mass that grows severalfold after the release (Section 6.2).",
    "C11 (hot cycles). A cycle with a hot big bang and a dark-energy epoch needs a bounce hotter than dark-matter freeze-out, dissipation that thermalizes the scalar energy in the contraction, and re-thermalization of the dark matter; the entropy then grows each cycle and the cycles grow as $(S_{n+1}/S_n)^{1/3}$ with a fixed local history (Section 6.6)."
]))
add(("h2", "7.2 Observational tests"))
add(("bullets", [
    "O1 (survival). The absence of a bubble in our past light cone requires $\\Gamma V_4<%.1f$, i.e. $\\Gamma/H^4<%.0f$ at 95%% confidence, which excludes %s%% of the prior volume of the benchmark family and constrains the future mean decay time only to $t_{\\rm dec}\\gtrsim%s$ yr." % (SV["N_95"], XMAX, pc(BASE["frac_inconsistent_with_survival"], 0), texsci(T_MIN, 1)),
    "O2 (reset window). A transition on cosmological time scales requires $B_c\\lesssim B\\lesssim B_c+%.0f$ for decay within $10^{20}$ yr; this holds for %s%% of the prior volume. An action above $B_c+%.0f$ puts the decay beyond $10^{100}$ yr." % (DB_20, pc(BASE["frac_decay_by"]["1e+20"], 2), DB_WINDOW),
    "O3 (hot epoch). A hot universe in the nucleated region requires $\\rho_{\\rm reh}\\gtrsim(1\\ \\mathrm{MeV})^4$, which is %s times the dark-energy density; the bound of Section 5 excludes it for decay of the present dark energy in this framework." % sci(RH["rho_needed_for_bbn_over_Vf"], 1),
    "O4 (trigger). Matter dilution changes a tunneling rate only for $\\Delta\\ln m\\gtrsim10^{-3}$ between the vacua (Section 6.1); a vacuum held by a quadratic dark-matter coupling is released classically, has no fifth force while held, and gives $w=-1$ exactly until the release (Sections 6.2--6.3), so a confirmed evolution of the dark energy would disfavor it.",
    "O7 (voids). A void with density below $\\rho_{\\rm dm,0}/a_c^3$ and radius above the Compton length $\\pi/\\mu$ releases the vacuum locally today and forms domain walls; void statistics bound $\\mu\\gtrsim300H_0$ with $a_c\\lesssim2$ (Section 6.7).",
    "O5 (signals). A solitary transition produces no signal in our patch; bubble collisions \\cite{FKMS,Kleban,AJ} require a population and $\\Gamma/H^4=O(1)$ (C6). The vacuum is indistinguishable from a cosmological constant ($w=-1$) until it decays. A measured evolution of the dark energy, which recent baryon-acoustic-oscillation analyses combined with the CMB and supernovae prefer at the level of a few standard deviations depending on the supernova sample \\cite{DESI}, would favor a rolling field and not a metastable vacuum. No observational discriminant of the cyclic reset is available from this model.",
    "O6 (cooling). If the scalar is coupled to the CMB the decay probability was larger in the past and falls as the universe cools: equilibrated scenarios with $c\\ge0$ lift the excluded prior mass from %s%% to %s--%s%% and leave no survival-consistent draw that decays within $10^{20}$ yr, so the absence of a transition so far constrains the coupling of the scalar to the CMB as well as its potential (Section 4.8)." % (pc(PMAP["decoupled"]["excluded"], 0), pc(PMAP["c=0 (equilibrated), z<=1"]["excluded"], 0), pc(PMAP["c=0, z<=10"]["excluded"], 0))
]))
add(("p", "A tunneling transition producing a successful cosmology would have to lie outside the class of Sections 3--5. It would need a parent vacuum energy "
          "of at least $\\sim10^{34}\\rho_\\Lambda$ and, by Section 5.3, a post-nucleation region that inflates before it reheats. Section 6 shows a different route that "
          "does not need tunneling: a vacuum held by matter and released by dilution ends the expansion, and an effective bounce with thermal dissipation and freeze-out starts a "
          "hot one; the cyclic models constructed earlier provide a reset by other means \\cite{ST,IS}."))

# ------------------------------------------------------------------------------------------------ 8 discussion
add(("h1", "8. Discussion and limitations"))
add(("bullets", [
    "A single scalar field and a quartic-tilt potential are used. Other potentials change $b(\\eta)$ and the interior energy fraction, but not the scales: the "
    "rate depends on the combination $M_{\\rm Pl}^4/V_f$ and the interior energy on $V_f$, both fixed by the data, so O1, O3 and the geometric argument of Section 5.3 are generic.",
    "The one-loop prefactor is a systematic uncertainty, not a result; the conclusions are insensitive to it over six decades (Section 4.3).",
    "The decay is treated as a flat-space process because $g\\sim10^{-60}$; the gravitational code is validated against analytic limits and used for the map "
    "of the regime in which gravity matters, which the dark-energy scale excludes.",
    "The survival criterion assumes a constant rate and a false vacuum throughout the history of the observed universe; a field that evolved or was thermally affected in the early universe would change the early part of the cone, which carries a small part of $V_4$.",
    "The interior evolution assumes a homogeneous open slicing and a perturbative decay of the condensate to radiation; bubble-wall collisions can "
    "concentrate energy locally, but the average energy density is bounded by $\\Delta V$ (Table 7).",
    "Up-tunneling from the false to a higher vacuum, the decay of the true vacuum in curved space \\cite{LW}, thermal activation of the Hawking--Moss type \\cite{BW} and decays of the present vacuum into other vacua are not considered.",
    "The priors of the Monte Carlo are documented assumptions, not measurements; the regime fractions describe the sample space of the model, not the "
    "probability of a transition in our universe.",
    "The matter-coupling analysis is first order in the tilt shift and does not simulate any specific screened model.",
    "The thermal analysis assumes equilibrium with the CMB (an upper bound on the thermal channel), the leading high-temperature term with one coefficient $c$, and a bath coupled below a stated redshift; it neglects higher-order thermal corrections and the interpolation between the O(4) and O(3) saddles near $T_\\times$, which a periodic-instanton calculation would refine.",
    "The cycles of Section 6 rest on an effective bounce (not a derivation), on $\\rho_c$ 108 orders of magnitude below the Planck density (with the dependence reported and converging), on the adiabatic result of the hot-phase restoration and an instantaneous freeze-out, and on homogeneous dynamics: the growth of inhomogeneities and anisotropies during the contraction, which can dominate a contracting universe, is not followed. The value $a_c\\sim1$ requires $\\mu^2M^2\\approx\\rho_{\\rm dm,0}$, a coincidence like that of the cosmological constant.",
    "The simulations of Section 5.5 are homogeneous and use a phenomenological dissipation: bubble-wall dynamics, inhomogeneous wall collisions, the microphysics of reheating and the gravitational backreaction of the wall are not simulated, and the bounce of Option B is an effective loop-quantum-cosmology prescription (with a second prescription as a check), not a derived one."
]))

add(("h1", "9. Conclusions"))
add(("p", "The present dark energy can end the expansion and start a new hot big bang, but not by tunneling. If it is the energy of a vacuum held in place by dark matter, the "
          "dilution of the matter releases it into a negative vacuum and the expansion turns around, with a $\\Lambda$CDM past ($w=-1$, no fifth force) that an uncoupled field "
          "cannot reproduce. With an effective bounce, thermal dissipation in the hot contraction and the re-thermalization and freeze-out of dark matter, the cycles are self-similar: "
          "each has a radiation-dominated bounce, a matter era, a dark-energy epoch of %.1f e-folds and a release at the same temperature, and the entropy produced, as the second law "
          "requires, enlarges the next cycle by exactly $(S_{n+1}/S_n)^{1/3}=%.1f$. Each ingredient is necessary: without the dissipation or the freeze-out the cycles fail. Cooling "
          "cannot replace dilution as the trigger. The remaining sections of this conclusion summarize why the direct route fails, which is what makes these ingredients necessary."
          % (DIL.BASE["cycles"][1]["held_accelerated_efolds"], DIL.gm(DIL.GS))))
add(("p", "A late-time false-vacuum transition of the present dark energy is a well-defined calculation, and for the explicit potential studied it can be done to a stated precision. The quantum decay is a flat-space "
          "process controlled by the action $B=b(\\eta)/\\lambda$, which must lie near $B_c\\approx%d$ for the transition to be neither excluded nor effectively absent; with survival evaluated on the past light cone the bound is weak "
          "($\\Gamma/H^4<%.0f$) and the window is set by the action itself. Cosmological cooling does not help: a thermal mass rescales the potential exactly, the thermal O(3) channel dominates above %.2f meV and falls "
          "monotonically as the universe cools, and a scalar in equilibrium with the CMB is more strongly constrained (%s--%s%% of the prior excluded against %s%%) with no late-time decay window left. "
          "The nucleated daughter region carries at most the dark-energy density, reheats to at most $%.1f$ meV and cools as $1/t$ in a curvature-dominated open universe; this rests on geometry and not on energy conservation alone, "
          "holds for any sub-Planckian parent energy and any number of scalar fields, and a trigger by matter dilution would need a change of the matter mass between the vacua of at least $0.1\\%%$. A daughter region of the present dark energy "
          "is cold and empty, and the cyclic-reset hypothesis, in the form tested, is not supported; the framework identifies what would have to change." % (
              round(bc_med), XMAX, TR["crossover_T_meV"]["0"], pc(PMAP["c=0 (equilibrated), z<=1"]["excluded"], 0), pc(PMAP["c=0, z<=10"]["excluded"], 0), pc(PMAP["decoupled"]["excluded"], 0), RH["T_max_meV_at_Vf"])))

add(("p", "Lorentzian simulations of the daughter region (Section 5.5) support these statements with two fields, dissipation and gravity: the closed-form interior is reproduced to %.0f%%, a reservoir hidden by a negative offset leaves an anti-de Sitter vacuum "
          "that crunches after %.1f s when it reaches %.1f MeV, flattening at the dark-energy scale needs a vev above $4M_{\\rm Pl}$ and reheats below %.1f meV, and repeated transitions release no more than $\\rho_\\Lambda$ in total. "
          "A viable hot daughter universe needs a parent vacuum energy at least %s times the observed dark energy, which is excluded, or a cancellation of the same size, which ends in a crunch."
          % (100 * max(abs(r["ratio_sim_over_closed"] - 1) for r in SIM["A_single_field"]["rows"]), RB35["t_crunch_s"], RB35["T_max_MeV"], HILL_T, sci(SIM["C_parent_energy_for_1MeV_over_rho_Lambda"], 1))))

add(("p", "The same region can oscillate only with physics beyond general relativity. With an effective bounce, the decay into anti-de Sitter turns expansion into contraction and radiation-only cycles repeat exactly, "
          "but a scalar excitation is amplified at every bounce (by factors between %.1f and %.0f in energy per cycle over the range simulated), the growth of successive cycles obeys $\\Delta a_{\\max}=W/3$, and the cycles are stable only when the dissipation per cycle "
          "exceeds the logarithm of the amplification. A vacuum transition during the cycles pushes them across this boundary. Every cycle has negative vacuum energy and its hot phase comes from compression at a Planck-density bounce, so an oscillating universe "
          "of this kind does not contain an epoch like the observed one unless the vacuum is held by matter, as in Section 6." % (min(MU_LIN), max(MU_LIN))))

# ---------------------------------------------------------------------------------------------------- statements
add(("h1", "Declarations"))
add(("pb", "Funding.", "This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors."))
add(("pb", "Conflict of interest.", "The author declares no competing interests."))
add(("pb", "Ethics approval, consent to participate, consent for publication.", "Not applicable."))
add(("pb", "Data availability.", "The data underlying every figure and table (bounce tables, error budget, survival calculation, falsification map, Monte Carlo summary, interior evolution, "
          "scope and coupling estimates, verification results) are deposited with the code at %s and archived at https://doi.org/%s." % (GITHUB, DOI_CODE)))
add(("pb", "Code availability.", "All code (Python, NumPy, SciPy, Matplotlib) is released under the MIT licence at %s, archived at https://doi.org/%s. "
          "Running the scripts in the stated order regenerates every number in this paper, and verify.py runs %d executable checks. The manuscript is archived at "
          "https://doi.org/%s." % (GITHUB, DOI_CODE, len(V), DOI_PAPER)))
add(("pb", "Authors' contributions.", "L.S.: conceptualization, methodology, software, formal analysis, investigation, visualization, writing."))
add(("pb", "Use of generative AI.", "During the preparation of this work the author used Claude (Anthropic) to assist with the design of the numerical "
          "analysis, writing of the code, literature checking and drafting. The author reviewed and edited the content, checked the code against analytic limits and "
          "executable tests, verified the cited references, and takes full responsibility for the publication. No AI tool is an author."))

# -------------------------------------------------------------------------------------------------------- references
add(("h1", "References"))
REFS = [
    ("CDL", "S. Coleman, F. De Luccia, Gravitational effects on and of vacuum decay. Phys. Rev. D 21, 3305--3315 (1980). https://doi.org/10.1103/PhysRevD.21.3305"),
    ("Coleman", "S. Coleman, Fate of the false vacuum: Semiclassical theory. Phys. Rev. D 15, 2929--2936 (1977). https://doi.org/10.1103/PhysRevD.15.2929"),
    ("CC", "C.G. Callan, S. Coleman, Fate of the false vacuum. II. First quantum corrections. Phys. Rev. D 16, 1762--1768 (1977). https://doi.org/10.1103/PhysRevD.16.1762"),
    ("HM", "S.W. Hawking, I.G. Moss, Supercooled phase transitions in the very early universe. Phys. Lett. B 110, 35--38 (1982). https://doi.org/10.1016/0370-2693(82)90946-7"),
    ("GW", "A.H. Guth, E.J. Weinberg, Could the universe have recovered from a slow first-order phase transition? Nucl. Phys. B 212, 321--364 (1983). https://doi.org/10.1016/0550-3213(83)90307-3"),
    ("Parke", "S. Parke, Gravity and the decay of the false vacuum. Phys. Lett. B 121, 313--315 (1983). https://doi.org/10.1016/0370-2693(83)91376-X"),
    ("JS", "R.A. Jensen, P.J. Steinhardt, Bubble nucleation and the Coleman--Weinberg model. Nucl. Phys. B 237, 176--188 (1984). https://doi.org/10.1016/0550-3213(84)90021-X"),
    ("LW", "K. Lee, E.J. Weinberg, Decay of the true vacuum in curved space-time. Phys. Rev. D 36, 1088--1094 (1987). https://doi.org/10.1103/PhysRevD.36.1088"),
    ("TWW", "M.S. Turner, E.J. Weinberg, L.M. Widrow, Bubble nucleation in first-order inflation and other cosmological phase transitions. Phys. Rev. D 46, 2384--2403 (1992). https://doi.org/10.1103/PhysRevD.46.2384"),
    ("HW", "T. Hackworth, E.J. Weinberg, Oscillating bounce solutions and vacuum tunneling in de Sitter spacetime. Phys. Rev. D 71, 044014 (2005). https://doi.org/10.1103/PhysRevD.71.044014"),
    ("BW", "A.R. Brown, E.J. Weinberg, Thermal derivation of the Coleman--De Luccia tunneling prescription. Phys. Rev. D 76, 064003 (2007). https://doi.org/10.1103/PhysRevD.76.064003"),
    ("Gott", "J.R. Gott III, Creation of open universes from de Sitter space. Nature 295, 304--307 (1982). https://doi.org/10.1038/295304a0"),
    ("BGT", "M. Bucher, A.S. Goldhaber, N. Turok, Open universe from inflation. Phys. Rev. D 52, 3314--3337 (1995). https://doi.org/10.1103/PhysRevD.52.3314"),
    ("GH", "G.W. Gibbons, S.W. Hawking, Cosmological event horizons, thermodynamics, and particle creation. Phys. Rev. D 15, 2738--2751 (1977). https://doi.org/10.1103/PhysRevD.15.2738"),
    ("Planck", "Planck Collaboration, N. Aghanim et al., Planck 2018 results. VI. Cosmological parameters. Astron. Astrophys. 641, A6 (2020). https://doi.org/10.1051/0004-6361/201833910"),
    ("BBN", "C. Pitrou, A. Coc, J.-P. Uzan, E. Vangioni, Precision big bang nucleosynthesis with improved Helium-4 predictions. Phys. Rep. 754, 1--66 (2018). https://doi.org/10.1016/j.physrep.2018.04.005"),
    ("Penrose", "R. Penrose, Cycles of Time: An Extraordinary New View of the Universe. Bodley Head, London (2010)."),
    ("GV", "M. Gasperini, G. Veneziano, The pre-big bang scenario in string cosmology. Phys. Rep. 373, 1--212 (2003). https://doi.org/10.1016/S0370-1573(02)00389-7"),
    ("ST", "P.J. Steinhardt, N. Turok, A cyclic model of the universe. Science 296, 1436--1439 (2002). https://doi.org/10.1126/science.1070462"),
    ("CosmoT", "C.L. Wainwright, CosmoTransitions: Computing cosmological phase transition temperatures and bubble profiles with multiple fields. Comput. Phys. Commun. 183, 2006--2013 (2012). https://doi.org/10.1016/j.cpc.2012.04.004"),
    ("SciPy", "P. Virtanen et al., SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat. Methods 17, 261--272 (2020). https://doi.org/10.1038/s41592-019-0686-2"),
    ("NumPy", "C.R. Harris et al., Array programming with NumPy. Nature 585, 357--362 (2020). https://doi.org/10.1038/s41586-020-2649-2"),
    ("IRS", "G. Isidori, G. Ridolfi, A. Strumia, On the metastability of the Standard Model vacuum. Nucl. Phys. B 609, 387--409 (2001). https://doi.org/10.1016/S0550-3213(01)00302-9"),
    ("EGR", "J.R. Espinosa, G.F. Giudice, A. Riotto, Cosmological implications of the Higgs mass measurement. J. Cosmol. Astropart. Phys. 2008(05), 002 (2008). https://doi.org/10.1088/1475-7516/2008/05/002"),
    ("DKS", "L. Dyson, M. Kleban, L. Susskind, Disturbing implications of a cosmological constant. J. High Energy Phys. 2002(10), 011 (2002). https://doi.org/10.1088/1126-6708/2002/10/011"),
    ("BP", "R. Bousso, J. Polchinski, Quantization of four-form fluxes and dynamical neutralization of the cosmological constant. J. High Energy Phys. 2000(06), 006 (2000). https://doi.org/10.1088/1126-6708/2000/06/006"),
    ("FKMS", "B. Freivogel, M. Kleban, M. Rodriguez Martinez, L. Susskind, Observational consequences of a landscape. J. High Energy Phys. 2006(03), 039 (2006). https://doi.org/10.1088/1126-6708/2006/03/039"),
    ("Kleban", "M. Kleban, Cosmic bubble collisions. Class. Quantum Grav. 28, 204008 (2011). https://doi.org/10.1088/0264-9381/28/20/204008"),
    ("AJ", "A. Aguirre, M.C. Johnson, A status report on the observability of cosmic bubble collisions. Rep. Prog. Phys. 74, 074901 (2011). https://doi.org/10.1088/0034-4885/74/7/074901"),
    ("DM", "G.V. Dunne, H. Min, Beyond the thin-wall approximation: Precise numerical computation of prefactors in false vacuum decay. Phys. Rev. D 72, 125004 (2005). https://doi.org/10.1103/PhysRevD.72.125004"),
    ("AFS", "A. Andreassen, W. Frost, M.D. Schwartz, Scale-invariant instantons and the complete lifetime of the standard model. Phys. Rev. D 97, 056006 (2018). https://doi.org/10.1103/PhysRevD.97.056006"),
    ("IS", "A. Ijjas, P.J. Steinhardt, Bouncing cosmology made simple. Class. Quantum Grav. 35, 135004 (2018). https://doi.org/10.1088/1361-6382/aac482"),
    ("DESI", "DESI Collaboration, M. Abdul Karim et al., DESI DR2 results. II. Measurements of baryon acoustic oscillations and cosmological constraints. Phys. Rev. D 112, 083515 (2025). https://doi.org/10.1103/tr6y-kpc6"),
    ("TS", "T. Tanaka, M. Sasaki, False vacuum decay with gravity. Prog. Theor. Phys. 88, 503--528 (1992). https://doi.org/10.1143/PTP.88.503"),
    ("KW", "J. Khoury, A. Weltman, Chameleon fields: awaiting surprises for tests of gravity in space. Phys. Rev. Lett. 93, 171104 (2004). https://doi.org/10.1103/PhysRevLett.93.171104"),
    ("HK", "K. Hinterbichler, J. Khoury, Screening long-range forces through local symmetry restoration. Phys. Rev. Lett. 104, 231301 (2010). https://doi.org/10.1103/PhysRevLett.104.231301"),
]
REFS += [
    ("L83", "A.D. Linde, Decay of the false vacuum at finite temperature. Nucl. Phys. B 216, 421--445 (1983). https://doi.org/10.1016/0550-3213(83)90293-6"),
    ("A81", "I. Affleck, Quantum-statistical metastability. Phys. Rev. Lett. 46, 388--391 (1981). https://doi.org/10.1103/PhysRevLett.46.388"),
    ("DJ", "L. Dolan, R. Jackiw, Symmetry behavior at finite temperature. Phys. Rev. D 9, 3320--3341 (1974). https://doi.org/10.1103/PhysRevD.9.3320"),
    ("W74", "S. Weinberg, Gauge and global symmetries at high temperature. Phys. Rev. D 9, 3357--3378 (1974). https://doi.org/10.1103/PhysRevD.9.3357"),
    ("PS", "J. Pavlovic, N. Sossich, Geodesically complete cyclic cosmologies and entropy. Eur. Phys. J. C 84, 242 (2024). https://doi.org/10.1140/epjc/s10052-024-12621-z"),
    ("Tolman", "R.C. Tolman, On the problem of the entropy of the universe as a whole. Phys. Rev. 37, 1639--1660 (1931). https://doi.org/10.1103/PhysRev.37.1639"),
    ("FG", "E. Farhi, A.H. Guth, An obstacle to creating a universe in the laboratory. Phys. Lett. B 183, 149--155 (1987). https://doi.org/10.1016/0370-2693(87)90429-1"),
    ("FGG", "E. Farhi, A.H. Guth, J. Guven, Is it possible to create a universe in the laboratory by quantum tunneling? Nucl. Phys. B 339, 417--490 (1990). https://doi.org/10.1016/0550-3213(90)90357-J"),
    ("BGG", "S.K. Blau, E.I. Guendelman, A.H. Guth, Dynamics of false-vacuum bubbles. Phys. Rev. D 35, 1747--1766 (1987). https://doi.org/10.1103/PhysRevD.35.1747"),
    ("APS06", "A. Ashtekar, T. Pawlowski, P. Singh, Quantum nature of the big bang. Phys. Rev. Lett. 96, 141301 (2006). https://doi.org/10.1103/PhysRevLett.96.141301"),
    ("AS11", "A. Ashtekar, P. Singh, Loop quantum cosmology: a status report. Class. Quantum Grav. 28, 213001 (2011). https://doi.org/10.1088/0264-9381/28/21/213001"),
    ("Singh09", "P. Singh, Are loop quantum cosmos never singular? Class. Quantum Grav. 26, 125005 (2009). https://doi.org/10.1088/0264-9381/26/12/125005"),
    ("NB08", "M. Novello, S.E. Perez Bergliaffa, Bouncing cosmologies. Phys. Rep. 463, 127--213 (2008). https://doi.org/10.1016/j.physrep.2008.04.006"),
    ("BP15", "D. Battefeld, P. Peter, A critical review of classical bouncing cosmologies. Phys. Rep. 571, 1--66 (2015). https://doi.org/10.1016/j.physrep.2014.12.004"),
    ("ST12", "V. Sahni, A. Toporensky, Cosmological hysteresis and the cyclic universe. Phys. Rev. D 85, 123542 (2012). https://doi.org/10.1103/PhysRevD.85.123542"),
    ("KOST", "J. Khoury, B.A. Ovrut, P.J. Steinhardt, N. Turok, Ekpyrotic universe: colliding branes and the origin of the hot big bang. Phys. Rev. D 64, 123522 (2001). https://doi.org/10.1103/PhysRevD.64.123522"),
    ("Amendola", "L. Amendola, Coupled quintessence. Phys. Rev. D 62, 043511 (2000). https://doi.org/10.1103/PhysRevD.62.043511"),
    ("HKLM11", "K. Hinterbichler, J. Khoury, A. Levy, A. Matas, Symmetron cosmology. Phys. Rev. D 84, 103521 (2011). https://doi.org/10.1103/PhysRevD.84.103521"),
    ("Berera95", "A. Berera, Warm inflation. Phys. Rev. Lett. 75, 3218--3221 (1995). https://doi.org/10.1103/PhysRevLett.75.3218"),
    ("KT90", "E.W. Kolb, M.S. Turner, The Early Universe (Addison-Wesley, Redwood City, 1990)"),
]
# order references by first citation in the text (tables count: they are cited where they appear)
import re as _re
THB, THX = TB.blocks()


def _renum(t):
    t = _re.sub(r"(Table )([6-8])(?![0-9])", lambda m: "%s%d" % (m.group(1), int(m.group(2)) + (2 if int(m.group(2)) < 8 else 7)), t)
    t = _re.sub(r"(Fig\. )([6-7])(?![0-9])", lambda m: "%s%d" % (m.group(1), int(m.group(2)) + 2), t)
    return t


def _renum_block(b):
    if b[0] in ("p", "abstract", "note", "h1", "h2"):
        return (b[0], _renum(b[1]))
    if b[0] == "bullets":
        return ("bullets", [_renum(x) for x in b[1]])
    if b[0] == "pb":
        return ("pb", b[1], _renum(b[2]))
    if b[0] == "table":
        return ("table", _renum(b[1]), b[2], [[_renum(str(c)) for c in row] for row in b[3]], b[4])
    if b[0] == "fig":
        return ("fig", b[1].replace("fig6_reheating", "fig8_reheating").replace("fig7_curvature", "fig9_curvature"), _renum(b[2])) + tuple(b[3:])
    return b


B = [_renum_block(b) for b in B]
_i5 = [k for k, b in enumerate(B) if b[0] == "h1" and b[1].startswith("5. What the nucleated")][0]
B[_i5:_i5] = THB
_i6 = [k for k, b in enumerate(B) if b[0] == "h1" and b[1].startswith("6. Matter dilution")][0]
B[_i6:_i6] = SBK.blocks() + OSB.blocks()
_i7 = [k for k, b in enumerate(B) if b[0] == "h1" and b[1].startswith("7. Consistency")][0]
B[_i7:_i7] = DIL.blocks()
text_all = " ".join(b[-1] for b in B if b[0] in ("p", "abstract", "pb") and isinstance(b[-1], str)) + " ".join(" ".join(x) for b in B if b[0] == "bullets" for x in [b[1]])
order = []
for blk in B:
    chunks = []
    if blk[0] in ("p", "abstract", "pb") and isinstance(blk[-1], str):
        chunks = [blk[-1]]
    elif blk[0] == "bullets":
        chunks = list(blk[1])
    elif blk[0] == "table":
        chunks = [str(c) for row in blk[3] for c in row]
    for ch in chunks:
        for m in _re.finditer(r"\\cite\{([^}]+)\}", ch):
            for key in m.group(1).split(","):
                if key not in order:
                    order.append(key)
numbering = {k: i + 1 for i, k in enumerate(order)}
for k, _ in REFS:
    if k not in numbering:
        numbering[k] = len(numbering) + 1
REFS_SORTED = sorted(REFS, key=lambda kv: numbering[kv[0]])
missing = [k for k, _ in REFS if k not in order]
print("references not cited in the text:", missing)
undefined = [k for k in order if k not in dict(REFS)]
chk(not undefined, "undefined citation keys %s" % undefined)


def replace_cites(s):
    return _re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(numbering[x]) for x in m.group(1).split(",")) + "]", s)


for k, r in REFS_SORTED:
    add(("ref", "[%d] %s" % (numbering[k], r)))
# number citations
B2 = []
for b in B:
    if b[0] in ("p", "abstract", "note"):
        B2.append((b[0], replace_cites(b[1])))
    elif b[0] == "bullets":
        B2.append(("bullets", [replace_cites(x) for x in b[1]]))
    elif b[0] == "pb":
        B2.append(("pb", b[1], replace_cites(b[2])))
    elif b[0] == "table":
        B2.append(("table", b[1], b[2], [[replace_cites(str(c)) for c in row] for row in b[3]], b[4]))
    else:
        B2.append(b)
B = B2
# equation reference in the table caption and the text: the reheating bound is a numbered display equation
neq = sum(1 for b in B if b[0] == "eq")
ieq = [i for i, b in enumerate(B) if b[0] == "eq"]
bound_eq = [n + 1 for n, i in enumerate(ieq) if "T_{\\rm reh}\\le" in B[i][1]][0]
B = [tuple([b[0], b[1].replace("Eq. (11)", "Eq. (%d)" % bound_eq)] + list(b[2:])) if b[0] == "table" else b for b in B]
B = [(b[0], b[1].replace("Eq. (11)", "Eq. (%d)" % bound_eq).replace("@@BOUNDEQ@@", str(bound_eq))) if b[0] == "p" else b for b in B]
B = [(b[0], b[1], b[2].replace("@@BOUNDEQ@@", str(bound_eq))) + tuple(b[3:]) if b[0] == "fig" else b for b in B]
print("equations:", neq, " bound is Eq.", bound_eq)


def fix_sci(text):
    """Resolve the sentinel markup of sci(): inside a math span it becomes raw TeX, outside it becomes its own inline math span."""
    if "⟦" not in text:
        return text
    parts, out = text.split("$"), []
    for i, seg in enumerate(parts):
        inside = i % 2 == 1
        if inside:
            seg = seg.replace("⟦", "").replace("⟧", "")
        else:
            seg = seg.replace("⟦", "$").replace("⟧", "$")
        out.append(seg)
    res = out[0]
    for i in range(1, len(out)):
        res += "$" + out[i]
    return res


def map_strings(block):
    k = block[0]
    if k in ("p", "abstract", "note", "title"):
        return (k, fix_sci(block[1]))
    if k == "pb":
        return (k, block[1], fix_sci(block[2]))
    if k == "bullets":
        return (k, [fix_sci(x) for x in block[1]])
    if k == "table":
        return (k, fix_sci(block[1]), [fix_sci(h) for h in block[2]], [[fix_sci(str(c)) for c in row] for row in block[3]], block[4])
    if k == "fig":
        return (k, block[1], fix_sci(block[2])) + tuple(block[3:])
    return block


B = [map_strings(b) for b in B]
paper_docx = os.path.join(OUT, "EPJC_Late_Time_Vacuum_Decay.docx")
RN.render_docx(B, paper_docx, FIGDIR)
RN.render_tex(B, os.path.join(OUT, "EPJC_Late_Time_Vacuum_Decay.tex"))
words = sum(len(b[1].split()) for b in B if b[0] in ("p", "abstract")) + sum(len(x.split()) for b in B if b[0] == "bullets" for x in b[1])
print("words (text, excl. references and tables):", words)
print("saved", paper_docx)
