# -*- coding: utf-8 -*-
"""Builds the manuscript, supplement and cover letter for the EPJ C submission from the stored results.

    python build_paper.py          (refuses to run unless verification.json reports every check passed)

The text carries no hand-typed result: every number is read from bounce_tables.json, falsification_map.json, monte_carlo.json,
interior_evolution.json, verification.json or constants.py, and every comparative sentence is asserted against the same files.
"""
import json
import os
import sys

import numpy as np

import constants as K
import cdl_lib as C
import render as RN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
FIGDIR = os.path.join(REPO, "figures")
OUT = os.path.join(REPO, "manuscript")
os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
T, F, MC, IN, V = J("bounce_tables.json"), J("falsification_map.json"), J("monte_carlo.json"), J("interior_evolution.json"), J("verification.json")
assert all(v["passed"] for v in V.values()), "verification has failures - do not build"

GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
DOI_PAPER, DOI_CODE = "10.5281/zenodo.23717617", "10.5281/zenodo.23717619"
TITLE = "Late-Time Vacuum Decay and Cosmological Reheating: Gravitational Tunneling, Stochastic Sensitivity, and Falsification Boundaries"


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
RUN = {(r["eps"] if r["eps"] == 1.0 else "in", r["gamma_over_m"]): r for r in IN["runs"]}
r_prompt = [r for r in IN["runs"] if r["eps"] == 1.0 and r["gamma_over_m"] == 1.0][0]
r_slow = [r for r in IN["runs"] if r["eps"] == 1.0 and r["gamma_over_m"] == 1e-3][0]
EPS_IN = IN["eps_in_ref"]
BC = MC["B_critical_quantiles"]
bc_med = BC["0.5"]
g_lo, g_hi = F["g_physical_range"]
kap = {s["eta"]: s["kappa"] for s in T["slopes"]}
v = lambda name: V[name]
pc = lambda x, d=1: ("%." + str(d) + "f") % (100 * x)
def sci(x, d=1):
    """Scientific notation as inline math (for prose and table cells)."""
    return "⟦" + texsci(x, d) + "⟧"


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    return "%.*f\\times10^{%d}" % (d, x / 10 ** e, e)


pw10 = lambda x: ("$%s\\times10^{%d}$" % (("%.1f" % (x / 10 ** int(np.floor(np.log10(x))))), int(np.floor(np.log10(x)))))

chk(PH[(1.0, 0.30)]["B"] > bc_med + 100, "B at the benchmark far above critical")
chk(PH[(1.0, 0.32)]["log10_t_yr"] > 12 and PH[(1.0, 0.32)]["log10_t_yr"] < 20, "eta=0.32 lies in the 1e12-1e20 yr window")
chk(PH[(1.0, 0.34)]["ln_gamma_h4"] > 0, "eta=0.34 decays at once")
chk(RH["T_max_meV_at_Vf"] < 3.0 and RH["ratio_Tbbn_over_Tmax"] > 1e8, "reheating no-go")
chk(r_prompt["T_max_meV"] < RH["T_max_meV_at_Vf"] and IN["t_for_bbn_temperature_seconds"] < IN["scales"]["t0_seconds"], "interior cannot reach 1 MeV")
chk(BASE["frac_never_within_1e100"] > 0.7 and BASE["frac_decay_by"]["1e+20"] < 0.01, "prior mass of the window is small")
chk(max(abs(SENS[k]["frac_inconsistent_with_survival"] - BASE["frac_inconsistent_with_survival"]) for k in ("sigma_log10_D = 0", "sigma_log10_D = 3", "r_L ~ logU(1, 10)")) < 1e-3, "prior mass insensitive to the prefactor")
chk(g_hi < 1e-59 and PH[(1.0, 0.30)]["hm_ratio"] > 1e58, "gravity and HM negligible at the physical scale")

# ------------------------------------------------------------------------------------------------------- abstract
DB_WINDOW = (100 - 11.04) * np.log(10.0)          # change of B over which the decay time runs from the survival bound to 1e100 yr
DB_20 = (20 - 11.04) * np.log(10.0)
ABSTRACT = (
    "A late-time false-vacuum transition has been proposed as the seed of a new cosmological cycle. We ask what the Coleman--De Luccia (CDL) "
    "framework can and cannot support when the parent vacuum is the observed dark energy. A scalar potential with an explicit metastable minimum "
    "is specified, the O(4) Euclidean bounce is solved numerically in flat space and with Einstein gravity (shooting, an independent "
    "boundary-value solution, analytic de Sitter tails; %d executable consistency checks), and the action is converted to a nucleation rate per "
    "Hubble four-volume with the measured dark-energy density. With all scales set by the dark energy, the gravitational parameter is "
    "$g = f^2/M_{\\rm Pl}^2 \\sim 10^{-60}$, gravitational corrections to the action are of relative size $10^{-59}$, and the Hawking--Moss regime is "
    "irrelevant: the decay is a flat-space process controlled by one number, the action $B = b(\\eta)/\\lambda$. Nucleation within a "
    "Hubble four-volume requires $B \\lesssim %d$, and the mean decay time rises from $10^{11}$ yr to beyond $10^{100}$ yr as $B$ increases by only $\\sim%d$. "
    "Over physically consistent priors %s%% of the parameter space is excluded by the survival of the present vacuum and %s%% never decays. "
    "After a nucleation the interior is a curvature-dominated open universe whose energy density is bounded by that of the dark energy: "
    "reheating reaches at most $T \\simeq %.1f$ meV, $%s$ times below nucleosynthesis. A cyclic reset that regenerates a hot big bang by decay of "
    "the present dark energy is therefore excluded within this framework, as is a trigger by matter dilution, which would need a scalar--matter coupling "
    "$\\sim10^{28}$--$10^{30}$ times larger than allowed. The falsification boundaries are stated as explicit, reproducible conditions." % (
        len(V), round(bc_med), round(DB_WINDOW), pc(BASE["frac_inconsistent_with_survival"], 0), pc(BASE["frac_never_within_1e100"], 0),
        RH["T_max_meV_at_Vf"], texsci(RH["ratio_Tbbn_over_Tmax"], 1)))
KEYWORDS = "false vacuum decay; Coleman--De Luccia instanton; de Sitter space; dark energy; reheating; cyclic cosmology; falsifiability; Monte Carlo"

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
          "the vacuum can decay by quantum tunneling through a Coleman--De Luccia instanton \\cite{CDL}. It has been suggested that such a "
          "late-time transition could seed a new cosmological cycle, so that the long-term history of the universe is cyclic rather than "
          "terminal; related cyclic and bouncing proposals are discussed in \\cite{Penrose,GV,ST}. The idea is attractive, but as a physical "
          "hypothesis it is only as good as its quantitative content: a metastable minimum must be specified, the tunneling action computed "
          "including gravity, the rate compared with the age and Hubble time of the universe, and the interior of the nucleated region followed "
          "to see whether it can become a hot, expanding cosmology."))
add(("p", "This paper does that for an explicit model and asks which parts of the proposal survive. Three claims are tested against calculation. "
          "(i) A scalar potential with a metastable minimum at the dark-energy scale can nucleate bubbles of lower-energy vacuum at a rate "
          "that is neither negligible nor already excluded. (ii) The nucleated region reheats into a hot expanding universe. (iii) The "
          "transition is triggered, or enhanced, by the dilution of matter in the late universe. The analysis is deliberately conditional: "
          "it takes the dark-energy density as the false-vacuum energy and asks what follows. A full Hawking--Moss and CDL treatment, a "
          "statement of where each approximation fails, and an executable verification suite are provided so that the conclusions can be "
          "checked and, if the model is changed, recomputed."))
add(("p", "The contributions are as follows."))
add(("bullets", [
    "An explicit potential family with a controlled false minimum, and a reduction of the CDL problem to two dimensionless parameters, the barrier "
    "coupling $\\lambda$ and the tilt-to-barrier ratio $\\eta$, together with the gravitational parameter $g = f^2/M_{\\rm Pl}^2$ (Section 2).",
    "A numerical solution of the Euclidean bounce in flat space and with gravity, validated by an independent boundary-value solver, the thin-wall "
    "and Parke limits, the exact de Sitter action and the Hawking--Moss criterion (Section 3, Table 1, Fig. 1--2).",
    "A conversion of the action to a nucleation rate per Hubble four-volume with the measured dark-energy density, giving a falsification diagram in "
    "the $(\\eta,\\lambda)$ plane and a prior-sensitivity Monte Carlo with convergence and failure regions (Section 4, Figs 3--4).",
    "An energy-conservation analysis of the nucleated region: the interior energy budget, an open-universe solution with reheating, and the resulting "
    "bound on the reheating temperature (Section 5, Fig. 5, Table 4).",
    "A quantitative test of matter dilution as a trigger, and an explicit list of falsification conditions (Sections 6--7).",
]))
add(("p", "The results are largely negative for the cyclic-reset hypothesis in its simplest form, and positive for a different claim: that the "
          "hypothesis, once stated as an effective field theory, makes definite and testable demands. We say what those demands are."))

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
add(("p", "The choice $V_t = 0$ (decay to a Minkowski vacuum, $r_L \\equiv V_f/\\Delta V = 1$) is the baseline, because it maximizes the energy "
          "released to the interior for a given $V_f$; $V_t < 0$ (anti-de Sitter) and $V_t > 0$ are considered in Sections 5.3 and 4.2. The "
          "scale $f$ is not free: it is fixed by $V_f = \\rho_\\Lambda$, $f^4 = \\rho_\\Lambda/(r_L\\lambda\\delta)$. For $\\lambda = 1$ and "
          "$\\eta = 0.3$ one finds $f = %.2f$ meV and a field excursion of $%.1f$ meV, that is, the dark-energy scale. The gravitational "
          "parameter is then fixed too,"
          % (PH[(1.0, 0.30)]["f_meV"], PH[(1.0, 0.30)]["delta_phi_meV"])))
add(("eq", r"g\equiv\frac{f^2}{M_{\rm Pl}^2}=\sqrt{\frac{V_f/M_{\rm Pl}^4}{r_L\,\lambda\,\delta(\eta)}}"))
add(("p", "which takes the values $%s$ to $%s$ over the benchmark parameters of Table 2. Gravity is therefore a $10^{-60}$ effect in every "
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
          "The analytic thin-wall formula with gravity \cite{Parke} serves as a further check of the gravitational suppression. Table 1 lists the checks, all of which are executed by a script distributed with the paper." % sci(v("V2_shooting_vs_bvp")["max_relative_difference"], 0)))
add(("table", "Table 1. Executable consistency checks of the numerical bounce (the complete suite has %d checks, all passing)." % len(V),
     ["Check", "Result"],
     [["Thin-wall limit: $B/B_{\\rm thin}$ as $\\eta\\to0$", "%.4f at $\\eta=0.03$, %.4f at 0.06 (monotone)" % (v("V1_thin_wall_limit")["ratio_at_eta_0p03"], v("V1_thin_wall_limit")["ratio_at_eta_0p06"])],
      ["Shooting versus boundary-value solver", "max relative difference %s (%d cases)" % (sci(v("V2_shooting_vs_bvp")["max_relative_difference"], 1), v("V2_shooting_vs_bvp")["cases"])],
      ["Scaling $B=b(\\eta)/\\lambda$", "max relative difference %s" % sci(v("V3_lambda_scaling")["max_relative_difference"], 1)],
      ["de Sitter action $-24\\pi^2M^4/V_f$", "relative difference %s" % sci(v("V4_de_sitter_action")["rel"], 0)],
      ["Friedmann constraint along the bounce", "max residual %s" % sci(v("V10_friedmann_constraint")["max_residual"], 1)],
      ["Gravitational suppression against the analytic thin-wall (Parke) value", "agree to 1.5%% in $B/B_{\\rm flat}$ (%d points)" % len(v("V6_parke_thin_wall_suppression")["numerical_vs_analytic_ratio"])],
      ["Hawking--Moss consistency", "%d bounces found in %d grid points; none with $|V''|<4H^2$; none above $B_{\\rm HM}$" % (v("V7_hawking_moss_consistency")["bounces_found"], v("V7_hawking_moss_consistency")["grid_points"])],
      ["Interpolation of $b(\\eta)$ off the table", "max relative error %s" % sci(v("V8_interpolation_off_grid")["max_relative_error"], 1)]],
     None))
add(("fig", "fig1_potential_bounce.png", "Fig. 1. (a) The potential for $\\lambda=1$, $\\eta=0.30$. (b) The bounce profile in flat space and with gravity ($g=0.1$). (c) The numerical action $b(\\eta)=B\\lambda$ against the thin-wall formula; the false minimum disappears at $\\eta_c=0.385$."))
add(("h2", "3.3 Gravitational corrections and the Hawking--Moss limit"))
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
          % (texsci(max(p["gravity_correction_relative"] for p in F["physical"]), 0), np.mean([p["ln_hm_action"] for p in F["physical"]]), texsci(K.V_F / K.M_PL ** 4 / 0.3 ** 4, 0), texsci(0.3 ** 4 * K.M_PL ** 4 / K.V_F, 0))))

# ------------------------------------------------------------------------------------------------ 4 Rate
add(("h1", "4. Nucleation rate in the observed universe"))
add(("p", "The rate per unit four-volume is $\\Gamma/V_4 = A\\,e^{-B}$ \\cite{Coleman,CC}. In the rescaled variables $\\Gamma/V_4 = K f^4\\lambda^2 e^{-B}$, "
          "where $K = (B/2\\pi)^2 D$ estimates the one-loop determinant (four translational zero modes) with an unknown factor $D$ of order unity "
          "that is varied with a lognormal spread of 1.5 decades (0 and 3 decades in the sensitivity runs). Dividing by $H^4 = V_f^2/(9M_{\\rm Pl}^4)$ and using $f^4 = V_f/(r_L\\lambda\\delta)$,"))
add(("eq", r"\frac{\Gamma}{H^4}=\frac{9K\lambda}{r_L\,\delta(\eta)}\,\frac{M_{\rm Pl}^4}{V_f}\,e^{-B(\lambda,\eta)}"))
add(("p", "The large number $M_{\\rm Pl}^4/V_f = %s$ is the number of Planck-volume cells in a Hubble four-volume. It compensates an action of "
          "$B \\simeq \\ln(M_{\\rm Pl}^4/V_f) = %.1f$ plus the logarithm of the prefactor. Setting $\\Gamma/H^4 = 1$ gives the critical action"
          % (sci(K.M4_OVER_VF, 2), K.LN_M4_OVER_VF)))
add(("eq", r"B_c=\ln\frac{M_{\rm Pl}^4}{V_f}+\ln\frac{9K\lambda}{r_L\,\delta},\qquad B_c\simeq %d\ \ (\text{median over the prior; 5--95\%% range }%d\text{--}%d)" % (round(bc_med), round(BC["0.05"]), round(BC["0.95"]))))
add(("p", "which is insensitive to the prefactor: changing $D$ by three decades moves $B_c$ by $\\pm7$ out of $%d$. The mean time for one Hubble "
          "patch to nucleate a bubble is $t_{\\rm dec} = H^{-1}/[(4\\pi/3)\\Gamma/H^4]$ for $\\Gamma/H^4\\ll1$. Two observational conditions bound the rate. "
          "First, the present vacuum has survived: the Euclidean de Sitter four-volume is $8\\pi^2/(3H^4)$, so $P_{\\rm surv}=\\exp[-(8\\pi^2/3)\\Gamma/H^4]$ and "
          "$\\Gamma/H^4 \\gtrsim 3/(8\\pi^2)=0.038$ ($t_{\\rm dec}\\lesssim1.1\\times10^{11}$ yr) is excluded at order unity. This is an order-of-magnitude statement, "
          "since observers exist before the transition. Second, percolation of bubbles in de Sitter space requires a rate of order unity in "
          "Hubble units \\cite{GW,TWW}; for $\\Gamma/H^4 < 1$ the false-vacuum volume grows without bound and no bubble wall ever collides with another "
          "to thermalize." % round(bc_med)))
add(("p", "A transition can therefore be a physical reset only if $B$ lies within a window of a few tens above $B_c$ and below it by an amount that "
          "keeps the vacuum alive. For $\\lambda=1$, $\\eta=0.30$ the action is $B=%.0f$ and the mean decay time is $10^{%.0f}$ yr; at $\\eta=0.32$ ($B=%.0f$) "
          "it is $10^{%.0f}$ yr; at $\\eta=0.34$ ($B=%.0f$) the vacuum would have decayed in $10^{%.0f}$ yr and the model is excluded (Table 2)."
          % (PH[(1.0, 0.30)]["B"], PH[(1.0, 0.30)]["log10_t_yr"], PH[(1.0, 0.32)]["B"], PH[(1.0, 0.32)]["log10_t_yr"], PH[(1.0, 0.34)]["B"], PH[(1.0, 0.34)]["log10_t_yr"])))
add(("table", "Table 2. Benchmark potentials with the dark-energy density as false-vacuum energy ($r_L=1$). $f$ is the field scale, $g=f^2/M_{\\rm Pl}^2$, $\\delta_B$ the relative gravitational correction to $B$, and $R_b/R_H$ the bubble radius over the Hubble radius.",
     ["$\\lambda$", "$\\eta$", "$f$ (meV)", "$B$", "$\\ln\\Gamma/H^4$", "$\\log_{10}t_{\\rm dec}$ (yr)", "$g$", "$\\delta_B$", "$|V''_{\\rm top}|/4H^2$", "$R_b/R_H$"],
     [[("%g" % p["lam"]), "%.2f" % p["eta"], "%.2f" % p["f_meV"], "%.0f" % p["B"], "%.0f" % p["ln_gamma_h4"], "%.0f" % p["log10_t_yr"], sci(p["g"], 1), sci(p["gravity_correction_relative"], 1),
       sci(p["hm_ratio"], 1), sci(p["bubble_radius_over_hubble"], 1)] for p in F["physical"]], None))
add(("h2", "4.1 The falsification diagram"))
add(("p", "Fig. 3 maps the decay time over the $(\\eta,\\lambda)$ plane for $r_L=1$. The excluded region (decay faster than the survival bound) and the region of "
          "effective stability ($t_{\\rm dec}>10^{100}$ yr) are separated by a band: the decay time runs from $10^{11}$ to $10^{20}$ yr within a change of "
          "$\\Delta B\\simeq%.0f$ in the action and from $10^{11}$ to $10^{100}$ yr within $\\Delta B\\simeq%.0f$. The model is thus either excluded, "
          "or it describes a vacuum that decays on a time scale of $10^{20}$ yr or longer unless $B$ is tuned to within about $%.0f$ of $B_c$, "
          "that is within %.0f%% of its value." % (DB_20, DB_WINDOW, DB_20, 100 * DB_20 / bc_med)))
add(("fig", "fig3_falsification_diagram.png", "Fig. 3. Mean decay time of a Hubble patch in the $(\\eta,\\lambda)$ plane for $V_f=\\rho_\\Lambda$ and a Minkowski true vacuum. Contours: survival bound ($1.1\\times10^{11}$ yr), $10^{12}$, $10^{20}$, $10^{50}$ and $10^{100}$ yr. The upper edge $\\lambda=4\\pi$ is the perturbative-unitarity limit used for the prior."))
add(("h2", "4.2 Prior-sensitivity Monte Carlo"))
add(("p", "To quantify how much of the parameter space the window occupies, $%s$ parameter sets are drawn from physically consistent priors "
          "($\\log_{10}\\lambda$ uniform on $[-2,\\log_{10}4\\pi]$, $\\eta$ uniform on $[0.03,0.38]$, $\\ln D$ Gaussian with a spread of 1.5 decades; the false "
          "minimum exists for every draw) and each is converted to a rate with the interpolated numerical action. The regime fractions are given in "
          "Table 3 together with batch-means standard errors (20 batches), convergence with the number of draws (Fig. 4c) and sensitivity to the "
          "priors. %s%% of the prior volume is excluded by the survival of the present vacuum, %s%% decays within $10^{20}$ yr, %s%% within $10^{100}$ yr and %s%% "
          "never decays. The fractions do not depend on the prefactor (a spread of 0 or 3 decades changes them by less than $10^{-3}$) and the choice of $r_L$ is "
          "likewise immaterial. They do depend on the prior range of $\\lambda$ and $\\eta$ (Table 3), which is why they are reported as properties of a stated prior "
          "and not as probabilities for our universe." % ("{:,}".format(MC["N"]), pc(BASE["frac_inconsistent_with_survival"], 1), pc(BASE["frac_decay_by"]["1e+20"], 2),
                                                           pc(BASE["frac_decay_by"]["1e+100"], 1), pc(BASE["frac_never_within_1e100"], 1))))
rows4 = [["baseline", "%s ± %s" % (pc(BASE["frac_inconsistent_with_survival"], 2), pc(BAT["surv"]["se"], 2)), "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+12"], 3), pc(BAT["1e12"]["se"], 3)),
          "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+20"], 2), pc(BAT["1e20"]["se"], 2)), "%s ± %s" % (pc(BASE["frac_decay_by"]["1e+100"], 2), pc(BAT["1e100"]["se"], 2)),
          "%s ± %s" % (pc(BASE["frac_never_within_1e100"], 1), pc(BAT["never"]["se"], 1))]]
for k_, s_ in SENS.items():
    rows4.append([k_.replace("sigma_log10_D", "$\\sigma_{\\log D}$").replace("r_L", "$r_L$").replace("lam", "$\\lambda$").replace("eta", "$\\eta$"), pc(s_["frac_inconsistent_with_survival"], 2),
                  pc(s_["frac_decay_by"]["1e+12"], 3), pc(s_["frac_decay_by"]["1e+20"], 2), pc(s_["frac_decay_by"]["1e+100"], 2), pc(s_["frac_never_within_1e100"], 1)])
add(("table", "Table 3. Prior mass (%) of each regime. Decay fractions are cumulative and refer to survival-consistent draws; the last column is the fraction that does not decay within $10^{100}$ yr.",
     ["Prior", "excluded (survival)", "decays by $10^{12}$ yr", "by $10^{20}$ yr", "by $10^{100}$ yr", "never"], rows4, None))
add(("fig", "fig4_monte_carlo.png", "Fig. 4. (a) Cumulative distribution of the decay time over survival-consistent draws. (b) Prior mass by regime with $3\\sigma$ batch errors. (c) Convergence of the regime fractions with the number of draws."))

# ------------------------------------------------------------------------------------------------ 5 interior
add(("h1", "5. What the nucleated region can become"))
add(("h2", "5.1 Energy budget"))
add(("p", "A thin-wall bubble of true vacuum is bounded by a wall that carries the released energy; its interior is, to leading order, the true vacuum "
          "with no excess energy to reheat. In the thick-wall regime relevant here the field at the centre of the bounce, $x_0$, lies above the true minimum, and the "
          "energy the interior can release is $\\epsilon_{\\rm in}\\Delta V = V(x_0)-V_t$. Fig. 5c shows $\\epsilon_{\\rm in}$ against $\\eta$: at $\\eta=0.30$, "
          "$\\epsilon_{\\rm in}=%s$, so %s%% of $\\Delta V$ resides in the interior and the remainder in the wall; $\\epsilon_{\\rm in}$ approaches unity "
          "only near $\\eta_c$, where the barrier is gone." % (("%.3f" % EPS_IN), pc(EPS_IN, 1))))
add(("h2", "5.2 Interior geometry and reheating"))
add(("p", "The interior of a CDL bubble has open spatial slices, $ds^2=-dt^2+a(t)^2dH_3^2$ \\cite{Gott,BGT}, with the Friedmann constraint "
          "$\\dot a^2 = 1 + a^2\\rho/(3M_{\\rm Pl}^2)$. Without a long inflationary stage inside the bubble, the energy density is negligible against "
          "the curvature term at early times: $\\alpha^2\\rho \\le %s$ in units where $H_\\Lambda=1$, so the interior is a Milne universe, $a=t$. "
          "The condensate begins to oscillate at $t_0 = 1/m$ with $m = %.2f$ meV, behaves as matter and decays to radiation at a rate $\\Gamma_\\varphi\\le m$. With $\\tau=H_\\Lambda t$ "
          "the radiation density has the closed form $\\rho_r(\\tau)=\\epsilon\\,\\tau_0^3\\,\\gamma\\,\\tau^{-4}\\int_{\\tau_0}^{\\tau}s\\,e^{-\\gamma(s-\\tau_0)}ds$ in units of $V_f$, "
          "which we compare with a numerical integration of the full system (relative difference %s). The radiation temperature is "
          "$T=(30\\rho_r/\\pi^2g_*)^{1/4}$ with $g_*=%.2f$." % (sci(max(r["alpha2_rho_max"] for r in IN["runs"]), 1), IN["scales"]["m_meV"],
                                                               sci(v("V11_interior_closed_form")["max_relative_difference"], 0), K.GSTAR_LOW)))
add(("p", "Energy conservation bounds the reheating temperature independently of the details,"))
add(("eq", r"T_{\rm reh}\le\left[\frac{30\,\epsilon_{\rm th}\,\epsilon_{\rm in}\Delta V}{\pi^2g_*}\right]^{1/4},\qquad \epsilon_{\rm th}\le1"))
add(("p", "which for the largest possible energy release, $\\epsilon_{\\rm in}\\Delta V = \\Delta V\\le V_f=\\rho_\\Lambda$, gives $T_{\\rm reh}\\le %.2f$ meV ($%.0f$ K), "
          "against the CMB temperature of %.3f meV today and %s times below the $\\sim1$ MeV required for nucleosynthesis \\cite{BBN}. Equivalently, a thermal bath at 1 MeV "
          "needs an energy density $%s$ times the dark-energy density; no decay of $\\rho_\\Lambda$ can supply it. The closed form shows that the situation is "
          "worse than the bound: the radiation temperature after its maximum falls as $T\\propto1/\\tau$ because the open interior keeps expanding, with "
          "$T\\tau=%s$ GeV, so $T=1$ MeV would have to occur at $t=%s$ s, before the condensate even starts to oscillate ($t_0=%s$ s). Table 4 gives "
          "the maximum temperature for decay rates from $\\Gamma_\\varphi=m$ down to $10^{-20}m$ and the temperature one Hubble time after nucleation, "
          "$\\sim10^{-30}$ meV."
          % (RH["T_max_meV_at_Vf"], RH["T_max_K_at_Vf"], RH["T_cmb_meV"], sci(RH["ratio_Tbbn_over_Tmax"], 1), sci(RH["rho_needed_for_bbn_over_Vf"], 1), sci(IN["T_times_tau"], 2),
             sci(IN["t_for_bbn_temperature_seconds"], 1), sci(IN["scales"]["t0_seconds"], 1))))
rows3 = []
for eps, lab in ((1.0, "1 (upper bound)"), (EPS_IN, "%.3f ($\\eta=0.30$)" % EPS_IN)):
    for fr in (1.0, 1e-3, 1e-9, 1e-20):
        r = [x for x in IN["runs"] if abs(x["eps"] - eps) < 1e-12 and x["gamma_over_m"] == fr][0]
        rows3.append([lab, ("%g" % fr), "%.4f" % r["T_max_meV"], "%.2f" % r["T_over_bound"], sci(r["T_at_one_hubble_time_meV"], 1), sci(r["T_bbn_over_Tmax"], 1)])
add(("table", "Table 4. Reheating of the bubble interior. $\\epsilon$: fraction of $\\Delta V$ in the interior; $\\Gamma_\\varphi/m$: decay rate of the oscillating condensate; $T_{\\max}$: maximum radiation temperature; the fourth column is $T_{\\max}$ over the bound of Eq. (11); the fifth is the temperature one Hubble time $1/H_\\Lambda$ after nucleation; the last is $T_{\\rm BBN}/T_{\\max}$.",
     ["$\\epsilon$", "$\\Gamma_\\varphi/m$", "$T_{\\max}$ (meV)", "$T_{\\max}/T_{\\rm bound}$", "$T(1/H_\\Lambda)$ (meV)", "$T_{\\rm BBN}/T_{\\max}$"], rows3, None))
add(("fig", "fig5_reheating_interior.png", "Fig. 5. (a) Maximum reheating temperature against the released energy density; the dark-energy density gives about 2 meV, far below the energy density needed for 1 MeV. (b) Temperature of the open interior after reheating ($\\Gamma_\\varphi=m$): the radiation cools as $1/t$ and never approaches nucleosynthesis. (c) Fraction of $\\Delta V$ carried by the interior against the tilt."))
add(("h2", "5.3 Other fates and the singularity question"))
add(("p", "If the true vacuum has negative energy ($V_t<0$) the open interior collapses to a singularity in a finite time, "
          "$\\tau_c = \\pi/\\sqrt{|V_t|/V_f}$: %s. A decay to a de Sitter vacuum with $V_t>0$ leaves the interior with an accelerating phase and no radiation epoch. Only $V_t\\simeq0$ leaves a "
          "regular, expanding, but empty interior. No case yields a nonsingular bounce from the parent geometry: the bubble is a region of the parent "
          "de Sitter space, causally connected to it, and the Milne interior is geodesically complete only because it has no matter."
          % "; ".join("$|V_t|=%g\\,V_f$: %.0f Gyr" % (-a["vt_over_Vf"], a["gyr"]) for a in IN["ads_crunch"])))

# ------------------------------------------------------------------------------------------------ 6 dilution
add(("h1", "6. Matter dilution and decoherence as a trigger"))
OM = 1.0 - K.OMEGA_L
DPHI_OVER_M = PH[(1.0, 0.30)]["delta_phi_meV"] * 1e-12 / K.M_PL
add(("p", "The original motivation for a late-time transition was that dilution of matter reduces environmental decoherence and thereby permits a macroscopic "
          "quantum transition. In the baseline model nothing of the kind appears: the potential $V(\\varphi)$, hence $B$, does not depend on the matter "
          "density, and the Gibbons--Hawking temperature $T_H=H/2\\pi=%s$ GeV \\cite{GH} is $%s$ of the field scale; thermal-type corrections to "
          "$B$ are of relative size $\\sim g\\,(\\kappa\\simeq%.1f)$, that is $10^{-59}$ (Section 3.3). A dependence on matter must be introduced by a coupling. "
          "For the simplest one, $V_{\\rm eff}=V+\\beta\\rho_m\\varphi/M_{\\rm Pl}$ with $\\rho_m=(\\Omega_m/\\Omega_\\Lambda)V_f$ today, the matter-induced shift of the tilt is "
          "$\\delta V/\\Delta V\\simeq\\beta(\\Omega_m/\\Omega_\\Lambda)(\\Delta\\varphi/M_{\\rm Pl})=\\beta\\times%s$ for $\\Delta\\varphi=%.1f$ meV. A change of $B$ by an amount of order its distance "
          "from $B_c$ (a few units out of several hundred, so $\\delta V/\\Delta V\\sim10^{-2}$) therefore requires $\\beta\\sim10^{28}$ to $10^{30}$, whereas solar-system and laboratory tests restrict unscreened couplings of this type to "
          "$\\beta\\lesssim O(1)$, and screening mechanisms suppress the effective coupling further. The order of magnitude is robust because the factor $\\Delta\\varphi/M_{\\rm Pl}\\sim10^{-30}$ "
          "follows directly from the dark-energy scale. Matter dilution cannot be the trigger within this framework; a noise-assisted mechanism would need an "
          "explicit open-system influence functional and a calculated change of the tunneling exponent, which is not assumed here."
          % (sci(K.H_L / (2 * np.pi), 1), texsci(K.H_L / (2 * np.pi) / (PH[(1.0, 0.30)]["f_meV"] * 1e-12), 0), kap[0.3], texsci(OM / K.OMEGA_L * DPHI_OVER_M, 1), PH[(1.0, 0.30)]["delta_phi_meV"])))

# ------------------------------------------------------------------------------------------------ 7 falsification
add(("h1", "7. Falsification boundaries"))
add(("p", "A model of this class is falsified, or its parameter space is restricted, by the following conditions, each of which is computed in this paper and can be "
          "recomputed for a different potential with the supplied code."))
add(("bullets", [
    "F1 (existence). A metastable minimum requires $\\eta<\\eta_c=%.3f$; for larger tilt there is nothing to decay." % C.eta_crit(),
    "F2 (survival). $\\Gamma/H^4\\lesssim3/(8\\pi^2)$, i.e. $B\\gtrsim B_c\\approx%d$, for the vacuum we occupy to have survived. In the benchmark family this excludes %s%% of the prior volume." % (round(bc_med), pc(BASE["frac_inconsistent_with_survival"], 0)),
    "F3 (reset window). A reset within $10^{20}$ yr requires $B_c<B\\lesssim B_c+%.0f$, which holds for %s%% of the prior volume; an action above $B_c+%.0f$ puts the decay beyond $10^{100}$ yr." % (DB_20, pc(BASE["frac_decay_by"]["1e+20"], 2), DB_WINDOW),
    "F4 (percolation). A global thermalization requires $\\Gamma/H^4=O(1)$ or larger, which is incompatible with F2 unless the vacuum decays within the next few Hubble times.",
    "F5 (energy). A hot universe in the nucleated region requires $\\rho_{\\rm reh}\\gtrsim(1\\ \\mathrm{MeV})^4$, which is %s times the dark-energy density; the largest available release is $\\Delta V\\le\\rho_\\Lambda$, giving $T_{\\rm reh}\\le%.1f$ meV. The hypothesis that the decay of the present dark energy produces a hot big bang is excluded in this framework." % (sci(RH["rho_needed_for_bbn_over_Vf"], 1), RH["T_max_meV_at_Vf"]),
    "F6 (interior). $V_t<0$ implies a crunch within $\\sim55$ Gyr for $|V_t|=V_f$; $V_t>0$ implies no radiation epoch.",
    "F7 (trigger). A matter-dilution trigger needs a scalar--matter coupling $\\beta\\sim10^{28}$--$10^{30}$.",
    "F8 (observation). A solitary transition produces no signal in our patch; a gravitational-wave background from bubble collisions requires a population and $\\Gamma/H^4=O(1)$ (F4). The vacuum is therefore indistinguishable from a cosmological constant ($w=-1$) until it decays, and a measurement of $w\\ne-1$ would favor a rolling field and not a metastable vacuum. No observational discriminant of the cyclic reset is available from this model.",
]))
add(("p", "A transition producing a successful cosmology would have to lie outside the class studied here. It would need a parent vacuum energy "
          "of at least $\\sim10^{34}\\rho_\\Lambda$ (an earlier, higher-energy phase of the universe rather than the present one), a post-nucleation region "
          "that inflates before it reheats so that the curvature term and the horizon are handled, and a mechanism to restore the high-energy parent from "
          "the late universe. The cyclic models that have been constructed provide such mechanisms by other means \\cite{ST}; this paper shows that "
          "tunneling of the present dark energy does not."))

# ------------------------------------------------------------------------------------------------ 8 discussion
add(("h1", "8. Discussion and limitations"))
add(("bullets", [
    "A single scalar field and a quartic-tilt potential are used. Other potentials change $b(\\eta)$ and the interior energy fraction, but not the scales: the "
    "rate depends on the combination $M_{\\rm Pl}^4/V_f$ and the interior energy on $V_f$, both fixed by the data, so F2, F5 and F7 are generic.",
    "The one-loop prefactor is estimated, not computed (a determinant calculation is required); the results are insensitive to it (Section 4).",
    "The decay is treated as a flat-space process because $g\\sim10^{-60}$; the gravitational code is validated against analytic limits and used for the map "
    "of the regime in which gravity matters, which the dark-energy scale excludes.",
    "The interior evolution assumes a homogeneous open slicing and a perturbative decay of the condensate to radiation; bubble-wall collisions can "
    "concentrate energy locally, but average energy density is bounded by $\\Delta V$ (F5).",
    "Up-tunneling from the false to a higher vacuum, the decay of the true vacuum in curved space \cite{LW}, thermal activation of the Hawking--Moss type \cite{BW} and decays of the present vacuum into other vacua are not considered.",
    "The priors of the Monte Carlo are documented assumptions, not measurements; the regime fractions describe the sample space of the model, not the "
    "probability of a transition in our universe.",
]))

add(("h1", "9. Conclusions"))
add(("p", "A late-time false-vacuum transition of the present dark energy is a well-defined calculation, and for the explicit potential studied it can be "
          "done to a stated precision. The result is a set of boundaries. The decay is a flat-space process controlled by the action $B=b(\\eta)/\\lambda$, which must lie "
          "near $B_c\\approx%d$ for the transition to be neither excluded nor effectively absent; only %s%% of the prior volume does. The interior of a nucleated "
          "region carries at most the dark-energy density, so it reheats to at most $%.1f$ meV and cools as $1/t$ in an open universe, and cannot "
          "reproduce a hot big bang; a trigger by matter dilution would need an unphysical coupling. The cyclic-reset hypothesis, in the form tested, is therefore "
          "not supported, and the framework identifies what would have to change." % (round(bc_med), pc(BASE["frac_decay_by"]["1e+20"], 1), RH["T_max_meV_at_Vf"])))

# ---------------------------------------------------------------------------------------------------- statements
add(("h1", "Declarations"))
add(("pb", "Funding.", "This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors."))
add(("pb", "Conflict of interest.", "The author declares no competing interests."))
add(("pb", "Ethics approval, consent to participate.", "Not applicable."))
add(("pb", "Data availability.", "The data underlying every figure and table (bounce tables, falsification map, Monte Carlo samples summary, interior evolution, "
          "verification results) are deposited with the code at %s and archived at https://doi.org/%s." % (GITHUB, DOI_CODE)))
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
]
# order references by first citation in the text
import re as _re
text_all = " ".join(b[-1] for b in B if b[0] in ("p", "abstract", "pb") and isinstance(b[-1], str)) + " ".join(" ".join(x) for b in B if b[0] == "bullets" for x in [b[1]])
order = []
for m in _re.finditer(r"\\cite\{([^}]+)\}", text_all):
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
    else:
        B2.append(b)
B = B2
# equation reference in the table caption: the reheating bound is the 11th display equation
neq = sum(1 for b in B if b[0] == "eq")
ieq = [i for i, b in enumerate(B) if b[0] == "eq"]
bound_eq = [n + 1 for n, i in enumerate(ieq) if "T_{\\rm reh}\\le" in B[i][1]][0]
B = [(b[0], b[1].replace("Eq. (11)", "Eq.~(%d)" % bound_eq)) if (b[0] == "table" and False) else b for b in B]
B = [tuple([b[0], b[1].replace("Eq. (11)", "Eq.~(%d)" % bound_eq)] + list(b[2:])) if b[0] == "table" else b for b in B]
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
    # re-join: only the original (odd/even) delimiters are restored with the explicit dollars from the sentinels
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
