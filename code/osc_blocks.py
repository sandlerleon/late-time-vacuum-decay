# -*- coding: utf-8 -*-
"""Manuscript blocks for Section 5.6 (Option B: the same region oscillating after a decay to anti-de Sitter); numbers from oscillation.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(HERE, "oscillation.json"), encoding="utf-8"))
OV = json.load(open(os.path.join(HERE, "oscillation_validation.json"), encoding="utf-8"))


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "⟦" + texsci(x, d) + "⟧"
PER = O["periodic"]
FL = O["stability"]["scalar_floquet"]
DIS = {r["upsilon"]: r for r in O["dissipation"]}
PHYS = {r["kappa"]: r for r in O["physical"]["rows"]}
HYS = O["hysteresis"]["a_max"]
MU = [f["energy_multiplier_per_cycle"] for f in FL]
LNMU = O["ln_multiplier_m005"]
RUN = [r["upsilon_times_period"] for r in O["dissipation"] if r["runaway"]]
STAB = [r["upsilon_times_period"] for r in O["dissipation"] if not r["runaway"]]
VAL = OV["validation_integrators"]
GOOD = {k: v for k, v in VAL.items() if "1e-9" not in k}
BC = OV["bounce_continuity"]
LAWS = {}
for lab, ct in OV["hysteresis"].items():
    res_ = [c for c in ct if abs(c["delta_a_max"]) > 1e-3 * c["a_max"] and c["a_max_next"] < 2e4]
    LAWS[lab] = max([abs(c["law"]) / max(abs(c["delta_a_max"]), abs(c["W"]) / 3.0) for c in res_] or [0.0])
G_SCALAR = [c["G"] for c in OV["hysteresis"]["phi0=1e-3, no dissipation"] if c["a_max"] > 1e8]
PD = OV["phase_diagram"]
NCLS = {}
for g in PD:
    NCLS[g["cls"]] = NCLS.get(g["cls"], 0) + 1
LONG = OV["long_runs"]
ROB = {r["label"]: r for r in OV["robustness"]}
TR_ = OV["transitions"]
MB = OV["model_B"]["rows"]


def blocks():
    out = []
    add = out.append
    add(("h2", "5.6 The same region: turnaround, bounce and repeated cycles (Option B)"))
    add(("p", "A cyclic universe is often pictured as a sinusoid. The quantity that can oscillate is the Hubble rate, not the scale factor, and the prescription"))
    add(("eq", r"H(t)=H_0\sin\omega t,\qquad a(t)=a_0\exp\left[-\frac{H_0}{\omega}\cos\omega t\right],\qquad \rho+p=-2M_{\rm Pl}^2\dot H"))
    add(("p", "keeps $a>0$, with a bounce at each zero of $H$ with positive slope and a turnaround at each zero with negative slope. In spatially flat general relativity the last relation shows that the null energy condition is violated whenever $\\dot H>0$, that is "
              "during exactly half of every period, so a sinusoid cannot be produced by conventional matter and has to be replaced by a dynamical model with a stated bounce mechanism \\cite{NB08,BP15}, as in ekpyrotic and cyclic constructions \\cite{KOST,ST}. "
              "The calculations of this paper supply the turnaround: a decay of the present vacuum into a true vacuum with $V_t<0$ makes the interior recollapse (Sections 5.3--5.5). For the bounce we use the effective Friedmann equation of loop quantum cosmology "
              "\\cite{APS06,AS11,Singh09}, an effective model and not established physics,"))
    add(("eq", r"H^2=\frac{\rho}{3M_{\rm Pl}^2}\left(1-\frac{\rho}{\rho_c}\right)+\frac{1}{a^2},\qquad \dot H=-\frac{\rho+p}{2M_{\rm Pl}^2}\left(1-\frac{2\rho}{\rho_c}\right)-\frac{1}{a^2}"))
    add(("p", "for the open interior, with radiation and the scalar oscillating about its anti-de Sitter minimum, $V=V_t+m^2\\varphi^2/2$, and an optional dissipation $\\Upsilon$ into radiation. The hierarchy $\\rho_c/|V_t|\\sim10^{120}$ cannot be integrated directly; "
              "the cycles are simulated at $\\rho_c/|V_t|=10^4$ to $10^8$ in Planck units, and the scaling of each quantity with $|V_t|$ and $\\rho_c$ is measured and extrapolated (oscillation.py; checks V25--V26)."))
    add(("p", "With radiation alone the cycles are exactly periodic: over %d bounces the period varies by %s and the turnaround radius returns to %s. The period is $%.3f\\,\\pi\\sqrt3M_{\\rm Pl}/\\sqrt{|V_t|}$, scales as $|V_t|^{%.3f}$ and does not depend on $\\rho_c$ (the factor below one is the radiation content; with negligible radiation it tends to one). "
              "The bounce is regular: the density reaches $\\rho_c$ to $%s$, the minimum scale factor stays positive and the curvature scalar remains below $%.1f\\,\\rho_c/M_{\\rm Pl}^2$; the constraint residual is at most %s. The trajectory is far from a sine: $H$ is sharply peaked at the bounce, "
              "the fundamental harmonic carries %.1f%% of its power, and $\\dot H>0$ (an effective violation of the null energy condition) only during %s of each cycle, at the bounce. Repeated expansion and contraction therefore emerge, and the sinusoid does not (Fig. 11a)."
              % (PER["n_bounces"], sci(PER["period_spread"], 0), sci(PER["a_max_closure"], 0), PER["period_over_ads_crunch"], O["scaling"]["period_slope_in_Vt"], sci(PER["rho_bounce_over_rho_c"] - 1, 0), PER["max_curvature_R_over_rho_c"], sci(PER["max_residual"], 0),
                 100 * PER["fundamental_power_fraction"], sci(PER["fraction_of_cycle_with_Hdot_positive"], 1))))
    add(("p", "The periodic orbit is unstable (Fig. 11b, c). A perturbation of the radiation content only shifts the period, but any oscillation of the scalar is amplified from cycle to cycle: in the linear regime its energy grows by a factor between %.0f and %.0f per cycle over $m/H_{\\rm AdS}=%.0f$--$%.0f$ "
              "(the field is frozen near the bounce and released with a phase that does not return the energy it gained). Once it is no longer small, successive cycles grow, the cosmological hysteresis of oscillating models \\cite{ST12}: the turnaround radius rises from $%.0f$ to $%s$ in %d cycles. "
              "Dissipation into radiation competes with the amplification: the cycles run away for $\\Upsilon P\\le%.2f$ and are stabilised for $\\Upsilon P\\ge%.1f$, consistent with the criterion $\\Upsilon P\\gtrsim\\ln\\mu=%.2f$ ($P$ the period, $\\mu$ the energy multiplier). Stabilised cycles keep a constant entropy after a "
              "one-time increase of %s; runaway cycles produce entropy without bound (a factor %s in ten cycles for $\\Upsilon P=%.2f$), which is the entropy problem of cyclic models \\cite{Tolman,PS}."
              % (min(MU), max(MU), FL[0]["m_over_H_ads"], FL[-1]["m_over_H_ads"], HYS[0], sci(HYS[-1], 0), len(HYS) - 1, max(RUN), min(STAB), LNMU, sci(DIS[1e-3]["total_entropy_growth"] - 1, 0), sci(DIS[1e-4]["total_entropy_growth"], 0), DIS[1e-4]["upsilon_times_period"])))
    add(("fig", "fig11_oscillating_cycles.png", "Fig. 11. Option B. (a) Hubble rate over three cycles with radiation alone (simulated at $|V_t|=10^{-6}$, $\\rho_c=1$ in Planck units) against a sinusoid of the same period. (b) Turnaround radius against the cycle number for a scalar excitation with dissipation $\\Upsilon P$ from 0 to 31: runaway (hysteresis) below $\\ln\\mu$, periodic above. (c) Energy multiplier of a scalar perturbation per cycle against $m/H_{\\rm AdS}$.", 6.3))
    add(("p", "The amplification is not a numerical artifact. The equations are integrated continuously through every bounce, with no reset. The energy multiplier agrees to %s among an explicit eighth-order Runge--Kutta method at two tolerances and two maximum steps, "
              "an implicit Radau method and LSODA (relative tolerance at most $10^{-10}$); at $10^{-9}$ it is %.0f%% low because the phase of the field at the bounce is no longer resolved. Across the bounces the largest step-to-step change of the density is %s of its scale, "
              "the constraint residual is at most %s and the residual of the continuity equation at most %s. The growth obeys an energy-balance law: the continuity equation gives $\\Delta(\\rho a^3)=-W_n$ with $W_n=\\oint p\\,d(a^3)$ over a cycle, and at a turnaround "
              "$\\rho a^3=-3a_{\\max}$, so that $\\Delta a_{\\max,n}=W_n/3$. The simulations satisfy it to %s while the turnaround is curvature dominated (at larger radii the turnaround energy $\rho a^3$ is a cancellation below the integration precision); the loop integral is dominated by the scalar, which is what does the work. Once the scalar dominates, the turnaround "
              "occurs at $\\rho\\simeq0$ and the cycles grow by a constant factor, $G_n=a_{\\max,n+1}/a_{\\max,n}\\simeq%s$ in the run shown, a law of the hysteresis type \\cite{ST12}."
              % (sci(max(v["multiplier_rel_dev"] for v in GOOD.values()), 0), 100 * VAL["DOP853 1e-9 P/200"]["multiplier_rel_dev"], sci(BC["max_relative_step_change"], 0), sci(BC["constraint_residual_at_bounces"], 0),
                 sci(BC["continuity_residual_at_bounces"], 0), sci(max(LAWS.values()) + 1e-16, 0), sci(float(np.median(G_SCALAR)), 1) if G_SCALAR else "--")))
    add(("p", "Over 100 or more cycles the grid of masses $m$, dissipation $\\Upsilon P$ from 0 to 10 and initial amplitudes $10^{-8}$ and $10^{-3}$ gives %d stable and %d growing solutions out of %d (Fig. 12b); none shrinks, none loses its turnaround, and the boundary follows $\\Upsilon P\\approx\\ln\\mu$. "
              "Stable solutions stay stable over 1000 cycles, with $|G_n-1|\\le%s$ over the last 500, as does the radiation-only control. The instability does not depend on the bounce prescription: with $H^2=(\\rho/3)(1-(\\rho/\\rho_c)^2)$ the multiplier is %.1f, and it is %.0f for $\\rho_c=10$, "
              "%.0f for $|V_t|=10^{-4}$ and %.0f for $|V_t|=10^{-8}$; for $\\rho_c=0.1$ it is %.2f, a window in which the scalar is damped without dissipation, so that the multiplier depends on the phase with which the field crosses the bounce and is not a fixed number."
              % (sum(v for k, v in NCLS.items() if k == "stable"), sum(v for k, v in NCLS.items() if k.startswith("growing")), len(PD), sci(max(l_["max_abs_G_minus_1_last_500"] for l_ in LONG), 0),
                 ROB["n=2"]["multiplier"], ROB["rho_c=10"]["multiplier"], ROB["Vt=-1e-4"]["multiplier"], ROB["Vt=-1e-8"]["multiplier"], ROB["rho_c=0.1"]["multiplier"])))
    add(("p", "Vacuum transitions during the cycles were added in the semiclassical regime only. Excluding the epochs with $R_b|H|>0.1$ removes %s of the four-volume of a cycle for $R_b=10$ (the bounce is short and the scale factor is small there), so the transition probability per cycle is "
              "$1-\\exp(-\\Gamma V_{4,\\rm cyc})$ with essentially the whole cycle four-volume (Model B). A transition to a vacuum deeper by $|V_t|/2$ at a turnaround, with the energy either thermalised (Model C1) or left in a displaced field (Model C2), turns a stable sequence "
              "($\\Upsilon P=3$, $G_n=1$) into a growing one ($G_n\\simeq%.2f$ and %.2f): the deeper vacuum shortens the period, which lowers $\\Upsilon P$, and raises the multiplier, which moves the cycle across the stability boundary. The transition destabilises the cycles rather than regulating them (Fig. 12c)."
              % (sci(1 - [r for r in MB if r["R_b"] == 10.0][0]["valid_fraction_of_four_volume"] + 1e-16, 0), TR_["C1"]["G"][-1], TR_["C2"]["G"][-1])))
    add(("p", "For the physical case the scalings give, with $\\rho_c=0.41\\rho_{\\rm Pl}$, a bounce temperature of %s GeV: the hot epoch of each cycle comes from compression at the bounce, not from the energy of the tunneling. The period is %.0f Gyr for $|V_t|=\\rho_\\Lambda$ "
              "and %.1f Gyr for $|V_t|=15.7\\rho_\\Lambda$. Four obstacles remain. First, every cycle has negative vacuum energy, so none contains an accelerating epoch like the observed one. Second, nothing returns the field to the positive-energy vacuum: in the CDL formalism up-tunneling needs a de Sitter parent \\cite{LW}, and "
              "the hot bounce restores the symmetric phase from which the field rolls into the true vacuum for $c\\ge0$ (Section 4.6). Third, the tunneling formalism does not apply near the bounce, where the background changes on the time $1/H$ while the bubble has the size $R_b$, with $R_bH=%s$. "
              "Fourth, the bounce itself is an effective description at Planck density. The quantum event is therefore a single trigger, the decay into anti-de Sitter, followed by classical cycles; tunneling does not recur at each zero crossing, and a transition at every bounce would in any case be probabilistic. "
              "Table 14 summarises the tests."
              % (sci(O["physical"]["T_bounce_GeV"], 1), PHYS[1.0]["period_Gyr"], PHYS[15.7]["period_Gyr"], sci(O["cdl_near_bounce"]["R_b_times_H"], 0))))
    rows = [
        ["Oscillation", "repeated expansion and contraction", "yes: periodic with radiation alone, period $\\propto|V_t|^{-1/2}$"],
        ["Nonsingular bounce", "$a_{\\min}>0$, finite curvature", "yes, with the effective bound $\\rho\\le\\rho_c$ (not derived here)"],
        ["Vacuum transition", "a physically justified rate", "once (the decay to $V_t<0$, Section 4); no transition near the bounce ($R_bH=%s$) and no return to $V>0$" % sci(O["cdl_near_bounce"]["R_b_times_H"], 0)],
        ["Reheating", "radiation after the bounce", "yes, by compression ($T_b\\sim%s$ GeV), independent of the tunneling" % sci(O["physical"]["T_bounce_GeV"], 0)],
        ["Energy conservation", "small constraint residual", "residual $\\le%s$" % sci(PER["max_residual"], 0)],
        ["Entropy", "consistent across cycles", "constant if dissipation stabilises; unbounded growth otherwise"],
        ["Stability", "cycles persist under perturbations", "no: scalar energy $\\times%.0f$--$%.0f$ per cycle; stable only for $\\Upsilon P\\gtrsim\\ln\\mu$ (stable cases hold 1000 cycles)" % (min(MU), max(MU))],
        ["Hysteresis law", "growth explained quantitatively", "$\\Delta a_{\\max}=W_n/3$ (curvature-dominated turnarounds), constant $G_n$ once the scalar dominates"],
        ["Transition during the cycles", "regulates or destabilises", "destabilises: a stable sequence becomes growing ($G_n\\simeq%.1f$)" % TR_["C1"]["G"][-1]],
        ["Observed universe in a cycle", "a long epoch with $\\rho_\\Lambda>0$", "no: every cycle has $V<0$"],
    ]
    add(("table", "Table 14. Tests of an oscillating universe in the same region (Option B), with an effective loop-quantum-cosmology bounce and the turnaround from a decay to anti-de Sitter.", ["Test", "Success condition", "Result"], rows, None))
    add(("fig", "fig12_hysteresis_stability.png", "Fig. 12. Validation and long-term behaviour of Option B. (a) Change of the turnaround radius against the pressure-volume loop $W_n/3$ for the curvature-dominated cycles of three runs; the dashed line is the law $\\Delta a_{\\max}=W/3$. (b) Classification after 100 or more cycles over the mass and the dissipation (circles $\\varphi_0=10^{-8}$, squares $10^{-3}$). (c) A vacuum transition at the third turnaround turns a stable sequence into a growing one.", 6.3))
    return out
