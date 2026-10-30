# -*- coding: utf-8 -*-
"""Manuscript blocks for Sections 6.8-6.15: the cycle without prescribed steps, its causal test, stability, bounce validation, relics, failure map,
observational consequences and status. Numbers from continuous_cycles.json, cycle_scan.json, causal_controls.json, contraction_stability.json,
bounce_validation.json, compact_relics.json and predictions.json."""
import json
import os

import numpy as np

import continuous_cycles as CCm

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
CR, SCN, CAU, CS, BV, RL, PR = (J("continuous_cycles.json"), J("cycle_scan.json"), J("causal_controls.json"), J("contraction_stability.json"),
                                 J("bounce_validation.json"), J("compact_relics.json"), J("predictions.json"))
MEV = CCm.T_UNIT_EV * 1e-6
fo = lambda c: c.get("Y_fo_over_kappa", c.get("Y_late_over_kappa"))       # freeze-out yield (older files: Y_late_over_kappa)
C100 = J("continuous_cycles_100MeV.json") if os.path.exists(os.path.join(HERE, "continuous_cycles_100MeV.json")) else None
CM = J("cycle_map.json")
NA = J("numerics_admissibility.json") if os.path.exists(os.path.join(HERE, "numerics_admissibility.json")) else None


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "$" + texsci(x, d) + "$"
CYC = [c for c in CR["cycles"] if c["n"] >= 1 and "held_accelerated_efolds" in c]
N_CYC = len(CYC)
arr = lambda k: np.array([c[k] for c in CYC if c.get(k) is not None], dtype=float)
HELD, YK, MM, LNG, LNGS, PHB, TB = arr("held_accelerated_efolds"), arr("Y_held_over_kappa"), arr("max_matter_fraction"), arr("lnG"), arr("lnG_entropy"), arr("phi_bounce_over_v"), arr("T_bounce")
GDEV = float(np.max(np.abs(LNG - LNGS))) if LNG.size else float("nan")
# depth of the restoration: smallest |phi|/v in the expanding branch of the cycles whose full history is kept
_pm = [min(h[9] for h in c["hist"] if h[2] > 0 and h[9] > 0) for c in CR["cycles"][1:3] if c.get("hist")]
PHMIN_DEC = "%d--%d" % (int(-np.log10(max(_pm))), int(-np.log10(min(_pm)))) if _pm else "many"
SHB = [c.get("bounce_shear_fraction", 0) for c in CR["cycles"]]
P = CR["params"]
COLD = [a for a in BV["A"] if abs(a["rho_c"] - 1e36) < 1e26][0]         # the bounce too cold for the baseline dark matter
MB = "%.3g" % (P["m_dm"] * MEV)
A44 = [a for a in BV["A"] if abs(a["rho_c"] - 1e44) < 1e34][0]
ROWS = SCN["rows"]
grid = [r for r in ROWS if set(r["params"]) == {"m_dm", "rho_c"}]
marg = [r for r in ROWS if set(r["params"]) != {"m_dm", "rho_c"} and r.get("baseline") != "robust"]      # about m = 100 MeV, T_b = 180 MeV
robust = [r for r in ROWS if r.get("baseline") == "robust"]                                                 # about m = 10 MeV, T_b = 180 MeV
oaat = marg + robust
short = lambda r: r["classification"].split(" (")[0]
lab_r = lambda r: lab({k: v for k, v in r["params"].items() if k != "m_dm"})
n_work = sum(r["classification"] == "working" for r in grid)
SUM = CAU["summary"]
NOEK = [s for s in CS["scan"] if s["ek"] is None][0]
EK = [s for s in CS["scan"] if s["ek"] is not None]
LQC = [c for c in BV["C"] if c["bounce"] == "lqc"][0]
GH = [c for c in BV["C"] if c["bounce"] == "ghost"][0]
ms_ = lambda x: "%.3g" % x
pm = lambda v: "%.2f\\pm%.2f" % (np.mean(v), np.std(v))
MODES = PR["perturbations"]["mode_exit"]["modes"]
DS = PR["dark_sector"]


def lab(p):
    d = dict(p)
    out = []
    for k, v in d.items():
        if k == "m_dm":
            out.append("$m=%s$ MeV" % ms_(v * MEV))
        elif k == "rho_c":
            out.append("$T_b\\approx%s$ MeV" % ms_(v ** 0.25 * MEV))
        elif k == "C_T":
            out.append("$C_T=%g$" % v)
        elif k == "shear0":
            out.append("$\\Omega_{\\sigma,0}=10^{%d}$" % round(np.log10(v)))
        elif k == "bounce":
            out.append("ghost bounce")
        else:
            out.append("$%s=%g$" % ({"a_c": "a_c", "mu": "\\mu/H_0", "q": "q"}[k], v))
    return ", ".join(out)


def blocks():
    B = []
    add = B.append
    # ------------------------------------------------------------------------------------------------ 6.8
    add(("h2", "6.8 The cycle without prescribed steps: continuous reheating"))
    add(("p", "The cycles of Section 6.6 contain three prescribed steps: the dark-matter abundance is reset at freeze-out, the hot-phase restoration of the vacuum is applied "
              "through its adiabatic result, and the dissipation is capped where it is already overdamped. Each is a statement about dynamics that the equations should produce "
              "themselves. They are removed here (continuous_cycles.py): the scalar and the radiation are never reset, and the dark matter is a thermal species with its own "
              "Boltzmann equation \\cite{KT90,GG91}. Its particles have the field-dependent mass $m_{\\rm eff}=mA(\\varphi)$ and are in kinetic equilibrium with the bath, so that their comoving "
              "number $N=na^3$ and their energy and pressure per particle follow"))
    add(("eq", r"\dot N=-\langle\sigma v\rangle\frac{N^2-N_{\rm eq}^2}{a^3},\qquad n_{\rm eq}=c_n\frac{x^2K_2(x)}{2}T^3,\qquad \epsilon=m_{\rm eff}\frac{K_1(x)}{K_2(x)}+3T,\qquad x=\frac{m_{\rm eff}}{T}"))
    add(("p", "The scalar exchanges energy with the bath through the dissipation and through the field dependence of the particle masses (each particle's energy changes by "
              "$(m_{\\rm eff}/E)\\,dm_{\\rm eff}$ at fixed momentum):"))
    add(("eq", r"\ddot\varphi+(3H+\Upsilon)\dot\varphi+V_0'+F=0,\qquad F=n\frac{K_1}{K_2}mA'(\varphi),\qquad \dot\rho_{\rm th}=-3H(\rho_{\rm th}+p_{\rm th})+\Upsilon\dot\varphi^2+F\dot\varphi"))
    add(("p", "with $\\rho_{\\rm th}=\\rho_r+n\\epsilon$, $p_{\\rm th}=\\rho_r/3+nT$ and $\\Upsilon=C_TT^3$ \\cite{Berera95}; the temperature is a state variable and the total stress-energy is "
              "conserved by construction. The single coupling $A(\\varphi)$ now produces both the hot-phase thermal mass (a relativistic gas gives $F\\simeq nm_{\\rm eff}mA'/2T$) and "
              "the late hold (a cold gas gives $F=\\rho_*A'$), so the restoration of the vacuum and the freeze-out of the dark matter are outputs. In physical units $T=1$ is 1.82 meV, "
              "$C_T=1$ is a dissipation $T^3/f^2$ with $f=(H_0M_{\\rm Pl}^3)^{1/4}=2.1$ TeV, and $\\langle\\sigma v\\rangle$ is fixed once, by requiring the present dark-matter density "
              "in the held epoch of the repeating cycle ($%s$ cm$^3$ s$^{-1}$ for $m=%s$ MeV). This is %s times its value for freeze-out at constant mass in a radiation "
              "background, because in the cycle the dark matter freezes out while the field is still displaced and the dark matter heavier, and the later restoration of the "
              "field lowers its mass and heats the bath (sigv_calibration.json). Two numerical devices remain, and are stated as such: "
              "where the dissipation exceeds every other rate by $10^3$ the field moves at its terminal velocity, $\\dot\\varphi=-(V_0'+F)/(3H+\\Upsilon)$ (the overdamped limit, error "
              "$O(m_{\\rm eff}^2/\\Upsilon^2)$), and $H$ is projected onto the constraint once per doubling of the scale factor away from the turning points (Section 6.6 explains why). The "
              "corrections are relative to $H$ and are largest, up to %.0f%%, in the first chunks after a turning point, where $H$ passes through zero; relative to the density, "
              "the constraint residual stays below %s throughout." % (texsci(PR.get("sigv_cm3s", 6.35e-26), 1), MB, "%.3g" % CR["params"].get("sigv_scale", 1.0),
                                                                  100 * CR["projections"]["max_rel"],
                                                                  sci(CR["max_residual"], 0))))
    add(("p", "The first result is negative. With $m=%s$ MeV and a bounce at %.1f MeV, our cycle completes (release, turnaround at $\\ln a=%.2f$, a radiation-dominated bounce) but "
              "the next cycle is empty: the dark matter reaches the bounce with $x=m_{\\rm eff}/T=%.0f$, is annihilated during the hot contraction and is not regenerated "
              "($Y/\\kappa=%s$ in the next cycle), the field is frozen at $|\\varphi|=%.2fv$ by the overdamping, there is no matter era (peak matter fraction %.3f) and no dark-energy "
              "epoch. The prescribed steps of Section 6.6 hid both effects. The cap on $\\Upsilon$ removed the overdamping that freezes the field, and the freeze-out reset restored dark "
              "matter that a bounce of this temperature cannot produce. The cap was introduced in hot_cycles.py as a numerical convenience; it is not one, and the self-similar cycles of "
              "Section 6.6 are best read as the prescribed-step version of the cycle that the present section tests."
              % (MB, COLD["T_bounce_MeV"], CR["cycles"][0]["ln_a_max"], COLD["bounce_x"], texsci(max(fo(COLD["next"]), 1e-30), 1), COLD["next"]["phi_bounce_over_v"],
                 COLD["next"]["max_matter_fraction"])))
    add(("p", "The second result is that the same equations give working cycles once the bounce re-thermalizes the dark matter, $T_b\\gtrsim m_{\\rm eff}/5$. With $m=%s$ MeV, "
              "a bounce at %.0f MeV and today's anisotropy set to $\\Omega_\\sigma=10^{-30}$, %d cycles were integrated with no reset of any variable (Fig. 17, Table 18). In every "
              "cycle the field is frozen at the bounce at $|\\varphi|/v=%s$, the dark matter returns to equilibrium ($x_b=%.1f$--$%.1f$) and freezes out while the field is still "
              "displaced, i.e. while the dark matter is %s times heavier than $m$ (freeze-out yield $Y_{\\rm fo}/\\kappa=%s$); the field is then "
              "restored, which lowers the dark-matter mass and heats the bath, and the held epoch has $Y/\\kappa=%s$, the value compared with today (it fixes "
              "$\\langle\\sigma v\\rangle$, see below), "
              "the field is released from the hot phase when the dissipation falls below its mass and is damped toward $\\varphi=0$ by %s orders of magnitude, a matter era follows "
              "(peak matter fraction $%s$), then a dark-energy epoch with the field held (%s e-folds of accelerated expansion), a release, a turnaround and the next bounce. The cycles "
              "are not exactly self-similar: the value at which the field freezes depends on the phase of its oscillation when the overdamping sets in, and the held e-folds vary between "
              "%.2f and %.2f. The growth per cycle, $\\ln G_n=%s$, equals the entropy growth $\\frac{1}{3}\\ln(S_{n+1}/S_n)$, with $S=a^3T^3$ the radiation entropy at the turnaround, to within %s in every cycle (this "
              "measure omits the small contributions of the dark matter and the scalar): the second law holds and is "
              "absorbed by Tolman growth, as in Section 6.6, now without any prescribed step. The constraint residual is %s over the whole run."
              % (MB, A44["T_bounce_MeV"], N_CYC, pm(PHB), min(c.get("bounce_x", 0) for c in CR["cycles"]), max(c.get("bounce_x", 0) for c in CR["cycles"]), "%.1f" % (1 + float(np.mean(PHB)) ** 2 * 2 * (1 + P["q"]) * CCm.MC.VC * P["a_c"] ** 3 / CCm.RHO_M0), pm(arr("Y_fo_over_kappa")), pm(YK),
                 PHMIN_DEC, pm(MM), pm(HELD), HELD.min(), HELD.max(), pm(LNG), sci(GDEV, 0), sci(CR["max_residual"], 0))))
    rows = []
    for c in CR["cycles"][:8] + ([CR["cycles"][-2]] if len(CR["cycles"]) > 9 else []):
        rows.append([str(c["n"]), "%.0f" % (c.get("T_bounce", 0) * MEV) if c.get("T_bounce") else "--", "%.1f" % c["bounce_x"] if c.get("bounce_x") else "--",
                     "%.3f" % c["phi_bounce_over_v"] if c.get("phi_bounce_over_v") is not None else "--",
                     "%.2f" % fo(c) if fo(c) else "(today)", "%.3f" % c["Y_held_over_kappa"] if c.get("Y_held_over_kappa") else "(today)",
                     "%.3f" % c.get("max_matter_fraction", 0),
                     "%.2f" % c.get("held_accelerated_efolds", 0), "%.2f" % c["lnG"] if c.get("lnG") else "--", "%.2f" % c["lnG_entropy"] if c.get("lnG_entropy") else "--",
                     sci(max(c.get("bounce_shear_fraction", 0), 1e-300), 0) if c.get("bounce_shear_fraction") else "--"])
    if C100:
        c1 = [c for c in C100["cycles"] if c["n"] >= 1 and "held_accelerated_efolds" in c]
        a1 = lambda k: np.array([c[k] for c in c1 if c.get(k) is not None], dtype=float)
        add(("p", "A second run of %d cycles with $m=100$ MeV and the same bounce (continuous_cycles_100MeV.json) works as well: held e-folds $%s$, freeze-out yield $Y_{\\rm fo}/\\kappa=%s$ (with the radiation-background $\\langle\\sigma v\\rangle$), peak matter "
                  "fraction $%s$, $\\ln G_n=%s$, with the field freezing at $\\varphi_b=%sv$. That point is marginal: it lies on the working branch of Section 6.13 only because the "
                  "field freezes far from $v$, and one-at-a-time changes of $\\mu$, $q$ or $C_T$ move it to the empty branch."
                  % (len(c1), pm(a1("held_accelerated_efolds")), pm(np.array([fo(c) for c in c1], dtype=float)), pm(a1("max_matter_fraction")), pm(a1("lnG")), pm(a1("phi_bounce_over_v")))))
    add(("table", "Table 18. Continuous cycles without resets (baseline: $m=%s$ MeV, $\\rho_c^{1/4}$ = %.0f MeV, $C_T=1$, $a_c=2$, $\\mu=30H_0$, $q=1$, $\\Omega_{\\sigma,0}=10^{-30}$). "
                  "Cycle 0 starts today; $x_b$ and $\\varphi_b$ at the bounce that ends the cycle; $Y/\\kappa$ the dark matter after freeze-out relative to the present value; "
                  "the last row is cycle %d. All cycles: continuous_cycles.csv." % (MB, A44["T_bounce_MeV"], CR["cycles"][-2]["n"]),
         ["cycle", "$T_b$ (MeV)", "$x_b$", "$\\varphi_b/v$", "$Y_{\\rm fo}/\\kappa$", "$Y/\\kappa$ held", "peak matter", "held e-folds", "$\\ln G_n$", "$\\frac13\\ln\\frac{S_{n+1}}{S_n}$", "shear at bounce"], rows, None))
    add(("fig", "fig17_continuous_cycles.png", "Fig. 17. Continuous cycles without resets. (a) Fractions of radiation, matter and vacuum energy against the bath temperature in the "
                "expanding branch, for today's cycle (dotted, from today) and the next (solid). (b) The field in three cycles: frozen at the bounce, released from the hot phase, "
                "damped to $\\varphi\\approx0$, held by the matter and released. (c) The dark matter of cycle 1 against $x=m_{\\rm eff}/T$: in equilibrium at the bounce, frozen out "
                "while the field is still displaced and the dark matter heavier (upper plateau), and reduced to the present ratio when the restoration of the field "
                "lowers its mass and heats the bath (lower plateau, the held epoch), up to the release. (d) Held e-folds, $Y/\\kappa$ and $\\ln G_n$ over %d cycles." % N_CYC, 6.3))
    # ------------------------------------------------------------------------------------------------ the cycle as a fixed point
    m10, m100 = CM["map"]["5500000000.0"], CM["map"]["55000000000.0"]
    HE = CM.get("held", {})
    add(("p", "Why the cycle repeats can be stated as a property of a map. At a bounce the density is fixed at $\\rho_c$, the field is overdamped with its velocity "
              "slaved, and the remaining freedom is the value $\\varphi_b$ at which the field is frozen (with the dark matter in equilibrium if it is light enough "
              "there). The bounce-to-bounce map $\\varphi_{b,n}\\mapsto\\varphi_{b,n+1}$ was probed by starting one cycle from the baseline bounce state with "
              "$\\varphi_b$ set to each of 15 values between $0.02v$ and $v$ and integrating to the next bounce (cycle_map.py, Fig. 18a). For $m=10$ MeV every start "
              "gives a working cycle, the map has a single fixed point at $\\varphi_b^*=%.3fv$, and its slope there is $%.2f$: a deviation from the fixed point "
              "shrinks by a factor of about %.0f in one cycle, so the repeating cycle is an attracting fixed point whose basin covers the whole interval tested. "
              "For $m=100$ MeV the fixed point lies at $%.3fv$, the value at which the 100-cycle run of that mass settles, and its slope is $%.2f$: still attracting, "
              "but with oscillating and slower convergence. The empty branch of Section 6.13 is not a second fixed point of this map; it is reached when the dark "
              "matter is not regenerated at the bounce, which is a second state variable that the one-dimensional map holds fixed. The map also shows that the "
              "dark-matter abundance of a cycle depends on $\\varphi_b$: the closer to zero the field is frozen, the lighter the dark matter at freeze-out and the "
              "larger the abundance that remains after the restoration, so the spread of $Y/\\kappa$ between cycles follows the spread of $\\varphi_b$ about the "
              "fixed point."
              % (m10["fixed_point"], m10["slope_at_fixed_point"], 1 / abs(m10["slope_at_fixed_point"]), m100["fixed_point"], m100["slope_at_fixed_point"])))
    if HE.get("points"):
        add(("p", "The held epoch itself has a closed form. Acceleration begins when the matter density falls to $2V_c$ and the field is released when it "
                  "falls to $\\rho_{\\rm crit}$, which takes $\\frac13\\ln(2V_c/\\rho_{\\rm crit})$ e-folds; the tachyonic field then needs $N_{\\rm grow}$ more "
                  "e-folds to grow from its restored value $\\varphi_{\\rm rest}$ to $0.1v$,"))
        add(("eq", r"N_{\rm held}=\frac{1}{3}\ln\frac{2V_c}{\rho_{\rm crit}}+N_{\rm grow},\qquad \int_0^{N_{\rm grow}}\frac{\lambda(N)}{H_\Lambda}\,dN=\ln\frac{0.1\,v}{\varphi_{\rm rest}},\qquad \lambda=\Big(\mu^2\big(1-e^{-3N}\big)+\tfrac94H_\Lambda^2\Big)^{1/2}-\tfrac32H_\Lambda"))
        add(("p", "with $H_\\Lambda=(V_c/3)^{1/2}$. With $\\varphi_{\\rm rest}$ read from each cycle of the 100-cycle run, this reproduces the integrated held "
                  "e-folds to %s e-folds (rms over %d cycles; largest deviation %s; Fig. 18b). The variation of the held epoch between cycles is therefore "
                  "explained rather than only reported: it is the logarithm of how deeply each cycle restored the field, divided by the tachyonic growth rate."
                  % ("%.3f" % HE["rms_error"], HE["n_points"], "%.3f" % HE["max_abs_error"])))
    add(("fig", "fig18_cycle_map.png", "Fig. 18. (a) Bounce-to-bounce return map of the frozen field value for $m=10$ and 100 MeV (bounce at %.0f MeV); dotted: the "
                "identity, whose intersection with each curve is the repeating cycle. (b) Held e-folds from the integration against the closed form, one point per "
                "cycle." % A44["T_bounce_MeV"], 6.3))
    if NA:
        base = [r for r in NA["rows"] if r["variant"] == "baseline"][0]
        others = [r for r in NA["rows"] if r["variant"] != "baseline"]
        worst = lambda r, k: r["max_rel_change_vs_baseline"].get(k, float("nan"))
        okv = [r for r in others if r["classification"] == "working"]
        bad = [r for r in others if r["classification"] != "working"]
        res_of = {r["variant"]: r["max_residual"] for r in NA["rows"]}
        sc = NA["baseline_cycle_scatter"]
        txt = ("The results do not depend on how the numerical devices are set (numerics_admissibility.py, three cycles per variant). A tighter tolerance "
               "($10^{-10}$), a ten times stricter threshold for the overdamped reduction, and the constraint projection applied only at every fourth chunk "
               "change the held e-folds by at most %.2g, the held-epoch abundance by at most %.2g and the cycle growth $\\ln G_n$ by at most %.2g (relative), "
               "against a scatter between cycles of %.2g, %.2g and %.2g in the baseline itself; every one of these variants is a working cycle. "
               % (max(worst(r, "held_accelerated_efolds") for r in okv), max(worst(r, "Y_held_over_kappa") for r in okv), max(worst(r, "lnG") for r in okv),
                  sc["held_accelerated_efolds"], sc["Y_held_over_kappa"], sc["lnG"]))
        if bad:
            txt += ("The projection cannot be removed altogether, however: without it the Friedmann-constraint residual grows to %.0e through the stiff "
                    "bounce and contraction phases and the integration stops on its evaluation budget before the first cycle closes. Projection is "
                    "therefore needed to keep the integrator on the constraint surface, which the exact equations never leave, while its frequency does not "
                    "matter: applying it four times less often raises the residual from %.0e to %.0e and leaves every reported quantity unchanged within the "
                    "scatter between cycles. It removes integration error, not physical energy, and it is not what makes the cycle repeat; a symplectic or "
                    "constraint-preserving integrator would make this explicit and is left for future work."
                    % (bad[0]["max_residual"], res_of["baseline"], res_of.get("projection every 4th chunk", float("nan"))))
        add(("p", txt))
    # ------------------------------------------------------------------------------------------------ 6.9
    add(("h2", "6.9 Is dilution the cause? Interventions on the turnaround"))
    add(("p", "Section 6.3 compared the coupled model with an uncoupled field. A sharper test intervenes on the cause itself (causal_controls.py). Five models are evolved from today "
              "over the grid of Table 16: P, a positive cosmological constant (expands forever); N, a constant negative vacuum with matter (recollapses whatever the matter does, but "
              "decelerates today, $q_0=%.2f$ for $q=1$, and is excluded); U, the scalar without the coupling; F, the coupled model with the matter density seen by the field frozen "
              "at today's value (an intervention on the field equation only, $do(\\rho_*=\\rho_{*,0})$; gravity still sees diluting matter); and D, the model of the paper. In all %d "
              "grid points F stays held and expands forever, since $m_{\\rm eff}^2=\\mu^2(a_c^3-1)>0$, while D recollapses for every $q>0$ (%d of %d) and releases but keeps "
              "expanding for $q<0$ (there $V_0\\ge-qV_c>0$, so $\\rho>0$ and $H$ cannot vanish); U is never held: it rolls, or for $\\mu=3H_0$, $q=10$ recollapses before reaching $v/2$. The outcome changes from eternal expansion to recollapse exactly when, and only because, the matter is allowed "
              "to dilute (Fig. 19). The transition here is a homogeneous roll of the field; the nucleation of local Coleman--De Luccia bubbles is the separate branch of Section 6.5."
              % (CAU["N"]["1.0"]["q0_deceleration"], SUM["runs"], SUM["dilution_causes_recollapse"], SUM["q_positive"])))
    add(("fig", "fig19_causal_controls.png", "Fig. 19. Causal test. (a) $H(t)$ for the five models at $a_c=2$, $\\mu=30H_0$, $q=1$: only D turns around after a $\\Lambda$CDM past; F, "
                "identical except that the matter seen by the field does not dilute, expands forever. (b) Outcome over the grid of $a_c$, $\\mu$ and $q$.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.10
    add(("h2", "6.10 Anisotropy in the contraction"))
    add(("p", "A contracting universe is unstable to anisotropy: for Bianchi I the shear enters the Friedmann equation as $\\rho_\\sigma=\\Sigma/a^6$, which grows relative to "
              "radiation as $a^{-2}$. In a single contraction from the turnaround to a bounce at $\\rho_c$ the shear stays below 10%% of the density at the bounce only if it is below "
              "$s_{\\max}=$%s of the matter density at the turnaround (contraction_stability.py; the analytic scaling is reproduced to %s). The cycles of Section 6.8 meet this "
              "condition in a way specific to them: the entropy produced in each cycle enlarges the next, so at a fixed temperature the scale factor is larger by $G_n$ and the "
              "shear, which is conserved in comoving terms, is diluted by $G_n^6$ per cycle. In the %d-cycle run the shear fraction at the bounce falls from %s to %s and then below "
              "%s. This controls the shear inherited from the past, not shear generated anew in each cycle. For that the margin is the tolerance $s_{\\max}$ itself."
              % (sci(NOEK["s_max"], 1), "%.1g" % abs(CS["analytic_check"]["numerical"] / CS["analytic_check"]["analytic"] - 1), N_CYC, sci(max(SHB[0], 1e-300), 0),
                 sci(max(SHB[1], 1e-300), 0), sci(max(min(SHB[2:]) if len(SHB) > 2 else 1e-300, 1e-300), 0))))
    add(("p", "An ekpyrotic phase widens the tolerance. A second scalar $\\chi$ with $V_{\\rm ek}=-V_ee^{-c\\chi/M_{\\rm Pl}}$, $c^2>6$, has $w=c^2/3-1>1$ and $\\rho_\\chi\\propto "
              "a^{-c^2}$, so it dilutes the shear as $a^{c^2-6}$ \\cite{KOST01,EWST04}. It needs its own dynamics, $\\ddot\\chi+(3H+\\Gamma)\\dot\\chi+V_{\\rm ek}'=0$, an exit (the potential switched off where "
              "$|V_{\\rm ek}|$ reaches $\\rho_{\\rm end}$) and a decay into radiation after the exit, $\\dot\\rho_r\\supset\\Gamma\\dot\\chi^2$; without the decay the stiff kinetic energy "
              "of $\\chi$ dominates the bounce. Starting with $\\rho_\\chi=10^{-12}\\rho_m$ at the turnaround, $s_{\\max}$ rises from %s to %s ($c=3$), %s ($c=5$) and %s ($c=7$), "
              "with %.1f, %.1f and %.1f e-folds of ekpyrotic contraction (Fig. 20). The tolerance is set by the onset: the ekpyrotic phase must dominate before the shear does, so "
              "$s_{\\max}$ follows $f_0=\\rho_\\chi/\\rho_m$ at the turnaround (Fig. 20b; %d cases with the earliest onset did not finish within the time budget and are "
              "listed as such in contraction_stability.json), and an earlier exit loses part of the gain to the radiation era before the bounce. "
              "Ekpyrosis is not part of the cycle of Section 6.8, which does not need it for the inherited shear. It is the mechanism the model would need if anisotropy were "
              "regenerated at the level $s>$%s each cycle." % (sci(NOEK["s_max"], 1), sci(EK[0]["s_max"], 1), sci(EK[1]["s_max"], 1), sci(EK[2]["s_max"], 1),
                                                                  EK[0]["ekpyrotic_efolds"], EK[1]["ekpyrotic_efolds"], EK[2]["ekpyrotic_efolds"], len(CS.get("not_completed", [])),
                                                                  sci(NOEK["s_max"], 0))))
    add(("fig", "fig20_contraction_stability.png", "Fig. 20. Shear in the contraction. (a) Shear fraction at the bounce against the anisotropy at the turnaround, without and "
                "with an ekpyrotic phase ($f_0=10^{-12}$, exit at $10^{-4}\\rho_c$). (b) The largest controlled anisotropy against the ekpyrotic fraction at the turnaround, "
                "for three exit densities ($c=5$).", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.11
    add(("h2", "6.11 Bounce validation"))
    add(("p", "Three tests address the effective bounce (bounce_validation.py). (A) Sensitivity to $\\rho_c$: bounces at %s MeV behave identically up to the bounce; the next "
              "cycle depends on $\\rho_c$ only through re-thermalization (next cycle: %s). (B) The effective equations through the bounce: "
              "without projection inside $\\rho>0.4\\rho_c$ the constraint holds to %s, $d\\ln\\rho/d\\ln a=-3(1+w)$ to %s, the density and the entropy are continuous across the bounce, "
              "and the curvature invariants are finite, $R=%.4f\\rho_c$ and $K=%.4f\\rho_c^2$ at the bounce against the analytic $3(\\rho+p)=4\\rho_c$ and $16\\rho_c^2/3$ for "
              "radiation. (C) An independent bounce: general relativity with a negative-energy stiff fluid $\\rho_g=-g/a^6$ in place of the LQC correction. Its bounce density is set "
              "by a comoving constant, so it is not fixed from cycle to cycle. The bounce temperature is %s MeV in successive cycles, against %s MeV for the LQC bounce. The cycle "
              "survives the change of mechanism %s, but the self-similarity of Section 6.8 is a property of a bounce at a fixed density, not of bounces in general (Fig. 21)."
              % (", ".join("%.3g" % a["T_bounce_MeV"] for a in BV["A"]),
                 "; ".join("%.3g MeV: %s" % (a["T_bounce_MeV"], "working" if (fo(a["next"]) or 0) > 0.1 and (a["next"].get("held_accelerated_efolds") or 0) > 0.5 else "empty") for a in BV["A"]),
                 sci(max(a["max_residual"] for a in BV["A"]), 0), sci(max(a["conservation_median_error"] for a in BV["A"] if a["conservation_median_error"]), 0),
                 A44["R_over_rho_c"], A44["K_over_rho_c2"], ", ".join("%.3g" % (c["T_bounce"] * MEV) for c in GH["cycles"] if c.get("T_bounce")),
                 ", ".join("%.3g" % (c["T_bounce"] * MEV) for c in LQC["cycles"] if c.get("T_bounce")),
                 "(%s)" % ("all cycles complete" if GH["status"] == "completed" else GH["status"]))))
    add(("fig", "fig21_bounce_validation.png", "Fig. 21. Bounce validation. (a) Curvature invariants and density through the first bounce of the baseline run; dashed lines: the "
                "analytic bounce values for radiation. (b) Test C: bounce temperature per cycle for the LQC bounce and for a ghost stiff fluid.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.12
    add(("h2", "6.12 Black holes through the bounce"))
    perc = RL["percolation"]
    add(("p", "The black holes of a cycle do not take part in its re-thermalization. Their mass function $n(M,t)$ is evolved through the contraction of our cycle "
              "(compact_relics.py) with $\\partial_tn+3Hn+\\partial_M(\\dot Mn)=C_{\\rm merge}+S_{\\rm form}$: accretion of radiation $\\dot M=27\\pi G^2M^2\\rho_r$ \\cite{CH74}, capped at the "
              "Hubble mass $4\\pi M_{\\rm Pl}^2/H$ (with $H$ replaced by $(\\rho/3M_{\\rm Pl}^2)^{1/2}$ near the bounce, where $H$ passes through zero); binned mergers with the capture cross-section; and the present population as the source (stellar remnants with $\\Omega=10^{-4}$ near $10\\,M_\\odot$, "
              "supermassive holes with $\\Omega=7.6\\times10^{-6}$ near $10^8M_\\odot$; log-normal mass functions with exponential cut-offs above 100 and $5\\times10^9M_\\odot$). Three experiments: no black holes (E1), the present population unchanged (E2), and the population "
              "with accretion and mergers (E3). The quantity that decides whether a homogeneous bounce is possible is the fraction of space inside horizons, $f_V$. At the bounce "
              "($T_b=%.0f$ MeV) it is %s in E2 and %s in E3. Holes above $%.0f\\,M_\\odot$ accrete runaway to the Hubble mass, which near the bounce is only about $%.0f\\,M_\\odot$, so "
              "the stellar population roughly doubles its mass (the mass in black holes grows by %.0f%%), and mergers move %s of it. Since $f_V\\propto a^{-3}\\propto T^3$, the "
              "evolved population would percolate, $f_V=1$, for a bounce above about %.1f GeV. A more conservative estimate puts all the supermassive mass into holes of a "
              "single mass, which gives percolation at %s MeV for $10^{10}$, $10^9$ and $10^8M_\\odot$, including the entropy growth of the cycle. "
              "The black holes therefore bound the bounce from above, while BBN bounds it from below: $5\\ \\mathrm{MeV}\\lesssim T_b\\lesssim%.1f$ GeV, or $\\lesssim%.1f$ GeV "
              "on the conservative estimate. The surviving holes enter "
              "the next cycle as relics, and their number per unit entropy falls by $G_n^3$ per cycle, so they do not accumulate."
              % (RL["T_bounce_MeV"], sci(RL["E2"]["f_V_bounce"], 0), sci(RL["E3"]["f_V_bounce"], 0), RL["M_runaway_Msun"], 4.64e22 / np.sqrt(P["rho_c"] / 3.0), 100 * (RL["E3"]["mass_density_bounce_over_today"] - 1),
                 sci(RL["E3"]["merged_mass_fraction"], 0), RL["T_perc_E3_MeV"] / 1e3,
                 ", ".join("%.3g" % perc[k]["T_MeV"] for k in ("SMBH 1e10 Msun", "SMBH 1e9 Msun", "SMBH 1e8 Msun")),
                 RL["T_perc_E3_MeV"] / 1e3, perc["SMBH 1e10 Msun"]["T_MeV"] / 1e3)))
    # ------------------------------------------------------------------------------------------------ 6.13
    add(("h2", "6.13 Where the cycles work, and how they fail"))
    fm = {}
    for r in ROWS:
        fm.setdefault(r["classification"].split(" (")[0], []).append(lab(r["params"]))
    n_rob = sum(short(r) == "working" for r in robust)
    add(("p", "Over a grid of dark-matter mass (1 MeV to 1 GeV) and bounce temperature (1.8 MeV to 1.8 GeV), with five cycles per run, the cycles work in %d of %d cells (Fig. 22b). "
              "Every failure in the grid is F1: the bounce is too cold to return the dark matter to equilibrium. The dark matter is then annihilated in the contraction, the next "
              "cycle has no matter to hold its vacuum, and it has no matter era and no dark-energy epoch. The dynamics are bistable. The field freezes at the bounce either at "
              "$\\varphi_b\\approx0.23$--$0.31v$, where the dark matter is light enough to re-thermalize and the cycle reproduces itself, or near $v$, where the dark matter is "
              "$A(v)\\approx85$ times heavier, is not regenerated, and nothing pulls the field back. Every failing run sits on the second branch from its first or second bounce "
              "on. Which branch a run reaches depends on where the field freezes, which is set by the phase of its oscillation when the overdamping begins; it is not a smooth "
              "function of the parameters. The condition that does not depend on that phase is a bounce that re-thermalizes even the heaviest dark matter, "
              "$T_b\\gtrsim mA(v)/5$. For $m=100$ MeV and $T_b=180$ MeV it is not met, and varying one parameter at a time about that point gives: %s. For $m=10$ MeV at the same "
              "bounce it is met, and %d of %d variations work: %s. The remaining failures come from the dissipation: strong dissipation keeps the field from relaxing (F2 at "
              "$C_T=10^3$), and weak dissipation lets the scalar's energy push the field far beyond $v$ (F1 at $C_T=10^{-3}$, and a run at $C_T=10^{-6}$ that could not be "
              "completed in the time allowed, F6). The classification looks for F1 (dark matter not regenerated), F2 (vacuum not restored), F3 (bounce dominated by the scalar's "
              "kinetic energy), F4 (bounce dominated by shear: $\\Omega_{\\sigma,0}=10^{-10}$ today, while $10^{-20}$ works), F5 (no bounce) and F6 (numerical); black-hole "
              "percolation (Section 6.12) is a further, external bound, and the ghost-fluid bounce is treated separately in Section 6.11."
              % (n_work, len(grid), "; ".join("%s: %s" % (lab(r["params"]), short(r)) for r in marg if "bounce" not in r["params"]), n_rob, len(robust),
                 "; ".join("%s: %s" % (lab_r(r), short(r)) for r in robust))))
    add(("fig", "fig22_relics_failure_map.png", "Fig. 22. (a) Fraction of space inside black-hole horizons through the contraction of our cycle, for the present population (E2) and "
                "with accretion and mergers (E3). (b) Outcome of five continuous cycles over dark-matter mass and bounce temperature: the cycles work where the bounce "
                "re-thermalizes the dark matter.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.14
    add(("h2", "6.14 Observational consequences"))
    m0 = MODES[0]
    add(("bullets", [
        "Dark energy. While held, the vacuum has $1+w<%s$; the model predicts no evolution of the dark energy before the release, the release near $a=a_c$ and a turnaround "
        "%.0f Gyr from now for $a_c=2$, $\\mu=30H_0$, $q=1$. A confirmed evolution of $w$ at the level suggested by recent analyses \\cite{DESI} excludes it." % (texsci(DS["one_plus_w_today_bound"], 0), DS["t_turn_Gyr"]),
        "Dark matter. The cycle requires a thermal relic re-thermalized at the bounce: $m\\lesssim5T_b/A(\\varphi_b)$ and, with Section 6.12, $m$ below a few hundred MeV for "
        "$T_b\\lesssim%.1f$ GeV; a non-thermal dark matter (an axion produced by misalignment, for instance) or a relic heavier than this is incompatible with the cycle; thermal relics below a few MeV are separately constrained by BBN and the CMB \\cite{Sabti20}." % (RL["T_perc_E3_MeV"] / 1e3),
        "Primordial perturbations. The scales observed today, $k=(1$--$1000)a_0H_0$, cross the Hubble radius of the preceding contraction at $T=%.0f$--$%.0f$ (in units of "
        "1.82 meV), where the contraction is radiation dominated ($w=%.3f$). A contraction with constant $w$ gives $n_s-1=12w/(1+3w)$ \\cite{Wands99,FB02}, i.e. $n_s=%.2f$, with an amplitude set "
        "by the bounce scale, $\\rho_c/M_{\\rm Pl}^4=%s$, against $n_s=0.965$ and $2.1\\times10^{-9}$ observed \\cite{PlanckX}. The background cycle does not produce the observed perturbations; "
        "an amplitude of the observed size from the bounce would need $\\rho_c^{1/4}\\sim%s$ GeV, which the black holes exclude. A separate mechanism (for example a curvaton, or "
        "an entropic mechanism in an ekpyrotic phase) is required, and none is part of this model. Non-Gaussianity is therefore not predicted." % (
            m0["T_exit"], MODES[-1]["T_exit"], m0["w_exit"], m0["n_s"], texsci(PR["perturbations"]["rho_c_over_MPl4"], 0), texsci(PR["perturbations"]["T_b_needed_GeV"], 0)),
        "Gravitational waves. Tensor modes from the contraction have the same tiny amplitude; the release is a homogeneous roll without bubble collisions and sources no "
        "gravitational waves; the black holes of Section 6.12 give the ordinary astrophysical background. No primordial gravitational-wave signal is predicted."]))
    # ------------------------------------------------------------------------------------------------ 6.15
    add(("h2", "6.15 What is shown, and what is assumed"))
    add(("table", "Table 19. Status of the cyclic model.",
         ["Element", "Status", "Where"],
         [["Vacuum held by matter, released by dilution", "derived within the model; causal test passed", "6.2, 6.3, 6.9"],
          ["Turnaround", "follows from $V_0(v)<0$", "6.3"],
          ["Bounce", "assumed (effective LQC); Test C: another mechanism gives cycles but not self-similar ones", "6.6, 6.11"],
          ["Reheating", "continuous, no reset: dissipation $\\Upsilon=C_TT^3$ (assumed form) and Boltzmann dark matter", "6.8"],
          ["Dark-matter abundance", "output of freeze-out in every cycle; $\\langle\\sigma v\\rangle$ fixed once by today", "6.8"],
          ["Vacuum restoration", "output (frozen at the bounce, damped in the expansion)", "6.8"],
          ["Entropy", "grows each cycle; absorbed by Tolman growth $G_n=(S_{n+1}/S_n)^{1/3}$", "6.8"],
          ["Anisotropy", "inherited shear diluted by $G_n^6$; regenerated shear needs $s<10^{-27}$ or ekpyrosis", "6.10"],
          ["Black holes", "bound $T_b\\lesssim$ %.1f GeV (%.1f GeV conservative); relics diluted" % (RL["T_perc_E3_MeV"] / 1e3, RL["percolation"]["SMBH 1e10 Msun"]["T_MeV"] / 1e3), "6.12"],
          ["Inhomogeneities, perturbations", "not followed; the observed spectrum is not produced", "6.14"],
          ["Bounce density", "far below the Planck density (18--180 MeV)", "6.11, 6.12"],
          ["Vacuum landscape (kinetic Monte Carlo over vacua)", "not attempted: the model has one field and two vacua", "--"]], None))
    add(("p", "A numerical model in which every cycle contains a hot big bang, a matter era and a dark-energy epoch, with no variable reset, is a statement about this set of "
              "equations, not evidence that the universe has undergone or will undergo repeated big bangs. Its value is that it makes the requirements explicit. These are a vacuum "
              "held by matter; a bounce at a fixed density hot enough to re-thermalize the dark matter but cold enough to keep black holes from percolating; dissipation; a "
              "thermal-relic dark matter; and an anisotropy that is inherited and not regenerated. It also identifies what is missing: a derivation of the bounce, the "
              "inhomogeneous dynamics, and a source for the primordial perturbations."))
    return B
