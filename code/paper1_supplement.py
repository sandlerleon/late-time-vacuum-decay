# -*- coding: utf-8 -*-
"""Supplementary information for Paper 1 (EPJ C).   python paper1_supplement.py   (after paper1_build.py)"""
import json
import os
import platform

import numpy as np
import scipy

import copyedit as CE
import cyc_blocks as CYB
import render as RN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "manuscript")
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
V = J("verification.json")
DVJ, SCJ = J("dilution_validation.json"), J("turnaround_scan.json")
OSV = J("oscillation_validation.json")
CRJ, CSCN, CSTJ, RLJ = J("continuous_cycles.json"), J("cycle_scan.json"), J("contraction_stability.json"), J("compact_relics.json")
DOIS = J("dois.json")
GITHUB = "https://github.com/sandlerleon/late-time-vacuum-decay"
import paper1_build as P1
tx = lambda x, d=1: CYB.texsci(x, d).join("$$")

S = []
add = S.append
add(("title", "Supplementary Information: " + P1.TITLE))
add(("author", "Leon Sandler\nIndependent Researcher, Northbrook, Illinois, USA"))
add(("h1", "S1. Overview"))
add(("p", "This supplement documents the numerical methods, the validation tables and the reproduction commands for the paper. Every number quoted in the "
          "paper is read from the JSON files produced by the scripts named here, and the manuscript is built only if the checks of Table S8 pass."))
add(("h1", "S2. The matter-held vacuum and its release"))
add(("p", "matter_coupling.py defines $V_0$, $A$, $\\rho_{\\rm crit}$ and $a_c$; dilution_background.py integrates the coupled background (flat, general "
          "relativity) from $z=1100$ with the field at $10^{-60}v$, with the absolute tolerance of the field set relative to its seed (an absolute tolerance on "
          "the background scale would swamp the seed); turnaround_scan.py runs the grid of Table 1 for the coupled model and the control; dilution_thermal.py the "
          "thermal comparison of Section 3.2; causal_controls.py the interventions of Section 3.3; dilution_validation.py the checks of Table S1."))
rows = [[k.split("_")[0], k.split("_", 1)[1].replace("_", " "), "pass" if v["passed"] else "FAIL"] for k, v in DVJ.items()]
add(("table", "Table S1. Validation of the dilution-trigger code (dilution_validation.py).", ["Check", "Content", "Result"], rows, None))
rows = [["%g" % r["a_c"], "%g" % r["mu"], "%g" % r["q"], "coupled" if r["coupled"] else "control", r["cls"], ("%.3g" % r["a_turn"]) if r["a_turn"] else "--",
         "%.0e" % r["max_residual"]] for r in SCJ["rows"] if r["q"] in (-0.5, 1.0) and (r["coupled"] or r["a_c"] == 2.0)]
add(("table", "Table S2. Release and turnaround over the grid ($q=-0.5$ and 1; the control does not depend on $a_c$). The full grid, including $q=0.1$ and "
              "10, is in turnaround_scan.csv.", ["$a_c$", "$\\mu/H_0$", "$q$", "model", "outcome", "turnaround $a$", "residual"], rows, None))
add(("h1", "S3. Stability of repeated cycles (Section 4.2)"))
add(("p", "oscillation.py integrates the open negative-vacuum universe with the effective bounce continuously through every bounce, with no reset; "
          "oscillation_validation.py repeats the runs with different integrators, tests the continuity at the bounces and the energy-balance law, and maps the "
          "phase diagram of Fig. 4b."))
rows = [[k, "%.4f" % v["multiplier"], "%.1e" % v["multiplier_rel_dev"], "%.1e" % v["a_max_rel_dev"]] for k, v in OSV["validation_integrators"].items()]
add(("table", "Table S3. Scalar energy multiplier per cycle (linear regime) and turnaround radii of a nonlinear run for each integrator setting, with their "
              "deviation from the reference.", ["Integrator", "multiplier", "deviation", "$a_{\\max}$ deviation"], rows, None))
rows = [["%.2f" % g["m"], "%.1f" % g["upsilon_P"], "%.0e" % g["phi0"], str(g["cycles"]), g["cls"]] for g in OSV["phase_diagram"]]
add(("table", "Table S4. Phase diagram of repeated cycles: classification after 100 or more cycles (runs stop when $a$ exceeds $10^{30}$).",
     ["$m$", "$\\Upsilon P$", "$\\varphi_0$", "cycles", "class"], rows, None))
add(("h1", "S4. Continuous cycles, stability, bounce and relics (Sections 4.3--6.2)"))
add(("p", "continuous_cycles.py integrates the cycle of Section 5.1 as described in Section 6.1; cycle_map.py probes the bounce-to-bounce map; "
          "numerics_admissibility.py repeats three baseline cycles with rtol $10^{-10}$, an overdamped threshold of $10^4$, projection at every fourth chunk, and "
          "no projection; cycle_scan.py maps the outcome over dark-matter mass and bounce density; contraction_stability.py, bounce_validation.py, "
          "compact_relics.py and predictions.py give Sections 4.3--4.5 and 7.1."))
rows = []
for c in CRJ["cycles"]:
    if c["n"] < 1 or "held_accelerated_efolds" not in c or (c["n"] > 20 and c["n"] % 10):
        continue
    rows.append([str(c["n"]), "%.4f" % c.get("phi_bounce_over_v", 0), "%.3f" % c.get("Y_fo_over_kappa", 0), "%.3f" % c.get("Y_held_over_kappa", 0),
                 "%.3f" % c.get("max_matter_fraction", 0), "%.3f" % c["held_accelerated_efolds"], "%.4f" % c["lnG"] if c.get("lnG") else "--",
                 "%.4f" % c["lnG_entropy"] if c.get("lnG_entropy") else "--"])
add(("table", "Table S5. Continuous cycles (baseline), cycles 1--20 and every tenth cycle thereafter; all cycles in continuous_cycles.csv.",
     ["cycle", "$\\varphi_b/v$", "$Y_{\\rm fo}/\\kappa$", "$Y/\\kappa$ held", "peak matter", "held e-folds", "$\\ln G$", "$\\frac13\\ln$ S ratio"], rows, None))
rows = [[CYB.lab(r["params"]), r["classification"], ", ".join("%.2f" % x for x in (r.get("held") or [])[:5] if x is not None),
         ", ".join("%.2g" % x for x in (r.get("Y") or [])[:5] if x is not None)] for r in sorted(CSCN["rows"], key=lambda r: str(r["params"]))]
add(("table", "Table S6. Outcome of five continuous cycles per run (cycle_scan.py): held e-folds and $Y/\\kappa$ of cycles 1--5.",
     ["Run", "Outcome", "held e-folds", "$Y/\\kappa$"], rows, None))
rows = [["%g" % r["f0"], "%g" % r["rho_end_over_rho_c"], tx(max(r["s_max"], 1e-300), 1)] for r in CSTJ["onset_exit"]]
rows += [[k, "", "%.3g MeV" % v["T_MeV"]] for k, v in RLJ["percolation"].items()]
add(("table", "Table S7. Upper part: ekpyrotic phase ($c=5$), largest controlled initial anisotropy against the ekpyrotic fraction at the turnaround and the "
              "exit density. Lower part: percolation temperature of black-hole populations in the contraction.", ["$f_0$ / population", "$\\rho_{\\rm end}/\\rho_c$",
                                                                                                                 "$s_{\\max}$ / T"], rows, None))
add(("h1", "S5. Executable verification"))
rows = []
for k in P1.P1_CHECKS:
    v = V[k]
    det = {kk: (("%.3g" % vv) if isinstance(vv, float) else str(vv)) for kk, vv in v.items() if kk != "passed"}
    rows.append([k.split("_")[0], P1.DESC.get(k.split("_")[0], k), "pass" if v["passed"] else "FAIL", "; ".join("%s = %s" % (a, b) for a, b in list(det.items())[:3])])
add(("table", "Table S8. Checks of this paper (verify.py) and their key figures.", ["No.", "What is checked", "Outcome", "Key figures"], rows, None))
add(("h1", "S6. Sensitivity of the black-hole bound (Section 4.5)"))
RS = J("relic_sensitivity.json")
add(("p", "relic_sensitivity.py evolves the black-hole population of compact_relics.py through the contraction of the baseline cycle with one prescription "
          "changed at a time, and extrapolates the fraction of space inside horizons from the bounce with $f_V\\propto T^3$. The bound is given for two "
          "thresholds: $f_V=1$ (horizon volumes added) and $f_V=%.2f$ (continuum percolation of randomly placed, overlapping spheres). The last column of the "
          "lower block gives the joint viability of Section 6.3 with the window closed at that estimate." % RS["f_c_overlap"]))
rows = [[r["variant"], tx(r["f_V_bounce"], 2), "%.2f" % r["T_perc_GeV_fc1"], "%.2f" % r["T_perc_GeV_overlap"]] for r in RS["rows"]]
add(("table", "Table S9. Black-hole bound on the bounce temperature under changes of the population and accretion prescriptions (baseline bounce %.0f MeV)."
              % RS["T_bounce_MeV"], ["Variant", "$f_V$ at the bounce", "$T_{\\rm perc}$, $f_V=1$ (GeV)", "$T_{\\rm perc}$, overlap (GeV)"], rows, None))
rows = [[k, "%.2f" % v["T_perc_GeV"], "%.1f%%" % (100 * v["window"]), "%.2f%%" % (100 * v["joint"])] for k, v in RS["held_viability"].items()]
add(("table", "Table S10. Joint viability of the held branch (viability_mc.py, same draws) with the bounce window closed at each estimate of the bound.",
     ["Estimate", "$T_{\\rm perc}$ (GeV)", "bounce window", "joint"], rows, None))
add(("h1", "S7. Independent solver (Section 6.1)"))
ISV = J("independent_solver.json")
add(("p", "independent_solver.py repeats the baseline (three cycles) with continuous_cycles.run(gravity='constrained'): away from the turning points the "
          "Hubble rate is computed from the Friedmann constraint, so no projection is applied; near a turnaround or bounce (where $H^2$ falls below "
          "$\\epsilon^2$ times the natural rate squared, $\\epsilon$ the bridge parameter, or the density exceeds $0.4\\rho_c$) it is evolved with the "
          "Raychaudhuri equation. Table S11 lists every cycle of every setting."))
rows = []
for r in ISV["rows"]:
    for c in r["cycles"]:
        rows.append([r["variant"], str(c["n"]), "%.5f" % c["ln_a_max"] if c.get("ln_a_max") is not None else "--",
                     "%.4f" % c["t_cycle"] if c.get("t_cycle") is not None else "--",
                     "%.4f" % c["held_accelerated_efolds"] if c.get("held_accelerated_efolds") is not None else "--",
                     "%.4f" % c["Y_held_over_kappa"] if c.get("Y_held_over_kappa") is not None else "--", "%.5f" % c["lnG"] if c.get("lnG") is not None else "--"])
add(("table", "Table S11. Cycles of the projected and constrained solvers (cycle 0 starts today).",
     ["Setting", "cycle", "$\\ln a_{\\max}$", "$t_{\\rm cycle}$", "held e-folds", "$Y/\\kappa$ held", "$\\ln G_n$"], rows, None))
add(("h1", "S8. Reproduction"))
add(("p", "Environment of the published run: Python %s, NumPy %s, SciPy %s (Windows). Requirements: NumPy, SciPy, Matplotlib, python-docx. From the code "
          "directory: matter_coupling.py and dilution_background.py (the background), turnaround_scan.py, dilution_thermal.py, causal_controls.py, "
          "dilution_validation.py, oscillation.py and oscillation_validation.py, continuous_cycles.py (calibrate, then run; several hours), cycle_map.py, "
          "numerics_admissibility.py, independent_solver.py, cycle_scan.py, contraction_stability.py, bounce_validation.py, compact_relics.py, relic_sensitivity.py, predictions.py, vacuum_phase_map.py "
          "(coherence), viability_mc.py, the figure scripts, verify.py, then paper1_build.py and paper1_supplement.py. Code: %s, archived at https://doi.org/%s."
          % (platform.python_version(), np.__version__, scipy.__version__, GITHUB, DOIS["code"])))
S = [CE.block(b) for b in S]
RN.render_docx(S, os.path.join(OUT, P1.BASENAME + "_Supplement.docx"), os.path.join(REPO, "figures"))
print("saved supplement")
