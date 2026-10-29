# -*- coding: utf-8 -*-
"""Manuscript blocks for Section 5.5 (Lorentzian simulations of the daughter region, Steps 6-7); numbers are read from interior_sim.json."""
import json
import os

import numpy as np

import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "interior_sim.json"), encoding="utf-8"))
SC = json.load(open(os.path.join(HERE, "scope.json"), encoding="utf-8"))


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "⟦" + texsci(x, d) + "⟧"
A, B1, B2, C1, DC, DE = S["A_single_field"], S["B_reservoir_ads"], S["B_reservoir_scan"], S["C_hilltop"], S["D_chain"], S["D_entropy"]
ROW_A = {r["gamma_over_m"]: r for r in A["rows"]}
SCAN = {r["log10_g"]: r for r in S["A_gravity_scan"]}
RB = {r["kappa"]: r for r in B1["rows"]}
RC = {r["w_over_M"]: r for r in C1["rows"]}
FLAT_G = min(r["log10_g"] for r in S["A_gravity_scan"] if r["flat_like_radiation_epoch"])
ENV_A = max(r["envelope_ratio"] for r in A["rows"])
ENV_B = max(r["envelope_ratio"] for r in B1["rows"])
ORD_MIN = SC["needed_parent_for_bbn_at_1s"]["kappa_m_min"]["orders_above_Planck"]
TT_B = [r["Tt_product_sim"] for r in B2]
RES_MAX = max(r["max_residual"] for r in S["A_gravity_scan"] if r["log10_g"] >= -3)
RES_C = max(r["max_residual"] for r in C1["rows"])


def blocks():
    out = []
    add = out.append
    add(("h2", "5.5 Lorentzian simulations of the daughter region: two fields, reservoirs and repeated transitions"))
    add(("p", "The interior was also simulated, not only bounded. The analytic continuation of the bounce gives the open homogeneous interior with the field starting at the bounce centre $x_0$ at rest and $a=t$ (Milne). In units of $f$ for the fields "
              "and $\\lambda f^4$ for energy densities, with $g=f^2/M_{\\rm Pl}^2$, the system is"))
    add(("eq", r"\ddot x_i+\left(3\frac{\dot a}{a}+\Upsilon_i\right)\dot x_i+\frac{\partial V}{\partial x_i}=0,\qquad \dot\rho_r+4\frac{\dot a}{a}\rho_r=\sum_i\Upsilon_i\dot x_i^2,\qquad \ddot a=-\frac{g\,a}{6}(\rho+3p)"))
    add(("p", "with $\\rho=\\tfrac12\\sum_i\\dot x_i^2+V+\\rho_r$, $p=\\tfrac12\\sum_i\\dot x_i^2-V+\\rho_r/3$ and the constraint $\\dot a^2=1+g\\,a^2\\rho/3$, whose relative residual is the energy-conservation check (eighth-order Runge--Kutta, relative tolerance $10^{-11}$). "
              "The friction $\\Upsilon_i$ is a phenomenological dissipation that moves field energy to radiation with total energy conserved. The simulations are the following (check V22--V24 and Table 10--13)."))
    # --------------------------------------------------------------------------------------------- A
    add(("p", "(A) Single field, physical benchmark ($\\lambda=1$, $\\eta=0.30$, $f=%.2f$ meV, $g=%s$). The full dynamics give a maximum radiation temperature within %.0f%% of the closed form of Section 5.2 for $\\Gamma_\\varphi/m$ from 1 to $10^{-3}$ "
              "(ratios %.2f, %.2f, %.2f and %.2f), so the assumption that the condensate starts to oscillate at $1/m$ and dilutes as matter is sound. The constraint residual is at most %s for $g\\ge10^{-3}$ (it scales with $g$ and is trivially small at $g=10^{-60}$, "
              "so the gravitational scan below is the real test). Scanning $g$ from $10^{-60}$ to $10^{3}$, the curvature parameter $Y=ga^2\\rho/3$ reaches %s at $g=10^{1}$ and %.2f at $g=10^{2}$; a flat-like radiation epoch ($Y\\ge1/4$ with radiation dominant) first appears at "
              "$g=10^{%d}$, that is $f=%d\\,M_{\\rm Pl}$. For the dark-energy scale $g=10^{-60}$ and $Y_{\\max}=%s$."
              % (A["f_meV"], sci(A["g"], 1), 100 * max(abs(ROW_A[k]["ratio_sim_over_closed"] - 1) for k in ROW_A), ROW_A[1.0]["ratio_sim_over_closed"], ROW_A[0.1]["ratio_sim_over_closed"], ROW_A[0.01]["ratio_sim_over_closed"],
                 ROW_A[0.001]["ratio_sim_over_closed"], sci(RES_MAX, 0), sci(SCAN[1]["Y_max"], 1), SCAN[2]["Y_max"], FLAT_G, round(10 ** (FLAT_G / 2.0)), sci(SCAN[-60]["Y_max"], 1))))
    # --------------------------------------------------------------------------------------------- B
    rows = []
    for k in (1.0, 1e10, 1e20, 1e30, 1e35):
        r = RB[k]
        rows.append([sci(k, 0), sci(r["f_GeV"], 1), sci(r["T_max_MeV"], 1), sci(r["t_peak_s"], 1), sci(r["t_crunch_s"], 1), sci(r["T_at_1s_GeV"], 1)])
    add(("p", "(B) Two fields with an energy reservoir. A second field $\\chi$ (here $V_\\chi=\\tfrac{\\lambda_\\chi}{4}(\\chi^2-w_\\chi^2)^2$ with $\\lambda_\\chi=1$, $w_\\chi=1.5f$, coupled by $\\tfrac12 g_c^2(\\varphi-\\varphi_t)^2\\chi^2$ so that it is stable in the false vacuum and tachyonic in the true one) stores %.2f $\\lambda f^4$, "
              "%.1f times the energy $\\Delta V$ released by $\\varphi$. A constant offset $V_{\\rm off}=-(\\Delta V+E_\\chi)$ makes the false vacuum have the observed dark-energy density, which is the only way the reservoir can sit in the parent without gravitating. "
              "The true vacuum is then anti-de Sitter with $|V_t|=\\Delta V+E_\\chi$, and the simulation (dissipation $\\Upsilon=m/2$ in both fields, $\\chi$ seeded at $10^{-6}f$) transfers the reservoir to radiation. The depth is set in units of the dark energy, $|V_t|=\\kappa\\rho_\\Lambda$, "
              "which fixes $f$ and $g$ (Table 10). For $\\kappa=10^{35}$ (enough for 1 MeV) the radiation reaches %.2f MeV after %s s, and the interior crunches after %.2f s, equal to the pure anti-de Sitter value $\\pi\\sqrt3M_{\\rm Pl}/\\sqrt{|V_t|}$ to better than $10^{-12}$. "
              "The temperature then falls as $1/t$, with $Tt=%.1f$ for this reservoir (between %.1f and %.1f for reservoirs from %.2f to %.0f in units of $\\lambda f^4$), so that at one second $T=%s$ GeV for every depth, %s times below 1 MeV. "
              "The envelope of Section 5.3 holds with $\\rho(mt)^3/E_{\\rm rel}\\le%.0f$ for the single field and %s for the reservoir, whose hilltop delays the roll by $\\sim16/m$; with that factor the parent scale needed for 1 MeV at 1 s falls by $%.1f$ orders of magnitude to still %.0f orders above the Planck scale. "
              % (1.5 ** 4 / 4.0, B1["rows"][0]["E_res_over_dtil"], RB[1e35]["T_max_MeV"], sci(RB[1e35]["t_peak_s"], 1), RB[1e35]["t_crunch_s"],
                 RB[1e35]["Tt_product_sim"], min(TT_B), max(TT_B), min(r["E_res"] for r in B2), max(r["E_res"] for r in B2), sci(RB[1e35]["T_at_1s_GeV"], 1), sci(1e-3 / RB[1e35]["T_at_1s_GeV"], 0),
                 ENV_A, sci(ENV_B, 1), np.log10(ENV_B), ORD_MIN - np.log10(ENV_B))
              + "The reservoir therefore does not make a hot, lasting epoch: it makes a hot instant followed by a crunch."))
    add(("table", "Table 10. Two-field reservoir compensated by a negative offset: the true vacuum has $|V_t|=\\kappa\\rho_\\Lambda$. $f$: field scale; $T_{\\max}$ and its time; crunch time of the interior; temperature one second after nucleation.",
         ["$\\kappa$", "$f$ (GeV)", "$T_{\\max}$ (MeV)", "time of $T_{\\max}$ (s)", "crunch (s)", "$T(1\\ {\\rm s})$ (GeV)"], rows, None))
    # --------------------------------------------------------------------------------------------- C
    rows = []
    for w in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        r = RC[w]
        rows.append(["%g" % w, sci(r["lam_c"], 1) if r["lam_c"] < 0.01 else "%.3f" % r["lam_c"], "%.1f" % r["N_accel"], "%.1f" % r["N_flatness"], sci(r["Omega_k_min"], 1), "%.2f" % r["T_max_meV"]])
    add(("p", "(C) Flattening at the dark-energy scale. With a field of vev $w$ and $V=\\tfrac{\\lambda_c}{4}(\\chi^2-w^2)^2$, $\\lambda_cw^4/4=3H_\\Lambda^2M_{\\rm Pl}^2$ (so that the false-vacuum energy is $\\rho_\\Lambda$) and $\\chi$ seeded at $10^{-5}w$, "
              "hilltop inflation inside the open interior gives the e-folds and curvature of Table 11 (gravity fully included, $g=1$ in these units, constraint residual at most %s). Flatness ($N\\ge60$, $\\Omega_k\\lesssim10^{-52}$) needs $w\\gtrsim4M_{\\rm Pl}$: a super-Planckian vev. "
              "The reheating temperature is $%.2f$--$%.2f$ meV in every case, below the bound of Eq. (%s) by construction, since the energy released is $\\rho_\\Lambda$. A hot epoch at 1 MeV needs the same dynamics at a parent energy $%s$ times the observed dark energy."
              % (sci(RES_C, 0), min(r["T_max_meV"] for r in C1["rows"]), max(r["T_max_meV"] for r in C1["rows"]), "@@BOUNDEQ@@", sci(S["C_parent_energy_for_1MeV_over_rho_Lambda"], 1))))
    add(("table", "Table 11. Hilltop inflation of a field with vev $w$ inside the open interior at the dark-energy scale ($H_\\Lambda=M_{\\rm Pl}=1$): e-folds with $q<-0.1$, flatness $N=-\\tfrac12\\ln\\Omega_{k,\\min}$, and the maximum radiation temperature.",
         ["$w/M_{\\rm Pl}$", "$\\lambda_c$", "$N_{\\rm accel}$", "$N_{\\rm flat}$", "$\\Omega_{k,\\min}$", "$T_{\\max}$ (meV)"], rows, None))
    add(("fig", "fig10_daughter_simulations.png", "Fig. 10. Simulated daughter region. (a) Radiation temperature after nucleation for a reservoir hidden by an anti-de Sitter vacuum of depth $|V_t|=\\kappa\\rho_\\Lambda$ (dots: maximum; dashed: crunch); the green band is the nucleosynthesis window, reached by neither. (b) Hilltop flattening at the dark-energy scale: e-folds against the vev (left) and the maximum temperature (right), with the bound of Eq. (@@BOUNDEQ@@).", 6.3))
    # --------------------------------------------------------------------------------------------- D
    rows = []
    for c in DC:
        rows.append(["%.2f / %.1f" % (c["eta1"], c["lam1"]), "%.2f / %.1f" % (c["eta2"], c["lam2"]), "%.1f / %.1f" % (c["dV1_over_rho"], c["dV2_over_rho"]), "%.0f" % c["log10_t1_yr"], "%.0f" % c["log10_t2_yr"],
                     "%.2f" % c["total_released_over_rho"], "%.2f" % c["T_max_total_meV"]])
    add(("p", "(D) Repeated transitions, entropy and causality. A chain of vacua $(\\varphi_f,\\chi_f)\\to(\\varphi_t,\\chi_f)\\to(\\varphi_t,\\chi_t)$ with a Minkowski end point releases $\\Delta V_1$ and then $\\Delta V_2$ out of the parent of energy $\\rho_\\Lambda$. "
              "The energy released over all generations telescopes, $\\sum_i\\Delta V_i=\\rho_\\Lambda-V_{\\rm final}\\le\\rho_\\Lambda$, so the total radiation, and its temperature $T\\le%.2f$ meV, is the same as for a single transition. "
              "Each generation has its own decay time (Table 12, from the rate of Eq. (7) with the parent energy of that generation); the second generation of the daughter has a Hubble rate $\\sqrt{\\Delta V_2/\\rho_\\Lambda}$ times that of the parent "
              "and decays no sooner than $10^{%.0f}$ yr in the cases that survive. Repeated transitions are therefore possible in principle, but they add no energy. "
              "The entropy per comoving curvature volume produced by the benchmark interior is $%.2f$ in units of $f^{-3}$ and is constant after reheating; the entropy inside a sphere of the bubble-wall radius at the time of the temperature maximum is $%.1f$, "
              "%s of the area bound $8\\pi^2M_{\\rm Pl}^2R_b^2=%s$ and %s of the de Sitter horizon entropy of the parent ($%s$), so the generalized second law is satisfied with enormous margin and gives no constraint. "
              "The interior is the future light cone of the nucleation event, with infinite comoving volume and a logarithmically divergent conformal time, so it contains no region causally connected to the parent except through the wall."
              % (DC[0]["T_max_total_meV"], min(c["log10_t2_yr"] for c in DC if c["log10_t2_yr"] > 0), DE["S_comoving_curvature_volume"], DE["S_in_wall_volume_at_Tmax"], sci(DE["ratio_to_bound"], 0), sci(DE["S_bound_wall_volume"], 1),
                 sci(DE["S_in_wall_volume_at_Tmax"] / DE["S_dS_parent"], 0), sci(DE["S_dS_parent"], 1))))
    add(("table", "Table 12. Chain of two transitions out of the dark-energy vacuum: tilt/coupling of each step, energy released in units of $\\rho_\\Lambda$, decay time of each generation, total released and the resulting temperature bound.",
         ["step 1 ($\\eta/\\lambda$)", "step 2 ($\\eta/\\lambda$)", "$\\Delta V_1/\\Delta V_2$", "$\\log_{10}t_1$ (yr)", "$\\log_{10}t_2$ (yr)", "$\\sum\\Delta V/\\rho_\\Lambda$", "$T_{\\max}$ (meV)"], rows, None))
    # criteria table
    rows = [
        ["Single field, decoupled (benchmark)", "1", "%.2f meV" % ROW_A[1.0]["T_max_meV"], "no", "$1$ (Milne)", "unbounded", "none"],
        ["Reservoir in $\\chi$, compensated ($\\kappa=10^{35}$)", "1, with a cancellation of $10^{35}$", "%.2f MeV" % RB[1e35]["T_max_MeV"], "no (%s GeV at 1 s)" % sci(RB[1e35]["T_at_1s_GeV"], 1), "$1$ (Milne)", "%.1f s" % RB[1e35]["t_crunch_s"], "crunch"],
        ["Hilltop at $\\rho_\\Lambda$ ($w=5M_{\\rm Pl}$)", "1", "%.2f meV" % RC[5.0]["T_max_meV"], "no", "%s" % sci(RC[5.0]["Omega_k_min"], 0), "unbounded", "energy"],
        ["Hilltop at a parent $%s\\,\\rho_\\Lambda$ ($w=5M_{\\rm Pl}$)" % sci(S["C_parent_energy_for_1MeV_over_rho_Lambda"], 1), "%s (excluded)" % sci(S["C_parent_energy_for_1MeV_over_rho_Lambda"], 1), "1 MeV", "yes", "%s" % sci(RC[5.0]["Omega_k_min"], 0), "unbounded", "survival"],
    ]
    add(("table", "Table 13. Criteria for a viable daughter universe, by scenario: parent vacuum energy in units of the observed dark energy, maximum temperature, a hot epoch (1 MeV at 1 s), flatness $\\Omega_{k,\\min}$, lifetime, and the criterion that fails.",
         ["Scenario", "parent energy$/\\rho_\\Lambda$", "$T_{\\max}$", "hot at 1 s", "$\\Omega_{k,\\min}$", "lifetime", "fails"], rows, None))
    add(("p", "Table 13 states the conditional criteria for renewal in the sense of Option A. A daughter region is a viable hot universe only if the parent vacuum energy is at least $%s$ times the observed one (excluded by the data), or if a reservoir is hidden by a cancellation of the same size, "
              "which leaves a true vacuum that crunches within %.1f s. No scenario with the observed dark energy as the parent satisfies the energy, flatness and lifetime criteria together."
              % (sci(S["C_parent_energy_for_1MeV_over_rho_Lambda"], 1), RB[1e35]["t_crunch_s"])))
    return out
