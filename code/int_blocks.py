# -*- coding: utf-8 -*-
"""Manuscript blocks for Section 7: density-dependent vacuum interfaces. Numbers from vacuum_phase_map.json, vacuum_wall.json,
interface_transport.json, bubble_gravity.json, bubble_to_cycle.json, bubble_entropy.json and interface_validation.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
PM, WJ, TJ, GJ, CJ, EJ, VJ = (J("vacuum_phase_map.json"), J("vacuum_wall.json"), J("interface_transport.json"), J("bubble_gravity.json"),
                              J("bubble_to_cycle.json"), J("bubble_entropy.json"), J("interface_validation.json"))
COH = PM["symmetron_coherence"]
PS, TW = WJ["physical"], WJ["trigger_wall"]
DM = TJ["dark_matter"]
VC = 3.0 * 0.6889


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "$" + texsci(x, d) + "$"


def ev(key, name, i=0):
    return [e for e in CJ[key]["events"] if e["event"] == name][i]


B_P, B_I = "B percolated", "B isolated"
LNG = dict(A=CJ["A"]["lnG_mean"], B=float(np.mean(CJ[B_P]["lnG"])), Bi=float(np.mean(CJ[B_I]["lnG"])),
           C=float(np.mean(CJ["C percolated (computed selectivity)"]["lnG"])), Cs=float(np.mean(CJ["C* percolated, fully selective, hot bounce"]["lnG"])))
NOREG = CJ["B percolated, no re-thermalization"]["lnG"]
DW = CJ["D"]["window_B0"]
SUM = dict(n_checks=VJ["n"], all_passed=VJ["all_passed"], a_trig=WJ["a_trigger"], B_trig=TW["B"], f_filt=DM["filtered_fraction"], lnG_B=LNG["B"],
           D_window=DW, seed_ratio=COH["seed_over_fluctuation_median"])
TB = WJ["walls"]["0.30"]["rows"]


def blocks():
    B = []
    add = B.append
    add(("h1", "7. Density-dependent vacuum interfaces"))
    add(("p", "Sections 6.2--6.15 treat the release of the held vacuum as homogeneous. A matter-dependent potential can also support localized "
              "configurations: walls between coexisting vacua, bubbles nucleated where the matter has diluted enough, and regions on either side of an "
              "interface whose contents need not be the same. This section asks whether such an interface exists in the two matter-coupled potentials "
              "of this paper, and whether the region it encloses can contract, bounce and reheat. The questions are taken in order, and a negative answer "
              "at any stage is kept: (A) does dilution create a metastable window, (B) does a wall form, (C) does the wall act selectively on what crosses "
              "it, (D) does the enclosed region genuinely contract, and (E) can it bounce and reheat? Four controls separate the contribution of each "
              "ingredient: A, the homogeneous held vacuum of Section 6.8; B, density-dependent bubble nucleation with every species transmitted; C, the "
              "same with the transmission computed in Section 7.3; and D, the matter coupling switched off."))
    # ---------------------------------------------------------------------------------------------- 7.1
    add(("h2", "7.1 Phase structure: where tunneling is meaningful"))
    add(("p", "For a conserved Einstein-frame density $\\rho_*$ the effective potential is $V_{\\rm eff}(\\varphi,\\rho_*)=V_0(\\varphi)+\\rho_*A(\\varphi)$, and the "
              "question is whether two minima coexist, separated by a barrier, for some range of $\\rho_*$ (vacuum_phase_map.py). In the symmetron form of "
              "Section 6.2 they never do for the occupied state. The held point $\\varphi=0$ loses stability at $\\rho_{\\rm crit}$ exactly when the broken "
              "minima appear, and with a tilt the occupied minimum is followed continuously into the deeper one; in no case is the occupied minimum ever "
              "metastable. The release of Section 6 is therefore a classical (spinodal) roll, and Coleman--De Luccia tunneling does not apply to it. Its "
              "only possible interfaces are domain walls between the two broken minima. They form if the sign of the roll is not set coherently, and they "
              "would carry up to %s%% of the critical density per Hubble volume (tension %s $H_0M_{\\rm Pl}^2$ after the release). Coherence is marginal "
              "in the integration: the restored value at the start of each held epoch is %s to %s times the Hubble-scale quantum fluctuation of the "
              "field, $H_\\Lambda/2\\pi$. Any explicit breaking of $\\varphi\\to-\\varphi$ with a tilt $\\epsilon\\gtrsim%s\\,\\mu^2v$ sets the sign coherently, so "
              "the homogeneous treatment needs that tilt or an equivalent seed. Matter inhomogeneity does not produce sharp interfaces either: the field "
              "averages the density over its Compton length ($%.0f$ Mpc at release), where linear fluctuations are $\\sigma_R=%.3f$, so the release time "
              "varies by $\\delta a/a\\approx%.3f$ between such regions and the field values differ by a factor of %.1f across them."
              % (round(100 * COH["domain_walls"][-1]["wall_energy_per_Hubble_volume_over_rho_crit"]), "%.2f" % COH["domain_walls"][-1]["sigma_H0MPl2"],
                 "%.0f" % (COH["phi_rest_over_v"]["min"] / COH["dphi_quantum_over_v"]), "%.0f" % (COH["phi_rest_over_v"]["max"] / COH["dphi_quantum_over_v"]),
                 texsci(COH["eps_hat_min"], 0), COH["compton_scale_Mpc"], COH["sigma_R_linear_today"], COH["release_spread_delta_a_over_a"],
                 COH["field_ratio_between_patches"])))
    add(("p", "The tunneling branch of Section 6.1 is different. With the coupling centred on the false vacuum, in units $x=\\varphi/f$ and $\\lambda f^4$,"))
    add(("eq", r"p(x;\eta,\theta)=\frac{(x^2-1)^2}{8}+\frac{\eta x}{2}+\frac{\theta}{2}(x-1)^2,\qquad \theta=\frac{\rho_*}{\lambda f^2M^2},\qquad \Delta V(\theta)=p(x_h)-p(x_t)"))
    add(("p", "two minima coexist for $\\theta_{\\rm low}<\\theta<\\theta_{\\rm high}$, with $\\theta_{\\rm low}=0$ for $\\eta<\\eta_c$ (Fig. 23). The matter-held "
              "minimum $x_h\\approx1$ is the true vacuum above the degeneracy density, which is exactly $\\theta_{\\rm deg}=\\eta/2$ because $p$ is then even in "
              "$x$, and metastable below it: dilution turns a stable held vacuum into a decaying one, and tunneling is meaningful for $0<\\theta<\\eta/2$. "
              "Above $\\theta_{\\rm high}$ (%.3f for $\\eta=0.30$) the other minimum disappears, which is what restores the held vacuum in a contraction "
              "(Section 7.5). For $\\eta>\\eta_c$ the held minimum itself disappears below $\\theta_{\\rm low}$ (for example %.3f at $\\eta=0.45$), and the release "
              "is a classical roll as in the symmetron form. The logarithmic coupling estimate of Section 6.1, $\\delta\\Delta V=\\rho_m\\Delta\\ln m$, agrees "
              "with the exact Einstein-frame term $\\rho_*\\Delta A$ to %.2f%% for a change of the rate by a factor $e$ and to %.0f%% in the reset window."
              % (PM["tunneling_profiles"]["0.30"]["window"][1], PM["tunneling_profiles"]["0.45"]["window"][0],
                 100 * (min(c["exact_over_log"] for c in PM["coupling_check"]) - 1), 100 * (max(c["exact_over_log"] for c in PM["coupling_check"]) - 1))))
    add(("fig", "fig23_vacuum_landscape.png", "Fig. 23. (a) Effective potential of the tunneling branch ($\\eta=0.30$) at five densities: dilution turns the matter-held "
                "minimum near $x=1$ from the true into the false vacuum at $\\theta_{\\rm deg}=\\eta/2$. (b) Phase diagram: number of minima and which is lower, "
                "with the degeneracy line and the lines where a minimum disappears; the grey line marks $\\eta_c$.", 6.3))
    # ---------------------------------------------------------------------------------------------- 7.2
    add(("h2", "7.2 The wall and its density-dependent nucleation"))
    r0, r6 = TB[0], [r for r in TB if abs(r["theta_over_deg"] - 0.6) < 1e-9][0]
    deg = WJ["walls"]["0.30"]["degenerate"]
    add(("p", "The interface follows from the field equations rather than being imposed. At the degeneracy density the static planar wall, "
              "$x''=\\partial_xp$, has the first integral $z(x)=\\int dx/\\sqrt{2(p-p_{\\rm min})}$, tension $\\sigma=\\int\\sqrt{2(p-p_{\\rm min})}\\,dx=%.3f\\,\\sqrt\\lambda f^3$ "
              "and 10--90%% thickness $%.1f/(\\sqrt\\lambda f)$ for $\\eta=0.30$; the symmetric case reproduces $\\sigma=2/3$ and $x=\\tanh(z/2)$. Below the degeneracy "
              "density a static planar wall does not exist, since the pressure $\\Delta V$ accelerates it, and the wall is read from the O(4) bounce "
              "(vacuum_wall.py): its tension $\\sigma_b=\\int x'^2dr$ and thickness change little with density, while the radius grows from $%.1f$ to $%.1f$ "
              "and the action from $b=%.0f$ to $%.0f$ between $\\theta=0$ and $0.6\\,\\theta_{\\rm deg}$ (Fig. 24a,b). The thin-wall action "
              "$B_{\\rm tw}=27\\pi^2\\sigma^4/2\\Delta V^3$, with $\\sigma$ computed after removing the energy difference, is recovered within 1%% from "
              "$0.6\\,\\theta_{\\rm deg}$ upwards, and underestimates the thick-wall bounce near zero density ($b/B_{\\rm tw}=%.2f$). Shooting becomes "
              "exponentially sensitive in the thin-wall limit, where collocation is used; the two agree where both work (Table 21)."
              % (deg["sigma"], deg["thickness"], r0["R_w"], r6["R_w"], r0["b"], r6["b"], r0["b_over_B_tw"])))
    add(("p", "With the normalization of Section 4 (false vacuum = observed dark energy, $V_t=-qV_f$ with $q=1$, $B_0=b(0)/\\lambda=%.0f$, "
              "$\\theta_0=%.1f\\,\\theta_{\\rm deg}$ today, $\\eta=0.30$), $f=%.2f$ meV and the wall scale $1/\\sqrt\\lambda f$ is %.3f mm. The rate "
              "$\\Gamma=\\mathcal A\\,e^{-B(\\theta)}$, with the prefactor convention of Section 4.1 and the actual Hubble rate, rises as the matter "
              "dilutes (Fig. 24c); the past light cone is safe because the coupling raises the action there, and the transition completes "
              "($\\Gamma/H^4>9/4\\pi$, so that bubbles percolate) at $a=%.2f$, when $\\theta=%.3f\\,\\theta_{\\rm deg}$ and $B=%.0f$. The nucleated bubble then has "
              "radius %.2f mm, wall thickness %.2f mm and tension %.0f meV$^3$. A rising nucleation probability marks the trigger; it does not by itself "
              "say what the enclosed region becomes, which Sections 7.4--7.5 compute."
              % (WJ["base"]["B0"], WJ["base"]["f0"], PS["f_meV"], PS["length_unit_mm"], WJ["a_trigger"], TW["theta_over_deg"], TW["B"], TW["R_w_mm"],
                 TW["thickness_mm"], TW["sigma_meV3"])))
    add(("fig", "fig24_interface_properties.png", "Fig. 24. (a) Wall tension (solid) and thickness/10 (dashed) of the O(4) bounce against density for three tilts. "
                "(b) Bounce action $b=\\lambda B$, numerical (points) and thin-wall (dotted). (c) $\\ln(\\Gamma/H^4)$ along the dilution history of the baseline, "
                "with the completion threshold and the trigger.", 6.6))
    # ---------------------------------------------------------------------------------------------- 7.3
    add(("h2", "7.3 Is the interface selective?"))
    sp = TJ["species"]
    g1 = sp["1.00"]
    add(("p", "A wall makes the mass of every field coupled to $\\varphi$ depend on position. For a test scalar with $\\mathcal L_{\\rm int}=-\\frac12g^2\\varphi^2\\psi^2$, "
              "$m_\\psi^2(z)=m_0^2+g^2f^2x(z)^2$, and the modes of the planar wall obey"))
    add(("eq", r"\left[-\frac{d^2}{dz^2}+m_\psi^2(z)\right]u=(\omega^2-k_\parallel^2)\,u,\qquad T=\frac{k_+}{k_-}\frac{1}{|A|^2},\qquad R=\left|\frac{B}{A}\right|^2"))
    add(("p", "with the transmitted wave on the false-vacuum side and the flux-normalized coefficients read on the true-vacuum side (interface_transport.py). "
              "The solver reproduces the reflectionless Poeschl--Teller result for the symmetric kink to %s and conserves $T+R$ to %s (Fig. 25b). "
              "Through the wall at the trigger, the interior is heavier for every coupling ($m_{\\rm in}/m_{\\rm out}=%.2f$, since $|x_t|>|x_h|$), modes below "
              "the interior threshold are reflected, and above it $T$ exceeds 0.5 within $0.1\\sqrt\\lambda f$ of the threshold for all couplings tested (Fig. 25a). "
              "The interface is selective, but kinematically: it filters by mass, as any step does, with little additional reflection from its shape. "
              "Photons, neutrinos and baryons do not couple to $\\varphi$ and cross unchanged."
              % (sci(TJ["max_PT_error"], 0), sci(TJ["max_unitarity_error"], 0), g1["m_in"] / g1["m_out"])))
    add(("p", "For the coupled dark matter the mass ratio across the wall is $A(x_t)/A(x_h)=%.2f$, far larger than the kinetic energy of cold dark matter, so a "
              "slow wall reflects all of it. The wall is not slow. Its driving pressure exceeds the largest pressure the dark matter can exert, "
              "$\\rho_*\\Delta A$ (the Boedeker--Moore limit for a particle that is always transmitted at high Lorentz factor \\cite{BM09}), by a factor %.0f, so "
              "it runs away ($\\Gamma=R/R_0$, Section 7.4), and dark matter at rest crosses once $\\Gamma>A(x_t)/A(x_h)$, i.e. within %.2f mm of the nucleation "
              "point. The fraction of the dark matter swept by the wall before it becomes transparent is %s. A thermal bath of $\\psi$ at the CMB temperature "
              "of the trigger epoch would exert less than %s of the driving pressure. Selective transmission therefore exists but has no cosmological effect: "
              "filtered dark matter at a first-order transition \\cite{BKL20} requires walls that stay slow, which the matter-coupled potential does not produce."
              % (DM["A_true"] / DM["A_false"], 1.0 / DM["P_dm_max_over_dV_vac"], DM["R_threshold_mm"], sci(DM["filtered_fraction"], 0),
                 sci(0.31 * TJ["T_cmb_at_trigger_units"] ** 2, 0))))
    add(("fig", "fig25_selective_transmission.png", "Fig. 25. (a) Transmission of a test scalar through the wall at the trigger against momentum above the "
                "interior threshold, for four couplings. (b) Validation on the symmetric kink: numerical transmission (dots) against the exact "
                "Poeschl--Teller result (lines), including the reflectionless coupling $g^2/\\lambda=1/2$.", 6.3))
    # ---------------------------------------------------------------------------------------------- 7.4
    add(("h2", "7.4 Gravity of the wall and the geometry of the enclosed region"))
    add(("p", "For a thin wall of tension $\\sigma$ and areal radius $R$ between an interior of density $\\rho_-$ and an exterior of density $\\rho_+$, the Israel "
              "junction condition can be written with the Misner--Sharp mass on each side \\cite{Israel66,BGG}:"))
    add(("eq", r"\Gamma_--\Gamma_+=\frac{\sigma R}{2M_{\rm Pl}^2},\qquad \Gamma_\pm=\sqrt{1+\dot R^2-\frac{\rho_\pm R^2}{3M_{\rm Pl}^2}}\ \Rightarrow\ \Gamma_-=\frac{(\rho_+-\rho_-)R}{3\sigma}+\frac{\sigma R}{4M_{\rm Pl}^2}"))
    add(("p", "(bubble_gravity.py). From $R_0=3\\sigma/\\Delta V$ (%.2f mm) to the Hubble radius the wall follows the flat-space runaway $\\Gamma=R/R_0$ to "
              "round-off and reaches $\\Gamma=$%s; its self-gravity, $\\sigma R/2\\Gamma$, is %s; and the Misner--Sharp mass it carries equals the vacuum-energy "
              "deficit of the region it encloses (Fig. 26a). The released energy is in the wall, not in the interior. The geometry of the enclosed region follows "
              "from the junction: the fluid inside keeps the Hubble rate of the parent at nucleation while its density drops by $\\Delta V_{\\rm eff}$, so an "
              "isolated bubble encloses an open region with $K/a^2=\\Delta V_{\\rm eff}/3$, and a percolated transition leaves a flat universe whose wall energy "
              "is thermalized by collisions. A collapsing wall is not a cosmological recollapse; the evolution of the enclosed region is computed separately below."
              % (GJ["wall"]["R0_mm"], sci(GJ["wall"]["Gamma_at_RH"], 1), sci(GJ["wall"]["max_grav"], 0))))
    # ---------------------------------------------------------------------------------------------- 7.5
    add(("h2", "7.5 Contraction, restoration, bounce and reheating of the daughter region"))
    nB, tB, rB, bB = ev(B_P, "nucleation"), ev(B_P, "turnaround"), ev(B_P, "restoration"), ev(B_P, "bounce")
    tI = ev(B_I, "turnaround")
    add(("p", "The daughter region is evolved with the effective bounce of Section 6.8, $H^2=(\\rho/3)(1-\\rho/\\rho_c)+K/a^2$, with radiation, baryons, dark "
              "matter of field-dependent mass and the scalar in one of its minima (its mass, of order meV, exceeds $H$ by thirty orders, so it follows the "
              "minimum adiabatically). Events are roots in $\\ln a$, $H$ comes from the constraint, and the proper time is a quadrature, so the constraint is "
              "satisfied exactly between events and energy is conserved across them. After the transition the region holds the negative true vacuum and "
              "turns around at %.2f times the nucleation scale factor (%.2f for the isolated, open region). It contracts; when the dark-matter density reaches "
              "$\\theta_{\\rm high}$ the true minimum disappears and the field returns to the matter-held minimum, releasing $%.2f\\,V_f$ into the bath. This "
              "restoration is the tunneling-branch analogue of the symmetry restoration of Section 6.8, and like it needs dissipation: undamped, the oscillation "
              "about the held minimum is an adiabatic invariant that returns to the same amplitude when the density falls back through $\\theta_{\\rm high}$, "
              "and the field could re-enter the true minimum without a held epoch. The region then bounces at the density of Section 6.8, radiation dominated, and "
              "re-expands with the field held; the dark matter is re-thermalized, and the vacuum is released again when the dilution completes the next "
              "transition (Fig. 26b,c). The cycles are self-similar after the first: $\\ln G=%.3f$ per cycle for a percolated transition and $%.3f$ for an "
              "isolated bubble, set by the entropy produced when the wall energy is thermalized and when the vacuum is restored."
              % (tB["a_max_over_a_nuc"], tI["a_max_over_a_nuc"], rB["dE_released"] / VC, LNG["B"], LNG["Bi"])))
    add(("p", "The comoving entropy never decreases (bubble_entropy.py): it is inherited across the wall, since every species crosses it, and is raised by "
              "a factor of about %.0f when the wall energy thermalizes. The interior bookkeeping $dE_{\\rm in}=F_E\\,dV+W_{\\rm wall}$, with the swept-in content "
              "$F_E=\\rho_+$ and the work $W_{\\rm wall}=-\\Delta V_{\\rm eff}dV$ handed to the wall, closes to round-off. The decay removes the cosmological "
              "horizon of the false vacuum, whose entropy ($%s$) exceeds that of the radiation in a Hubble volume ($%s$) by 34 orders of magnitude; this is the "
              "usual situation for decays out of de Sitter space and is accounted for by the action itself, not a violation of the balance of the matter. "
              "The interface redistributes energy; it does not remove entropy, and the entropy growth of the cycle is that of the homogeneous model."
              % (EJ["entropy"][B_P]["nucleation_jumps"][1], texsci(EJ["horizon"]["S_dS"], 1), texsci(EJ["horizon"]["S_radiation_Hubble_volume"], 1))))
    add(("fig", "fig26_daughter_evolution.png", "Fig. 26. (a) Lorentz factor of the wall from the Israel junction against the flat-space runaway, and its self-gravity "
                "(right axis). (b) Scale factor of a percolated transition through three cycles: nucleation, turnaround, restoration of the held vacuum, "
                "bounce. (c) The same for the open region enclosed by an isolated bubble.", 6.6))
    # ---------------------------------------------------------------------------------------------- 7.6
    add(("h2", "7.6 Controls and viability"))
    rows = [["A", "homogeneous held vacuum (Sec. 6.8)", "classical roll", "%.2f" % LNG["A"], "coherent seed or tilt; $T_b\\gtrsim m/5$"],
            ["B", "bubbles, all species transmitted", "nucleation, $B=%.0f$" % TW["B"], "%.3f" % LNG["B"], "dissipation at restoration"],
            ["B", "isolated bubble (open region)", "nucleation", "%.3f" % LNG["Bi"], "as above"],
            ["B", "no re-thermalization at the bounce", "nucleation", "%.3f, %.3f, ..." % (NOREG[0], NOREG[1]), "cycles continue, dark matter dilutes away"],
            ["C", "computed selectivity", "nucleation", "%.3f" % LNG["C"], "identical to B (filtered fraction %s)" % texsci(DM["filtered_fraction"], 0).join("$$")],
            ["C*", "fully selective wall, hot bounce", "nucleation", "%.3f" % LNG["Cs"], "restoration only at the bounce"],
            ["C*", "fully selective wall, cold bounce", "nucleation", "fails", "no held epoch after the first cycle"],
            ["D", "no matter coupling", "constant rate", "fails", "window $%.0f\\le B_0\\le%.0f$; no restoration" % tuple(DW)]]
    add(("table", "Table 20. Controls for the interface mechanism (baseline $\\eta=0.30$, $q=1$, $B_0=%.0f$, $\\theta_0=%.1f\\,\\theta_{\\rm deg}$). $\\ln G$: growth "
                  "of the scale factor at successive transitions." % (WJ["base"]["B0"], WJ["base"]["f0"]),
         ["Model", "Description", "Release", "$\\ln G$ per cycle", "Requires / outcome"], rows, None))
    add(("p", "Table 20 answers the five questions. (A) Dilution creates a metastable window in the tunneling branch, not in the symmetron form. (B) A wall forms, "
              "with a tension fixed by the field equations and nearly independent of density. (C) It is selective by mass, but it runs away and becomes "
              "transparent within a millimetre, so the computed selectivity changes nothing: models B and C coincide to %s. Only a hypothetical wall that kept "
              "all dark matter out (C*) would differ, and then the cycle survives only if the bounce re-thermalizes the dark matter, which restores the field "
              "at the bounce instead of during the contraction. (D) The region enclosed genuinely contracts for any negative true vacuum ($q>0$) and is solved "
              "as a cosmology, not inferred from the wall radius. (E) It bounces and reheats with the effective bounce and the restoration by dissipation, "
              "and the cycles are self-similar when the dark matter is re-thermalized; without re-thermalization they continue but the dark matter dilutes away "
              "relative to the entropy. Without the coupling (D) there is no restoration, and a single transition that neither has already happened nor never "
              "completes requires $B_0$ in a window of width %.0f around %.0f, the fine tuning of Section 4; with the coupling, the region of the "
              "$(B_0,\\theta_0)$ plane that survives the past light cone and completes in the future extends from $B_0=%.0f$ to $%.0f$ for suitable $\\theta_0$ "
              "and covers %.0f%% of the plane scanned (Fig. 27a)."
              % (sci(DM["filtered_fraction"], 0), DW[1] - DW[0], 0.5 * (DW[0] + DW[1]), CJ["map"]["B0_range_viable"][0], CJ["map"]["B0_range_viable"][1],
                 100 * CJ["map"]["fraction_viable"])))
    add(("p", "The interface therefore gives a second, independent route to the conditional cycle, with a trigger that is first order rather than spinodal, "
              "but it adds no new condition that the homogeneous model escapes and removes none: the turnaround needs $q>0$, the restoration needs "
              "dissipation, and self-similarity needs re-thermalization. It does not filter entropy, does not restore the universe to an earlier state and "
              "does not prove repeated big bangs. The idea of a matter-selective interface was suggested by membranes that pass some species and retain "
              "others; in this potential the analogy fails quantitatively because the wall runs away. All %d interface checks pass (Table 21)."
              % VJ["n"]))
    add(("fig", "fig27_viability_map.png", "Fig. 27. (a) Outcome over the vacuum action $B_0$ and the coupling today $\\theta_0/\\theta_{\\rm deg}$ ($\\eta=0.30$, $q=1$): "
                "already decayed in the past light cone, no completed transition, or nucleation followed by recollapse, restoration, bounce and reheating; "
                "the red band is the window without coupling (model D). (b) Contraction requires $q>0$; re-thermalization of the dark matter "
                "($T_b\\gtrsim mA/5$) is required only by a fully selective wall (C*).", 6.4))
    def clean(t):
        for a, b_ in (("e^d - 1", "exp(d) - 1"), ("(k1 + k2)^2", "(k1 + k2) squared"), ("H^2", "H squared"), ("Gamma/H^4", "Gamma/H4"),
                      ("theta_deg", "theta(deg)"), ("R/R_0", "R/R0"), ("rho_c", "rho(c)"), ("dE_in", "dE(in)"), ("F_E", "F(E)"), ("W_wall", "W(wall)"),
                      ("B_tw", "B(tw)"), ("R_0", "R0")):
            t = t.replace(a, b_)
        return t.replace("_", " ").replace("^", "")
    vrows = [[clean(c["name"]), "pass" if c["passed"] else "FAIL"] for c in VJ["checks"]]
    add(("table", "Table 21. Validation of the interface calculations (interface_validation.py).", ["Check", "Result"], vrows, None))
    return B


if __name__ == "__main__":
    b = blocks()
    print(len(b), "blocks;", SUM)
