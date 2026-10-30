# -*- coding: utf-8 -*-
"""Manuscript blocks for Sections 6.2-6.8: matter dilution as the trigger, from a held vacuum to a cyclic hot big bang.
Numbers from turnaround_scan.json, dilution_thermal.json, density_bounce.json, cyclic_extension.json, hot_cycles.json and dilution_validation.json."""
import json
import os

import numpy as np

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
S, TH, DB, CE, HC, DV = (J("turnaround_scan.json"), J("dilution_thermal.json"), J("density_bounce.json"), J("cyclic_extension.json"),
                         J("hot_cycles.json"), J("dilution_validation.json"))


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "⟦" + texsci(x, d) + "⟧"
GYR = MC.H0_GYR
REF = MC.Model(2.0, 100.0, 1.0)
ROWS = S["rows"]
row = lambda a_c, mu, q, c: [r for r in ROWS if r["a_c"] == a_c and r["mu"] == mu and r["q"] == q and r["coupled"] == c][0]
RC, RB = row(2.0, 100.0, 1.0, True), row(2.0, 100.0, 1.0, False)
N_C = sum(1 for r in ROWS if r["coupled"])
N_REC = sum(1 for r in ROWS if r["coupled"] and r["cls"] == "transition and recollapse")
N_NOREC = sum(1 for r in ROWS if r["coupled"] and r["cls"] == "transition, no recollapse")
CTRL_BEFORE = sorted({r["mu"] for r in ROWS if (not r["coupled"]) and r["cls"] == "rolled before today"})
AV = [r["A_at_v"] for r in ROWS if r["coupled"]]
SEEDS = S["seeds"]
SEC = {s["seed"]: s for s in SEEDS if s["coupled"]}
SEB = {s["seed"]: s for s in SEEDS if not s["coupled"]}
TM = TH["models"]
TV = TH["thermal_viability"]
TV100 = [r for r in TV["rows"] if r["mu"] == 100.0 and r["a_T"] == 2.0][0]
HIST = DB["histories"]
h_ = lambda eta, B0, f0: [h for h in HIST if h["eta"] == eta and h["B0"] == B0 and h["theta0_over_deg"] == f0][0]
CYC = HC["runs"]
BASE = CYC[0]
BC = [c for c in BASE["cycles"] if c["n"] >= 1]
GN = [c["G"] for c in BC if c.get("G")]
GS = [c["G_from_entropy"] for c in BC if c.get("G_from_entropy")]
by = lambda **kw: [r for r in CYC if all(r["params"][k] == v for k, v in kw.items())][0]
NORESET, NODISS = by(reset=False), by(c_T=0.0)
COLD = {(r["params"]["rho_c_over_Vc"], r["params"]["ups"], r["params"]["mu"], r["params"]["q"]): r for r in CE["runs"]}
cold = lambda rc=1e4, ups=0.0, mu=30.0, q=1.0: COLD[(rc, ups, mu, q)]
CVALID = all(v["passed"] for v in DV.values())


def gm(xs):
    xs = [x for x in xs if x]
    return float(np.median(xs)) if xs else float("nan")


def blocks():
    B = []
    add = B.append
    # ------------------------------------------------------------------------------------------------ 6.2 held vacuum
    add(("h2", "6.2 A vacuum held by matter"))
    add(("p", "The estimate above concerns a coupling that changes a tunneling rate. A stronger form of the trigger is a coupling that holds the vacuum itself in place while the "
              "matter density is high, so that dilution releases it classically. The minimal realization couples the scalar conformally to dark matter only (baryons and radiation are "
              "uncoupled, which avoids the stringent visible-sector fifth-force tests),"))
    add(("eq", r"S=\int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2R}{2}-\frac{1}{2}(\nabla\varphi)^2-V_0(\varphi)\right]+S_{\rm dm}\left[A^2(\varphi)g_{\mu\nu},\psi_{\rm dm}\right]"))
    add(("p", "For pressureless dark matter the conserved density $\\rho_*=\\rho_{*,0}a^{-3}$ enters the field equation through the effective potential, and the physical dark-matter "
              "density is $A(\\varphi)\\rho_*$ \\cite{Amendola,HK}. With"))
    add(("eq", r"V_0(\varphi)=V_c-\frac{\mu^2}{2}\varphi^2+\frac{\lambda}{4}\varphi^4,\qquad A(\varphi)=1+\frac{\varphi^2}{2M^2},\qquad V_{\rm eff}=V_0+\rho_*A(\varphi)"))
    add(("p", "the effective mass at the symmetric point is $m_{\\rm eff}^2=\\rho_*/M^2-\\mu^2$. The point $\\varphi=0$, with vacuum energy $V_c>0$, is held while "
              "$\\rho_*>\\rho_{\\rm crit}=\\mu^2M^2$ and released when the expansion dilutes the matter below it, at"))
    add(("eq", r"a_c=\left(\frac{\rho_{*,0}}{\mu^2M^2}\right)^{1/3},\qquad V_c-V_0(v)=\frac{\mu^2v^2}{4}=\frac{A(v)-1}{2}\,\rho_{\rm crit}"))
    add(("p", "This is the symmetron form \\cite{HK,HKLM11} with two changes: the coupling is to dark matter, and the broken vacua $\\varphi=\\pm v$, $v^2=\\mu^2/\\lambda$, carry "
              "negative energy $V_0(v)=-qV_c$. Today $V_c=3\\Omega_\\Lambda H_0^2M_{\\rm Pl}^2$, so the model reproduces the observed dark energy while it is held, and $a_c>1$ is required "
              "for it to be held now. The second relation is exact and has a consequence that cannot be avoided: the vacuum energy given up at the release equals half the coupling energy "
              "stored at the critical density, and since $\\rho_{\\rm crit}<\\rho_{\\rm dm,0}<V_c$, a release into a negative vacuum needs $A(v)-1>2V_c/\\rho_{\\rm crit}$, i.e. the "
              "dark-matter mass must grow severalfold once the field rolls ($A(v)=%.0f$ for $a_c=2$, $q=1$; $%.1f$--$%s$ over the parameter grid below). At $\\varphi=0$ the coupling "
              "$\\beta=M_{\\rm Pl}\\,d\\ln A/d\\varphi$ vanishes, so there is no fifth force today. Fig. 13a shows the effective potential as the matter dilutes."
              % (REF.A(REF.v), min(AV), sci(max(AV), 1))))
    add(("p", "The background equations (flat, general relativity) are"))
    add(("eq", r"3M_{\rm Pl}^2H^2=\rho_r+\rho_b+A\rho_*+\frac{\dot\varphi^2}{2}+V_0,\qquad \ddot\varphi+3H\dot\varphi+V_0'+\rho_*A'=0,\qquad \dot\rho_*=-3H\rho_*"))
    add(("p", "with $\\dot H=-(\\tfrac43\\rho_r+\\rho_b+A\\rho_*+\\dot\\varphi^2)/2M_{\\rm Pl}^2$. Total stress-energy is conserved: the scalar and the coupled matter exchange energy "
              "through $A'$, and the coupling energy is counted once, in $A\\rho_*$. They are integrated from recombination ($z=1100$) with the field at the de Sitter quantum floor, "
              "$\\varphi_0=10^{-60}v$ (about $H/2\\pi$ in Planck units), in units of $H_0$ and $M_{\\rm Pl}$ (dilution_background.py)."))
    # ------------------------------------------------------------------------------------------------ 6.3 release and recollapse
    add(("h2", "6.3 Release and recollapse"))
    add(("p", "Three models are compared with identical initial data: $\\Lambda$CDM (A), the uncoupled scalar with the same potential (B, the control) and the coupled scalar (C). "
              "For the reference case $a_c=2$, $\\mu=100H_0$, $q=1$ (Fig. 13), $\\Lambda$CDM expands forever. The coupled field stays at $|\\varphi|/v=%s$ up to the present, so the past "
              "history is that of $\\Lambda$CDM ($w_0=-1$ to all digits; $H(a)$ agrees with $\\Lambda$CDM to the integration accuracy), is released after $a_c$, rolls at $a=%.2f$ and "
              "turns the expansion around at $a=%.2f$, %.0f Gyr from now; the total density reaches zero there and the contraction follows. The uncoupled control has been unstable "
              "since recombination and turns around at $a=%.2f$, %.0f Gyr from now."
              % (sci(RC["phi_today_over_v"], 1), RC["a_roll"], RC["a_turn"], RC["t_turn"] * GYR, RB["a_turn"], RB["t_turn"] * GYR)))
    add(("p", "The comparison the trigger has to pass is C against B. Over a grid of %d coupled models ($a_c=1.25$--$5$, $\\mu=3$--$300H_0$, $q$ from $-0.5$ to $10$) every one "
              "reaches the present with the vacuum held; %d release into a negative vacuum and recollapse, and the %d with $q<0$ settle in a lower de Sitter vacuum and expand forever "
              "(Fig. 14b, Table 16). The control rolls before today for $\\mu\\ge%.0fH_0$, so it cannot describe the observed universe there, while the coupled model can at every "
              "$\\mu$. The coupling is what makes a late release compatible with an observed $\\Lambda$CDM past. The release is not instantaneous: the field grows from the quantum "
              "floor at a rate of order $\\mu$, so the roll lags $a_c$ by about $\\ln(v/\\varphi_0)/\\mu$ Hubble times, from $a=%.2f$ to $%.2f$ for seeds $10^{-6}$ to $10^{-60}v$; "
              "the control rolls before today for every seed above $10^{-40}v$ (Fig. 14c). Large $\\mu$ makes the release a sharp trigger at $a_c$."
              % (N_C, N_REC, N_NOREC, CTRL_BEFORE[0], SEC[1e-6]["a_roll"], SEC[1e-60]["a_roll"])))
    rowsT = []
    for mu in sorted({r["mu"] for r in ROWS}):
        rc_, rb_ = row(2.0, mu, 1.0, True), row(2.0, mu, 1.0, False)
        rowsT.append(["%g" % mu, "%.3g" % rc_["a_roll"], "%.3g" % rc_["a_turn"], "%.0f" % (rc_["t_turn"] * GYR),
                      ("rolled before today" if rb_["cls"] == "rolled before today" else "%.3g" % rb_["a_turn"]), "%.0f" % rc_["A_at_v"]])
    add(("table", "Table 16. Release and turnaround of the coupled model and the uncoupled control ($a_c=2$, $q=1$, seed $10^{-60}v$ at $z=1100$). Times from today; $A(v)$ is the "
                  "factor by which the dark-matter mass grows after the release.",
         ["$\\mu/H_0$", "roll, $a$", "turnaround, $a$", "turnaround (Gyr)", "control turnaround, $a$", "$A(v)$"], rowsT, None))
    add(("fig", "fig13_matter_held_vacuum.png", "Fig. 13. The matter-held vacuum ($a_c=2$, $\\mu=100H_0$, $q=1$). (a) Effective potential for several matter densities; the symmetric point, "
                "with energy $V_c$, is a minimum while $\\rho_*>\\rho_{\\rm crit}$. (b) Hubble rate against time from today for the coupled model, the uncoupled control and "
                "$\\Lambda$CDM. (c) Field amplitude from recombination: the coupled field is held at the quantum floor until $a_c$, the control grows from the start.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.4 cooling vs dilution
    add(("h2", "6.4 Cooling or dilution?"))
    add(("p", "A thermal mass $cT^2\\varphi^2/2$ from a bath of temperature $T$ acts in the same way: the symmetric point is held while $\\rho_*/\\rho_{\\rm crit}+(T/T_c)^2>1$, "
              "$T_c=\\mu/\\sqrt c$, and the three possible triggers are compared in Fig. 14a. With the critical scale factor set to 2 for each, matter alone releases the field at $a=2$, "
              "temperature alone at $a=2$, both together at $a=%.2f$, and all three recollapse ($a=%.1f$, $%.1f$ and $%.1f$). The thermal free energy is used in its saturating "
              "form, $V_T=(cT^2\\varphi^2/2)/(1+c\\varphi^2/2T^2)$, which follows the high-temperature result while the bath particles are light and the free energy of the species "
              "($\\sim T^4$) when they become heavy; the truncated quadratic form fails during the roll."
              % (TH["a_release_C_analytic"], TM["A_matter"]["a_turn"], TM["B_temperature"]["a_turn"], TM["C_both"]["a_turn"])))
    add(("p", "Mathematically, then, cooling could trigger the release; physically it cannot. In physical units $\\sqrt c=\\mu a_T/T_0$, so a release at $a_T\\sim2$ with "
              "$\\mu=100H_0$ needs $\\sqrt c=%s$, and a scalar with so weak a coupling interacts with the bath at a rate $\\sim c^2T\\approx%s\\,H_0$: it never reaches "
              "equilibrium, and the thermal mass is never established. A coupling strong enough for equilibrium today, $\\sqrt c>%s$, holds the field until "
              "$a_T\\sim%s$. The two requirements exclude each other by twenty orders of magnitude, and the CMB photons couple to a neutral scalar only through higher-dimension "
              "operators whose thermal mass falls faster still. Cooling does not trigger the release; dilution of dark matter can. This complements the result of Section 4.8 that "
              "cooling cannot raise the tunneling probability."
              % (sci(TV100["sqrt_c_needed"], 1), sci(TV100["gamma_over_H0"], 0), sci(TV["sqrt_c_for_equilibrium_today"], 0), sci(TV["a_T_if_equilibrated"]["100.0"], 0))))
    add(("fig", "fig14_release_scan.png", "Fig. 14. (a) The plane of matter density and temperature: the symmetric point is held above the line; trajectories of the matter-only, "
                "temperature-only and combined triggers (dots: release). (b) Scale factor at turnaround against $\\mu$ for the coupled model at five $a_c$ and for the uncoupled "
                "control, which rolls before today at $\\mu=300H_0$. (c) Scale factor of the release against the initial seed for the coupled model and the control.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.5 tunneling branch
    add(("h2", "6.5 The tunneling branch: dilution can open the survival window"))
    add(("p", "When the coupling is centred on the false vacuum of Section 2, $A=1+(\\varphi-\\varphi_{f})^2/2M^2$, matter deepens the false minimum and raises the barrier, and in the "
              "dimensionless family the potential becomes $p(x)+\\theta(x-1)^2/2$ with $\\theta=\\rho_*/(\\lambda f^2M^2)\\propto a^{-3}$ (density_bounce.py). Above a degeneracy "
              "density $\\theta_{\\rm deg}(\\eta)$ the matter-held state is the true vacuum; below it, it is metastable, and the bounce action falls from infinity toward its vacuum "
              "value as matter dilutes (Fig. 15a; at $\\theta=0$ the table of Section 3 is reproduced to $%s$). Dilution can therefore only switch on a vacuum instability that the "
              "potential already contains, never exceed it, but it changes which vacua are allowed. The survival integral of Section 4.2 is evaluated with the time-dependent rate: a vacuum "
              "with $B_0=150$, excluded without the coupling by a factor $e^{%.0f}$, survives with it ($\\ln I=%.0f$ for $\\theta_0=0.3\\theta_{\\rm deg}$, $\\eta=0.2$) and reaches one "
              "nucleation per Hubble four-volume at $a_*=%.2f$; for $B_0=287$ the trigger point moves to $a_*=%.1f$--$%.1f$, and for $B_0=400$ the rate never reaches $H^4$ "
              "(Fig. 15b, c). The coupling turns the window below $B_c$, which survival excludes for a matter-independent rate, into a future trigger. Two limits are stated rather "
              "than extrapolated through: the homogeneous mean density applies to a bubble larger than the scale of matter inhomogeneity, and near the classical threshold the barrier "
              "vanishes and the instanton approximation fails (for $\\eta>\\eta_c$ the matter-held minimum exists only for $%.3f<\\theta<%.3f$ at $\\eta=0.40$ and the release is "
              "classical)."
              % (sci(max(abs(v["b"] / v["table"] - 1) for v in DB["check_theta0"].values()) + 1e-16, 0), h_(0.2, 150.0, 0.1)["lnI_without_coupling"], h_(0.2, 150.0, 0.3)["lnI_past"],
                 h_(0.2, 150.0, 0.3)["a_star"], min(h["a_star"] for h in HIST if h["B0"] == 287.0 and h["a_star"]), max(h["a_star"] for h in HIST if h["B0"] == 287.0 and h["a_star"]),
                 DB["classical"]["0.40"]["theta_low"], DB["classical"]["0.40"]["theta_high"])))
    add(("fig", "fig15_tunneling_branch.png", "Fig. 15. Tunneling branch. (a) Bounce action against the matter density, relative to its vacuum value, for three tilts. (b) Expected number of "
                "nucleations in the past light cone with the coupling, against the density today, for vacuum actions $B_0=150$--$400$ (dotted: without the coupling). (c) Scale "
                "factor at which one nucleation per Hubble four-volume is reached, for the surviving cases.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.6 cycles
    add(("h2", "6.6 From recollapse to a cyclic hot big bang"))
    c4, c5, c6, cu = cold(1e4), cold(1e5), cold(1e6), cold(1e4, ups=0.1)
    gmed = lambda r: gm([c.get("G") for c in r["cycles"][1:]])
    add(("p", "A recollapse becomes a cycle only if the contraction reverses, which needs physics beyond general relativity; the effective loop-quantum-cosmology equations for a "
              "flat universe, for which they are derived \\cite{APS06,AS11},"))
    add(("eq", r"H^2=\frac{\rho}{3M_{\rm Pl}^2}\left(1-\frac{\rho}{\rho_c}\right),\qquad \dot H=-\frac{\rho+p}{2M_{\rm Pl}^2}\left(1-\frac{2\rho}{\rho_c}\right)"))
    add(("p", "are used as the bounce, with no curvature term appended (cyclic_extension.py, hot_cycles.py). The physical $\\rho_c\\sim\\rho_{\\rm Pl}$ is $10^{122}V_c$ and "
              "cannot be integrated; $\\rho_c$ is varied and the dependence on it reported. A bounce alone is not enough. Without dissipation the contraction is dominated by the "
              "scalar's kinetic energy, which grows as $a^{-6}$, faster than radiation, and the bounce occurs at low density and is cold: at $\\rho_c=10^4V_c$ the cycles repeat with "
              "$G_n\\simeq%.2f$ and each has a vacuum-dominated phase, but no hot dense epoch, and they grow ($G_n\\simeq%.2f$ and $%.2f$) at $\\rho_c=10^5$ and $10^6V_c$, the "
              "hysteresis of Section 5.6. A constant dissipation $\\Upsilon$ heats the bounce but fills the next cycle with radiation, and the vacuum fraction falls cycle by cycle "
              "(from %.2f to %.2f over six cycles at $\\Upsilon=0.1H_0$): the energy released by the vacuum, once thermalized, swamps the matter that would hold the next vacuum, "
              "by the factor $\\Delta V/\\rho_{\\rm crit}=(A(v)-1)/2$ of Section 6.2."
              % (gmed(c4), gmed(c5), gmed(c6), cu["cycles"][1].get("max_vacuum_fraction") or 0, cu["cycles"][-1].get("max_vacuum_fraction") or 0)))
    add(("p", "Two pieces of standard early-universe physics remove both failures. The first is thermal dissipation, $\\Upsilon=C_TT^3$, the form used for dissipative scalar "
              "dynamics in warm inflation \\cite{Berera95}: it is negligible in the cold expanding universe and dominant in the hot contraction, where it turns the scalar's energy into "
              "radiation, so the bounce is radiation dominated and hot. The second is that the dark-matter abundance is not an initial condition but is set by thermal freeze-out as a "
              "fixed ratio to the entropy \\cite{KT90}. A bounce hotter than the dark-matter mass returns the dark matter to equilibrium, and as the bath cools after the bounce "
              "freeze-out restores the same ratio. The same coupling then restores the vacuum: while the dark matter is relativistic its field-dependent mass gives the scalar a thermal "
              "mass $m^2T^2/12M^2$, which exceeds $H$ once $T\\lesssim mM_{\\rm Pl}/6M$, long before freeze-out at $T\\sim m/20$, and damps the field toward $\\varphi=0$ "
              "(high-temperature symmetry restoration). The resulting oscillation is $10^3$--$10^9$ times faster than the expansion and is applied through its adiabatic result; the matter "
              "era that follows is integrated explicitly. In the equations, the comoving matter is reset at freeze-out ($T_{\\rm fo}=5T_{\\rm eq}$) to $N_{\\rm dm}=\\kappa_{\\rm dm}S$, "
              "$N_b=\\kappa_bS$, with $S=a^3\\rho_r^{3/4}$ and $\\kappa$ fixed by the present universe, the energy taken from the bath; the bounce must be hotter than $T_{\\rm fo}$, "
              "which needs $\\rho_c\\gtrsim10^{13}V_c$ and is satisfied with a margin of 30 orders of magnitude by the physical $\\rho_c$."))
    add(("p", "With these, the cycles are self-similar (Fig. 16, Table 17). In the reference run ($a_c=2$, $\\mu=30H_0$, $q=1$, $\\rho_c=10^{14}V_c$, $C_T=1$) every cycle after ours "
              "has a radiation-dominated bounce (radiation fraction %.4f at the bounce), a matter era (matter fraction %.3f at its peak), a dark-energy epoch with the field held "
              "(vacuum fraction %.4f, %.2f e-folds of accelerated expansion, against about half an e-fold in our universe so far), a release at the same temperature in every cycle "
              "($T_{\\rm rel}=%.4f$ in units where $T_0=%.4f$ today) and a turnaround. From the second cycle on, these values agree from cycle to cycle to five digits or better. The second law is satisfied, not evaded: "
              "the radiation entropy grows by a factor $%s$ per cycle, and because the matter-to-entropy ratio and therefore the whole thermal history as a function of temperature are "
              "reset, the entropy does not accumulate in the local state but enlarges the next cycle, $G_n=a_{\\max,n+1}/a_{\\max,n}=%.2f=(S_{n+1}/S_n)^{1/3}=%.2f$ (Tolman's growing "
              "cycles \\cite{Tolman} with a fixed local history). The constraint of the effective equations is satisfied to $%s$, enforced by projection whenever the scale factor doubles "
              "(the largest correction %s), because a violation would otherwise act as a spurious cosmological constant."
              % (BC[1]["bounce_radiation_fraction"], BC[1]["max_matter_fraction"], BC[1]["max_vacuum_fraction"], BC[1]["held_accelerated_efolds"], BC[1]["T_release"],
                 MC.RHO_R0 ** 0.25, sci(gm(GS) ** 3, 2), gm(GN), gm(GS), sci(BASE["max_residual"], 0), sci(BASE["projections"]["max_rel"], 0))))
    rowsH = []
    for r in CYC:
        p = r["params"]
        cs = [c for c in r["cycles"] if c["n"] >= 1]
        if not cs:
            continue
        lab = ("reference" if r is BASE else "no freeze-out reset" if not p["reset"] else "no dissipation ($C_T=0$)" if p["c_T"] == 0 else
               ", ".join(s for s in ["$C_T=%g$" % p["c_T"] if p["c_T"] != 1 else "", "$\\rho_c=10^{%d}V_c$" % round(np.log10(p["rho_c_over_Vc"])) if p["rho_c_over_Vc"] != 1e14 else "",
                                     "$\\mu=%g$" % p["mu"] if p["mu"] != 30 else "", "$q=%g$" % p["q"] if p["q"] != 1 else "", "$a_c=%g$" % p["a_c"] if p["a_c"] != 2 else ""] if s))
        c1 = cs[min(1, len(cs) - 1)]
        rowsH.append([lab, "%.3f" % (c1.get("bounce_radiation_fraction") or 0), "%.3f" % c1.get("max_matter_fraction", 0), "%.2f" % c1.get("held_accelerated_efolds", 0),
                      ("%.4f" % c1["T_release"]) if c1.get("T_release") else "none",
                      ("%.1f" % gm([c.get("G") for c in cs])) if (r["status"] == "completed" and p["reset"] and p["c_T"] > 0) else ("runaway" if r["status"] == "runaway" else "--"),
                      ("%.1f" % gm([c.get("G_from_entropy") for c in cs])) if (r["status"] == "completed" and p["reset"] and p["c_T"] > 0) else "--"])
    add(("table", "Table 17. Hot cycles: values of the second cycle (identical in later cycles for the completed runs). Reference: $a_c=2$, $\\mu=30H_0$, $q=1$, "
                  "$\\rho_c=10^{14}V_c$, $C_T=1$; other rows change one parameter. The two controls fail: without the freeze-out reset the matter-to-entropy ratio falls by the entropy "
                  "growth every cycle and no matter era or dark-energy epoch remains; without dissipation the bounce is cold and kinetic dominated and the cycles run away.",
         ["Run", "radiation at bounce", "peak matter fraction", "held e-folds", "$T_{\\rm rel}$", "$G_n$", "$(S_{n+1}/S_n)^{1/3}$"], rowsH, None))
    add(("fig", "fig16_hot_cycles.png", "Fig. 16. Hot cycles (reference run). (a) Fractions of radiation, matter and vacuum energy against the radiation temperature in three successive "
                "cycles, up to the release; the curves coincide. (b) Turnaround and bounce scale factors and the radiation entropy per cycle: the growth factor equals "
                "$(S_{n+1}/S_n)^{1/3}$. (c) E-folds of accelerated expansion with the field held, per cycle, for the full model and the two controls.", 6.3))
    # ------------------------------------------------------------------------------------------------ 6.7 conditions, predictions and limits
    add(("h2", "6.7 What the cycle requires and how it can fail"))
    add(("p", "The calculation identifies four ingredients, each necessary: a vacuum held by matter and released by dilution (which gives the dark-energy epoch and the turnaround), a "
              "nonsingular bounce, dissipation that thermalizes the scalar energy in the hot contraction, and re-thermalization with freeze-out that resets the matter-to-entropy "
              "ratio and, through the same coupling, the vacuum. Removing the third or the fourth destroys the cycle (Table 17). Here the last two are prescribed (a reset at freeze-out and the adiabatic result of the restoration); Section 6.8 obtains them from the equations and finds an additional condition on the bounce temperature. The result does not depend on the details within the "
              "ranges tested: $C_T=0.1$--$10$, $\\rho_c=10^{14}$--$10^{16}V_c$ (with $G_n=%.1f$, $%.1f$ and $%.1f$, converging), $\\mu=30$ and $100H_0$, $q=0.3$ and 1, $a_c=2$ and 3."
              % (gm([c.get("G") for c in by(rho_c_over_Vc=1e14, c_T=1.0, mu=30.0, q=1.0, a_c=2.0, reset=True)["cycles"][1:]]),
                 gm([c.get("G") for c in by(rho_c_over_Vc=1e15)["cycles"][1:]]), gm([c.get("G") for c in by(rho_c_over_Vc=1e16)["cycles"][1:]]))))
    add(("p", "The model makes statements that can be tested now. The dark energy is exactly a cosmological constant until the release ($w=-1$), so a confirmed evolution of the dark "
              "energy at the level suggested by recent baryon-acoustic-oscillation analyses \\cite{DESI} would disfavor it. The coupling vanishes at $\\varphi=0$, so there is no fifth force "
              "in the cosmological background or in collapsed structures. The weakest point is the deepest voids: a void with density below $\\rho_{\\rm crit}=\\rho_{\\rm dm,0}/a_c^3$ "
              "($1/8$ of the mean for $a_c=2$) and radius larger than the Compton length $\\pi/\\mu$ ($\\sim140$ Mpc for $\\mu=100H_0$) would release the vacuum locally today and "
              "form domain walls; for $\\mu\\gtrsim300H_0$ and $a_c\\lesssim2$ the largest, emptiest voids reach that condition, which bounds the parameter space from void "
              "statistics. In the future, the dark-matter mass grows by $A(v)$ after the release and the recollapse follows within tens of Gyr (Table 16)."))
    add(("p", "The limits are equally definite. The bounce is an effective prescription, not a derivation; $\\rho_c$ is $10^{108}$ times below the Planck value in the simulations; the "
              "hot-phase restoration is applied through its adiabatic result and freeze-out is instantaneous (both removed in Section 6.8); the dynamics are homogeneous, so the growth of inhomogeneities in "
              "the contraction, which can dominate a contracting universe, is not followed (Bianchi I shear is treated in Section 6.10); and $a_c\\sim1$ requires $\\mu^2M^2\\approx\\rho_{\\rm dm,0}$, a coincidence "
              "of the same kind as that of the cosmological constant. The calculation shows that a cyclic hot big bang is consistent within this model with the stated ingredients, and "
              "which ingredients it cannot do without; it does not show that the universe is cyclic."))
    return B
