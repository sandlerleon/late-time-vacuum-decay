# -*- coding: utf-8 -*-
"""Manuscript blocks for the finite-temperature analysis (Sections 4.6-4.8 of the paper); numbers are read from thermal_results.json / thermal_tables.json."""
import json
import os

import numpy as np

import constants as K
import cdl_lib as C
import rates as R
import thermal as TH

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
TR, TT, SV, MC = J("thermal_results.json"), J("thermal_tables.json"), J("survival.json"), J("monte_carlo.json")


def texsci(x, d=1):
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10 ** e, d)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return "%.*f\\times10^{%d}" % (d, m, e)


sci = lambda x, d=1: "⟦" + texsci(x, d) + "⟧"
pc = lambda x, d=1: ("%." + str(d) + "f") % (100 * x)

BMK = TR["benchmark"]
F_MEV, T0, THC = BMK["f_meV"], BMK["T_cmb_meV"], BMK["theta_c"]
TX = TR["crossover_T_meV"]["0"]
A_X = T0 / TX
PM = TR["prior_maps"]
BS = TR["benchmark_survival"]
C_MAX_1100 = THC * F_MEV ** 2 / (T0 * 1101.0) ** 2
LAM, ETA = 1.0, 0.30
B4_0 = float(R.b_of_eta(ETA) / LAM)
S3_0 = F_MEV * float(TH.s3_of_eta(ETA))          # S_3 in meV at theta = 0


def row_z(z):
    T = T0 * (1.0 + z)
    out = [("%g" % z), "%.3f" % T]
    B4, S3T, th, ex = TH.exponents(LAM, ETA, 0.0, T)
    out += ["%.0f" % B4, "%.0f" % S3T]
    out.append("%.0f" % float(R.ln_gamma_over_h4(LAM, ETA)[0]))
    for c in (0.0, 0.1, -1.0):
        lr, ch = TH.ln_rate_over_hl4(LAM, ETA, c, T)
        out.append("%.0f (O(%d))" % (float(lr), int(ch)) if np.isfinite(lr) else "no false minimum")
    return out


def blocks():
    B = []
    add = B.append
    # --------------------------------------------------------------------------------------------------------- 4.6
    add(("h2", "4.6 Cooling and the thermally corrected potential"))
    add(("p", "The calculation above treats the universe as a clock: the CMB temperature $T_\\gamma=T_0/a$ falls with the expansion and enters nothing. It can enter the "
              "decay only if the scalar interacts with a thermal bath, and then through a thermal correction to the potential whose coefficient $c$ is fixed by those "
              "interactions and is not a universal constant. Two regimes are therefore kept apart throughout: the scalar is decoupled and the decay is the zero-temperature "
              "O(4) channel of the previous sections, or it is in thermal equilibrium with the CMB, in which case the leading high-temperature term is"))
    add(("eq", r"V_{\rm eff}(\varphi,T)=V_0(\varphi)+\tfrac12\,c\,T^2\varphi^2,\qquad \theta\equiv\frac{c\,T^2}{\lambda f^2}"))
    add(("p", "The thermal term is a rescaling of the zero-temperature family, which makes the analysis exact. With $x=sy$ and $s=\\sqrt{1-2\\theta}$ the potential $(x^2-1)^2/8+\\eta x/2+\\theta x^2/2$ becomes "
              "$s^4[(y^2-1)^2/8+(\\eta/s^3)y/2]$ up to a constant. Hence a false minimum exists only for"))
    add(("eq", r"\eta_{\rm eff}\equiv\frac{\eta}{(1-2\theta)^{3/2}}<\eta_c\quad\Longleftrightarrow\quad\theta<\theta_c(\eta)=\frac12\left[1-\left(\frac{\eta}{\eta_c}\right)^{2/3}\right]"))
    add(("p", "and the bounce actions are those of the zero-temperature table at the shifted tilt: $B_4=b(\\eta_{\\rm eff})/\\lambda$ and $S_3=(fs/\\sqrt\\lambda)\\,s_3(\\eta_{\\rm eff})$, where $s_3=4\\pi\\int\\rho^2[x'^2/2+w-w_f]d\\rho$ "
              "is the O(3) action at $\\lambda=1$, $\\theta=0$ (tabulated here for the first time for this family, with the boundary-value solver for $\\eta<0.1$, and tending to the thin-wall value $16\\pi\\sigma^3/3\\Delta V^2$ as $\\eta\\to0$: the ratio is %.3f at $\\eta=0.03$). "
              "Direct solutions of the O(4) and O(3) bounce with the thermal term in the potential agree with these scalings to %s (check V19). For the benchmark $\\lambda=1$, $\\eta=0.30$: $f=%.2f$ meV, "
              "$\\theta_c=%.4f$, and today ($T_0=%.3f$ meV) $\\theta=%.4f\\,c$."
              % (TT["s3_thin_wall"][0]["ratio"], sci(max(max(r["rel4"], r["rel3"]) for r in TT["reduction_check"]), 0), F_MEV, THC, T0, BMK["theta_today_per_c"])))
    zc = TR["z_c"]
    add(("p", "Cooling relaxes the thermal term, and the thermal history has a consequence that does not depend on the decay rate: the false minimum disappears at $T_c=f\\sqrt{\\lambda\\theta_c/c}$, which is %.1f meV (redshift %.0f) "
              "for $c=0.1$ and %.2f meV (redshift %.1f) for $c=1$. A scalar in equilibrium with the CMB at higher temperatures would have settled into the true vacuum at the symmetric-to-broken transition, and the "
              "present dark energy would not be the false-vacuum energy. The observed universe in the false vacuum therefore requires that the scalar either decoupled before $T_c$ or has $c<%s$ if the bath is coupled all the way back to "
              "recombination ($z=1100$). The high-temperature form is an upper bound on the effect for particles heavier than $T$ in the loop, since the true correction is then Boltzmann suppressed; the results below are therefore "
              "the largest thermal effect that the coefficient $c$ can produce." % (TR["T_c_meV"]["0.1"], zc["0.1"], TR["T_c_meV"]["1"], zc["1"], sci(C_MAX_1100, 1))))
    add(("fig", "fig6_thermal_exponents.png", "Fig. 6. (a) Extrema of the thermally corrected potential for $\\eta=0.30$ against the thermal strength $\\theta$; the false minimum disappears at $\\theta_c$. (b) The quantum exponent $B_4$ and the thermal exponent $S_3/T$ against the bath temperature for the benchmark; the O(3) saddle dominates above the crossover temperature $T_\\times$ (where $S_3/T=B_4$) and the O(4) saddle below. Solid: $c=0$ (equilibrium without thermal mass); dashed: $c=-1$; dotted: $c=+0.1$.", 6.3))
    # --------------------------------------------------------------------------------------------------------- 4.7
    add(("h2", "4.7 The cooling history and the three decay channels"))
    add(("p", "The temperature follows the observed history, $T_\\gamma(a)=T_0/a$ with $H(a)$ of Section 4.2, and cosmic time $t(a)-t_0=\\int_1^a da'/[a'H(a')]$. Three channels compete. "
              "The Hawking--Moss transition is the horizon-scale channel of de Sitter space; its action is $\\ln B_{\\rm HM}\\simeq279$ (Section 3.4) and it plays no role. The de Sitter horizon temperature $T_{\\rm dS}=H_\\Lambda/2\\pi=%s$ meV is "
              "$%s$ of $T_0$, would equal the CMB temperature only after %.0f e-folds ($\\sim10^{12}$ yr), and is not a homogeneous radiation bath; it is therefore kept separate and not combined with $T_\\gamma$ by a maximum. "
              "The thermal exponent at the horizon temperature, $S_3/T_{\\rm dS}=%s$, rules out any activation by it. That leaves quantum tunneling (the O(4) bounce with action $B_4$) and thermal activation (the O(3) bounce with $S_3/T$). "
              "At a given temperature the saddle with the smaller exponent dominates, and the rate takes its exponent and prefactor; the two rates are not added, because they are the two ends of one family of periodic instantons. "
              "The thermal prefactor is $\\Gamma_T=D_T\\,T^4(S_3/2\\pi T)^{3/2}e^{-S_3/T}$ with $D_T$ of order unity (a systematic of 1.5 decades, as for the quantum prefactor)."
              % (sci(TR["T_dS_meV"], 2), sci(TR["T_dS_over_T_cmb"], 1), TR["e_folds_until_T_cmb_equals_T_dS"], sci(TR["S3_over_T_dS"], 1))))
    add(("p", "For the benchmark the crossover is at $T_\\times=%.3f$ meV, so that today the thermal saddle already dominates ($S_3/T_0=%.0f<B_4=%.0f$), the crossover being reached when the CMB has cooled by a factor %.2f, "
              "and in the past the thermal exponent was far smaller (Table 6, Fig. 6b). Cooling lowers the rate for $c\\ge0$ at every epoch: the thermal exponent grows as $1/T$ and the thermal mass relaxes, so $\\ln\\Gamma$ falls monotonically with time "
              "(check V20). The late-time decay is therefore the zero-temperature O(4) decay of Section 4, because the thermal contribution switches off exponentially fast as $T\\propto e^{-H_\\Lambda t}$. For $c<0$ (symmetry non-restoration, outside the usual sign of "
              "the thermal mass) the barrier grows with $T$ and cooling lowers $B_4$, so the sign of the effect is reversed for the quantum channel, but the thermal exponent still has a minimum near $z\\simeq2$ for $c=-1$ and the past rate is not small."
              % (TX, S3_0 / T0, B4_0, A_X)))
    rows = [row_z(z) for z in (0.0, 0.5, 1.0, 2.0, 3.0)]
    add(("table", "Table 6. Exponents and rates for the benchmark ($\\lambda=1$, $\\eta=0.30$) as the universe cools. $B_4$ and $S_3/T$ at $c=0$; $\\ln\\Gamma/H_\\Lambda^4$ for the decoupled scalar (O(4) only) and for the equilibrated scalar with the dominant saddle in brackets.",
         ["$z$", "$T$ (meV)", "$B_4$", "$S_3/T$", "decoupled", "$c=0$", "$c=0.1$", "$c=-1$"], rows, None))
    # --------------------------------------------------------------------------------------------------------- 4.8
    add(("h2", "4.8 Survival with a time-dependent rate and the thermal prior maps"))
    add(("p", "With a rate that depends on the temperature, the expected number of nucleations in the past light cone is"))
    add(("eq", r"I=\frac{4\pi}{3}\int_{t_i}^{t_0}dt\,\Gamma(T_\gamma(t))\,a(t)^3\chi(t)^3,\qquad P_{\rm surv}=e^{-I}"))
    add(("p", "with $\\chi$ as in Eq. (9). It assumes a flat universe and a homogeneous rate per physical four-volume, and walls at the speed of light. For a constant rate it reduces to Eq. (9), reproduced by the quadrature to %s. "
              "For the benchmark, $\\ln I=%.0f$ for the decoupled scalar (the vacuum survives), while an equilibrated scalar gives $\\ln I=%.0f$ ($c=0$, bath coupled for $z\\le1$), $%.0f$ ($z\\le3$) and $%.0f$ ($z\\le10$): the benchmark is excluded by many orders "
              "of magnitude as soon as it is in equilibrium with the CMB. The same calculation over the priors of Section 4.5 ($2\\times10^5$ draws, the same seed, plus a thermal prefactor spread of 1.5 decades) gives Table 7 and Fig. 7."
              % (sci(abs(TR["light_cone_check"]["V4_HL4_grid"] / TR["light_cone_check"]["V4_HL4_survival_json"] - 1), 0), BS["decoupled"]["ln_I"], BS["c=0, z<=1"]["ln_I"], BS["c=0, z<=3"]["ln_I"], BS["c=0, z<=10"]["ln_I"])))
    labs = [("decoupled", "decoupled"), ("c=0 (equilibrated), z<=1", "$c=0$, $z\\le1$"), ("c=0, z<=3", "$c=0$, $z\\le3$"), ("c=0, z<=10", "$c=0$, $z\\le10$"), ("c=0.1, z<=1", "$c=0.1$, $z\\le1$"), ("c=0.1, z<=3", "$c=0.1$, $z\\le3$"),
            ("c=1, z<=1", "$c=1$, $z\\le1$"), ("c=-0.1, z<=1100", "$c=-0.1$, $z\\le1100$"), ("c=-1, z<=1100", "$c=-1$, $z\\le1100$")]
    rows = []
    for k, lab in labs:
        v = PM[k]
        rows.append([lab, pc(v["excluded"], 1), pc(v["inconsistent"], 1), pc(v["allowed"], 1), pc(v["decay_by_1e20"], 2), pc(v["no_decay_1e100"], 1)])
    add(("table", "Table 7. Prior mass (%) by thermal scenario ($2\\times10^5$ draws; $z$ is the redshift below which the scalar is in equilibrium with the CMB). “Inconsistent” draws have no false minimum at some redshift of the equilibrium epoch. Decay fractions refer to the allowed draws and to the zero-temperature decay of the future.",
         ["Scenario", "above survival bound", "no false minimum in the past", "allowed", "decays by $10^{20}$ yr", "no decay in $10^{100}$ yr"], rows, None))
    add(("fig", "fig7_thermal_survival.png", "Fig. 7. (a) Rate history of the benchmark: the decoupled O(4) rate is constant, the equilibrated rate falls as the universe cools; the dashed line is the survival bound on $\\Gamma/H_\\Lambda^4$. (b) Prior mass by thermal scenario.", 6.3))
    dec = PM["decoupled"]
    add(("p", "The results answer the question posed by cooling. First, a scalar that is not coupled to the CMB is unaffected; the decay of the previous sections stands. Second, a scalar in equilibrium with the CMB is constrained more strongly, not less: "
              "the prior mass above the survival bound rises from %s%% to %s%% ($z\\le1$), %s%% ($z\\le3$) and %s%% ($z\\le10$) for $c=0$, and a further %s%% to %s%% of the draws have no false minimum in the equilibrium epoch for $c=0.1$ and 1. "
              "Third, in no equilibrated scenario with $c\\ge0$ does a survival-consistent draw decay within $10^{20}$ yr: the decay window of Section 4.4 closes completely. The reason is that the allowed draws must have a thermal exponent large enough at $z\\sim1$, "
              "and because $S_3/T<B_4$ above $T_\\times$ this requires $B_4$ well above the critical action, so that the zero-temperature decay time of the future exceeds $10^{100}$ yr. Fourth, only the non-standard sign $c=-1$ leaves a window (%s%% of the prior against %s%% for the decoupled scalar). "
              "In short, cooling cannot raise the probability of a late-time transition: it lowers the thermal channel monotonically, the late-time decay is the zero-temperature O(4) decay, and the thermal coupling only adds past constraints."
              % (pc(dec["excluded"], 1), pc(PM["c=0 (equilibrated), z<=1"]["excluded"], 0), pc(PM["c=0, z<=3"]["excluded"], 0), pc(PM["c=0, z<=10"]["excluded"], 0), pc(PM["c=0.1, z<=1"]["inconsistent"], 1),
                 pc(PM["c=1, z<=1"]["inconsistent"], 0), pc(PM["c=-1, z<=1100"]["decay_by_1e20"], 2), pc(dec["decay_by_1e20"], 2))))
    return B, dict(c_max_1100=C_MAX_1100, T_x=TX, a_x=A_X)
