# -*- coding: utf-8 -*-
"""Survival of the present vacuum: the four-volume of our PAST LIGHT CONE in the observed expansion history.

A bubble nucleated at an event p expands at (almost) the speed of light, so it engulfs an observer at O if and only if p lies in the
causal past of O. The probability that no nucleation occurred in the past light cone of the present observer is therefore

        P_surv = exp( - Gamma * V4 ),        V4 = Integral_0^{t0} dt a(t)^3 (4 pi / 3) chi(t)^3,    chi(t) = Integral_t^{t0} dt' / a(t'),

with the comoving distance chi(t) of the light cone at cosmic time t (the observer sits at the comoving origin, flat FLRW). This is the
Lorentzian four-volume used for the electroweak vacuum in the Standard Model (Isidori, Ridolfi, Strumia 2001). The Euclidean four-sphere
volume 8 pi^2 / (3 H^4) of the first draft is not that quantity.

Expansion history (flat, Planck 2018): H(a)^2 = H0^2 (Om a^-3 + Or a^-4 + OL),  Om = 1 - OL - Or, Or from T_CMB and N_eff = 3.046.
The rate Gamma is taken constant in time (the scalar sits in its false minimum throughout; the early universe contributes
negligibly because V4 converges at small a).

    python survival.py          (writes survival.json)
"""
import json
import os

import numpy as np
from scipy.integrate import quad, solve_ivp

import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
NEFF = 3.046
H_REDUCED = K.H0_KMS_MPC / 100.0
# photons + massless neutrinos: Omega_r h^2 = 2.4728e-5 (T_cmb / 2.7255)^4 (1 + 0.22711 N_eff)
OMEGA_R = 2.4728e-5 * (K.T_CMB_K / 2.7255) ** 4 * (1.0 + 0.22711 * NEFF) / H_REDUCED ** 2
OMEGA_M = 1.0 - K.OMEGA_L - OMEGA_R


def e_of_a(a, om=OMEGA_M, ol=K.OMEGA_L, orad=OMEGA_R):
    """E(a) = H / H0."""
    return np.sqrt(om / a ** 3 + orad / a ** 4 + ol)


def v4_past_light_cone(om=OMEGA_M, ol=K.OMEGA_L, orad=OMEGA_R, a_min=1e-12, method="grid", n=40001):
    """V4 in units of H0^-4 (c = 1), computed in two independent ways. With s = -ln a measured back from the present:
        d chi / ds = 1 / (a E),        d V4 / ds = (4 pi / 3) a^3 chi^3 / E        (chi = 0 = V4 at s = 0).

    'grid' : cumulative Simpson rule for chi on a uniform grid in s, then the trapezoid rule for V4.
    'ode'  : the coupled system integrated with solve_ivp (DOP853, rtol 1e-12)."""
    s_max = -np.log(a_min)
    if method == "ode":
        def rhs(s, y):
            a = np.exp(-s)
            E = e_of_a(a, om, ol, orad)
            return [1.0 / (a * E), (4.0 * np.pi / 3.0) * a ** 3 * y[0] ** 3 / E]
        sol = solve_ivp(rhs, [0.0, s_max], [0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-30)
        return float(sol.y[1, -1])
    from scipy.integrate import cumulative_simpson
    s = np.linspace(0.0, s_max, n)
    a = np.exp(-s)
    E = e_of_a(a, om, ol, orad)
    chi = cumulative_simpson(1.0 / (a * E), x=s, initial=0.0)
    integrand = (4.0 * np.pi / 3.0) * a ** 3 * chi ** 3 / E
    return float(np.trapezoid(integrand, s))


def v4_einstein_de_sitter():
    """Analytic check: matter only, V4 = (4 pi / 3) 81 B(9, 4) t0^4 with t0 = 2 / (3 H0)."""
    from scipy.special import beta
    t0 = 2.0 / 3.0
    return float((4.0 * np.pi / 3.0) * 81.0 * beta(9, 4) * t0 ** 4)


def v4_de_sitter_window():
    """Asymptotic de Sitter growth rate of the past-cone four-volume per Hubble time: dV4/dt = (4 pi / 3) H^-3  ->  V4 grows by 4 pi / 3 per H^-1."""
    return 4.0 * np.pi / 3.0


if __name__ == "__main__":
    out = {}
    v4 = v4_past_light_cone()
    v4_ode = v4_past_light_cone(method="ode")
    # normalise with the asymptotic de Sitter rate of the dark energy, H_L^2 = OL H0^2
    to_hl4 = K.OMEGA_L ** 2
    v4_hl = v4 * to_hl4
    euclid = 8.0 * np.pi ** 2 / 3.0
    out["omega_r"], out["omega_m"] = OMEGA_R, OMEGA_M
    out["V4_over_H0m4"] = v4
    out["V4_over_H0m4_ode"] = v4_ode
    out["V4_HL4"] = v4_hl
    out["euclid_HL4"] = euclid
    out["ratio_euclid_to_lorentzian"] = euclid / v4_hl
    # checks against an analytic limit and convergence in the lower cutoff and grid
    out["check_EdS"] = dict(numeric=v4_past_light_cone(om=1.0, ol=0.0, orad=0.0), analytic=v4_einstein_de_sitter())
    out["convergence"] = dict(
        a_min=[dict(a_min=a, V4=v4_past_light_cone(a_min=a)) for a in (1e-6, 1e-9, 1e-12, 1e-15)],
        grid=[dict(n=n, V4=v4_past_light_cone(n=n)) for n in (2001, 10001, 40001, 160001)])
    # variation with the cosmological parameters (Planck 1 sigma: Omega_Lambda 0.6889 +- 0.0056; h = 0.6766 +- 0.0042, H0 factored out)
    var = []
    for dol in (-0.0056, 0.0, 0.0056):
        ol = K.OMEGA_L + dol
        var.append(dict(omega_lambda=ol, V4_HL4=v4_past_light_cone(om=1 - ol - OMEGA_R, ol=ol) * ol ** 2))
    out["parameter_variation"] = var
    # criterion: the expected number of nucleations in the past light cone is N = Gamma V4; excluded at 95% confidence if exp(-N) < 0.05
    out["N_95"] = float(-np.log(0.05))
    out["gamma_over_hl4_max_95"] = float(out["N_95"] / v4_hl)
    out["gamma_over_hl4_max_euclid_old"] = 3.0 / (8.0 * np.pi ** 2)
    out["t_dec_min_yr"] = float(K.T_HUBBLE_YR / ((4.0 * np.pi / 3.0) * out["gamma_over_hl4_max_95"]))
    # cumulative four-volume contributed by events more recent than redshift z (for the figure)
    from scipy.integrate import cumulative_simpson
    sg = np.linspace(0.0, -np.log(1e-4), 20001)
    ag = np.exp(-sg)
    Eg = e_of_a(ag)
    chig = cumulative_simpson(1.0 / (ag * Eg), x=sg, initial=0.0)
    cum = cumulative_simpson((4.0 * np.pi / 3.0) * ag ** 3 * chig ** 3 / Eg, x=sg, initial=0.0) * to_hl4
    pick = np.unique(np.round(np.logspace(0, np.log10(len(sg) - 1), 300)).astype(int))
    out["cumulative"] = dict(z=(1.0 / ag[pick] - 1.0).tolist(), V4_HL4=cum[pick].tolist(), chi_over_hubble_radius=(chig[pick] * 1.0).tolist())
    out["fraction_of_V4_below_z"] = {str(zz): float(cum[int(np.searchsorted(sg, np.log(1.0 + zz)))] / cum[-1]) for zz in (0.5, 1.0, 2.0, 5.0)}
    # age at the present (for the record) and the comoving particle horizon
    json.dump(out, open(os.path.join(HERE, "survival.json"), "w"), indent=1)
    print("Omega_r = %.3e  Omega_m = %.4f" % (OMEGA_R, OMEGA_M))
    print("V4 = %.5f H0^-4 (conformal), %.5f (ODE) ; = %.5f H_L^-4 ; Euclidean %.3f H_L^-4 ; ratio %.1f" % (v4, v4_ode, v4_hl, euclid, euclid / v4_hl))
    print("EdS check", out["check_EdS"])
    print("convergence", out["convergence"])
    print("param variation", var)
    print("Gamma/H^4 max (95%%) = %.3f (old Euclidean %.3f) ; t_dec min = %.3e yr" % (out["gamma_over_hl4_max_95"], out["gamma_over_hl4_max_euclid_old"], out["t_dec_min_yr"]))
