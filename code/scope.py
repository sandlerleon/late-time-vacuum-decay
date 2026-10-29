# -*- coding: utf-8 -*-
"""Scope of the reheating bound: which energy sources are excluded by this model and which only by energy conservation.

Energy conservation alone gives T_reh <= 2.2 meV when the released energy density is at most the dark-energy density (V_t >= 0). It does not
exclude a deeper (anti-de Sitter) true vacuum, an extra energy reservoir, or a different potential scale. What does is the geometry of the
interior. With the open slicing, a Friedmann constraint  (da/dt)^2 = 1 + a^2 rho / (3 M^2)  and the curvature parameter

        Y(t) = a^2 rho / (3 M^2)          (Y << 1: curvature dominated, a = t;  Y >~ 1: a radiation- or matter-dominated, flat-like cosmology),

a hot big bang at nucleosynthesis needs Y(t_BBN) ~ 1 (standard radiation domination has H t = 1/2, i.e. rho t^2 = 3 M^2 / 4). In the curvature
dominated phase  rho t^2  falls as 1/t (matter) or 1/t^2 (radiation), so Y can only decrease unless the interior inflates. The consequences,
all evaluated here with the scales of the potential:

  (1) The energy density of the condensate plus its decay products obeys  rho(t) <= eps DeltaV (t0 / t)^3  for t > t0 = 1/m (matter dilution is the
      slowest dilution; decay into radiation can only make it faster), whatever DeltaV is.
  (2) Y(t0) = eps DeltaV / (3 M^2 m^2) = c sqrt(DeltaV) / M^2, tiny for any sub-Planckian DeltaV; Y ~ 1 needs DeltaV ~ M_Pl^4.
  (3) The number of e-folds available while the field rolls inside the bubble is N = g Integral v/|v'| dx, of order g ~ 1e-60.
  (4) A deeper true vacuum V_t < 0 raises DeltaV = V_f + |V_t| but the open interior crunches at t_c = pi sqrt(3) M / sqrt(|V_t|).
  (5) An extra reservoir with energy density rho_x at t_i must obey the same constraint: Y(t_i) ~ 1 needs rho_x ~ 3 M^2 / t_i^2.

    python scope.py          (writes scope.json)
"""
import json
import os

import numpy as np
from scipy.integrate import quad

import cdl_lib as C
import constants as K
import interior_evolution as I

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
FL = {r["eta"]: r for r in T["flat"]}
M = K.M_PL
HBAR = K.HBAR_GEV_S
GSTAR_BBN = K.GSTAR_BBN
RHO_BBN = (np.pi ** 2 * GSTAR_BBN / 30.0) * K.T_BBN_GEV ** 4
T_BBN_START_S = 1.0                                # time after the start of the hot phase by which T ~ 1 MeV in standard cosmology
out = {"rho_bbn_gev4": float(RHO_BBN), "g_star_bbn": GSTAR_BBN, "t_bbn_s": T_BBN_START_S}


def mass_factor(lam, eta):
    """m = kappa_m * DeltaV^(1/4) with kappa_m = lam^(1/4) sqrt((3 x_t^2 - 1) / 2) / dtil^(1/4)  (m: mass at the true minimum)."""
    p = C.Potential(lam, eta)
    return lam ** 0.25 * np.sqrt((3.0 * p.x_t ** 2 - 1.0) / 2.0) / p.dtil ** 0.25


# ---- (2) curvature parameter at t0 and the energy density at which it reaches one
rows = []
for lam, eta in [(1.0, 0.30), (1e-2, 0.30), (4 * np.pi, 0.30), (1.0, 0.15), (1.0, 0.36)]:
    p = C.Potential(lam, eta)
    km = mass_factor(lam, eta)
    eps = float(FL[eta]["v_x0"] / FL[eta]["dtil"]) if eta in FL else 1.0
    c0 = eps / (3.0 * km ** 2)                       # Y(t0) = c0 * sqrt(DeltaV) / M^2   (m^2 = km^2 sqrt(DeltaV))
    dv_unit = (1.0 / c0) ** 2                        # DeltaV / M^4 at which Y(t0) = 1
    rows.append(dict(lam=float(lam), eta=eta, kappa_m=float(km), eps_in=eps, Y0_at_Vf=float(c0 * np.sqrt(K.V_F) / M ** 2), c0=float(c0),
                     DeltaV_over_M4_for_Y1=float(dv_unit)))
out["curvature"] = rows
out["kappa_m_max"] = float(max(mass_factor(l, e) for l in np.logspace(-2, np.log10(4 * np.pi), 30) for e in np.linspace(0.05, 0.37, 30)))
out["kappa_m_min"] = float(min(mass_factor(l, e) for l in np.logspace(-2, np.log10(4 * np.pi), 30) for e in np.linspace(0.05, 0.37, 30)))

# ---- (1) the envelope rho(t) <= eps DeltaV (m t)^-3 against the density needed at t_BBN, and the parent scale this would require
t_gev = T_BBN_START_S / HBAR
need = {}
for tag, km in (("kappa_m_max", out["kappa_m_max"]), ("kappa_m_min", out["kappa_m_min"]), ("kappa_m_benchmark", mass_factor(1.0, 0.30))):
    dv14 = RHO_BBN * (km * t_gev) ** 3 / 1.0            # (DeltaV)^(1/4) >= rho_BBN (kappa_m t)^3 / eps, eps <= 1
    need[tag] = dict(DeltaV_quarter_root_gev=float(dv14), over_M_Pl=float(dv14 / M), orders_above_Planck=float(np.log10(dv14 / M)))
out["needed_parent_for_bbn_at_1s"] = need
# the density the envelope allows at 1 s for the dark-energy scale
km1 = mass_factor(1.0, 0.30)
rho_env = K.V_F / (km1 * K.V_F ** 0.25 * t_gev) ** 3
out["envelope_at_1s_for_Vf"] = dict(rho_gev4=float(rho_env), over_rho_bbn=float(rho_env / RHO_BBN))

# ---- (3) number of e-folds while the field is inside the bubble
efold = []
for eta in (0.15, 0.20, 0.30, 0.36):
    p = C.Potential(1.0, eta)
    x0 = FL[eta]["x0"]
    val, err = quad(lambda x: p.v(x) / abs(p.dv(x)), p.x_t + 1e-9, x0, limit=400)
    g = float(np.sqrt(K.V_F / (M ** 4 * 1.0 * p.dtil)))
    efold.append(dict(eta=eta, integral=float(val), g=g, N=float(g * val)))
out["e_folds"] = efold

# ---- (4) anti-de Sitter true vacuum: DeltaV, the crunch time, and what is reached at the moment of the crunch
ads = []
for kappa in (0.0, 1.0, 10.0, 1e10, 1e20, 1e30, 1e34):
    dv = K.V_F * (1.0 + kappa)
    T_max = K.t_reh_max(dv)
    vt = K.V_F * kappa
    t_c = (np.pi * np.sqrt(3.0) * M / np.sqrt(vt)) * HBAR if kappa > 0 else np.inf          # seconds
    ads.append(dict(kappa=float(kappa), T_max_gev=float(T_max), T_max_meV=float(T_max * 1e12), t_crunch_s=(float(t_c) if np.isfinite(t_c) else None),
                    t_crunch_gyr=(float(t_c / K.YEAR_S / 1e9) if np.isfinite(t_c) else None)))
out["ads"] = ads
# the largest |V_t| for which the interior outlives 13.8 Gyr, and the temperature bound that goes with it
t_life = 13.8e9 * K.YEAR_S / HBAR
vt_max = 3.0 * np.pi ** 2 * M ** 2 / t_life ** 2
out["ads_lifetime_13p8_gyr"] = dict(Vt_max_over_Vf=float(vt_max / K.V_F), T_max_meV=float(K.t_reh_max(K.V_F + vt_max) * 1e12))
# the opposite question: |V_t| that would give T_max = 1 MeV, and the resulting crunch time
vt_bbn = RHO_BBN
out["ads_for_1MeV"] = dict(Vt_over_Vf=float(vt_bbn / K.V_F), t_crunch_s=float(np.pi * np.sqrt(3.0) * M / np.sqrt(vt_bbn) * HBAR))

# ---- (5) an extra reservoir: energy density needed at t_i = 1/m for Y = 1, relative to what the scalar releases
res = []
for lam, eta in [(1.0, 0.30)]:
    km = mass_factor(lam, eta)
    rho_x = 3.0 * M ** 2 * (km * K.V_F ** 0.25) ** 2
    res.append(dict(lam=lam, eta=eta, rho_x_over_DeltaV=float(rho_x / K.V_F), orders=float(np.log10(rho_x / K.V_F))))
out["extra_reservoir"] = res[0]

# ---- consistency of the envelope with the closed-form radiation density: rho_phi + rho_r <= eps DeltaV (t0/t)^3 on a grid
tau0, eps = 1e-3, 1.0
worst = 0.0
for gamma in (1e-3, 1.0, 10.0, 500.0, 1e4):
    tau = np.logspace(np.log10(tau0 * 1.0000001), 3, 4000)
    rr = I.rho_r_milne(tau, eps, gamma, tau0)
    rphi = eps * (tau0 / tau) ** 3 * np.exp(-gamma * (tau - tau0))
    worst = max(worst, float(np.max((rr + rphi) / (eps * (tau0 / tau) ** 3))))
out["envelope_check_max_ratio"] = worst
json.dump(out, open(os.path.join(HERE, "scope.json"), "w"), indent=1)
print("kappa_m range [%.3f, %.3f]" % (out["kappa_m_min"], out["kappa_m_max"]))
for r in rows:
    print("lam %.3g eta %.2f  kappa_m %.3f  eps_in %.3f  Y(t0) at V_f = %.2e   DeltaV/M^4 for Y=1: %.2e" % (r["lam"], r["eta"], r["kappa_m"], r["eps_in"], r["Y0_at_Vf"], r["DeltaV_over_M4_for_Y1"]))
print("parent scale for rho(1 s) = rho_BBN:", need)
print("envelope at 1 s for V_f:", out["envelope_at_1s_for_Vf"])
for e in efold:
    print("e-folds eta %.2f : %.2e" % (e["eta"], e["N"]))
for a in ads:
    print("kappa %.0e  T_max %.4g meV  t_crunch %s s" % (a["kappa"], a["T_max_meV"], a["t_crunch_s"]))
print("13.8 Gyr:", out["ads_lifetime_13p8_gyr"], " 1 MeV:", out["ads_for_1MeV"], " reservoir:", out["extra_reservoir"], " envelope worst ratio:", worst)
