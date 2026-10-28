# -*- coding: utf-8 -*-
"""Falsification-region map in the (eta, lambda) plane and the physical size of the gravitational and Hawking-Moss effects.

    python falsification_map.py          (writes falsification_map.json)
"""
import json
import os

import numpy as np
from scipy.optimize import brentq

import cdl_lib as C
import constants as K
import rates as R

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
out = {}

# ---- grid of ln(Gamma/H^4), B and decay time
etas = np.linspace(0.03, 0.38, 176)
log_lams = np.linspace(-2.0, 2.0, 161)
E, L = np.meshgrid(etas, log_lams)
lam = 10.0 ** L
lg, B = R.ln_gamma_over_h4(lam, E)
out["grid"] = dict(eta=etas.tolist(), log10_lam=log_lams.tolist(), B=B.tolist(), ln_gamma_h4=lg.tolist(),
                   log10_t_yr=R.log10_decay_time_years(lg).tolist())

# ---- contours: for each lambda the eta at which the decay time equals a given value (b(eta) is monotonic decreasing)
def eta_for_log10t(lam_, log10t):
    f = lambda e: float(R.log10_decay_time_years(R.ln_gamma_over_h4(lam_, e)[0]) - log10t)
    lo, hi = 0.03, 0.38
    if f(lo) < 0 or f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-9)

cont = {}
for name, lt in [("survival_bound", float(np.log10(K.T_HUBBLE_YR / ((4 * np.pi / 3) * 3.0 / (8 * np.pi ** 2))))), ("1e12yr", 12.0), ("1e20yr", 20.0),
                 ("1e50yr", 50.0), ("1e100yr", 100.0)]:
    cont[name] = [dict(log10_lam=float(l), eta=eta_for_log10t(10 ** l, lt)) for l in np.linspace(-2.0, 2.0, 41)]
out["contours"] = cont

# ---- the physical gravity parameter and its consequences
phys = []
for lam_, eta_ in [(1.0, 0.30), (1.0, 0.32), (1.0, 0.34), (0.3, 0.34), (3.0, 0.25), (10.0, 0.18)]:
    pot = C.Potential(lam_, eta_)
    g = float(np.sqrt(K.V_F / (K.M_PL ** 4 * lam_ * pot.dtil)))        # g = f^2/M^2 with f^4 = V_f / (lam dtil)
    f_gev = (K.V_F / (lam_ * pot.dtil)) ** 0.25
    H2 = g * pot.v_f / 3.0
    kappa = np.interp(eta_, [s["eta"] for s in T["slopes"]], [s["kappa"] for s in T["slopes"]])
    lgv, Bv = R.ln_gamma_over_h4(lam_, eta_)
    phys.append(dict(lam=lam_, eta=eta_, g=g, f_meV=f_gev * 1e12, delta_phi_meV=2 * f_gev * 1e12 * float(pot.x_f - pot.x_t) / 2.0, B=float(Bv),
                     gravity_correction_relative=float(kappa * g), hm_ratio=float(abs(pot.m2_top) / (4.0 * H2)),
                     ln_hm_action=float(np.log(C.hm_action(lam_, eta_, g))), ln_gamma_h4=float(lgv),
                     log10_t_yr=float(R.log10_decay_time_years(lgv)), bubble_radius_over_hubble=float(
                         (3.0 * pot.wall_tension() / (lam_ * pot.dtil)) / (1.0 / np.sqrt(H2)))))
out["physical"] = phys

# ---- where gravity would matter: the value of f/M_Pl needed, against the value implied by the measured dark energy
out["g_for_order_one_effect"] = 0.1
out["f_over_M_for_order_one_effect"] = float(np.sqrt(0.1))
out["g_physical_range"] = [min(p["g"] for p in phys), max(p["g"] for p in phys)]

# ---- reheating energy bookkeeping
dV_grid = np.logspace(np.log10(K.V_F) - 2, np.log10(K.T_BBN_GEV ** 4) + 6, 200)
out["reheat"] = dict(
    dV_gev4=dV_grid.tolist(), T_gev=[float(K.t_reh_max(d)) for d in dV_grid],
    T_max_meV_at_Vf=float(K.t_reh_max(K.V_F) * 1e12), T_max_K_at_Vf=float(K.t_reh_max(K.V_F) / K.K_TO_GEV),
    T_cmb_meV=float(K.T_CMB_GEV * 1e12), ratio_Tbbn_over_Tmax=float(K.T_BBN_GEV / K.t_reh_max(K.V_F)),
    rho_needed_for_bbn_over_Vf=float(((np.pi ** 2 * K.GSTAR_LOW / 30.0) * K.T_BBN_GEV ** 4) / K.V_F),
    vacuum_quarter_root_meV=float(K.V_F ** 0.25 * 1e12), gstar=K.GSTAR_LOW)
json.dump(out, open(os.path.join(HERE, "falsification_map.json"), "w"), indent=1)
for p in phys:
    print("lam %.1f eta %.2f  g %.2e  f %.2f meV  B %.1f  grav corr %.1e  HM ratio %.2e  ln B_HM %.1f  log10 t %.1f  R_bubble/R_H %.1e"
          % (p["lam"], p["eta"], p["g"], p["f_meV"], p["B"], p["gravity_correction_relative"], p["hm_ratio"], p["ln_hm_action"], p["log10_t_yr"], p["bubble_radius_over_hubble"]))
r = out["reheat"]
print("T_max at dV=V_f: %.2f meV = %.1f K (CMB now %.3f meV); T_BBN/T_max = %.2e ; rho for BBN / V_f = %.2e" % (r["T_max_meV_at_Vf"], r["T_max_K_at_Vf"], r["T_cmb_meV"], r["ratio_Tbbn_over_Tmax"], r["rho_needed_for_bbn_over_Vf"]))
print({k: [c for c in v if c["eta"] is not None][:2] for k, v in cont.items()})
