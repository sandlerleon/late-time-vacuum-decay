# -*- coding: utf-8 -*-
"""How strong must a matter coupling be to change the decay rate? (Section 6 of the paper.)

Any coupling of matter to the scalar enters the effective potential as  V_eff = V(phi) + rho_m [ m(phi) / m(phi_ref) ]-weighted term, i.e.
        delta V(phi) = rho_m * ln[ m(phi) / m_ref ]            (for non-relativistic matter of mass m(phi); rho_m is the matter density)
so that the change of the energy difference between the two vacua is  delta(DeltaV) = rho_m * Delta ln m,  Delta ln m the fractional change of the
matter mass between the two field values. The change of the action follows from the tilt:
        delta eta = 2 (delta V / DeltaV) dtil(eta) / (x_f - x_t),     delta B = B |d ln b / d ln eta| (delta eta / eta).
A change delta B changes the rate by the factor e^{delta B}. The required Delta ln m is therefore model independent; a specific coupling
function only decides which coupling constant produces it:
        linear     m = m0 exp(beta phi / M_Pl)          Delta ln m = beta Delta phi / M_Pl
        quadratic  m = m0 (1 + phi^2 / (2 L^2))         Delta ln m = f^2 (x_f^2 - x_t^2) / (2 L^2)         (screened / symmetron-type)

    python coupling.py          (writes coupling.json)
"""
import json
import os

import numpy as np

import cdl_lib as C
import constants as K
import rates as R

HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "falsification_map.json"), encoding="utf-8"))
PH = {(p["lam"], p["eta"]): p for p in F["physical"]}
out = {"omega_m": 1.0 - K.OMEGA_L}
rho_m_over_Vf = (1.0 - K.OMEGA_L) / K.OMEGA_L
out["rho_m_over_Vf"] = rho_m_over_Vf
rows = []
for eta in (0.30, 0.32):
    p = C.Potential(1.0, eta)
    ph = PH[(1.0, eta)]
    h = 1e-4
    slope = abs((np.log(R.b_of_eta(eta * (1 + h))) - np.log(R.b_of_eta(eta * (1 - h)))) / (np.log(1 + h) - np.log(1 - h)))     # |d ln b / d ln eta|
    dx = float(p.x_f - p.x_t)
    to_deta = 2.0 * p.dtil / dx / eta                                      # delta eta / eta  per unit delta V / DeltaV
    row = dict(eta=eta, B=ph["B"], dlnb_dlneta=float(slope), deta_over_eta_per_dV=float(to_deta), x_f_minus_x_t=dx, f_meV=ph["f_meV"],
               delta_phi_meV=ph["delta_phi_meV"], x_f=float(p.x_f), x_t=float(p.x_t))
    rows.append(row)
MC = json.load(open(os.path.join(HERE, "monte_carlo.json"), encoding="utf-8"))
BC = MC["B_critical_quantiles"]["0.5"]
out["B_c"] = BC
for row in rows:
    p_ = C.Potential(1.0, row["eta"])
    S = row["dlnb_dlneta"] * row["deta_over_eta_per_dV"]                  # delta B / B per unit delta V / DeltaV
    row["dB_over_B_per_dV"] = S
    row["cases"] = []
    for label, dB in (("change of the rate by a factor e", 1.0), ("reset window: B lowered to B_c", max(row["B"] - BC, 1.0))):
        dV = dB / (row["B"] * S)                                            # delta V / DeltaV  (DeltaV = V_f for r_L = 1)
        dlnm = dV / rho_m_over_Vf
        beta = dlnm / (row["delta_phi_meV"] * 1e-12 / K.M_PL)
        Lq = row["f_meV"] * np.sqrt(abs(row["x_f"] ** 2 - row["x_t"] ** 2) / (2.0 * dlnm))
        row["cases"].append(dict(label=label, delta_B=float(dB), dV_over_DV=float(dV), delta_ln_m=float(dlnm), beta_linear=float(beta), Lambda_quadratic_meV=float(Lq)))
out["rows"] = rows
json.dump(out, open(os.path.join(HERE, "coupling.json"), "w"), indent=1)
for row in rows:
    print("eta %.2f  B %.0f  dlnb/dlneta %.2f  dB/B per dV/DV %.2f" % (row["eta"], row["B"], row["dlnb_dlneta"], row["dB_over_B_per_dV"]))
    for c in row["cases"]:
        print("   %-34s dB %.1f  dV/DV %.2e  dln m %.2e  beta %.2e  Lambda_q %.1f meV" % (c["label"], c["delta_B"], c["dV_over_DV"], c["delta_ln_m"], c["beta_linear"], c["Lambda_quadratic_meV"]))
