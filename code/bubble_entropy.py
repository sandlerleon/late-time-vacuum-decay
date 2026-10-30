# -*- coding: utf-8 -*-
"""Section 7.6: energy and entropy bookkeeping across the vacuum interface. No subsystem may succeed by violating a balance elsewhere.

(1) Wall energy. With the junction of bubble_gravity.py, the Misner-Sharp mass carried by the wall equals the vacuum-energy deficit of the region it
    encloses, M_wall = (4 pi/3) Delta_rho R^3, to round-off for every R: the released vacuum energy is in the wall (kinetic energy, Gamma = R/R_0), not
    in the interior, until walls collide.
(2) Interior bookkeeping while the wall sweeps outward (quasi-static, R << 1/H): for the region inside the wall,
        dE_in = F_E dV + W_wall,    F_E = rho_out (content swept in; every species transmitted),    W_wall = -Delta V_eff dV (energy handed to the wall),
    so dE_in = rho_in dV: the closure residual is reported.
(3) Entropy. Photons, neutrinos and baryons cross the wall unchanged (they do not couple to phi); dark matter crosses once the wall is faster than
    gamma_th, which happens within a distance gamma_th R_0 of the nucleation point; the comoving entropy of the region is therefore inherited. Entropy is
    produced when the wall energy thermalizes at collisions (percolated case) and when the restoration energy is dissipated. The comoving entropy after
    every event is compared with the one before (it may not decrease). The cosmological-horizon entropy of the false vacuum, S_dS = 8 pi^2 M_Pl^2 / H^2,
    is reported for comparison: decay into an anti-de Sitter vacuum removes the horizon, as in any decay out of de Sitter space.

    python bubble_entropy.py          (writes bubble_entropy.json)
"""
import json
import os

import numpy as np

import bubble_gravity as BG
import constants as K
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    GJ = json.load(open(os.path.join(HERE, "bubble_gravity.json"), encoding="utf-8"))
    CJ = json.load(open(os.path.join(HERE, "bubble_to_cycle.json"), encoding="utf-8"))
    TJ = json.load(open(os.path.join(HERE, "interface_transport.json"), encoding="utf-8"))
    W_ = GJ["wall"]
    out = {}
    # (1) wall energy = deficit
    out["wall_energy_closure_max_dev"] = float(max(abs(c - 1.0) for c in W_["closure"]))
    # (2) interior bookkeeping along R (quasi-static)
    R = np.array(W_["R_over_RH"]) / W_["H"]
    V = 4.0 * np.pi / 3.0 * R ** 3
    dV = np.diff(V)
    dE_in = W_["rho_in"] * dV
    F_E = W_["rho_out"] * dV
    W_wall = -W_["dV_eff"] * dV
    out["interior_bookkeeping_max_rel_residual"] = float(np.max(np.abs(dE_in - (F_E + W_wall)) / np.abs(dE_in)))
    # (3) entropy along the cycles (percolated and isolated, baseline)
    ent = {}
    for key in ("B percolated", "B isolated", "C* percolated, fully selective, hot bounce"):
        ev = CJ[key]["events"]
        S = [e["S_comoving"] for e in ev]
        jumps = []
        for e in ev:
            if "S_after" in e:
                jumps.append(dict(event=e["event"], ratio=float(e["S_after"] / e["S_comoving"])))
        ent[key] = dict(min_step_ratio=float(min(S[i + 1] / S[i] for i in range(len(S) - 1))), nondecreasing=bool(all(S[i + 1] >= S[i] * (1 - 1e-12)
                        for i in range(len(S) - 1))), per_cycle_ratio=float(np.exp(3.0 * np.mean(CJ[key]["lnG"]))) if CJ[key]["lnG"] else None,
                        nucleation_jumps=[j["ratio"] for j in jumps if j["event"] == "nucleation"])
        # restoration releases
        rel = [e for e in ev if e["event"].startswith("restoration")]
        ent[key]["restoration_energy_over_Vf"] = [float(e["dE_released"] / MC.VC) for e in rel]
    out["entropy"] = ent
    # selective transport changes the interior content only by the filtered fraction
    out["filtered_fraction"] = TJ["dark_matter"]["filtered_fraction"]
    # horizon entropy of the false vacuum vs radiation entropy in a Hubble volume at nucleation
    HL = np.sqrt(K.V_F / 3.0) / K.M_PL                                        # GeV
    S_dS = 8.0 * np.pi ** 2 * K.M_PL ** 2 / HL ** 2
    T0 = K.T_CMB_GEV
    s_gamma = 2.0 * np.pi ** 2 / 45.0 * 3.91 * T0 ** 3                        # entropy density today (photons + neutrinos), GeV^3
    out["horizon"] = dict(S_dS=float(S_dS), S_radiation_Hubble_volume=float(s_gamma * 4.0 * np.pi / 3.0 / HL ** 3),
                          log10_ratio=float(np.log10(S_dS / (s_gamma * 4.0 * np.pi / 3.0 / HL ** 3))))
    json.dump(out, open(os.path.join(HERE, "bubble_entropy.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1)[:2500])
