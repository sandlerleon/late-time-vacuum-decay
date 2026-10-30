# -*- coding: utf-8 -*-
"""Observational consequences, at the level of estimates (no perturbation code is part of this work).

(1) Dark sector. The held vacuum has w = -1 until the release: the field sits at phi ~ 1e-60 v (its seed) and 1 + w = phi'^2/rho_DE. The release,
    turnaround and recollapse times follow from the background (causal_controls.py, model D). The dark-matter mass grows by A(v) at the release.
(2) Primordial perturbations from the contraction. A mode k leaves the Hubble radius in the contraction when |aH| = k; for a contraction with a
    constant equation of state w the duality-invariant spectrum of the growing curvature mode has n_s - 1 = 12 w/(1 + 3w) (w = 0: scale invariant,
    the matter bounce; w = 1/3: n_s = 3). The scales observed in the next cycle, k = (1 - 1000) a_0' H_0 (a_0' the scale factor of the next cycle at
    T = T0), are followed back into the present cycle's contraction (the scale factor is continuous through the bounce) and w is read off there.
    The amplitude of such a spectrum is set by the bounce scale, P ~ rho_c/M_Pl^4 up to O(1-100) factors.
(3) Gravitational waves: the tensor modes obey the same equation without the matter-specific growth; their amplitude is also set by rho_c/M_Pl^4.
    The transition is a homogeneous roll (no bubbles), so it sources no gravitational waves; the black holes of the cycle (compact_relics.py)
    give the ordinary astrophysical background.

    python predictions.py     (writes predictions.json)
"""
import json
import os

import numpy as np

import causal_controls as CCt
import continuous_cycles as CC
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))


def mode_exit(c0, c1, k_range=(1.0, 1000.0)):
    """For k = k_range x a_0' H0, the epoch of Hubble exit in the contraction of cycle 0 and w there."""
    h0 = np.array(sorted(c0["hist"], key=lambda h: h[1]))
    con = h0[h0[:, 2] < 0]
    h1 = np.array(sorted(c1["hist"], key=lambda h: h[1]))
    ex1 = h1[h1[:, 2] > 0]
    # a_0' : the scale factor of the next cycle where T falls through T0 (if it does)
    below = np.where(ex1[:, 3] <= CC.T0)[0]
    if below.size:
        lna0p = float(ex1[below[0], 0])
    else:
        # the next cycle does not cool to T0: the mapping assumes a next cycle with our thermal history, a T = const from the bounce
        lna0p = float(c0["ln_a_min"] + np.log(c0["T_bounce"] / CC.T0))
    out = dict(ln_a0_next=lna0p, next_cycle_reaches_T0=bool(below.size), modes=[])
    aH = np.exp(con[:, 0]) * np.abs(con[:, 2])
    for kf in np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 7):
        k = kf * np.exp(lna0p) * 1.0                       # H0 = 1
        # first time in the contraction with |aH| > k (the comoving Hubble radius shrinks through 1/k)
        idx = np.where((aH > k) & (np.abs(con[:, 12]) < 10))[0]
        if not idx.size:
            out["modes"].append(dict(k_over_a0H0=float(kf), exits=False))
            continue
        i = idx[0]
        w = float(con[i, 12])
        ns = 1.0 + 12.0 * w / (1.0 + 3.0 * w) if abs(1 + 3 * w) > 1e-6 else None
        out["modes"].append(dict(k_over_a0H0=float(kf), exits=True, T_exit=float(con[i, 3]), w_exit=w, n_s=ns, ln_a_exit=float(con[i, 0])))
    return out


if __name__ == "__main__":
    out = {}
    # (1) dark sector
    m = MC.Model(2.0, 30.0, 1.0)
    D = CCt.evolve(m, "D")
    out["dark_sector"] = dict(one_plus_w_today_bound=float((CCt.SEED * m.v) ** 2 * m.mu ** 2 / MC.VC), t_turn_Gyr=D["t_turn"] * MC.H0_GYR,
                              A_v=float(m.A(m.v)), note="w = -1 until the release; the release happens at a ~ a_c; the turnaround follows")
    print(out["dark_sector"])
    # (2) and (3)
    P = CC.Params()
    r = CC.run(P, n_cycles=1)
    me = mode_exit(r["cycles"][0], r["cycles"][1])
    rc_planck = P.rho_c * CC.H0_OVER_MPL ** 2
    out["perturbations"] = dict(mode_exit=me, rho_c_over_MPl4=rc_planck, amplitude_estimate=rc_planck, observed_amplitude=2.1e-9,
                                rho_c_needed_over_MPl4=2.1e-9, T_b_needed_GeV=(2.1e-9) ** 0.25 * 2.435e18)
    out["sigv_cm3s"] = float(P.sigv * 1.4385e-33 / CC.T_UNIT_EV ** 3 * (1.973e-5) ** 2 * 2.998e10)
    out["gravitational_waves"] = dict(tensor_amplitude_estimate=rc_planck, r_bound="r ~ P_t/P_s ~ O(1) of a spectrum that is itself ~1e-81: no observable primordial background",
                                      transition="homogeneous roll: no bubble collisions, no GW from the release")
    print(json.dumps(out["perturbations"], indent=1)[:2000])
    json.dump(out, open(os.path.join(HERE, "predictions.json"), "w"), indent=1)
