# -*- coding: utf-8 -*-
"""Bounce validation for the continuous cycles (continuous_cycles.py).

Test A  rho_c sensitivity: the present cycle (today -> turnaround -> bounce) and the next cycle for rho_c = 1e36, 1e40, 1e44 (bounce temperatures
        1e9, 1e10, 1e11 in units of 1.82 meV, i.e. 1.8, 18 and 180 MeV).
Test B  the full effective equations through the bounce: the Friedmann constraint 3H^2 = rho (1 - rho/rho_c) is not projected within
        rho > 0.4 rho_c, so its residual there measures the consistency of the evolved Raychaudhuri equation with the constraint; energy
        conservation d rho/dt + 3H (rho + p) = 0 is checked by finite differences on the stored history; continuity of a, H, rho and of the
        entropy a^3 T^3 across the bounce; the curvature invariants R = 6 (dH/dt + 2H^2) and K = 12 [(dH/dt + H^2)^2 + H^4] stay finite, with the
        analytic bounce values R = 3 (rho + p) and K = 3 (rho + p)^2 (= 4 rho_c and 16 rho_c^2/3 for radiation).
Test C  an independent bounce: general relativity with a negative-energy stiff fluid, rho_g = -g/a^6 (bounce='ghost'). The bounce density is then
        set by a comoving constant, so the bounce temperature is not fixed from cycle to cycle: T_b ~ S/g^(1/2) for radiation.

    python bounce_validation.py      (writes bounce_validation.json)
"""
import json
import os
import sys
import time

import numpy as np

import continuous_cycles as CC

HERE = os.path.dirname(os.path.abspath(__file__))


def bounce_window(r, P, k=0):
    """History rows of cycle k around its bounce (end of cycle k + start of cycle k+1)."""
    a = np.array(r["cycles"][k]["hist"])
    b = np.array(r["cycles"][k + 1]["hist"]) if len(r["cycles"]) > k + 1 else np.zeros((0, a.shape[1]))
    return a, b


def invariants(P, rho, p):
    """dH/dt and H^2 from the effective equations at a point with density rho and pressure p (LQC)."""
    H2 = max(rho * (1 - rho / P.rho_c) / 3.0, 0.0)
    dH = -0.5 * (rho + p) * (1 - 2 * rho / P.rho_c)
    R = 6 * (dH + 2 * H2)
    K = 12 * ((dH + H2) ** 2 + H2 ** 2)
    return R, K


def run_C(bounce, n_cycles=4, wall=None):
    P = CC.Params(bounce=bounce)
    r = CC.run(P, n_cycles=n_cycles, wall=wall)
    return dict(bounce=bounce, status=r["status"], max_residual=r["max_residual"],
                cycles=[dict((k, c.get(k)) for k in ("n", "T_bounce", "bounce_radiation_fraction", "ln_a_max", "ln_S_turn", "held_accelerated_efolds",
                                                       "Y_fo_over_kappa", "Y_held_over_kappa", "lnG")) for c in r["cycles"]])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ghost":
    # Test C alone, with a longer time budget (the ghost bounce is slower to integrate); replaces the ghost entry
    out = json.load(open(os.path.join(HERE, "bounce_validation.json"), encoding="utf-8"))
    g = run_C("ghost", n_cycles=3, wall=10000)
    out["C"] = [c for c in out["C"] if c["bounce"] != "ghost"] + [g]
    print("C ghost", g)
    json.dump(out, open(os.path.join(HERE, "bounce_validation.json"), "w"), indent=1)
elif __name__ == "__main__":
    t0 = time.time()
    out = dict(A=[], C=[])
    for rc in (1e36, 1e40, 1e44):
        P = CC.Params(rho_c=rc)
        r = CC.run(P, n_cycles=2)
        c0, c1 = r["cycles"][0], r["cycles"][1]
        a, b = bounce_window(r, P)
        # continuity across the bounce: last stored row of cycle 0 = bounce point, first row of cycle 1
        rho_end = a[-1, 4] + a[-1, 6] + a[-1, 7] + a[-1, 8]
        rho_start = b[0, 4] + b[0, 6] + b[0, 7] + b[0, 8]
        S_end = 3 * a[-1, 0] + 3 * np.log(a[-1, 3])
        S_start = 3 * b[0, 0] + 3 * np.log(b[0, 3])
        # energy conservation along the stored history of cycle 0 (finite differences in ln a: d ln rho/d ln a = -3 (1 + w))
        la, rho, w = a[:, 0], a[:, 4] + a[:, 6] + a[:, 7] + a[:, 8], a[:, 12]
        sel = (np.abs(np.diff(la)) > 1e-3) & (rho[1:] > 10 * CC.MC.VC) & (rho[:-1] > 10 * CC.MC.VC)
        dl = np.diff(np.log(np.abs(rho))) / np.diff(la)
        pred = -3 * (1 + 0.5 * (w[1:] + w[:-1]))
        cons = float(np.median(np.abs(dl[sel] - pred[sel]))) if sel.any() else None
        # curvature at the bounce
        rb = c0["bounce_rho"]
        Rb, Kb = invariants(P, rb, rb / 3.0)
        out["A"].append(dict(rho_c=rc, T_bounce=c0["T_bounce"], T_bounce_MeV=c0["T_bounce"] * CC.T_UNIT_EV * 1e-6, bounce_radiation_fraction=c0["bounce_radiation_fraction"],
                             bounce_x=c0["bounce_x"], phi_bounce_over_v=c0["phi_bounce_over_v"], next=dict((k, c1.get(k)) for k in
                             ("held_accelerated_efolds", "Y_fo_over_kappa", "Y_held_over_kappa", "max_matter_fraction", "T_release", "phi_bounce_over_v", "ln_a_max")),
                             max_residual=r["max_residual"], rho_jump=float(abs(rho_start / rho_end - 1)), lnS_jump=float(abs(S_start - S_end)),
                             conservation_median_error=cons, R_over_rho_c=float(Rb / rc), K_over_rho_c2=float(Kb / rc ** 2), status=r["status"]))
        print("A", out["A"][-1], "[%.0fs]" % (time.time() - t0), flush=True)
    for bounce in ("lqc", "ghost"):
        P = CC.Params(bounce=bounce)
        r = CC.run(P, n_cycles=4)
        out["C"].append(dict(bounce=bounce, status=r["status"], max_residual=r["max_residual"],
                             cycles=[dict((k, c.get(k)) for k in ("n", "T_bounce", "bounce_radiation_fraction", "ln_a_max", "ln_S_turn", "held_accelerated_efolds",
                                                                    "Y_fo_over_kappa", "Y_held_over_kappa", "lnG")) for c in r["cycles"]]))
        print("C", bounce, out["C"][-1], "[%.0fs]" % (time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, "bounce_validation.json"), "w"), indent=1)
