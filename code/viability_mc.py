# -*- coding: utf-8 -*-
"""Section 7.7: how often are all the conditions for a cycle satisfied at once, over stated parameter ranges?

The conditions are the ones derived in this paper; each is evaluated for every draw and the fractions are reported singly and jointly. The ranges
are stated assumptions (log-uniform unless noted), so the fractions describe the sample space of the model, not a probability for our universe.

Held vacuum (symmetron form, Section 6.8):
    a_c in [0.5, 10], q in [-1, 3] (uniform), dark-matter mass m in [1 MeV, 1 GeV], bounce temperature T_b in [1 MeV, 10 GeV]
    H1 held today (a_c > 1)             H2 recollapse (q > 0)              H3 bounce above BBN and below black-hole percolation (5 MeV < T_b < 2.17 GeV)
    H4 re-thermalization T_b > m A(v)/5, with A(v) - 1 = 2 (1 + q) V_c a_c^3 / rho_dm,0 (the exact energy relation of Section 6.2)
Tunneling branch (Section 7):
    eta in {0.20, 0.30, 0.36}, B0 in [50, 1000], theta_0/theta_deg in [1e-3, 10], q in [-1, 3], m and T_b as above
    T1 the past light cone contains no bubble     T2 the transition completes (Gamma/H^4 > 9/4pi) in the future     T3 recollapse (q > 0)
    T4 bounce window as H3                        T5 re-thermalization T_b > m A(x_t)/5 (needed for self-similar cycles)
Common to both and not sampled: dissipation at the restoration, a coherent seed (symmetron), and shear inherited rather than regenerated.

    python viability_mc.py          (writes viability_mc.json, ~5 min)
"""
import json
import os
import time

import numpy as np

import bubble_to_cycle as BC
import constants as K
import matter_coupling as MC
import thermal as TH
import vacuum_phase_map as VP
import vacuum_wall as W

HERE = os.path.dirname(os.path.abspath(__file__))
T_BBN, T_PERC = 5.0, 2174.0                                             # MeV (Section 6.12, population E3)
rng = np.random.default_rng(20261030)
logu = lambda lo, hi, n: np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def held(n=200000):
    a_c, q = logu(0.5, 10.0, n), rng.uniform(-1.0, 3.0, n)
    m, Tb = logu(1.0, 1000.0, n), logu(1.0, 1e4, n)
    A_v = 1.0 + 2.0 * (1.0 + q) * MC.VC * a_c ** 3 / MC.RHO_DM0
    c = {"H1 held today": a_c > 1.0, "H2 recollapse": q > 0.0, "H3 bounce window": (Tb > T_BBN) & (Tb < T_PERC),
         "H4 re-thermalization": Tb > m * A_v / 5.0}
    allc = np.logical_and.reduce(list(c.values()))
    return dict(n=n, single={k: float(v.mean()) for k, v in c.items()}, joint=float(allc.mean()),
                joint_given_H1H2=float(allc[c["H1 held today"] & c["H2 recollapse"]].mean()))


def tunneling(WJ, n=3000):
    z_lc, w_lc = TH.light_cone_grid()
    tabs = {e: BC.rate_tables(WJ, e) for e in ("0.20", "0.30", "0.36")}
    eta = rng.choice(list(tabs.keys()), n)
    B0, f0, q = logu(50.0, 1000.0, n), logu(1e-3, 10.0, n), rng.uniform(-1.0, 3.0, n)
    m, Tb = logu(1.0, 1000.0, n), logu(1.0, 1e4, n)
    T1, T2 = np.zeros(n, bool), np.zeros(n, bool)
    Arat = np.ones(n)
    for i in range(n):
        code, lnI, a_t = BC.classify_point(B0[i], f0[i], max(q[i], -0.99), tabs[eta[i]], z_lc, w_lc)
        T1[i] = lnI < 0
        T2[i] = a_t is not None
        if a_t is not None:
            e = float(eta[i])
            th_deg = VP.theta_deg_tb(e)
            th = f0[i] * th_deg / a_t ** 3
            n_, x_t, _, x_h, _, _ = VP.classify_tb(e, th)
            # A(x) - 1 = (x-1)^2 f^2/(2M^2) with f^2/M^2 = theta lam f^4 / rho_*; lam f^4 = (1+q) V_f / dtil0
            d0 = VP.classify_tb(e, 0.0)[4]
            E0 = (1.0 + max(q[i], 1e-3)) * MC.VC / d0
            rho_s = MC.RHO_DM0 / a_t ** 3
            Arat[i] = (1.0 + (x_t - 1.0) ** 2 * th * E0 / rho_s / 2.0) / (1.0 + (x_h - 1.0) ** 2 * th * E0 / rho_s / 2.0)
    c = {"T1 past light cone clear": T1, "T2 completes": T2, "T3 recollapse": q > 0.0, "T4 bounce window": (Tb > T_BBN) & (Tb < T_PERC),
         "T5 re-thermalization": Tb > m * Arat / 5.0}
    allc = np.logical_and.reduce(list(c.values()))
    back = T1 & T2 & (q > 0)
    return dict(n=n, single={k: float(v.mean()) for k, v in c.items()}, joint=float(allc.mean()), joint_background=float(back.mean()),
                joint_given_background=float(allc[back].mean()) if back.any() else None,
                by_eta={e: float(allc[eta == e].mean()) for e in tabs})


if __name__ == "__main__":
    t0 = time.time()
    WJ = json.load(open(os.path.join(HERE, "vacuum_wall.json"), encoding="utf-8"))
    out = dict(held=held())
    print("held", out["held"], "[%.0fs]" % (time.time() - t0), flush=True)
    out["tunneling"] = tunneling(WJ)
    print("tunneling", out["tunneling"], "[%.0fs]" % (time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, "viability_mc.json"), "w"), indent=1)
