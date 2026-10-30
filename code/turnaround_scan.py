# -*- coding: utf-8 -*-
"""Parameter scan of the dilution trigger: coupled model (C) against the uncoupled control (B) with identical initial conditions at z = 1100
(phi = quantum floor, pi = 0), and LambdaCDM (A) for reference. Grid: a_c (critical scale factor), mu (tachyonic mass, H_0), q (depth of the broken
vacuum, V = -q V_c; q < 0 a lower de Sitter vacuum), seed phi_0/v (10^-60 ~ the de Sitter quantum floor H/2pi in Planck units).

Outcome classes (per run): 'rolled before today' (inconsistent with the observed history), 'eternal expansion', 'transition, no recollapse',
'transition and recollapse'. Observational flags today: |1 + w_0| of the scalar sector and phi(today)/v.

    python turnaround_scan.py          (writes turnaround_scan.json and turnaround_scan.csv; about 15 minutes)
"""
import csv
import json
import os
import time

import numpy as np

import dilution_background as D
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
A_C = [1.25, 1.5, 2.0, 3.0, 5.0]
MU = [3.0, 10.0, 30.0, 100.0, 300.0]
Q = [-0.5, 0.1, 1.0, 10.0]
SEED = 1e-60
T_MAX = 40.0


def today_state(m, r):
    sol = r["sol"]
    # interpolate the state at a = 1 from the stored steps
    la = sol.y[0]
    k = int(np.searchsorted(la, 0.0))
    if k == 0 or k >= la.size:
        return None
    f = (0.0 - la[k - 1]) / (la[k] - la[k - 1])
    return sol.y[:, k - 1] * (1 - f) + sol.y[:, k] * f


def one(a_c, mu, q, coupled, seed=SEED, a_i=1e-3, t_max=T_MAX, rtol=1e-10):
    m = MC.Model(a_c, mu, q, coupled=coupled)
    # q < 0: V_0 >= V_min > 0, every density is positive and H cannot vanish in flat GR, so no turnaround is possible; only the roll is computed
    t_max = max(t_max, 40.0 + 400.0 / mu)
    r = D.evolve(m, seed * m.v, a_i=a_i, t_max=t_max, rtol=rtol, stop_at_roll=(q < 0))
    y = today_state(m, r)
    w0 = None
    phi_today = None
    if y is not None:
        K_ = 0.5 * y[3] ** 2
        Vde = m.V0(y[2]) + MC.RHO_DM0 * (m.A(y[2]) - 1.0)        # scalar sector incl. the coupling energy
        w0 = float((K_ - m.V0(y[2])) / (K_ + Vde)) if (K_ + Vde) != 0 else None
        phi_today = float(abs(y[2]) / m.v)
    rolled_before = (r["a_roll"] is not None and r["a_roll"] < 1.0) or not r["reached_today"]
    if rolled_before:
        cls = "rolled before today"
    elif q < 0:
        cls = "transition, no recollapse" if r["t_roll"] is not None else "eternal expansion (within t_max)"
    else:
        cls = D.classify(m, r)
    row = dict(a_c=a_c, mu=mu, q=q, coupled=coupled, seed=seed, cls=cls, t_roll=r["t_roll"], t_turn=r["t_turn"], t_crunch=r["t_crunch"],
               a_roll=r["a_roll"], a_turn=r["a_turn"], D_turn=r.get("D_turn"), one_plus_w0=(None if w0 is None else float(1 + w0)),
               phi_today_over_v=phi_today, max_residual=r["max_residual"], A_at_v=float(m.A(m.v)) if coupled else 1.0,
               M=float(np.sqrt(m.M2)), v=float(m.v))
    return row


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for a_c in A_C:
        for mu in MU:
            for q in Q:
                for coupled in (True, False):
                    if not coupled and a_c != A_C[0]:
                        # the uncoupled control does not depend on a_c except through M (irrelevant): reuse
                        base = [r for r in rows if (not r["coupled"]) and r["mu"] == mu and r["q"] == q][0]
                        rows.append(dict(base, a_c=a_c))
                        continue
                    rows.append(one(a_c, mu, q, coupled))
        print("a_c %.2f done [%.0fs]" % (a_c, time.time() - t0), flush=True)
    # seed dependence for the reference model
    seeds = []
    for s in (1e-6, 1e-20, 1e-40, 1e-60):
        for coupled in (True, False):
            seeds.append(one(2.0, 100.0, 1.0, coupled, seed=s))
    out = dict(grid=dict(a_c=A_C, mu=MU, q=Q, seed=SEED, t_max=T_MAX), rows=rows, seeds=seeds, H0_Gyr=MC.H0_GYR,
               lcdm=D.lcdm_future())
    json.dump(out, open(os.path.join(HERE, "turnaround_scan.json"), "w"), indent=1)
    with open(os.path.join(HERE, "turnaround_scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved [%.0fs]" % (time.time() - t0))
