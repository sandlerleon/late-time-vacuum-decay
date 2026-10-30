# -*- coding: utf-8 -*-
"""Where do the continuous cycles work? Failure-mode map of continuous_cycles.py.

A grid of dark-matter mass m and bounce density rho_c (the two scales that decide whether the bounce re-thermalizes the dark matter), and
one-at-a-time variations about the baseline (m = 100 MeV, rho_c = 1e44, i.e. a bounce at ~180 MeV) of the dissipation C_T, the coupling
parameters a_c, mu, q, the initial shear and the bounce mechanism. Every run: today + 5 cycles. Outcomes (continuous_cycles.classify):
working, F1 dark matter not regenerated (bounce too cold), F2 vacuum not restored, F3 kinetic-dominated bounce, F4 shear-dominated bounce,
F5 no bounce, F6 integration failure / limit.

    python cycle_scan.py          (writes cycle_scan.json; runs in parallel)
    python cycle_scan.py robust   (adds one-at-a-time variations about the robust baseline m = 10 MeV)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_CYC = 5


def job(kw):
    import continuous_cycles as CC
    t0 = time.time()
    try:
        P = CC.Params(**kw)
        r = CC.run(P, n_cycles=N_CYC, keep_history=False, wall=1500)
        cls = CC.classify(r)
    except Exception as e:                                     # recorded, not hidden
        return dict(params=kw, classification="F6 exception: %s" % e, seconds=time.time() - t0)
    cs = [c for c in r["cycles"] if c["n"] >= 1]
    stat = lambda k: [c.get(k) for c in cs]
    return dict(params=kw, classification=cls, status=r["status"], max_residual=r["max_residual"], sigv=P.sigv, seconds=time.time() - t0,
                T_bounce_MeV=[c.get("T_bounce", 0) * CC.T_UNIT_EV * 1e-6 for c in r["cycles"]], bounce_x=[c.get("bounce_x") for c in r["cycles"]],
                held=stat("held_accelerated_efolds"), Y=[c.get("Y_fo_over_kappa", c.get("Y_late_over_kappa")) for c in cs], Y_held=stat("Y_held_over_kappa"), mmax=stat("max_matter_fraction"), lnG=stat("lnG"),
                lnG_entropy=stat("lnG_entropy"), phi_b=stat("phi_bounce_over_v"), K_b=[c.get("bounce_kinetic_fraction") for c in r["cycles"]],
                shear_b=[c.get("bounce_shear_fraction") for c in r["cycles"]])


ROBUST = dict(m_dm=5.5e9)          # 10 MeV dark matter with the 180 MeV bounce: T_b ~ m A(v)/5, re-thermalized whatever the phase of the frozen field


def robust_jobs():
    jobs = []
    for CT in (1e-3, 1e3):
        jobs.append(dict(ROBUST, C_T=CT))
    for a_c in (1.5, 3.0):
        jobs.append(dict(ROBUST, a_c=a_c))
    for mu in (10.0, 100.0):
        jobs.append(dict(ROBUST, mu=mu))
    for q in (0.3, 3.0):
        jobs.append(dict(ROBUST, q=q))
    return jobs


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "robust":
    # one-at-a-time variations about the robust baseline, appended to cycle_scan.json
    t0 = time.time()
    data = json.load(open(os.path.join(HERE, "cycle_scan.json"), encoding="utf-8"))
    with Pool(6) as pool:
        for row in pool.imap_unordered(job, robust_jobs()):
            row["baseline"] = "robust"
            data["rows"].append(row)
            print("%-45s %-48s [%.0fs]" % (row["params"], row["classification"], time.time() - t0), flush=True)
    json.dump(data, open(os.path.join(HERE, "cycle_scan.json"), "w"), indent=1)
elif __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for m in (5.5e8, 5.5e9, 5.5e10, 5.5e11):                    # 1 MeV ... 1 GeV
        for rc in (1e36, 1e40, 1e44, 1e48):                     # bounce at ~1.8 MeV ... 1.8 GeV
            jobs.append(dict(m_dm=m, rho_c=rc))
    for CT in (1e-6, 1e-3, 1e3):
        jobs.append(dict(C_T=CT))
    for a_c in (1.5, 3.0):
        jobs.append(dict(a_c=a_c))
    for mu in (10.0, 100.0):
        jobs.append(dict(mu=mu))
    for q in (0.3, 3.0):
        jobs.append(dict(q=q))
    for s0 in (1e-20, 1e-10):
        jobs.append(dict(shear0=s0))
    jobs.append(dict(bounce="ghost"))
    with Pool(6) as pool:
        rows = []
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print("%-45s %-48s [%.0fs]" % (row["params"], row["classification"], time.time() - t0), flush=True)
    json.dump(dict(n_cycles=N_CYC, rows=rows), open(os.path.join(HERE, "cycle_scan.json"), "w"), indent=1)
