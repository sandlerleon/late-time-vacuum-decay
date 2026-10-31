# -*- coding: utf-8 -*-
"""Sensitivity of the black-hole percolation bound on the bounce temperature (compact_relics.py) to the population and accretion prescriptions.

The bound is T_perc = T_b (f_c / f_V(T_b))^(1/3): the bounce temperature at which the fraction of space inside horizons reaches the percolation
threshold f_c, extrapolated with f_V ~ T^3 from the evolved population at the bounce of the baseline cycle. Varied: the abundances of stellar and
supermassive black holes (x 1/3 and x 3), the supermassive cut-off mass, the accretion rate (x 0.1, x 2, x 3, x 10, none), mergers (off), and the threshold:
f_c = 1 (volumes simply added, as in the paper's estimate) or f_c = 0.34, the critical reduced density for continuum percolation of randomly placed,
overlapping spheres, for which the occupied fraction is 1 - exp(-f_V). Writes relic_sensitivity.json.

    python relic_sensitivity.py
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import compact_relics as CR
import continuous_cycles as CC

HERE = os.path.dirname(os.path.abspath(__file__))
F_C_OVERLAP = 0.34
BASE_POP = dict(omega=(1e-4, 7.6e-6), centre=(1.0, 8.0), width=(0.5, 1.0), cutoff=(100.0, 5e9))
VARIANTS = [("baseline (E3)", {}, {}),
            ("stellar abundance x 1/3", dict(omega=(1e-4 / 3, 7.6e-6)), {}), ("stellar abundance x 3", dict(omega=(3e-4, 7.6e-6)), {}),
            ("supermassive abundance x 1/3", dict(omega=(1e-4, 7.6e-6 / 3)), {}), ("supermassive abundance x 3", dict(omega=(1e-4, 2.28e-5)), {}),
            ("supermassive cut-off 1e9", dict(cutoff=(100.0, 1e9)), {}), ("supermassive cut-off 2e10", dict(cutoff=(100.0, 2e10)), {}),
            ("supermassive centre 1e9", dict(centre=(1.0, 9.0)), {}),
            ("accretion x 0.1", {}, dict(k_acc=0.1 * CR.K_ACC)), ("accretion x 2", {}, dict(k_acc=2 * CR.K_ACC)),
            ("accretion x 3", {}, dict(k_acc=3 * CR.K_ACC)), ("accretion x 10", {}, dict(k_acc=10 * CR.K_ACC)),
            ("no accretion, no mergers (E2)", {}, dict(experiment="E2")), ("no mergers", {}, dict(merge=False))]
_HIST = None


def init(hist):
    global _HIST
    _HIST = hist


def job(item):
    name, pop, kw = item
    t = time.time()
    p = dict(BASE_POP)
    p.update(pop)
    o = CR.evolve(_HIST, kw.get("experiment", "E3"), merge=kw.get("merge", True), pop=p, k_acc=kw.get("k_acc", CR.K_ACC))
    return dict(variant=name, f_V_bounce=o["f_V_bounce"], mass_growth=o.get("mass_density_bounce_over_today"), seconds=time.time() - t)


if __name__ == "__main__":
    r = CC.run(CC.Params(), n_cycles=0)
    c0 = r["cycles"][0]
    hist = sorted([h for h in c0["hist"] if h[2] < 0], key=lambda h: h[1])
    Tb = c0["T_bounce"] * CC.T_UNIT_EV * 1e-6
    with Pool(6, initializer=init, initargs=(hist,)) as pool:
        rows = pool.map(job, VARIANTS)
    for row in rows:
        fv = row["f_V_bounce"]
        row["T_perc_GeV_fc1"] = Tb * fv ** (-1 / 3.0) / 1e3
        row["T_perc_GeV_overlap"] = Tb * (F_C_OVERLAP / fv) ** (1 / 3.0) / 1e3
        print("%-34s f_V %.3e  T_perc %.2f GeV (f_c = 1)  %.2f GeV (overlap)  [%.0fs]" % (row["variant"], fv, row["T_perc_GeV_fc1"],
                                                                                     row["T_perc_GeV_overlap"], row["seconds"]), flush=True)
    allT = [x["T_perc_GeV_fc1"] for x in rows] + [x["T_perc_GeV_overlap"] for x in rows]
    phys = [x for x in rows if "accretion x" not in x["variant"] or "0.1" in x["variant"]]
    by = {x["variant"]: x for x in rows}
    # clustering: supermassive holes trace galaxies; a region overdense by 1 + delta percolates at a bounce temperature lower by (1 + delta)^(-1/3)
    clustered = {d: by["baseline (E3)"]["T_perc_GeV_overlap"] * (1.0 + d) ** (-1 / 3.0) for d in (10.0, 100.0)}
    # joint viability of the held branch (viability_mc.held, same seed) with the bounce window closed at each estimate
    import viability_mc as VM
    held = {}
    for lab, T in (("baseline estimate", by["baseline (E3)"]["T_perc_GeV_fc1"]), ("overlap threshold", by["baseline (E3)"]["T_perc_GeV_overlap"]),
                   ("lowest population variant, overlap", min(x["T_perc_GeV_overlap"] for x in phys)),
                   ("region overdense by 100, overlap", clustered[100.0]), ("accretion x 10, overlap", by["accretion x 10"]["T_perc_GeV_overlap"])):
        VM.T_PERC = 1e3 * T
        VM.rng = np.random.default_rng(20261030)
        h = VM.held()
        held[lab] = dict(T_perc_GeV=T, joint=h["joint"], window=h["single"]["H3 bounce window"])
        print("held-branch joint viability, %-36s T_perc %.2f GeV: %.2f%%" % (lab, T, 100 * h["joint"]))
    json.dump(dict(T_bounce_MeV=Tb, f_c_overlap=F_C_OVERLAP, rows=rows, T_perc_min_GeV=min(allT), T_perc_max_GeV=max(allT),
                   physical_accretion_range_GeV=[min(min(x["T_perc_GeV_fc1"], x["T_perc_GeV_overlap"]) for x in phys),
                                                 max(max(x["T_perc_GeV_fc1"], x["T_perc_GeV_overlap"]) for x in phys)],
                   clustered_overlap_GeV=clustered, held_viability=held),
              open(os.path.join(HERE, "relic_sensitivity.json"), "w"), indent=1)
    print("range %.2f - %.2f GeV" % (min(allT), max(allT)))
