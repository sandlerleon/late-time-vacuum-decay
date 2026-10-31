# -*- coding: utf-8 -*-
"""Independent check of the continuous cycles with a constraint-preserving formulation of the gravitational equations.

The baseline (continuous_cycles.Params(), today + 3 cycles) uses H as a variable evolved with the Raychaudhuri equation and projected onto the
Friedmann constraint at the start of each chunk. Here the same matter equations are integrated with H eliminated by the constraint away from the
turning points (continuous_cycles.run(gravity='constrained')), so no projection is applied; close to the turnaround and the bounce, where the sign of
H changes, H is evolved across a short Raychaudhuri bridge and the constraint mismatch accumulated there is recorded. Two bridge widths and a tighter
tolerance are run. Writes independent_solver.json.

    python independent_solver.py
"""
import json
import os
import time
from multiprocessing import Pool

import continuous_cycles as CC

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [("projected (baseline)", {}), ("constrained, bridge 0.05", dict(gravity="constrained", near=0.05)),
            ("constrained, bridge 0.02", dict(gravity="constrained", near=0.02)),
            ("constrained, bridge 0.05, rtol 1e-10", dict(gravity="constrained", near=0.05, rtol=1e-10))]
KEYS = ("ln_a_max", "t_cycle", "T_turn", "held_accelerated_efolds", "Y_held_over_kappa", "Y_fo_over_kappa", "lnG", "phi_bounce_over_v", "T_bounce",
        "T_release")


def job(item):
    name, kw = item
    t = time.time()
    r = CC.run(CC.Params(), n_cycles=3, keep_history=False, wall=6000, **kw)
    return dict(variant=name, settings=kw, status=r["status"], classification=CC.classify(r), max_residual=r["max_residual"], projections=r["projections"],
                cycles=[dict((k, c.get(k)) for k in ("n",) + KEYS) for c in r["cycles"]], seconds=time.time() - t)


if __name__ == "__main__":
    with Pool(len(VARIANTS)) as pool:
        rows = pool.map(job, VARIANTS)
    base = rows[0]
    for r in rows:
        dev = {}
        for k in KEYS:
            pairs = [(x.get(k), y.get(k)) for x, y in zip(base["cycles"], r["cycles"]) if x["n"] >= 1]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None and x != 0]
            if pairs:
                dev[k] = float(max(abs(y / x - 1) for x, y in pairs))
        r["max_rel_change_vs_projected"] = dev
        print("%-38s %-10s res %.1e bridges %s %s [%.0fs]" % (r["variant"], r["classification"][:10], r["max_residual"], r["projections"],
                                                              {k: "%.2g" % v for k, v in dev.items()}, r["seconds"]), flush=True)
    json.dump(dict(rows=rows), open(os.path.join(HERE, "independent_solver.json"), "w"), indent=1)
