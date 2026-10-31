# -*- coding: utf-8 -*-
"""Numerical admissibility of the continuous cycles: are the results properties of the equations or of the numerical devices?

The baseline (continuous_cycles.Params(), today + 3 cycles) is repeated with each device changed in turn: integration tolerance (rtol 1e-8 -> 1e-10),
the overdamped reduction (used where Upsilon exceeds every other rate by 1e3 -> 1e4), and the constraint projection of H (every chunk -> every 4th
chunk -> never). A result that changes beyond the scatter between cycles under these changes is an artifact; one that does not is admissible.
Without projection a constraint violation is conserved and acts like a cosmological constant (Section 6.6), so that run also measures how large the
spurious term becomes.

    python numerics_admissibility.py           (writes numerics_admissibility.json)
    python numerics_admissibility.py reuse     (keeps the stored no-projection row, which takes about an hour to fail)
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import continuous_cycles as CC

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [("baseline", {}), ("rtol 1e-10", dict(rtol=1e-10)), ("overdamped threshold 1e4", dict(od_ratio=1e4)),
            ("projection every 4th chunk", dict(proj_every=4)), ("no projection", dict(proj_every=10 ** 9))]
KEYS = ("ln_a_max", "t_cycle", "held_accelerated_efolds", "Y_held_over_kappa", "Y_fo_over_kappa", "max_matter_fraction", "lnG", "phi_bounce_over_v", "bounce_x", "T_bounce", "T_release")


def job(item):
    name, kw = item
    t = time.time()
    r = CC.run(CC.Params(), n_cycles=3, keep_history=False, wall=6000, **kw)
    return dict(variant=name, settings=kw, status=r["status"], classification=CC.classify(r), max_residual=r["max_residual"], projections=r["projections"],
                cycles=[dict((k, c.get(k)) for k in ("n",) + KEYS) for c in r["cycles"]], seconds=time.time() - t)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    reuse = len(sys.argv) > 1 and sys.argv[1] == "reuse"
    old = json.load(open(os.path.join(HERE, "numerics_admissibility.json"), encoding="utf-8")) if reuse else None
    todo = [v for v in VARIANTS if not (reuse and v[0] == "no projection")]
    with Pool(len(todo)) as pool:
        rows = pool.map(job, todo)
    if reuse:
        rows += [r for r in old["rows"] if r["variant"] == "no projection"]
    base = rows[0]
    for r in rows:
        dev = {}
        for k in KEYS:
            a = [c.get(k) for c in base["cycles"] if c["n"] >= 1]
            b = [c.get(k) for c in r["cycles"] if c["n"] >= 1]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and x != 0]
            if pairs:
                dev[k] = float(max(abs(y / x - 1) for x, y in pairs))
        r["max_rel_change_vs_baseline"] = dev
        print("%-28s %-10s res %.1e  %s [%.0fs]" % (r["variant"], r["classification"][:10], r["max_residual"],
                                                     {k: "%.2g" % v for k, v in dev.items()}, r["seconds"]), flush=True)
    # scatter between cycles of the baseline itself, for comparison
    cs = [c for c in base["cycles"] if c["n"] >= 1]
    scatter = {k: float(np.std([c[k] for c in cs if c.get(k) is not None]) / abs(np.mean([c[k] for c in cs if c.get(k) is not None])))
               for k in KEYS if sum(c.get(k) is not None for c in cs) > 1}
    json.dump(dict(rows=rows, baseline_cycle_scatter=scatter), open(os.path.join(HERE, "numerics_admissibility.json"), "w"), indent=1)
    print("baseline scatter between cycles:", {k: "%.2g" % v for k, v in scatter.items()}, "[%.0fs]" % (time.time() - t0))
