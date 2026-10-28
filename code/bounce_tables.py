# -*- coding: utf-8 -*-
"""Numerical bounce tables.

  (1) flat-space bounce action b(eta) = B * lam for the potential family of cdl_lib (B = b(eta)/lam exactly, by scaling), with the
      thin-wall value and an independent boundary-value solution as checks;
  (2) the gravitational suppression B(g)/B_flat over a grid of (g, eta) at lam = 1, a Minkowski true vacuum, together with the
      Hawking-Moss criterion |V''_top| / (4 H^2) and the Hawking-Moss action, which delimit where an oscillating bounce exists.

    python bounce_tables.py          (about 10 minutes; writes bounce_tables.json)
"""
import json
import os
import time

import numpy as np

import cdl_lib as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ----------------------------------------------------------------------------------------------------- (1) b(eta)
eta_bvp = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
eta_sh = sorted(set(list(np.round(np.arange(0.10, 0.371, 0.01), 3)) + [0.335, 0.345, 0.355, 0.365, 0.372, 0.375, 0.378, 0.38]))
rows = []
t0 = time.time()
for eta in eta_bvp + eta_sh:
    thin = C.thin_wall_B(1.0, eta)
    if eta < 0.10:
        r = C.flat_bounce_bvp(1.0, eta)
        B, x0, method = r["B"], r["x0"], "bvp"
    else:
        r = C.flat_bounce(1.0, eta)
        B, x0, method = r["B"], r["x0"], "shooting"
    pot = C.Potential(1.0, eta)
    cross = None
    if 0.10 <= eta <= 0.20:                    # independent check where both methods work
        rb = C.flat_bounce_bvp(1.0, eta)
        cross = rb["B"] if rb.get("success") else None
    rows.append(dict(eta=float(eta), b=float(B), thin=float(thin), ratio=float(B / thin), x0=float(x0), method=method, bvp_check=cross,
                     dtil=float(pot.dtil), btil=float(pot.btil), x_t=float(pot.x_t), x_top=float(pot.x_top), x_f=float(pot.x_f),
                     m2_top=float(pot.m2_top), m2_f=float(pot.m2_f), v_f=float(pot.v_f), v_top=float(pot.v_top),
                     v_x0=float(pot.v(x0))))
    print("eta %.3f  b %.4f  thin %.4f  ratio %.4f  [%s]  %.0fs" % (eta, B, thin, B / thin, method, time.time() - t0))
OUT["flat"] = rows

# scaling check: B(lam, eta) = b(eta)/lam
chk = []
for lam in (0.3, 3.0):
    for eta in (0.2, 0.3):
        B = C.flat_bounce(lam, eta)["B"]
        b = [r["b"] for r in rows if abs(r["eta"] - eta) < 1e-9][0]
        chk.append(dict(lam=lam, eta=eta, B=float(B), b_over_lam=float(b / lam), rel=float(abs(B * lam / b - 1))))
OUT["scaling_check"] = chk

# --------------------------------------------------------------------------------- (2) gravitational suppression map
gs = list(np.logspace(-3, 0.3, 14))
etas = [0.12, 0.15, 0.20, 0.25, 0.30, 0.34, 0.37]
grid = []
for eta in etas:
    Bf = [r["b"] for r in rows if abs(r["eta"] - eta) < 1e-9]
    Bf = Bf[0] if Bf else C.flat_bounce(1.0, eta)["B"]
    pot = C.Potential(1.0, eta)
    for g in gs:
        H2 = g * pot.v_f / 3.0
        hm_ratio = abs(pot.m2_top) / (4.0 * H2)
        res = C.grav_bounce(1.0, eta, g)
        if res is None:
            B, found = None, False
        else:
            B, found = float(C.grav_action(res, g)), True
        grid.append(dict(eta=eta, g=float(g), B=B, found=found, ratio=(B / Bf if found else None), hm_ratio=float(hm_ratio),
                         B_hm=float(C.hm_action(1.0, eta, g)), B_flat=float(Bf)))
    print("grid eta", eta, "done  %.0fs" % (time.time() - t0))
OUT["grid"] = grid

# small-g slope kappa(eta): B(g) = B_flat * (1 - kappa g + ...), measured at g = 1e-3 and 2e-3 (below that the 1/g^2 cancellation in the action
# limits the numerical precision to about 1e-4 in B/B_flat)
slopes = []
for eta in etas:
    Bf = [r["b"] for r in rows if abs(r["eta"] - eta) < 1e-9]
    Bf = Bf[0] if Bf else C.flat_bounce(1.0, eta)["B"]
    vals = []
    for g in (1e-3, 2e-3):
        res = C.grav_bounce(1.0, eta, g)
        vals.append(float(C.grav_action(res, g) / Bf - 1.0))
    slopes.append(dict(eta=eta, kappa=float(-vals[0] / 1e-3), kappa2=float(-vals[1] / 2e-3)))
OUT["slopes"] = slopes

json.dump(OUT, open(os.path.join(HERE, "bounce_tables.json"), "w"), indent=1)
print("saved bounce_tables.json  (%.0fs)" % (time.time() - t0))
