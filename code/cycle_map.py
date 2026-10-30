# -*- coding: utf-8 -*-
"""The continuous cycle as a return map, and a closed-form account of the held epoch.

(1) Return map. The state at a bounce is fixed by rho = rho_c up to the value at which the field is frozen (the field is overdamped there, its velocity
slaved, and the dark matter is in, or out of, equilibrium according to m A(phi_b)/T_b). The bounce-to-bounce map phi_b,n -> phi_b,n+1 is therefore a
one-dimensional reduction of the cycle. It is probed by starting one cycle from the baseline bounce state with the field set to phi_b (velocity set to
its slaved value by the integrator) and integrating to the next bounce. A fixed point with |slope| < 1 is an attracting cycle; the set of phi_b that
reach it is its basin, and a second fixed point (the empty branch) and the separatrix between the basins are read off the same map.

(2) Closed form for the held epoch. With the field restored to phi_rest, the epoch of accelerated expansion with the field held lasts
    N_held = (1/3) ln(2 V_c / rho_crit)  +  N_grow,     integral over N_grow of lambda(N)/H_L dN = ln(0.1 v / phi_rest),
the first term the e-folds between the onset of acceleration (rho_m = 2 V_c) and the release density (rho_m = rho_crit), the second the e-folds the
tachyonic field needs to grow from phi_rest to 0.1 v with growth rate lambda = (mu_eff^2 + 9H^2/4)^(1/2) - 3H/2, mu_eff^2 = mu^2 (1 - e^(-3N)),
H = H_L = (V_c/3)^(1/2). It predicts how the held e-folds change from cycle to cycle with the depth of the restoration, which the 100-cycle run
records.

    python cycle_map.py      (writes cycle_map.json)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import continuous_cycles as CC
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
PHI = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def bounce_state(m_dm):
    P = CC.Params(m_dm=m_dm)
    r = CC.run(P, n_cycles=0, keep_history=False)
    return r["y_end"], r["aux_end"]


def probe(args):
    m_dm, phi, y_b, aux_b = args
    P = CC.Params(m_dm=m_dm)
    y = list(y_b)
    y[2] = phi * P.mod.v
    y[3] = 0.0
    t = time.time()
    r = CC.run(P, n_cycles=0, y0=y, aux0=aux_b, keep_history=False, wall=1500)
    c = r["cycles"][0] if r["cycles"] else {}
    return dict(m_dm=m_dm, phi_in=phi, phi_out=c.get("phi_bounce_over_v"), Y_fo=c.get("Y_fo_over_kappa"), Y_held=c.get("Y_held_over_kappa"),
                held=c.get("held_accelerated_efolds"), x_next=c.get("bounce_x"), status=r["status"], classification=CC.classify(dict(r, cycles=[dict(c, n=1)])) if c else r["status"],
                seconds=time.time() - t)


def n_held_closed_form(phi_rest, mu=30.0, a_c=2.0):
    VC = MC.VC
    rho_crit = CC.RHO_M0 / a_c ** 3
    HL = np.sqrt(VC / 3.0)
    gate = np.log(2.0 * VC / rho_crit) / 3.0
    target = np.log(0.1 / max(phi_rest, 1e-300))
    lam = lambda N: np.sqrt(mu ** 2 * (1 - np.exp(-3 * N)) + 2.25 * HL ** 2) - 1.5 * HL
    f = lambda Ng: quad(lambda N: lam(N) / HL, 0, Ng)[0] - target
    grow = brentq(f, 1e-6, 50.0) if target > 0 else 0.0
    return gate + grow, gate, grow




def main_map(out):
    t0 = time.time()

    jobs = []
    for m in (5.5e9, 5.5e10):
        y_b, aux_b = bounce_state(m)
        out["map"][str(m)] = dict(bounce_state_phi_over_v=float(y_b[2] / CC.Params(m_dm=m).mod.v))
        jobs += [(m, phi, y_b, aux_b) for phi in PHI]
        print("bounce state for m = %.3g ready [%.0fs]" % (m, time.time() - t0), flush=True)
    with Pool(4) as pool:
        rows = []
        for row in pool.imap_unordered(probe, jobs):
            rows.append(row)
            print("m %.3g phi_in %.2f -> phi_out %s held %s Y_held %s %s [%.0fs]" % (row["m_dm"], row["phi_in"], row["phi_out"], row["held"], row["Y_held"],
                                                                                 row["classification"][:20], time.time() - t0), flush=True)
    for m in (5.5e9, 5.5e10):
        rr = sorted([r for r in rows if r["m_dm"] == m], key=lambda r: r["phi_in"])
        out["map"][str(m)]["rows"] = rr
        # slope of the map near the working fixed point (finite difference over the probes in the working basin)
        w = [r for r in rr if r["classification"] == "working" and r["phi_out"] is not None]
        if len(w) >= 2:
            x = np.array([r["phi_in"] for r in w]); yv = np.array([r["phi_out"] for r in w])
            out["map"][str(m)]["working_basin"] = [float(x.min()), float(x.max())]
            out["map"][str(m)]["slope_in_basin"] = float(np.polyfit(x, yv, 1)[0])
            out["map"][str(m)]["phi_out_spread_in_basin"] = float(yv.max() - yv.min())


def fixed_points(out):
    """Fixed point phi* (linear interpolation of phi_out - phi_in through zero) and the local slope of the map there."""
    for key, v in out["map"].items():
        rr = [r for r in sorted(v.get("rows", []), key=lambda r: r["phi_in"]) if r.get("phi_out") is not None]
        g = [(r["phi_in"], r["phi_out"] - r["phi_in"], r["phi_out"]) for r in rr]
        for (x0, d0, y0), (x1, d1, y1) in zip(g, g[1:]):
            if d0 > 0 >= d1:
                xs = x0 + (x1 - x0) * d0 / (d0 - d1)
                v["fixed_point"] = float(xs)
                v["slope_at_fixed_point"] = float((y1 - y0) / (x1 - x0))
                break
        v["all_working"] = all(r["classification"] == "working" for r in rr)


def held_part(out):
    # held epoch: closed form vs the 100-cycle run
    R = json.load(open(os.path.join(HERE, "continuous_cycles.json"), encoding="utf-8"))
    pts = []
    for c in R["cycles"]:
        if c["n"] < 1 or not c.get("hist") or "held_accelerated_efolds" not in c:
            continue
        h = np.array(c["hist"])
        exp_ = h[:, 2] > 0
        first_neg = np.where(~exp_)[0]
        end = first_neg[0] if first_neg.size else len(h)
        seg = h[:end]
        rel = np.where(seg[:, 9] > 0.5)[0]
        rel = rel[rel > 3]
        seg = seg[: rel[0]] if rel.size else seg
        # restored level: envelope (max |phi|) of the held plateau just before the release, where rho_m lies between rho_crit and 8 rho_crit
        rho_crit = CC.RHO_M0 / R["params"]["a_c"] ** 3
        late = seg[(seg[:, 5] > rho_crit) & (seg[:, 5] < 8 * rho_crit) & (seg[:, 10] > 1e3)]
        if not late.size:
            continue
        phi_rest = float(late[:, 9].max()) if (late[:, 9] > 0).any() else None
        if not phi_rest:
            continue
        pred, gate, grow = n_held_closed_form(phi_rest, R["params"]["mu"], R["params"]["a_c"])
        pts.append(dict(n=c["n"], phi_rest=phi_rest, held=c["held_accelerated_efolds"], predicted=pred, gate=gate, grow=grow))
    if pts:
        e = np.array([p["held"] - p["predicted"] for p in pts])
        lp = np.log([p["phi_rest"] for p in pts]); hv = np.array([p["held"] for p in pts])
        out["held"] = dict(points=pts, mean_error=float(e.mean()), rms_error=float(np.sqrt((e ** 2).mean())), max_abs_error=float(np.abs(e).max()),
                           fitted_slope_dN_dlnphi=float(np.polyfit(lp, hv, 1)[0]) if len(pts) > 2 else None,
                           predicted_slope_asymptotic=float(-np.sqrt(MC.VC / 3.0) / R["params"]["mu"]), n_points=len(pts))
        print("held epoch: %d cycles, mean error %.3f, rms %.3f e-folds; slope fit %s vs asymptotic %.4f" % (
            len(pts), out["held"]["mean_error"], out["held"]["rms_error"], out["held"]["fitted_slope_dN_dlnphi"], out["held"]["predicted_slope_asymptotic"]))


if __name__ == "__main__":
    t0 = time.time()
    f = os.path.join(HERE, "cycle_map.json")
    out = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else dict(map={}, held={})
    if len(sys.argv) > 1 and sys.argv[1] == "fixed":
        pass
    elif len(sys.argv) == 1 or sys.argv[1] == "map":
        main_map(out)
        json.dump(out, open(f, "w"), indent=1)
    fixed_points(out)
    json.dump(out, open(f, "w"), indent=1)
    if len(sys.argv) == 1 or sys.argv[1] == "held":
        held_part(out)
        json.dump(out, open(f, "w"), indent=1)
    print("[%.0fs]" % (time.time() - t0))
