# -*- coding: utf-8 -*-
"""Sections 7.6-7.7: can the region enclosed by the vacuum interface contract, bounce and reheat, and which models need what?

Controls (one row each in Table 20):
  A   the homogeneous held vacuum of Section 6.8 (symmetron form; classical release, no interface) - from continuous_cycles.json
  B   density-dependent bubble nucleation of the tunneling branch, every species transmitted (bubble_gravity.Daughter, percolated and isolated)
  C   as B with the selective transport computed in interface_transport.py (dark matter reflected while the wall is slower than gamma_th)
  C*  as B with a fully selective wall (all dark matter kept out of the true-vacuum region), with and without re-thermalization at the bounce
  D   matter coupling switched off: constant decay rate, no restoration
Viability map (Fig. 27a): over the action scale B0 and the present density theta_0 / theta_deg of the baseline eta, each point is classified as
  already decayed (the past light cone would contain a bubble: ln I_past > 0), no completion (Gamma / H^4 never exceeds 9/(4 pi): isolated bubbles in
  an eternally inflating false vacuum, whose interiors are empty), or a completed transition followed by recollapse, restoration and bounce.
Fig. 27b: depth q of the true vacuum (contraction needs q > 0) against the bounce temperature relative to the dark-matter mass (re-thermalization).

    python bubble_to_cycle.py          (writes bubble_to_cycle.json, ~2 min)
"""
import json
import os
import time

import numpy as np

import bubble_gravity as BG
import density_bounce as DB
import matter_coupling as MC
import thermal as TH
import vacuum_phase_map as VP
import vacuum_wall as W
import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))


def lnG_of(events):
    nuc = [e for e in events if e["event"] == "nucleation"]
    return [float(nuc[i + 1]["lna"] - nuc[i]["lna"]) for i in range(len(nuc) - 1)]


def summary(r):
    ev = r["events"]
    b = [e for e in ev if e["event"] == "bounce"]
    fail = [e["event"] for e in ev if e["event"].startswith("no ")]
    return dict(n_bounces=len(b), T_b_MeV=b[-1]["T_b_MeV"] if b else None, lnG=lnG_of(ev), failure=fail[0] if fail else None,
                restorations=[e["event"] for e in ev if e["event"].startswith("restoration")])


def rate_tables(WJ, eta="0.30"):
    T = WJ["walls"][eta]
    good = [r for r in T["rows"] if not r.get("failed")]
    fr = np.array([r["theta_over_deg"] for r in good])
    return T["theta_deg"], fr, np.log([r["b"] for r in good]), np.log([r["dV"] for r in good]), good[0]["b"]


def classify_point(B0, f0, q, tabs, z_lc, w_lc):
    th_deg, fr, lb, ld, b0 = tabs
    lam = b0 / B0

    def lr(f, H_over_HL=1.0):
        if f > fr[-1]:
            return -np.inf
        return W.ln_gamma_h4(lam, np.exp(np.interp(f, fr, lb)), np.exp(np.interp(f, fr, ld)), q, H_over_HL)
    lrs = np.array([lr(f0 * (1 + z) ** 3) for z in z_lc])
    with np.errstate(divide="ignore"):
        lnI = float(np.logaddexp.reduce(lrs + 2.0 * np.log(K.OMEGA_L) + np.log(w_lc)))
    a_trig = None
    for a in np.logspace(0.0, 3.0, 300):
        if lr(f0 / a ** 3, W.hubble_over_HL(a)) >= W.LN_COMPLETE:
            a_trig = float(a)
            break
    code = 0 if lnI >= 0 else (1 if a_trig is None else 2)
    return code, lnI, a_trig


if __name__ == "__main__":
    t0 = time.time()
    WJ = json.load(open(os.path.join(HERE, "vacuum_wall.json"), encoding="utf-8"))
    TJ = json.load(open(os.path.join(HERE, "interface_transport.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "interface_transport.json")) else None
    CR = json.load(open(os.path.join(HERE, "continuous_cycles.json"), encoding="utf-8"))
    base, TW = WJ["base"], WJ["trigger_wall"]
    th_deg = WJ["walls"]["0.30"]["theta_deg"]
    out = {}
    # ---- controls
    lnGA = [c["lnG"] for c in CR["cycles"] if c.get("lnG") is not None and c["n"] >= 1]
    out["A"] = dict(model="homogeneous held vacuum (Section 6.8)", lnG_mean=float(np.mean(lnGA)), lnG_spread=float(np.std(lnGA)), cycles=len(lnGA),
                    release="classical (spinodal) roll", needs="coherent seed or tilt (Section 7.1)")
    tabs = rate_tables(WJ)
    lam0 = tabs[4] / base["B0"]

    def lnrate(th, tabs=tabs):
        f = th / tabs[0]
        if f > tabs[1][-1]:
            return -np.inf
        return W.ln_gamma_h4(lam0, np.exp(np.interp(f, tabs[1], tabs[2])), np.exp(np.interp(f, tabs[1], tabs[3])), base["q"])
    D = BG.Daughter(base["eta"], base["f0"] * th_deg, base["q"], TW["theta"], lnrate=lnrate)
    f_filt = TJ["dark_matter"]["filtered_fraction"] if TJ else 0.0
    runs = {"B percolated": dict(mode="percolated"), "B isolated": dict(mode="isolated"),
            "B percolated, no re-thermalization": dict(mode="percolated", regen_allowed=False),
            "C percolated (computed selectivity)": dict(mode="percolated", filter_frac=f_filt),
            "C* percolated, fully selective, hot bounce": dict(mode="percolated", filter_frac=1.0),
            "C* percolated, fully selective, cold bounce": dict(mode="percolated", filter_frac=1.0, regen_allowed=False)}
    for k, kw in runs.items():
        r = D.run(n_cycles=4, **kw)
        out[k] = dict(summary(r), events=r["events"])
        print("%-48s %s [%.0fs]" % (k, {kk: vv for kk, vv in out[k].items() if kk != "events"}, time.time() - t0), flush=True)
    out["filtered_fraction_used"] = f_filt
    # ---- D: no matter coupling (constant rate): window in B0 where the past survives and the transition completes
    z_lc, w_lc = TH.light_cone_grid()
    b0, d0 = tabs[4], np.exp(tabs[3][0])
    Dw = []
    for B0 in np.arange(270.0, 300.01, 0.25):
        lam = b0 / B0
        lr_vac = W.ln_gamma_h4(lam, b0, d0, base["q"])
        lnI0 = float(lr_vac + 2.0 * np.log(K.OMEGA_L) + np.log(np.sum(w_lc)))
        Dw.append(dict(B0=float(B0), lnI=lnI0, ln_gamma_h4_vacuum=lr_vac, survives=lnI0 < 0, completes=lr_vac >= W.LN_COMPLETE))
    ok = [d["B0"] for d in Dw if d["survives"] and d["completes"]]
    out["D"] = dict(model="no matter coupling", window_B0=[min(ok), max(ok)] if ok else None, width=(max(ok) - min(ok)) if ok else 0.0, scan=Dw,
                    restoration=False)
    print("D window", out["D"]["window_B0"], flush=True)
    # ---- viability map over (B0, theta_0/theta_deg)
    B0s = np.arange(100.0, 460.01, 10.0)
    f0s = np.logspace(-2.0, np.log10(3.0), 31)
    code = np.zeros((f0s.size, B0s.size), dtype=int)
    atr = np.full((f0s.size, B0s.size), np.nan)
    for i, f0 in enumerate(f0s):
        for j, B0 in enumerate(B0s):
            c, lnI, a_t = classify_point(B0, f0, base["q"], tabs, z_lc, w_lc)
            code[i, j] = c
            if a_t:
                atr[i, j] = a_t
    out["map"] = dict(B0=B0s.tolist(), f0=f0s.tolist(), code=code.tolist(), a_trigger=np.where(np.isnan(atr), None, atr).tolist(),
                      fraction_viable=float((code == 2).mean()))
    rangeB = [float(B0s[j]) for j in range(B0s.size) if (code[:, j] == 2).any()]
    out["map"]["B0_range_viable"] = [min(rangeB), max(rangeB)] if rangeB else None
    print("map: viable fraction %.2f, B0 range %s [%.0fs]" % (out["map"]["fraction_viable"], out["map"]["B0_range_viable"], time.time() - t0))
    # ---- q and re-thermalization (Fig. 27b): analytic boundaries from the runs above
    out["q_rethermalization"] = dict(contraction_requires="q > 0 (V_t < 0)", B_requires_rethermalization=False,
                                     Cstar_requires_rethermalization=out["C* percolated, fully selective, cold bounce"]["failure"] is not None,
                                     threshold_T_over_m=float(1.0 / 5.0))
    json.dump(out, open(os.path.join(HERE, "bubble_to_cycle.json"), "w"), indent=1)
    print("saved [%.0fs]" % (time.time() - t0))
