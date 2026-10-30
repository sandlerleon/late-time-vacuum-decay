# -*- coding: utf-8 -*-
"""Section 7: numerical convergence, conservation and analytic-limit tests of the interface calculations.

    python interface_validation.py          (writes interface_validation.json; run after vacuum_wall, interface_transport, bubble_gravity,
                                             bubble_to_cycle, bubble_entropy)
"""
import json
import os

import numpy as np
from scipy.integrate import quad

import bubble_gravity as BG
import interface_transport as IT
import thermal as TH
import vacuum_phase_map as VP
import vacuum_wall as W

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
checks = []


def chk(name, ok, **info):
    checks.append(dict(name=name, passed=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int)) and not isinstance(v, bool) else v)
                                                       for k, v in info.items()}))
    print(("PASS  " if ok else "FAIL  ") + name, info)


# 1. phase structure: analytic degeneracy theta_deg = eta/2 (p becomes even in x) and the cubic window
for eta in (0.2, 0.3, 0.36, 0.45):
    th = VP.theta_deg_tb(eta)
    chk("degeneracy density theta_deg = eta/2 (eta = %.2f)" % eta, abs(th - eta / 2.0) < 1e-10, theta_deg=th)
lo, hi = VP.window_tb(0.45)
d = lambda t: -(4 * (2 * t - 1) ** 3 + 27 * (0.45 - 2 * t) ** 2)
chk("metastable window edges are roots of the cubic discriminant", abs(d(lo)) < 1e-9 and abs(d(hi)) < 1e-9, theta_low=lo, theta_high=hi)
PM = J("vacuum_phase_map.json")
chk("symmetron: the occupied minimum is never metastable (no tunneling)", not any(v["occupied_ever_metastable"] for v in PM["symmetron"].values()))
cc = PM["coupling_check"]
chk("coupling check: exact / logarithmic = (e^d - 1)/d", all(abs(c["exact_over_log"] - np.expm1(c["delta_ln_m"]) / c["delta_ln_m"]) < 1e-12 for c in cc),
    max_ratio=max(c["exact_over_log"] for c in cc))

# 2. walls: the symmetric kink (eta = theta = 0): sigma = 2/3, profile tanh(z/2), 10-90 % thickness 4 artanh(0.8)
sig = quad(lambda x: np.sqrt(2.0 * VP.p_tb(x, 0.0, 0.0)), -1.0, 1.0)[0]
chk("kink tension 2/3", abs(sig - 2.0 / 3.0) < 1e-10, sigma=sig)
zs = [quad(lambda x: 1.0 / np.sqrt(2.0 * VP.p_tb(x, 0.0, 0.0)), 0.0, xx)[0] for xx in (-0.8, 0.8)]
chk("kink thickness 4 artanh(0.8)", abs((zs[1] - zs[0]) - 4.0 * np.arctanh(0.8)) < 1e-8, thickness=zs[1] - zs[0])

# 3. bounce actions: against the independent table of density_bounce.json at theta = 0, thin-wall limit, shooting vs collocation
WJ, DBJ = J("vacuum_wall.json"), J("density_bounce.json")
for e in ("0.20", "0.30", "0.36"):
    b_new = WJ["walls"][e]["rows"][0]["b"]
    b_old = DBJ["b_table"][e]["rows"][0]["b"]
    chk("bounce action at theta = 0 reproduces density_bounce (eta %s)" % e, abs(b_new / b_old - 1) < 1e-6, b=b_new, ref=b_old)
for e in ("0.20", "0.30"):
    late = [r for r in WJ["walls"][e]["rows"] if r["theta_over_deg"] >= 0.6 - 1e-9]
    chk("thin-wall limit: b / B_tw within 1 %% for theta >= 0.6 theta_deg (eta %s)" % e, all(abs(r["b_over_B_tw"] - 1) < 0.01 for r in late),
        worst=max(abs(r["b_over_B_tw"] - 1) for r in late))
eta, th = 0.30, 0.6 * 0.15
n_, x_t, x_top, x_f, dV, _ = VP.classify_tb(eta, th)
v = lambda x: VP.p_tb(x, eta, th)
dv = lambda x: VP.dp_tb(x, eta, th)
b_sh = TH.bounce_shoot(v, dv, x_t, x_top, x_f, 4)[0]
tw = W.thin_wall(eta, th)
b_bvp = TH.bounce_bvp(v, dv, x_t, x_f, 4, tw["sigma_tw"], dV, m_f=np.sqrt(VP.d2p_tb(x_f, eta, th)))
chk("shooting and collocation agree (eta 0.30, 0.6 theta_deg)", abs(b_sh / b_bvp - 1) < 2e-3, shooting=b_sh, collocation=b_bvp)

# 4. transport: Poeschl-Teller, unitarity, sharp step, tolerance convergence
TJ = J("interface_transport.json")
chk("transmission reproduces the Poeschl-Teller result", TJ["max_PT_error"] < 1e-4, max_error=TJ["max_PT_error"])
chk("unitarity T + R = 1", TJ["max_unitarity_error"] < 1e-8, max_error=TJ["max_unitarity_error"])
m1, m2 = 0.5, 1.0
step = lambda z: m1 ** 2 + (m2 ** 2 - m1 ** 2) * 0.5 * (1 + np.tanh(z / 1e-3))
errs = []
for k2 in (1.2, 2.0, 5.0):
    T, _ = IT.transmission(step, k2, -5.0, 5.0)
    ka, kb = np.sqrt(k2 - m1 ** 2), np.sqrt(k2 - m2 ** 2)
    errs.append(abs(T - 4 * ka * kb / (ka + kb) ** 2))
chk("sharp mass step: T = 4 k1 k2 / (k1 + k2)^2", max(errs) < 2e-3, max_error=max(errs))
kp = IT.kink_profile()
m2f = lambda z: 1.0 * kp(z) ** 2
Ta = IT.transmission(m2f, 1.3, -40, 40, rtol=1e-8)[0]
Tb = IT.transmission(m2f, 1.3, -40, 40, rtol=1e-11)[0]
chk("transmission converged in the integration tolerance", abs(Ta - Tb) < 1e-6, difference=abs(Ta - Tb))

# 5. gravity: junction in the flat limit, wall energy = deficit, gravitational parameter
GJ = J("bubble_gravity.json")
chk("junction reduces to the flat-space runaway Gamma = R/R_0", GJ["wall"]["max_rel_dev_from_flat"] < 1e-10, deviation=GJ["wall"]["max_rel_dev_from_flat"])
EJ = J("bubble_entropy.json")
chk("wall energy equals the enclosed vacuum-energy deficit", EJ["wall_energy_closure_max_dev"] < 1e-12, deviation=EJ["wall_energy_closure_max_dev"])
chk("interior bookkeeping dE_in = F_E dV + W_wall closes", EJ["interior_bookkeeping_max_rel_residual"] < 1e-12, residual=EJ["interior_bookkeeping_max_rel_residual"])
chk("wall self-gravity negligible (sigma R / 2 Gamma < 1e-50)", GJ["wall"]["max_grav"] < 1e-50, value=GJ["wall"]["max_grav"])

# 6. daughter regions: constraint at events, bounce density, entropy, self-similarity
CJ = J("bubble_to_cycle.json")
for key in ("B percolated", "B isolated"):
    ev = CJ[key]["events"]
    rel = max(abs(e["H2_constraint"]) / (max(abs(e["rho"]), e["rho_r"]) / 3.0) for e in ev if e["event"] in ("turnaround", "bounce"))
    chk("%s: H^2 = 0 at turnaround and bounce (relative)" % key, rel < 1e-6, worst=rel)
    rb = [e["rho"] for e in ev if e["event"] == "bounce"]
    chk("%s: bounce at rho_c" % key, all(abs(r / 1e44 - 1) < 1e-6 for r in rb))
    nj = [e.get("H_jump", 0.0) for e in ev if e["event"] == "nucleation"]
    chk("%s: Hubble rate continuous across nucleation" % key, max(abs(x) for x in nj) < 1e-10, worst=max(abs(x) for x in nj))
    lg = CJ[key]["lnG"]
    chk("%s: self-similar cycles (ln G constant to 1e-3)" % key, max(lg) - min(lg) < 1e-3, lnG=float(np.mean(lg)))
for key, e in EJ["entropy"].items():
    chk("%s: comoving entropy never decreases" % key, e["nondecreasing"])
chk("C equals B (computed selectivity changes nothing)", CJ["C percolated (computed selectivity)"]["lnG"] == CJ["B percolated"]["lnG"])
chk("C* fails without re-thermalization", CJ["C* percolated, fully selective, cold bounce"]["failure"] is not None)

out = dict(checks=checks, all_passed=all(c["passed"] for c in checks), n=len(checks))
json.dump(out, open(os.path.join(HERE, "interface_validation.json"), "w"), indent=1)
print("%d/%d passed" % (sum(c["passed"] for c in checks), len(checks)))
