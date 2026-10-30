# -*- coding: utf-8 -*-
"""Contraction stability: Bianchi I shear in the contracting phase, with and without an ekpyrotic phase.

Shear. For Bianchi I the shear enters the Friedmann equation as a stiff density rho_sigma = Sigma/a^6 (Sigma conserved). In a contraction from the
turnaround to the bounce it grows relative to radiation as a^-2 and relative to matter as a^-3, so a bounce at rho_c needs an initial anisotropy
below roughly (rho_r/rho_c)^(1/2) of the radiation density at the turnaround.

Ekpyrotic phase. A second scalar chi with V_ek = -V_e exp(-c chi) (c^2 > 6, w = c^2/3 - 1 > 1, rho_chi ~ a^-(c^2)) dominates the contraction and
dilutes the shear as a^(c^2 - 6). Its own dynamics: chi'' + (3H + Gamma(chi)) chi' + V_ek'(chi) = 0, with the potential switched off below
chi_end (V_ek -> V_ek S(chi), S = 1/(1 + exp(-(chi - chi_end)/delta))), where |V_ek| reaches rho_end, and decay into radiation after the exit,
Gamma = Gamma_0 (1 - S), d rho_r/dt = -4H rho_r + Gamma chi'^2 (the specified interaction). chi starts on the ekpyrotic scaling solution
(K/|V| = c^2/(c^2 - 6)) with a fraction f_0 of the matter density at the turnaround.

Background: contraction from a turnaround with matter rho_m, radiation r_t rho_m and a negative vacuum -q V_c (rho = 0 at the turnaround fixes q V_c),
effective LQC Friedmann equation H^2 = rho (1 - rho/rho_c)/3 (H < 0). Integrated in ln a down to rho = rho_c/2; the remaining stretch to the bounce
(rho = rho_c) is evolved analytically (radiation a^-4, stiff components a^-6, matter a^-3; the ekpyrotic potential is off and chi has decayed by then).

The roadmap asks whether rho_sigma/rho_total at the bounce stays controlled over a range of initial anisotropies; the scan below gives the largest
initial shear fraction s_max (rho_sigma/rho_m at the turnaround) for which the bounce shear fraction stays below 0.1, with and without ekpyrosis,
and the analytic cycle map for the shear produced by the entropy growth of the cycles (continuous_cycles.py).

    python contraction_stability.py      (writes contraction_stability.json)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DEBUG = False
HERE = os.path.dirname(os.path.abspath(__file__))


def contract(s, rho_c=1e40, r_t=1e-4, rho_m=2.0, ek=None):
    """s = rho_sigma/rho_m at the turnaround (a = 1). ek = dict(c, f0, rho_end_over_rho_c, gamma_fac) or None."""
    Sig = s * rho_m
    rr0 = r_t * rho_m
    if ek:
        c = ek["c"]
        alpha = c * c / (c * c - 6.0)
        rho_chi0 = ek["f0"] * rho_m
        Ve = rho_chi0 * (c * c - 6.0) / 6.0                      # |V| at chi = 0 on the scaling solution
        K0 = alpha * Ve
        rho_end = ek["rho_end_over_rho_c"] * rho_c
        chi_end = -np.log(rho_end / Ve) / c
        delta = 0.05 / c
        G0 = ek["gamma_fac"] * np.sqrt(rho_end / 3.0)
    else:
        c, Ve, K0, chi_end, delta, G0, rho_chi0 = 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    neg_vac = rho_m + rr0 + Sig + rho_chi0                       # rho = 0 at the turnaround

    def S(chi):
        z = (chi - chi_end) / delta
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

    def V(chi):
        return -Ve * np.exp(-c * chi) * S(chi) if Ve else 0.0

    def dV(chi):
        if not Ve:
            return 0.0
        s_ = S(chi)
        return -Ve * np.exp(-c * chi) * (-c * s_ + s_ * (1 - s_) / delta)

    def comps(la, y):
        chi, pi, rr = y
        a = np.exp(la)
        return dict(m=rho_m / a ** 3, r=rr, sig=Sig / a ** 6, K=0.5 * pi * pi, V=V(chi), vac=-neg_vac)

    def Hof(la, y):
        d = comps(la, y)
        rho = sum(d.values())
        return -np.sqrt(max(rho * (1 - rho / rho_c), 0.0) / 3.0), rho, d

    def rhs(la, y):
        chi, pi, rr = y
        H, rho, d = Hof(la, y)
        G = G0 * (1.0 - S(chi)) if Ve else 0.0
        return [pi / H, (-(3 * H + G) * pi - dV(chi)) / H, (-4 * H * rr + G * pi * pi) / H]

    def stop(la, y):
        return Hof(la, y)[1] - 0.5 * rho_c
    stop.terminal = True

    y0 = [0.0, -np.sqrt(2.0 * K0), rr0]
    # start slightly after the turnaround (H = 0 exactly is a fixed point of the ln a parametrization)
    la0 = -1e-6
    sol = solve_ivp(rhs, (la0, -200.0), y0, method="LSODA", rtol=1e-9, atol=[1e-12, 1e-12, 1e-30], events=[stop])
    if DEBUG:
        print(sol.status, sol.message, sol.t[-1], sol.y[:, -1], Hof(sol.t[-1], sol.y[:, -1])[1])
    la_s = sol.t_events[0][0] if sol.t_events[0].size else sol.t[-1]
    ys = sol.y_events[0][0] if sol.t_events[0].size else sol.y[:, -1]
    d = comps(la_s, ys)
    ekp_on = bool(Ve) and abs(V(ys[0])) > 1e-3 * (0.5 * rho_c)
    # analytic stretch to the bounce
    stiff = d["sig"] + d["K"]
    f = lambda x: d["r"] * np.exp(4 * x) + stiff * np.exp(6 * x) + d["m"] * np.exp(3 * x) + d["vac"] + d["V"] - rho_c
    x = brentq(f, 0.0, 5.0)
    rb = dict(r=d["r"] * np.exp(4 * x), sig=d["sig"] * np.exp(6 * x), K=d["K"] * np.exp(6 * x), m=d["m"] * np.exp(3 * x))
    tot = sum(rb.values())
    # e-folds of ekpyrotic domination (|V| + K > half the total)
    n_ek = 0.0
    if Ve:
        las = sol.t
        dom = []
        for i in range(las.size):
            di = comps(las[i], sol.y[:, i])
            pos = di["m"] + di["r"] + di["sig"] + di["K"]
            dom.append((di["K"] + di["V"]) > 0.5 * pos and S(sol.y[0, i]) > 0.5)
        dom = np.array(dom)
        n_ek = float(np.sum(np.where(dom[:-1], -np.diff(las), 0.0)))
    return dict(s=s, bounce_shear_fraction=float(rb["sig"] / tot), bounce_radiation_fraction=float(rb["r"] / tot), bounce_kinetic_fraction=float(rb["K"] / tot),
                ln_a_bounce=float(la_s - x), ekpyrosis_active_at_stop=bool(ekp_on), ekpyrotic_efolds=n_ek)


def s_max(**kw):
    """Largest initial shear fraction with bounce shear fraction < 0.1 (bisection in log10 s, 0.003 dex)."""
    lo, hi = -60.0, 0.0
    if contract(10 ** lo, **kw)["bounce_shear_fraction"] > 0.1:
        return 0.0
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        if contract(10 ** mid, **kw)["bounce_shear_fraction"] < 0.1:
            lo = mid
        else:
            hi = mid
    return 10 ** lo


S_GRID = np.logspace(-40, -1, 40)


def scan_case(args):
    lab, ek = args
    rows = [contract(s, ek=ek) for s in S_GRID]
    r0 = contract(1e-40, ek=ek)
    return dict(case=lab, ek=ek, s=S_GRID.tolist(), bounce_shear_fraction=[r["bounce_shear_fraction"] for r in rows],
                bounce_radiation_fraction=[r["bounce_radiation_fraction"] for r in rows], s_max=s_max(ek=ek),
                ekpyrotic_efolds=r0["ekpyrotic_efolds"], ekpyrosis_active_at_stop=r0["ekpyrosis_active_at_stop"])


def onset_case(args):
    f0, re = args
    ek = dict(c=5.0, f0=f0, rho_end_over_rho_c=re, gamma_fac=1000.0)
    r0 = contract(1e-40, ek=ek)
    return dict(f0=f0, rho_end_over_rho_c=re, s_max=s_max(ek=ek), ekpyrotic_efolds=r0["ekpyrotic_efolds"],
                bounce_radiation_fraction=r0["bounce_radiation_fraction"])


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    out = dict(settings=dict(rho_c=1e40, r_t=1e-4, rho_m=2.0), not_completed=[])
    cases = [("no ekpyrosis", None)] + [("c = %g" % c, dict(c=c, f0=1e-12, rho_end_over_rho_c=1e-4, gamma_fac=1000.0)) for c in (3.0, 5.0, 7.0)]
    onsets = [(f0, re) for f0 in (1e-20, 1e-12, 1e-8, 1e-6) for re in (1e-8, 1e-4, 1e-2)]
    pool = Pool(6)
    a_scan = [(c, pool.apply_async(scan_case, (c,))) for c in cases]
    a_on = [(o, pool.apply_async(onset_case, (o,))) for o in onsets]
    deadline = t0 + 3000.0                                    # per-run budget: cases not done by then are reported, not dropped silently
    out["scan"], out["onset_exit"] = [], []
    for key, lst in (("scan", a_scan), ("onset_exit", a_on)):
        for arg, a in lst:
            try:
                res = a.get(timeout=max(1.0, deadline - time.time()))
                out[key].append(res)
                print(key, arg if key == "onset_exit" else arg[0], "s_max %.3g" % res["s_max"], "N_ek %.2f" % res["ekpyrotic_efolds"], "[%.0fs]" % (time.time() - t0), flush=True)
            except Exception as e:
                out["not_completed"].append(dict(kind=key, args=str(arg), reason=type(e).__name__))
                print(key, arg, "NOT COMPLETED", type(e).__name__, flush=True)
    pool.terminate()
    r = contract(1e-30)
    out["analytic_check"] = dict(numerical=r["bounce_shear_fraction"], analytic=1e-30 / 1e-4 * np.exp(-2 * r["ln_a_bounce"]))
    print(out["analytic_check"])
    json.dump(out, open(os.path.join(HERE, "contraction_stability.json"), "w"), indent=1)
    print("[%.0fs]" % (time.time() - t0))
