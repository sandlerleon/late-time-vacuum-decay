# -*- coding: utf-8 -*-
"""Section 7.6: is the hand-off from the bubble wall to the cycle of the enclosed region physically consistent?

(1) Percolation. With the rate history of vacuum_wall.py and the false-vacuum background, the fraction of space still in the false vacuum is
        f(t) = exp[-I(t)],   I(t) = (4 pi/3) Integral_{t_0}^{t} Gamma(t') a(t')^3 r(t, t')^3 dt',   r(t, t') = Integral_{t'}^{t} dt''/a(t'')
    (Guth and Tye; Turner, Weinberg and Widrow). It gives the epoch of completion, beta = d ln Gamma / dt, and the mean bubble radius at collision
    R_coll = (8 pi)^(1/3) / beta. If R_coll << 1/H the transition is completed in a small fraction of a Hubble time and the region is homogeneous on
    scales above R_coll: the homogeneous hand-off of the percolated case is then exact for the background, whatever the collisions produce, provided
    the products are relativistic (w = 1/3: ultra-relativistic shells and scalar radiation) or dilute (w = 0: massive scalar quanta). Both limits are
    run through the cycle (the products thermalize at the latest in the hot bounce).
(2) Bounds on the ultra-relativistic wall: thickness after Lorentz contraction against the radius; pressure of the dark matter (saturated, computed in
    interface_transport.py); the next-to-leading pressure ~ gamma g^2 Delta m_V T^3 that a dark gauge boson acquiring mass from phi would add
    (Boedeker and Moore 2017), giving a terminal Lorentz factor; the fraction of the wall energy in gravitational waves, ~ kappa^2 (H/beta)^2; and
    primordial black holes, which need order-one overdensities on the Hubble scale.
(3) Isolated bubbles. The enclosed open region keeps its Friedmann evolution only while the Misner-Sharp mass inside the wall stays below that of
    the exterior at the same areal radius (rho_in < rho_out; positive wall energy with Gamma_+- > 0). The wall is followed as a null ray in the
    interior (R = a sinh(sqrt(K) chi)/sqrt(K), dchi/dt = 1/a) and compared with the exterior false-vacuum universe; once rho_in > rho_out the junction
    requires a sign change of Gamma_+ (the region is cut off behind a horizon), and once rho_in R^2/3 > 1 the wall is a trapped surface.
(4) Bounce form: rho_c over eight orders and H^2 = (rho/3)(1 - (rho/rho_c)^2) instead of the loop-quantum form.
(5) Independent reproduction: the percolated daughter history integrated as an initial-value problem (dH/dt from the Raychaudhuri equation, clocks
    restarted per decade of a) against the root-finding solution; the wall trajectory integrated from the junction against Gamma = R/R_0.

    python bubble_handoff.py          (writes bubble_handoff.json, ~2 min)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

import bubble_gravity as BG
import bubble_to_cycle as BC
import constants as K
import matter_coupling as MC
import vacuum_wall as W

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))


def percolation(WJ):
    base = WJ["base"]
    tabs = BC.rate_tables(WJ)
    lam = tabs[4] / base["B0"]
    th_deg = tabs[0]
    th0 = base["f0"] * th_deg
    HL = np.sqrt(MC.VC / 3.0)
    a = np.logspace(np.log10(1.0), np.log10(3.0), 6000)
    H = HL * W.hubble_over_HL(a)
    t = cumulative_trapezoid(1.0 / (a * H), np.log(a), initial=0.0)
    lnG = []
    for x in a:
        f = th0 / x ** 3 / th_deg
        if f > tabs[1][-1]:
            lnG.append(-np.inf)
        else:
            lnG.append(W.ln_gamma_h4(lam, np.exp(np.interp(f, tabs[1], tabs[2])), np.exp(np.interp(f, tabs[1], tabs[3])), base["q"]))
    lnG = np.array(lnG)
    Gam = np.exp(lnG) * HL ** 4                                        # H_0^4 units
    eta_c = cumulative_trapezoid(1.0 / a, t, initial=0.0)              # conformal time
    idx = np.arange(0, a.size, 10)
    Ii = np.zeros(idx.size)
    for k, i in enumerate(idx):
        if i == 0:
            continue
        r = eta_c[i] - eta_c[:i + 1]
        Ii[k] = 4.0 * np.pi / 3.0 * np.trapezoid(Gam[:i + 1] * a[:i + 1] ** 3 * r ** 3, t[:i + 1])
    I = np.interp(np.arange(a.size), idx, Ii)
    ff = np.exp(-I)
    a50 = float(np.interp(0.5, ff[::-1], a[::-1]))
    a01 = float(np.interp(0.01, ff[::-1], a[::-1]))
    i50 = np.argmin(abs(a - a50))
    beta = float(np.gradient(lnG[np.isfinite(lnG)], t[np.isfinite(lnG)])[np.searchsorted(a[np.isfinite(lnG)], a50)])
    Hn = float(H[i50])
    R_coll_H = (8.0 * np.pi) ** (1.0 / 3.0) / beta * Hn
    return dict(a_half=a50, a_one_percent=a01, a_trigger_criterion=WJ["a_trigger"], beta_over_H=beta / Hn, R_coll_times_H=R_coll_H,
                completion_duration_over_hubble_time=float((t[np.argmin(abs(a - a01))] - t[np.argmin(abs(a - a50 * 0.98))]) * Hn),
                a=a[::30].tolist(), f_false=ff[::30].tolist())


def wall_bounds(WJ, TJ, GJ, perc):
    PS, TW = WJ["physical"], WJ["trigger_wall"]
    unit = np.sqrt(PS["lam"]) * PS["f_meV"]                            # meV
    dV = TW["dV"] * PS["lam"] * PS["f_meV"] ** 4                       # meV^4 (driving pressure)
    R0_mm, RH_mm = GJ["wall"]["R0_mm"], TJ["dark_matter"]["hubble_radius_mm"]
    R_coll_mm = perc["R_coll_times_H"] * RH_mm
    gamma_coll = R_coll_mm / R0_mm
    thick_mm = TW["thickness_mm"]
    T_cmb = K.T_CMB_GEV * 1e12 / WJ["a_trigger"]                       # meV
    dmV = abs(TW["x_t"] - TW["x_f"]) * PS["f_meV"]                     # g = 1: m_V = g f |x|
    out = dict(gamma_at_collision=float(gamma_coll), R_coll_mm=float(R_coll_mm), contracted_thickness_over_radius=float(thick_mm / gamma_coll / R_coll_mm),
               dm_pressure_over_vacuum_drive=TJ["dark_matter"]["P_dm_max_over_dV_vac"])
    rows = []
    for g in (0.1, 1.0, 3.0):
        for T in (T_cmb, 1.0, 4.0):
            P_per_gamma = g ** 2 * (g * dmV) * T ** 3
            rows.append(dict(g=g, T_meV=float(T), gamma_terminal=float(dV / P_per_gamma)))
    out["gauge_nlo_terminal_gamma"] = rows
    out["gamma_threshold_dm"] = TJ["dark_matter"]["gamma_threshold"]
    out["gw_fraction_bound"] = float(0.1 * (1.0 / perc["beta_over_H"]) ** 2)          # kappa^2 (H/beta)^2 with kappa <= 1, prefactor 0.1 (envelope)
    out["pbh_scale_ratio"] = float(perc["R_coll_times_H"])
    return out


def isolated_consistency(GJ):
    tr = [p for p in GJ["isolated"]["traj"]]
    ev = GJ["isolated"]["events"]
    nuc = [e for e in ev if e["event"] == "nucleation"][0]
    bnc = [e for e in ev if e["event"] == "bounce"][0]
    Kc = None
    seg = np.array([p["segment"] != "held (false vacuum)" for p in tr])
    t = np.array([p["tau"] for p in tr]); la = np.array([p["lna"] for p in tr]); rho = np.array([p["rho"] for p in tr])
    sel = (t >= nuc["tau"] - 1e-12) & (t <= bnc["tau"] + 1e-12) & seg
    t, la, rho = t[sel], la[sel], rho[sel]
    o = np.argsort(t, kind="stable"); t, la, rho = t[o], la[o], rho[o]
    t, uniq = np.unique(t, return_index=True); la, rho = la[uniq], rho[uniq]
    a = np.exp(la)
    # curvature: K/a^2 = dV_eff/3 at nucleation
    Kc = nuc["dV_eff"] / 3.0 * np.exp(2.0 * nuc["lna"])
    chi = cumulative_trapezoid(1.0 / a, t, initial=0.0)
    sk = np.sqrt(Kc)
    R = a * np.sinh(sk * chi) / sk
    # exterior: flat false-vacuum universe from the same nucleation state, same cosmic time (approximation stated in the text)
    a0 = np.exp(nuc["lna"])
    rho_out_of = lambda x: MC.RHO_R0 / x ** 4 + (MC.RHO_B0 + MC.RHO_DM0) / x ** 3 + MC.VC
    sol = solve_ivp(lambda tt, y: [y[0] * np.sqrt(rho_out_of(y[0]) / 3.0)], (t[0], t[-1]), [a0], rtol=1e-10, dense_output=True)
    rho_out = rho_out_of(sol.sol(t)[0])
    drho = rho_out - rho
    trap = rho * R ** 2 / 3.0
    k_sign = int(np.argmax(drho < 0)) if (drho < 0).any() else None
    k_trap = int(np.argmax(trap > 1.0)) if (trap > 1.0).any() else None
    ta = [e for e in ev if e["event"] == "turnaround"][0]
    rs = [e for e in ev if e["event"] == "restoration"][0]
    return dict(t_nucleation=nuc["tau"], t_turnaround=ta["tau"], t_restoration=rs["tau"], t_bounce=bnc["tau"],
                t_interior_denser=float(t[k_sign]) if k_sign is not None else None, rho_in_at_crossing=float(rho[k_sign]) if k_sign is not None else None,
                R_H_at_crossing=float(R[k_sign] * np.sqrt(max(rho_out[k_sign], 1e-300) / 3.0)) if k_sign is not None else None,
                t_trapped=float(t[k_trap]) if k_trap is not None else None, R_at_bounce_over_RH0=float(R[-1]),
                max_trap_parameter=float(trap.max()), curve_t=t[::4].tolist(), curve_drho=drho[::4].tolist(), curve_trap=trap[::4].tolist(),
                curve_rho_in=rho[::4].tolist(), curve_rho_out=rho_out[::4].tolist())


def bounce_variants(WJ, lnrate):
    base, TW = WJ["base"], WJ["trigger_wall"]
    th_deg = WJ["walls"]["0.30"]["theta_deg"]
    rows = []
    for lab, kw in (("baseline (LQC, rho_c = 1e44)", {}), ("rho_c = 1e40", dict(rho_c=1e40)), ("rho_c = 1e48", dict(rho_c=1e48)),
                    ("quadratic bounce, H^2 = rho/3 (1 - (rho/rho_c)^2)", dict(bounce_form="quadratic")),
                    ("wall energy as dark radiation until the bounce (w = 1/3)", dict(wall_w=1.0 / 3.0)),
                    ("wall energy as massive scalar quanta until the bounce (w = 0)", dict(wall_w=0.0))):
        D = BG.Daughter(base["eta"], base["f0"] * th_deg, base["q"], TW["theta"], lnrate=lnrate, **kw)
        r = D.run("percolated", n_cycles=3)
        sm = BC.summary(r)
        ta = [e for e in r["events"] if e["event"] == "turnaround"]
        bb = [e for e in r["events"] if e["event"] == "bounce"]
        rows.append(dict(variant=lab, lnG=sm["lnG"], T_b_MeV=sm["T_b_MeV"], failure=sm["failure"],
                         a_max_over_a_nuc=[e["a_max_over_a_nuc"] for e in ta], bounce_rho_over_rho_c=[e["rho"] / D.rho_c for e in bb]))
    return rows


def independent_reproduction(WJ, lnrate):
    """Percolated cycle 0 from nucleation: initial-value integration (ln a, H) with the Raychaudhuri equation, against the root-finding solution."""
    base, TW = WJ["base"], WJ["trigger_wall"]
    th_deg = WJ["walls"]["0.30"]["theta_deg"]
    D = BG.Daughter(base["eta"], base["f0"] * th_deg, base["q"], TW["theta"], lnrate=lnrate)
    ref = D.run("percolated", n_cycles=1)["events"]
    nuc, ta, rs = [[e for e in ref if e["event"] == k][0] for k in ("nucleation", "turnaround", "restoration")]
    D.bscale, D.cf, D.Sx = 1.0, 1.0, 0.0
    Sr = (nuc["S_after"]) ** (4.0 / 3.0)
    nd = MC.RHO_DM0
    N0 = nuc["lna"]
    H0_ = np.sqrt(D.F(N0, Sr, nd, "true", 0.0))

    def rhs(tt, y):
        rho, rpp, *_ = D.parts(y[0], Sr, nd, "true")
        return [y[1], -rpp / 2.0 * (1.0 - 2.0 * rho / D.rho_c)]
    ev_ta = lambda tt, y: y[1]; ev_ta.terminal, ev_ta.direction = True, -1
    s1 = solve_ivp(rhs, (0.0, 50.0), [N0, H0_], method="DOP853", rtol=1e-12, atol=1e-14, events=ev_ta)
    t_ta, N_ta = s1.t_events[0][0], s1.y_events[0][0][0]
    N_res = rs["lna"]
    ev_rs = lambda tt, y: y[0] - N_res; ev_rs.terminal, ev_rs.direction = True, -1
    s2 = solve_ivp(rhs, (0.0, 50.0), [N_ta, -1e-12], method="DOP853", rtol=1e-12, atol=1e-14, events=ev_rs)
    t_rs = s2.t_events[0][0]
    yend = s2.y_events[0][0]
    res = abs(yend[1] ** 2 - D.F(yend[0], Sr, nd, "true", 0.0)) / D.parts(yend[0], Sr, nd, "true")[0]
    return dict(turnaround_lna_ivp=float(N_ta), turnaround_lna_root=ta["lna"], turnaround_time_ivp=float(t_ta),
                turnaround_time_root=float(ta["tau"] - nuc["tau"]), restoration_time_ivp=float(t_ta + t_rs),
                restoration_time_root=float(rs["tau"] - nuc["tau"]), constraint_residual_ivp=float(res))


if __name__ == "__main__":
    t0 = time.time()
    WJ, TJ, GJ = J("vacuum_wall.json"), J("interface_transport.json"), J("bubble_gravity.json")
    tabs = BC.rate_tables(WJ)
    lam0 = tabs[4] / WJ["base"]["B0"]

    def lnrate(th):
        f = th / tabs[0]
        if f > tabs[1][-1]:
            return -np.inf
        return W.ln_gamma_h4(lam0, np.exp(np.interp(f, tabs[1], tabs[2])), np.exp(np.interp(f, tabs[1], tabs[3])), WJ["base"]["q"])
    out = {}
    out["percolation"] = perc = percolation(WJ)
    print("percolation", {k: v for k, v in perc.items() if k not in ("a", "f_false")}, "[%.0fs]" % (time.time() - t0), flush=True)
    out["wall_bounds"] = wall_bounds(WJ, TJ, GJ, perc)
    print("bounds", out["wall_bounds"], flush=True)
    out["isolated"] = isolated_consistency(GJ)
    print("isolated", out["isolated"], flush=True)
    out["variants"] = bounce_variants(WJ, lnrate)
    for r in out["variants"]:
        print(r, flush=True)
    out["reproduction"] = independent_reproduction(WJ, lnrate)
    print("reproduction", out["reproduction"], "[%.0fs]" % (time.time() - t0))
    json.dump(out, open(os.path.join(HERE, "bubble_handoff.json"), "w"), indent=1)
