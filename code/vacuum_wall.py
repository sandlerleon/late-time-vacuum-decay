# -*- coding: utf-8 -*-
"""Sections 7.2-7.3: the vacuum interface of the tunneling branch and its density-dependent nucleation.

For p(x; eta, theta) (vacuum_phase_map.py) and theta below the degeneracy density:
  * the O(4) bounce is solved by overshoot/undershoot (thermal.bounce_shoot) and its profile is kept: wall radius R_w (where x crosses the barrier
    top), wall thickness (10-90 % of the field change), wall tension sigma_b = Integral x'^2 dr across the wall, and the action b(theta) (lam = 1);
  * the thin-wall estimates: sigma_tw = Integral sqrt(2 [p - p_lin]) dx between the minima (p_lin the straight line through the two minima, which
    removes the energy difference), R_tw = 3 sigma_tw / dV and B_tw = 27 pi^2 sigma_tw^4 / (2 dV^3);
  * at the degeneracy density the static planar wall exists and is computed from the first integral z(x) = Integral dx / sqrt(2 (p - p_min)).
Units: x = phi/f, lengths 1/(sqrt(lam) f), tension sqrt(lam) f^3, energy densities lam f^4. B = b / lam.

Physical scale (Section 4 normalization): the false vacuum carries the observed dark energy V_f, the true vacuum V_t = -q V_f, so
lam f^4 = (1 + q) V_f / dtil0 with dtil0 = p(x_f) - p(x_t) at theta = 0, and lam = b(theta = 0) / B0.

Nucleation: ln(Gamma / H^4) along the dilution history theta = theta_0 / a^3 with the rate convention of rates.py (density_bounce.ln_rate) and the
actual Hubble rate H(a) of the false-vacuum universe; the transition completes (bubbles percolate) once Gamma / H^4 > 9 / (4 pi) (Guth and Weinberg).

    python vacuum_wall.py          (writes vacuum_wall.json, ~3 min)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import quad

import constants as K
import density_bounce as DB
import matter_coupling as MC
import thermal as TH
import vacuum_phase_map as VP

HERE = os.path.dirname(os.path.abspath(__file__))
LN_COMPLETE = float(np.log(9.0 / (4.0 * np.pi)))
BASE = dict(eta=0.30, B0=200.0, f0=0.3, q=1.0)               # baseline of Section 7 (survives the past light cone, density_bounce.json)


def bounce_profile(eta, theta, n=4001):
    """O(4) bounce with profile. Returns dict or None (no false vacuum / shooting failed)."""
    n_min, x_t, x_top, x_f, dV, bar = VP.classify_tb(eta, theta)
    if n_min < 2 or dV <= 0:
        return None
    v = lambda x: VP.p_tb(x, eta, theta)
    dv = lambda x: VP.dp_tb(x, eta, theta)
    res = TH.bounce_shoot(v, dv, x_t, x_top, x_f, 4)
    if res is None:
        return None
    b, x0 = res
    _, sol = TH._shoot(x0, dv, x_f, 4)
    r = np.linspace(1e-6, sol.t[-1], n)
    x, xp = sol.sol(r)
    span = x_f - x_t
    frac = (x - x_t) / span
    r_w = float(np.interp(x_top, x, r)) if np.all(np.diff(x) >= -1e-12) else float(r[np.argmin(np.abs(x - x_top))])
    r10 = float(r[np.argmax(frac >= 0.1)]) if (frac >= 0.1).any() else 0.0
    r90 = float(r[np.argmax(frac >= 0.9)]) if (frac >= 0.9).any() else float(r[-1])
    sig_b = float(np.trapezoid(xp ** 2, r))
    method = "shooting"
    if (x0 - x_t) / span < 1e-5:
        # thin-wall regime: shooting is exponentially sensitive to x0; use collocation (thermal.bounce_bvp) for the action
        tw = thin_wall(eta, theta)
        bb = TH.bounce_bvp(v, dv, x_t, x_f, 4, tw["sigma_tw"], dV, m_f=np.sqrt(VP.d2p_tb(x_f, eta, theta)))
        if bb is not None:
            b, method = bb, "collocation"
            r_w, sig_b = tw["R_tw"], tw["sigma_tw"]
    return dict(b=float(b), x0=float(x0), method=method, x_t=float(x_t), x_top=float(x_top), x_f=float(x_f), dV=float(dV), barrier=float(bar), R_w=r_w,
                thickness=r90 - r10, sigma_bounce=sig_b, x_center_over_true=float((x0 - x_t) / span), r=r[::20].tolist(), x=x[::20].tolist())


def thin_wall(eta, theta):
    n_min, x_t, x_top, x_f, dV, _ = VP.classify_tb(eta, theta)
    if n_min < 2 or dV <= 0:
        return None
    pt, pf = VP.p_tb(x_t, eta, theta), VP.p_tb(x_f, eta, theta)
    lin = lambda x: pt + (pf - pt) * (x - x_t) / (x_f - x_t)
    sig = quad(lambda x: np.sqrt(max(2.0 * (VP.p_tb(x, eta, theta) - lin(x)), 0.0)), x_t, x_f, limit=200)[0]
    return dict(sigma_tw=float(sig), R_tw=float(3.0 * sig / dV), B_tw=float(27.0 * np.pi ** 2 * sig ** 4 / (2.0 * dV ** 3)))


def degenerate_wall(eta):
    """Static planar wall at theta_deg (degenerate minima): tension, thickness (10-90 %), profile."""
    th = VP.theta_deg_tb(eta)
    _, x_t, x_top, x_f, _, _ = VP.classify_tb(eta, th)
    pmin = VP.p_tb(x_t, eta, th)
    g = lambda x: np.sqrt(max(2.0 * (VP.p_tb(x, eta, th) - pmin), 1e-300))
    sig = quad(g, x_t, x_f, limit=400)[0]
    xs = x_t + (x_f - x_t) * np.linspace(0.02, 0.98, 481)
    z = np.array([quad(lambda x: 1.0 / g(x), x_top, xx, limit=200)[0] for xx in xs])
    frac = (xs - x_t) / (x_f - x_t)
    return dict(theta_deg=float(th), sigma=float(sig), thickness=float(np.interp(0.9, frac, z) - np.interp(0.1, frac, z)), z=z[::8].tolist(),
                x=xs[::8].tolist())


def physical_scale(eta, B0, q):
    """lam, f (meV), sigma unit (meV^3), length unit (mm), energy unit (GeV^4)."""
    b0 = bounce_profile(eta, 0.0)["b"]
    lam = b0 / B0
    _, x_t, _, x_f, d0, _ = VP.classify_tb(eta, 0.0)
    E0 = (1.0 + q) * K.V_F / d0                                       # lam f^4, GeV^4
    f = (E0 / lam) ** 0.25                                            # GeV
    L_GeVinv = 1.0 / (np.sqrt(lam) * f)                               # length unit, GeV^-1
    L_mm = L_GeVinv * K.HBAR_GEV_S * 2.99792458e11                    # mm
    return dict(b0=b0, lam=lam, E0_GeV4=E0, f_meV=f * 1e12, sigma_unit_meV3=np.sqrt(lam) * f ** 3 * 1e36, length_unit_mm=L_mm, dtil0=d0)


def ln_gamma_h4(lam, b, dtil, q, H_over_HL=1.0):
    """ln(Gamma / H^4) for the actual Hubble rate H = H_over_HL * H_Lambda."""
    return float(DB.ln_rate(lam, b, dtil, r_L=1.0 / (1.0 + q)) - 4.0 * np.log(H_over_HL))


def hubble_over_HL(a, cycle_scale=1.0):
    """Flat false-vacuum universe (today's composition), H / H_Lambda."""
    rho = (MC.RHO_R0 / a ** 4 + (MC.RHO_B0 + MC.RHO_DM0) / a ** 3) * cycle_scale + MC.VC
    return np.sqrt(rho / MC.VC)


if __name__ == "__main__":
    t0 = time.time()
    out = {"base": BASE, "ln_completion": LN_COMPLETE}
    tabs = {}
    for eta in (0.20, 0.30, 0.36):
        th_deg = VP.theta_deg_tb(eta)
        rows = []
        for fr in np.linspace(0.0, 0.8, 17):
            bp = bounce_profile(eta, fr * th_deg)
            tw = thin_wall(eta, fr * th_deg)
            if bp is None:
                rows.append(dict(theta_over_deg=float(fr), failed=True))
                continue
            rows.append(dict(theta=float(fr * th_deg), theta_over_deg=float(fr), **{k: v for k, v in bp.items() if k not in ("r", "x")}, **tw,
                             b_over_B_tw=float(bp["b"] / tw["B_tw"]), sigma_b_over_tw=float(bp["sigma_bounce"] / tw["sigma_tw"])))
        tabs["%.2f" % eta] = dict(theta_deg=th_deg, rows=rows, degenerate=degenerate_wall(eta))
        ok = [r for r in rows if not r.get("failed")]
        print("eta %.2f: %d/%d bounces; b %.0f -> %.0f; b/B_tw %.2f -> %.2f; R_w %.1f -> %.1f; thick %.1f -> %.1f [%.0fs]"
              % (eta, len(ok), len(rows), ok[0]["b"], ok[-1]["b"], ok[0]["b_over_B_tw"], ok[-1]["b_over_B_tw"], ok[0]["R_w"], ok[-1]["R_w"],
                 ok[0]["thickness"], ok[-1]["thickness"], time.time() - t0), flush=True)
    out["walls"] = tabs
    # profiles for the figure (baseline eta)
    th_deg = tabs["0.30"]["theta_deg"]
    out["profiles"] = {"%.2f" % fr: bounce_profile(0.30, fr * th_deg) for fr in (0.0, 0.4, 0.8)}
    # physical scale and the nucleation history of the baseline (and the family used by the viability map)
    PS = physical_scale(BASE["eta"], BASE["B0"], BASE["q"])
    out["physical"] = PS
    T = tabs["0.30"]
    good = [r for r in T["rows"] if not r.get("failed")]
    fr_ = np.array([r["theta_over_deg"] for r in good]); lb = np.log([r["b"] for r in good]); ld = np.log([r["dV"] for r in good])
    th0 = BASE["f0"] * th_deg
    hist = []
    a_trig = None
    for a in np.logspace(np.log10(0.3), 1.0, 400):
        f_ = th0 / a ** 3 / th_deg
        if f_ > fr_[-1]:
            lg = -np.inf
        else:
            lg = ln_gamma_h4(PS["lam"], np.exp(np.interp(f_, fr_, lb)), np.exp(np.interp(f_, fr_, ld)), BASE["q"], hubble_over_HL(a))
        hist.append(dict(a=float(a), theta_over_deg=float(f_), ln_gamma_h4=float(lg) if np.isfinite(lg) else None))
        if a_trig is None and np.isfinite(lg) and lg >= LN_COMPLETE:
            a_trig = float(a)
    out["nucleation_history"] = hist
    out["a_trigger"] = a_trig
    # physical wall at the trigger
    if a_trig:
        th_tr = th0 / a_trig ** 3
        bp = bounce_profile(0.30, th_tr)
        out["trigger_wall"] = dict(theta=float(th_tr), theta_over_deg=float(th_tr / th_deg), b=bp["b"], B=bp["b"] / PS["lam"], dV=bp["dV"],
                                   R_w_mm=bp["R_w"] * PS["length_unit_mm"], thickness_mm=bp["thickness"] * PS["length_unit_mm"],
                                   sigma_meV3=bp["sigma_bounce"] * PS["sigma_unit_meV3"], x_t=bp["x_t"], x_f=bp["x_f"])
    json.dump(out, open(os.path.join(HERE, "vacuum_wall.json"), "w"), indent=1)
    print("physical:", PS)
    print("a_trigger", a_trig, out.get("trigger_wall"))
    print("saved [%.0fs]" % (time.time() - t0))
