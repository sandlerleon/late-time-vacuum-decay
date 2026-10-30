# -*- coding: utf-8 -*-
"""Tunneling branch of the dilution trigger: a coupling centred on the false vacuum makes the decay rate depend on the matter density.

With the coupling A(phi) = 1 + (phi - phi_f0)^2 / (2 M^2) centred on the zero-density false vacuum, the matter term deepens the false minimum and
raises the barrier; dilution lowers the barrier again. In the dimensionless family of Section 2 (x = phi/f, potential / lam f^4):
    p(x; eta, theta) = (x^2 - 1)^2 / 8 + eta x / 2 + theta (x - 1)^2 / 2,        theta(a) = rho_*(a) / (lam f^2 M^2) = theta_0 a^-3
(the coupling is centred on x = 1, the false vacuum of the untilted potential). Two regimes follow:
  eta < eta_c   a false minimum exists at zero density; matter raises B(eta, theta) = b(eta, theta)/lam above b(eta)/lam and dilution restores it:
                quantum tunneling whose rate rises toward its vacuum value as matter dilutes (never above it);
  eta > eta_c   no false minimum at zero density; matter creates one, which disappears at theta_crit(eta): a classical roll at
                a_crit = (theta_0/theta_crit)^(1/3), unless tunneling empties the false vacuum first.
The O(4) action b(eta, theta) is computed by overshoot/undershoot shooting (thermal.bounce_shoot). The rate is that of rates.py with the bounce
action and energy difference of the matter-corrected potential; the survival integral uses the past-light-cone weights of thermal.py.

Validity. The effective potential uses the homogeneous mean density. A bubble of radius R_b sees that density only if R_b exceeds the scale of
matter inhomogeneity; and near theta_crit the barrier vanishes and the semiclassical (instanton) approximation fails (B -> 0). Both limits are
reported, not extrapolated through.

    python density_bounce.py           (writes density_bounce.json)
"""
import json
import os
import time

import numpy as np
from scipy.optimize import brentq

import cdl_lib as C
import constants as K
import thermal as TH

HERE = os.path.dirname(os.path.abspath(__file__))
ETA_C = C.eta_crit()


def stationary(eta, theta):
    """Real stationary points of p (sorted): x^3 + (2 theta - 1) x + (eta - 2 theta) = 0."""
    r = np.roots([1.0, 0.0, 2.0 * theta - 1.0, eta - 2.0 * theta])
    return np.sort(r[np.abs(r.imag) < 1e-10].real)


def has_false(eta, theta):
    return stationary(eta, theta).size == 3


def metastable_window(eta, th_max=5.0, n=4001):
    """theta interval on which three stationary points exist (two minima): (theta_low, theta_high). For eta < eta_c theta_low = 0."""
    th = np.linspace(0.0, th_max, n)
    ok = np.array([has_false(eta, t) for t in th])
    if not ok.any():
        return None, None
    k = np.where(ok)[0]
    lo, hi = th[k[0]], th[k[-1]]
    if k[0] > 0:
        lo = brentq(lambda t: (1.0 if has_false(eta, t) else -1.0), th[k[0] - 1], th[k[0]], xtol=1e-12) if False else lo
    return float(lo), float(hi)


def theta_degenerate(eta):
    """Density at which the matter-held state x ~ 1 and the other minimum are degenerate (above it x ~ 1 is the true vacuum)."""
    def g(th):
        st = stationary(eta, th)
        if st.size < 3:
            return np.nan
        v, _ = potential(eta, th)
        return v(st[2]) - v(st[0])
    lo, hi = metastable_window(eta)
    th = np.linspace(lo, hi, 2001)
    gv = np.array([g(t) for t in th])
    sgn = np.where(np.isfinite(gv), np.sign(gv), 0)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    return float(brentq(g, th[idx[0]], th[idx[0] + 1], xtol=1e-13)) if idx.size else None


def potential(eta, theta):
    v = lambda x: (x * x - 1.0) ** 2 / 8.0 + eta * x / 2.0 + theta * (x - 1.0) ** 2 / 2.0
    dv = lambda x: x * (x * x - 1.0) / 2.0 + eta / 2.0 + theta * (x - 1.0)
    return v, dv


def b_action(eta, theta):
    """b(eta, theta) at lam = 1 and the energy difference dtil (false minus true, / lam f^4). None if no false minimum."""
    st = stationary(eta, theta)
    if st.size < 3:
        return None, None
    x_t, x_top, x_f = st
    v, dv = potential(eta, theta)
    dtil = v(x_f) - v(x_t)
    if dtil <= 0:
        return None, None
    res = TH.bounce_shoot(v, dv, x_t, x_top, x_f, 4)
    return (None if res is None else float(res[0])), float(dtil)


def ln_rate(lam, b, dtil, r_L=1.0, ln_D=0.0):
    """ln(Gamma / H_Lambda^4) with the prefactor convention of rates.py, for action B = b / lam."""
    B = b / lam
    Kp = (B / (2.0 * np.pi)) ** 2 * np.exp(ln_D)
    return np.log(9.0 * Kp * lam / (r_L * dtil)) + K.LN_M4_OVER_VF - B


if __name__ == "__main__":
    t0 = time.time()
    out = {"eta_c": ETA_C}
    # ---- (1) b(eta, theta) from zero density to 70% of the degeneracy density (beyond it the wall is thin and B >> b(eta): the rate is negligible)
    tab = {}
    for eta in (0.20, 0.30, 0.36):
        th_deg = theta_degenerate(eta)
        rows = []
        for f in np.linspace(0.0, 0.7, 15):
            b, dt = b_action(eta, float(f * th_deg))
            rows.append(dict(theta=float(f * th_deg), theta_over_deg=float(f), b=b, dtil=dt))
        tab["%.2f" % eta] = dict(theta_deg=th_deg, rows=rows)
        print("eta %.2f: theta_deg %.4f  b(0) %.2f  b(0.35) %.2f  b(0.7) %.2f  [%.0fs]" % (eta, th_deg, rows[0]["b"], rows[7]["b"], rows[-1]["b"], time.time() - t0), flush=True)
    out["b_table"] = tab
    out["check_theta0"] = {k: dict(b=v["rows"][0]["b"], table=float(TH.R.b_of_eta(float(k)))) for k, v in tab.items()}
    # ---- (2) classical branch, eta > eta_c: the matter-held minimum exists only for theta_low < theta < theta_high
    out["classical"] = {"%.2f" % e: dict(zip(("theta_low", "theta_high"), metastable_window(e))) for e in (0.40, 0.45, 0.50)}
    # ---- (3) dilution histories: theta(z) = theta_0 (1+z)^3 over the past light cone and theta_0/a^3 into the future
    z_lc, w_lc = TH.light_cone_grid()
    hist = []
    for eta in (0.20, 0.30, 0.36):
        T_ = tab["%.2f" % eta]
        th_deg = T_["theta_deg"]
        fr = np.array([r["theta_over_deg"] for r in T_["rows"]])
        lb = np.log([r["b"] for r in T_["rows"]])
        ld = np.log([r["dtil"] for r in T_["rows"]])
        b0, d0 = T_["rows"][0]["b"], T_["rows"][0]["dtil"]

        def lr_at(lam, th):
            f = th / th_deg
            if f > fr[-1]:
                return -np.inf                                    # B larger than at 0.7 theta_deg: negligible rate (conservative for the past)
            return ln_rate(lam, np.exp(np.interp(f, fr, lb)), np.exp(np.interp(f, fr, ld)))
        for B0 in (150.0, 200.0, 250.0, 287.0, 400.0):
            lam = b0 / B0
            lr_vac = float(ln_rate(lam, b0, d0))
            lnI0 = float(lr_vac + 2.0 * np.log(K.OMEGA_L) + np.log(np.sum(w_lc)))
            for f0 in (0.1, 0.3, 0.7, 2.0):
                th0 = f0 * th_deg
                lr = np.array([lr_at(lam, th0 * (1 + z) ** 3) for z in z_lc])
                with np.errstate(divide="ignore"):
                    lnI = float(np.logaddexp.reduce(lr + 2.0 * np.log(K.OMEGA_L) + np.log(w_lc)))
                a_grid = np.logspace(0, 3, 600)
                lr_f = np.array([lr_at(lam, th0 / a ** 3) for a in a_grid])
                a_star = float(a_grid[np.argmax(lr_f >= 0)]) if (lr_f >= 0).any() else None
                hist.append(dict(eta=eta, B0=B0, lam=float(lam), theta0_over_deg=f0, ln_rate_today=float(lr_at(lam, th0)), ln_rate_vacuum=lr_vac,
                                 lnI_past=lnI, lnI_without_coupling=lnI0, past_suppression=float(lnI0 - lnI) if np.isfinite(lnI) else None,
                                 a_star=a_star, survives=bool(lnI < 0)))
    out["histories"] = hist
    for h in hist:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()})
    json.dump(out, open(os.path.join(HERE, "density_bounce.json"), "w"), indent=1)
    print("saved [%.0fs]" % (time.time() - t0))
