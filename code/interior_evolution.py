# -*- coding: utf-8 -*-
"""The interior of a nucleated bubble: energy budget, open-FRW Lorentzian evolution with reheating, and the AdS crunch time.

Part A. Energy budget from the numerical bounce: the field at the centre of the bounce, x0, sets the energy that the interior can
        release, eps_in = (v(x0) - v_t) / DeltaV. The remainder of DeltaV is carried by the bubble wall.
Part B. Lorentzian evolution in the open slicing, ds^2 = -dt^2 + a(t)^2 dH_3^2, Friedmann constraint (da/dt)^2 = 1 + a^2 rho / (3 M_Pl^2).
        In units of the asymptotic de Sitter rate H_L (tau = H_L t, alpha = H_L a, densities in units of V_f):
            (d alpha / d tau)^2 = 1 + alpha^2 (rho_phi + rho_r + rho_vac).
        The condensate rho_phi starts oscillating at tau_0 = H_L / m (m the mass at the true minimum), behaves as matter and decays to
        radiation at the rate gamma = Gamma_phi / H_L <= m / H_L:
            rho_phi' = -3 (alpha'/alpha) rho_phi - gamma rho_phi,   rho_r' = -4 (alpha'/alpha) rho_r + gamma rho_phi.
        In terms of the comoving quantities Phi = rho_phi alpha^3 and Y = rho_r alpha^4, Phi = Phi_0 e^{-gamma (tau - tau_0)} exactly and
            alpha' = sqrt(1 + Phi/alpha + Y/alpha^2 + rho_vac alpha^2),     Y' = gamma Phi alpha.
        When alpha^2 rho << 1 the interior is curvature dominated (Milne, alpha = tau) and the radiation density has the closed form
            rho_r(tau) = (gamma eps Phi_scale / tau^4) [ (tau_0/gamma + 1/gamma^2) - (tau/gamma + 1/gamma^2) e^{-gamma (tau - tau_0)} ],
        Phi_scale = tau_0^3. The closed form is compared with a numerical integration of the full system in verify.py.
Part C. A true vacuum with V_t < 0 (AdS) crunches in a finite open-slicing time, tau_c = pi / sqrt(|V_t| / V_f).

    python interior_evolution.py          (writes interior_evolution.json)
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp

import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
GSTAR = K.GSTAR_LOW


def _phi(x):
    """2 (1 - (1 + x) e^{-x}) / x^2, with a series for small x (avoids cancellation)."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-2
    xs = np.where(small, 1.0, x)
    full = 2.0 * (1.0 - (1.0 + xs) * np.exp(-np.minimum(xs, 700.0))) / xs ** 2
    ser = 1.0 - 2.0 * x / 3.0 + x ** 2 / 4.0 - x ** 3 / 15.0
    return np.where(small, ser, full)


def rho_r_milne(tau, eps, gamma, tau0):
    """Radiation density (units of V_f) in the curvature-dominated interior, closed form; zero before tau0.
    rho_r = eps tau0^3 gamma / tau^4 * Integral_{tau0}^{tau} s exp(-gamma (s - tau0)) ds, evaluated without cancellation."""
    tau = np.asarray(tau, dtype=float)
    u = np.maximum(tau - tau0, 0.0)
    x = gamma * u
    xc = np.minimum(x, 700.0)                      # only the exponentials are clipped; the algebraic ratios use the true x
    integral = tau0 * u * (-np.expm1(-xc)) / np.where(x > 0, x, 1.0) + 0.5 * u ** 2 * _phi(x)
    integral = np.where(x > 0, integral, 0.0)
    val = eps * tau0 ** 3 * gamma / tau ** 4 * integral
    return np.where(tau > tau0, val, 0.0)


def integrate_full(eps, gamma, tau0, tau_end, rho_vac=0.0, n=2000):
    """Numerical integration of the full system (alpha, Y) with Phi analytic; returns tau, alpha, rho_r, alpha'."""
    lt = np.linspace(np.log(tau0), np.log(tau_end), n)
    Phi0 = eps * tau0 ** 3

    def rhs(l, y):
        tau = np.exp(l)
        al, Y = y
        Phi = Phi0 * np.exp(-gamma * (tau - tau0))
        da = np.sqrt(1.0 + Phi / al + Y / al ** 2 + rho_vac * al ** 2)
        return [tau * da, tau * gamma * Phi * al]

    sol = solve_ivp(rhs, [lt[0], lt[-1]], [tau0, 0.0], method="Radau", t_eval=lt, rtol=1e-10, atol=[1e-18, 1e-30])
    tau = np.exp(lt)
    al, Y = sol.y
    Phi = Phi0 * np.exp(-gamma * (tau - tau0))
    return tau, al, Y / al ** 4, np.sqrt(1.0 + Phi / al + Y / al ** 2 + rho_vac * al ** 2)


if __name__ == "__main__":
    out = {}
    # ---- Part A
    rows = []
    for r in T["flat"]:
        if r["eta"] < 0.10:
            continue
        rows.append(dict(eta=r["eta"], eps_in=float(r["v_x0"] / r["dtil"]), x0=r["x0"], x_t=r["x_t"]))
    out["energy_budget"] = rows
    # ---- Part B
    ref = [r for r in T["flat"] if abs(r["eta"] - 0.30) < 1e-9][0]
    m_f_gev = float(np.sqrt((3.0 * ref["x_t"] ** 2 - 1.0) / 2.0) * (K.V_F / ref["dtil"]) ** 0.25)          # lam = 1
    m_dim = m_f_gev / K.H_L                                                                                  # m / H_L
    tau0 = 1.0 / m_dim
    out["scales"] = dict(m_meV=m_f_gev * 1e12, m_over_H=m_dim, tau0=tau0, t0_seconds=tau0 * K.T_HUBBLE_YR * K.YEAR_S)
    eps_in = float(np.interp(0.30, [r["eta"] for r in rows], [r["eps_in"] for r in rows]))
    out["eps_in_ref"] = eps_in
    tau_grid = np.logspace(np.log10(tau0), 4.0, 6000)
    runs = []
    for eps in (1.0, eps_in):
        for frac in (1.0, 1e-3, 1e-9, 1e-20):
            gamma = frac * m_dim
            rr = rho_r_milne(tau_grid, eps, gamma, tau0)
            Tg = (30.0 * rr * K.V_F / (np.pi ** 2 * GSTAR)) ** 0.25
            i = int(np.argmax(Tg))
            T1 = float(np.interp(1.0, tau_grid, Tg))
            runs.append(dict(eps=eps, gamma_over_m=frac, T_max_meV=float(Tg[i] * 1e12), tau_peak=float(tau_grid[i]), T_over_bound=float(Tg[i] / K.t_reh_max(K.V_F, eps)),
                             T_at_one_hubble_time_meV=T1 * 1e12, T_bbn_over_Tmax=float(K.T_BBN_GEV / Tg[i]), alpha2_rho_max=float(np.max(tau_grid ** 2 * (rr + eps * (tau0 / tau_grid) ** 3)))))
    out["runs"] = runs
    # the invariant T * t after the peak and the time at which T = T_BBN would have to occur
    best = max(runs, key=lambda r: r["T_max_meV"])
    out["T_times_tau"] = float(best["T_max_meV"] * 1e-12 * best["tau_peak"])
    out["tau_for_bbn_temperature"] = float(out["T_times_tau"] / K.T_BBN_GEV)
    out["t_for_bbn_temperature_seconds"] = float(out["tau_for_bbn_temperature"] * K.T_HUBBLE_YR * K.YEAR_S)
    out["curvature_dominated"] = True
    # one trace for the figure (gamma = m, eps = 1): T(tau)
    rr = rho_r_milne(tau_grid, 1.0, m_dim, tau0)
    Tg = (30.0 * rr * K.V_F / (np.pi ** 2 * GSTAR)) ** 0.25
    sel = np.linspace(0, len(tau_grid) - 1, 400).astype(int)
    out["trace"] = dict(tau=tau_grid[sel].tolist(), T_meV=(Tg[sel] * 1e12).tolist())
    # ---- Part C
    out["ads_crunch"] = [dict(vt_over_Vf=-x, tau_crunch=float(np.pi / np.sqrt(x)), gyr=float(np.pi / np.sqrt(x) * K.T_HUBBLE_YR / 1e9)) for x in (0.01, 0.1, 1.0, 10.0)]
    out["bbn"] = dict(rho_needed_over_Vf=float((np.pi ** 2 * GSTAR / 30.0) * K.T_BBN_GEV ** 4 / K.V_F), T_max_meV_bound=float(K.t_reh_max(K.V_F) * 1e12))
    json.dump(out, open(os.path.join(HERE, "interior_evolution.json"), "w"), indent=1)
    print("m = %.2f meV (m/H = %.2e), tau0 = %.2e (t0 = %.2e s), eps_in(eta=0.3) = %.3f" % (out["scales"]["m_meV"], m_dim, tau0, out["scales"]["t0_seconds"], eps_in))
    for r in runs:
        print("eps %.3f gamma/m %.0e  T_max %.4f meV (x bound %.3f) at tau %.2e ; T(1/H) = %.2e meV ; T_BBN/T_max %.1e ; alpha^2 rho max %.1e" % (
            r["eps"], r["gamma_over_m"], r["T_max_meV"], r["T_over_bound"], r["tau_peak"], r["T_at_one_hubble_time_meV"], r["T_bbn_over_Tmax"], r["alpha2_rho_max"]))
    print("T * tau = %.3e GeV  -> T = T_BBN only at tau = %.2e (%.2e s)" % (out["T_times_tau"], out["tau_for_bbn_temperature"], out["t_for_bbn_temperature_seconds"]))
    print("AdS crunch", out["ads_crunch"])
