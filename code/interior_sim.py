# -*- coding: utf-8 -*-
"""Steps 6-7: Lorentzian simulation of the daughter region, two-field energy reservoirs, hilltop inflation, repeated transitions and entropy.

All simulations use the open (k = -1) homogeneous interior of a CDL bubble, the analytic continuation of the bounce: the field starts at the bounce centre x0 with zero velocity and the scale
factor starts as a = t (Milne). In units of f (fields), 1/f (time), lambda f^4 (energy density), with g = f^2 / M_Pl^2:

    (da/dt)^2 = 1 + g a^2 rho / 3            (constraint; its residual measures energy conservation)
    d2a/dt2   = -(g a / 6) (rho + 3 p)
    x'' + (3 a'/a + Y_x) x' + dV/dx = 0,   y'' + (3 a'/a + Y_y) y' + dV/dy = 0
    rho_r' + 4 (a'/a) rho_r = Y_x x'^2 + Y_y y'^2
    rho = x'^2/2 + y'^2/2 + V + rho_r,   p = x'^2/2 + y'^2/2 - V + rho_r / 3

Part A  single field, physical benchmark (lambda = 1, eta = 0.30, V_t = 0, f = 3.05 meV, g = 1.6e-60): the full dynamics against the closed-form interior of interior_evolution.py,
        and a scan in g from 1e-60 to 10 that finds where a flat-like radiation epoch (curvature parameter Y = g a^2 rho / 3 of order one) can occur.
Part B  two fields (phi, chi) with a reservoir stored in chi, compensated by a negative offset so that the false vacuum has the observed dark-energy density. The released energy then
        sits in a deep anti-de Sitter true vacuum: reheating, the temperature history and the crunch time (full simulation, then exact radiation + AdS solution).
Part C  hilltop (new) inflation of a field with a Planck-scale vev at the dark-energy scale, in units H_Lambda = M_Pl = 1: e-folds, flatness and the reheating temperature against w / M_Pl.
Part D  repeated transitions (a three-vacuum chain: rates and the energy sum rule), and entropy accounting.

    python interior_sim.py          (writes interior_sim.json; about 5 minutes)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import cdl_lib as C
import constants as K
import rates as R
import interior_evolution as I

HERE = os.path.dirname(os.path.abspath(__file__))
T_TAB = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
IN = json.load(open(os.path.join(HERE, "interior_evolution.json"), encoding="utf-8"))
GSTAR = K.GSTAR_LOW
HBAR_S = K.HBAR_GEV_S
M = K.M_PL
out = {}
T_START = time.time()


# ----------------------------------------------------------------------------------------------------- homogeneous simulation
def simulate(pot, g, upsilon, x0, y0=0.0, t0=1e-4, t_end=500.0, rho_extra=0.0, rtol=1e-11, atol=1e-14, n_out=4000, stop_frac=1e-7):
    """Integrate the open-universe interior. pot(x, y) -> (V, Vx, Vy). Returns dict of arrays and the final state."""
    V0, Vx0, Vy0 = pot(x0, y0)
    y_init = [t0, 1.0, x0 - Vx0 * t0 ** 2 / 8.0, -Vx0 * t0 / 4.0, y0 - Vy0 * t0 ** 2 / 8.0, -Vy0 * t0 / 4.0, 0.0]
    ups_x, ups_y = upsilon

    def rhs(t, s):
        a, ad, x, xd, y, yd, rr = s
        V, Vx, Vy = pot(x, y)
        kin = 0.5 * xd * xd + 0.5 * yd * yd
        rho = kin + V + rr
        p = kin - V + rr / 3.0
        add = -(g * a / 6.0) * (rho + 3.0 * p)
        H = ad / a
        return [ad, add, xd, -(3.0 * H + ups_x) * xd - Vx, yd, -(3.0 * H + ups_y) * yd - Vy, -4.0 * H * rr + ups_x * xd * xd + ups_y * yd * yd]

    sol = solve_ivp(rhs, [t0, t_end], y_init, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    t = np.geomspace(t0, sol.t[-1], n_out)
    s = sol.sol(t)
    a, ad, x, xd, y, yd, rr = s
    V = np.array([pot(xx, yy)[0] for xx, yy in zip(x, y)])
    kin = 0.5 * xd ** 2 + 0.5 * yd ** 2
    rho = kin + V + rr
    resid = (ad ** 2 - 1.0 - g * a ** 2 * rho / 3.0) / (1.0 + ad ** 2)
    return dict(t=t, a=a, ad=ad, x=x, y=y, rho_r=rr, rho=rho, V=V, kin=kin, resid=resid, final=s[:, -1], sol=sol)


def temperature_tilde(rho_r, gstar=GSTAR):
    return (30.0 * np.maximum(rho_r, 0.0) / (np.pi ** 2 * gstar)) ** 0.25


def fluid_tail(a1, t1, rho_r1, rho_v, g):
    """Exact evolution after the fields have settled: radiation rho_r = R / a^4 and a constant vacuum energy rho_v in the open universe,
    ad^2 = 1 + g rho_v a^2 / 3 + g R / (3 a^2). Returns the crunch time (inf for rho_v >= 0) and the functions a -> t, a -> ad."""
    Rr = rho_r1 * a1 ** 4
    c = g * Rr / 3.0
    if rho_v >= 0:
        return np.inf, None
    b = g * abs(rho_v) / 3.0
    s_p = (1.0 + np.sqrt(1.0 + 4.0 * b * c)) / (2.0 * b)           # a_max^2
    s_m = (np.sqrt(1.0 + 4.0 * b * c) - 1.0) / (2.0 * b)           # |A_-|
    arc = lambda s: np.arcsin(np.clip((2.0 * s - s_p + s_m) / (s_p + s_m), -1.0, 1.0))
    t_fl = lambda s: (arc(s) - arc(0.0)) / (2.0 * np.sqrt(b))      # time from a = 0 on the fluid solution
    t_half = t_fl(s_p)
    t_c = t1 + (t_half - t_fl(a1 ** 2)) + t_half
    return t_c, dict(b=b, s_p=s_p, s_m=s_m, t_fl=t_fl, a_max=np.sqrt(s_p), t_half=t_half)


# ----------------------------------------------------------------------------------------------------------------- Part A
lam, eta = 1.0, 0.30
P = C.Potential(lam, eta)
flat = {r["eta"]: r for r in T_TAB["flat"]}
x0 = flat[eta]["x0"]
f_mev = float(IN["scales"]["m_meV"] / np.sqrt((3 * P.x_t ** 2 - 1) / 2.0))        # f in meV (m = sqrt((3 x_t^2 - 1)/2) f)
f_gev = f_mev * 1e-12
g_phys = (f_gev / M) ** 2
m_sim = float(np.sqrt((3 * P.x_t ** 2 - 1) / 2.0))
dtil = P.dtil
eps_in = float(P.v(x0) / dtil)
pot1 = lambda x, y: (float(P.v(x)), float(P.dv(x)), 0.0)

rowsA = []
for fr in (1.0, 1e-1, 1e-2, 1e-3):
    ups = fr * m_sim
    sm = simulate(pot1, g_phys, (ups, 0.0), x0, t_end=max(60.0 / ups, 400.0), n_out=6000)
    Tt = temperature_tilde(sm["rho_r"])
    i = int(np.argmax(Tt))
    tau_grid = np.logspace(np.log10(IN["scales"]["tau0"]), 4.0, 6000)
    rr_c = I.rho_r_milne(tau_grid, IN["eps_in_ref"], fr * IN["scales"]["m_over_H"], IN["scales"]["tau0"])
    closed = dict(T_max_meV=float(np.max((30.0 * rr_c * K.V_F / (np.pi ** 2 * GSTAR)) ** 0.25) * 1e12))
    rho_exc = sm["kin"] + sm["V"] + sm["rho_r"]
    sel_env = sm["t"] >= 2.0 / m_sim
    env_ratio = float(np.max(rho_exc[sel_env] * (m_sim * sm["t"][sel_env]) ** 3) / float(P.v(x0)))
    rowsA.append(dict(envelope_ratio=env_ratio, gamma_over_m=fr, T_max_meV=float(Tt[i] * f_mev), t_peak=float(sm["t"][i]), T_max_closed_meV=closed["T_max_meV"], ratio_sim_over_closed=float(Tt[i] * f_mev / closed["T_max_meV"]),
                      max_residual=float(np.max(np.abs(sm["resid"]))), T_tau_final=float(Tt[-1] * f_mev * 1e-12 * sm["t"][-1] / f_gev),
                      field_energy_final_over_rad=float((sm["kin"][-1] + sm["V"][-1]) / max(sm["rho_r"][-1], 1e-300))))
    print("A gamma/m %.0e: T_max %.4f meV (closed form %.4f, ratio %.3f) resid %.1e [%.0fs]" % (fr, rowsA[-1]["T_max_meV"], closed["T_max_meV"], rowsA[-1]["ratio_sim_over_closed"], rowsA[-1]["max_residual"], time.time() - T_START), flush=True)
out["A_single_field"] = dict(f_meV=f_mev, g=g_phys, eps_in=eps_in, m_sim=m_sim, rows=rowsA)

# gravity scan: where does a flat-like radiation epoch appear?
scan = []
for lg in (-60, -40, -20, -10, -5, -3, -2, -1, 0, 1, 2, 3):
    gg = 10.0 ** lg
    sm = simulate(pot1, gg, (m_sim, 0.0), x0, t_end=300.0, n_out=3000)
    Y = gg * sm["a"] ** 2 * sm["rho"] / 3.0
    ok = np.isfinite(Y).all()
    # radiation-dominated, flat-like: Y >= 1/4 while rho_r dominates the field energy
    rad_dom = (sm["rho_r"] > 0.5 * np.maximum(sm["rho"], 1e-300)) & (Y >= 0.25)
    scan.append(dict(log10_g=lg, Y_max=float(np.max(Y)), flat_like_radiation_epoch=bool(rad_dom.any()), max_residual=float(np.max(np.abs(sm["resid"])))))
    print("A g=1e%d: Y_max %.2e  flat-like radiation %s  resid %.1e" % (lg, scan[-1]["Y_max"], scan[-1]["flat_like_radiation_epoch"], scan[-1]["max_residual"]), flush=True)
out["A_gravity_scan"] = scan

# ----------------------------------------------------------------------------------------------------------------- Part B
# potential in (x, y): V = V_phi(x) + (lam_y/4)(y^2 - w_y^2)^2 + (g_c^2/2)(x - x_c)^2 y^2 + V_off, x_c = x_t, V_off = -(dtil + E_res) so the false vacuum (x_f, 0) has energy ~ 0 (= rho_Lambda / (lam f^4))
def make_pot_B(w_y, lam_y=1.0, gc2=None):
    E_res = lam_y * w_y ** 4 / 4.0
    if gc2 is None:
        gc2 = 1.6 * lam_y * w_y ** 2 / (P.x_f - P.x_t) ** 2          # m_y^2(x_f) > 0: the reservoir field is stable in the false vacuum
    V_off = -(dtil + E_res)

    def pot(x, y):
        V = P.v(x) + 0.25 * lam_y * (y * y - w_y * w_y) ** 2 + 0.5 * gc2 * (x - P.x_t) ** 2 * y * y + V_off
        Vx = P.dv(x) + gc2 * (x - P.x_t) * y * y
        Vy = lam_y * y * (y * y - w_y * w_y) + gc2 * (x - P.x_t) ** 2 * y
        return float(V), float(Vx), float(Vy)
    return pot, V_off, E_res, gc2


rowsB = []
for kappa in (1.0, 1e5, 1e10, 1e20, 1e30, 1e35):
    w_y = 1.5
    pot, V_off, E_res, gc2 = make_pot_B(w_y)
    depth = abs(V_off)                                           # |V_t| in lam f^4 units
    f4 = kappa * K.V_F / depth                                    # f^4 in GeV^4 so that |V_t| = kappa rho_Lambda
    fB = f4 ** 0.25
    gB = (fB / M) ** 2
    ups = 0.5 * m_sim
    # false-vacuum energy sanity: V(x_f, 0) = V_off + dtil + E_res = 0
    sm = simulate(pot, gB, (ups, ups), x0, y0=1e-6, t_end=400.0, n_out=6000, rho_extra=0.0)
    Tt = temperature_tilde(sm["rho_r"], K.GSTAR_BBN)
    i = int(np.argmax(Tt))
    fin = sm["final"]
    a1, t1 = fin[0], sm["t"][-1]
    rho_r1 = fin[6]
    kin_f = 0.5 * fin[3] ** 2 + 0.5 * fin[5] ** 2
    V_final, _, _ = pot(fin[2], fin[4])
    rho_v = V_final + kin_f                                       # residual field energy counted with the vacuum (negligible after dissipation)
    t_c, tail = fluid_tail(a1, t1, rho_r1, V_off, gB)
    # analytic pure-AdS crunch time pi sqrt(3) M / sqrt(|V_t|) in the same units
    t_c_ads = np.pi / np.sqrt(gB * depth / 3.0)
    Tmax_gev = Tt[i] * fB
    # the radiation temperature when the interior is 1 s old (physical), and when it is as old as the crunch of the AdS-only solution
    tp = t_c * 1.0 / fB                                           # GeV^-1
    sec = fB / HBAR_S                                           # sim time units per second
    T_at_1s = None
    if tail is not None and t_c > sec:
        a_1s = np.sin(np.sqrt(tail["b"]) * sec) / np.sqrt(tail["b"])
        T_at_1s = float(temperature_tilde(rho_r1 * a1 ** 4 / a_1s ** 4, K.GSTAR_BBN) * fB)
    rho_exc_B = sm["kin"] + sm["V"] - V_off + sm["rho_r"]
    selB = sm["t"] >= 2.0 / m_sim
    E_rel_B = pot(x0, 1e-6)[0] - V_off
    env_B = float(np.max(rho_exc_B[selB] * (m_sim * sm["t"][selB]) ** 3) / E_rel_B)
    rowsB.append(dict(envelope_ratio=env_B, E_release_sim=float(E_rel_B), kappa=kappa, f_GeV=float(fB), g=float(gB), E_res_over_dtil=float(E_res / dtil), T_max_GeV=float(Tmax_gev), T_max_MeV=float(Tmax_gev * 1e3), t_peak_s=float(sm["t"][i] / fB * HBAR_S),
                      t_crunch_s=float(t_c / fB * HBAR_S), t_crunch_ads_only_s=float(t_c_ads / fB * HBAR_S), crunch_ratio=float(t_c / t_c_ads), T_at_1s_GeV=T_at_1s,
                      max_residual=float(np.max(np.abs(sm["resid"]))), Tt_product_sim=float(Tt[-1] * sm["t"][-1]), t_peak_sim=float(sm["t"][i])))
    print("B kappa %.0e: f=%.2e GeV T_max %.3g MeV  t_crunch %.3g s (AdS only %.3g s) resid %.1e T(1s) %s [%.0fs]" % (kappa, fB, rowsB[-1]["T_max_MeV"], rowsB[-1]["t_crunch_s"], rowsB[-1]["t_crunch_ads_only_s"],
          rowsB[-1]["max_residual"], T_at_1s, time.time() - T_START), flush=True)
out["B_reservoir_ads"] = dict(rows=rowsB, w_y=1.5, note="V_off compensates the stored reservoir so that V(false vacuum) = rho_Lambda; the true vacuum has V_t = V_off < 0")
# reservoir size at fixed depth ratio: T_max/f against w_y
rowsB2 = []
for wy in (0.5, 1.0, 1.5, 2.0, 3.0):
    pot, V_off, E_res, gc2 = make_pot_B(wy)
    sm = simulate(pot, 1e-43, (0.5 * m_sim, 0.5 * m_sim), x0, y0=1e-6, t_end=400.0, n_out=3000)
    Tt = temperature_tilde(sm["rho_r"], K.GSTAR_BBN)
    rowsB2.append(dict(w_y=wy, E_res=float(E_res), release_total=float(E_res + dtil), T_max_over_f=float(np.max(Tt)), Tt_product_sim=float(Tt[-1] * sm["t"][-1])))
out["B_reservoir_scan"] = rowsB2
print("B scan", [(r["w_y"], round(r["T_max_over_f"], 3), round(r["Tt_product_sim"], 2)) for r in rowsB2], flush=True)

# ----------------------------------------------------------------------------------------------------------------- Part C
# hilltop inflation at the dark-energy scale. Units H_Lambda = M_Pl = 1: rho_Lambda = 3, V = (lam_c/4)(chi^2 - w^2)^2 with lam_c w^4 / 4 = 3, Minkowski true vacuum.
rowsC = []
for w in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    lam_c = 12.0 / w ** 4
    pot = lambda x, y, lam_c=lam_c, w=w: (0.25 * lam_c * (x * x - w * w) ** 2, lam_c * x * (x * x - w * w), 0.0)
    m_vac = np.sqrt(2.0 * lam_c) * w
    chi0 = 1e-5 * w
    sm = simulate(pot, 1.0, (0.7 * m_vac, 0.0), chi0, t_end=4000.0, n_out=12000, t0=1e-4, rtol=1e-11, atol=1e-16)
    a = sm["a"]
    t = sm["t"]
    qdec = -a * np.gradient(sm["ad"], t) / sm["ad"] ** 2          # deceleration parameter q = -a addot / adot^2 (Milne: 0, de Sitter: -1)
    acc = qdec < -0.1
    # e-folds of clearly accelerated expansion
    N_acc = float(np.log(a[acc].max() / a[acc].min())) if acc.any() else 0.0
    Omk = 1.0 / (sm["ad"] ** 2)                                  # curvature density parameter
    Tt = temperature_tilde(sm["rho_r"], K.GSTAR_LOW)
    rowsC.append(dict(w_over_M=w, lam_c=float(lam_c), N_accel=N_acc, N_flatness=float(-0.5 * np.log(np.min(Omk))), slow_roll_estimate=float(w ** 2 / 4.0 * np.log(w / chi0)), Omega_k_min=float(np.min(Omk)), T_max_meV=float(np.max(Tt) * np.sqrt(K.H_L * M) * 1e12),
                      max_residual=float(np.max(np.abs(sm["resid"]))), t_end_over_Hinv=float(t[-1])))
    print("C w/M=%.0f: e-folds %.1f, flatness N %.1f (slow-roll %.1f) Omega_k,min %.1e  T_max %.2f meV  resid %.1e [%.0fs]" % (w, N_acc, rowsC[-1]["N_flatness"], rowsC[-1]["slow_roll_estimate"], rowsC[-1]["Omega_k_min"], rowsC[-1]["T_max_meV"], rowsC[-1]["max_residual"], time.time() - T_START), flush=True)
out["C_hilltop"] = dict(rows=rowsC, note="units H_Lambda = M_Pl = 1, chi0 = 1e-5 w; T_max in meV for the observed dark-energy density")
# required parent energy for a 1 MeV hot epoch with the same dynamics (the dynamics scale with H)
out["C_parent_energy_for_1MeV_over_rho_Lambda"] = float(((np.pi ** 2 * K.GSTAR_BBN / 30.0) * K.T_BBN_GEV ** 4) / K.V_F)

# ----------------------------------------------------------------------------------------------------------------- Part D
# D1: three-vacuum chain (f,f) -> (t,f) -> (t,t) with V_tt = 0: energy sum rule and the rate of each generation
def ln_rate_chain(lam_i, eta_i, dV, V_parent, lnD=0.0):
    """ln(Gamma/H^4) for a transition releasing dV out of a parent vacuum of energy V_parent (rates.py generalised: ln(M^4/V_parent), r_L = V_parent/dV)."""
    B = float(R.b_of_eta(eta_i)) / lam_i
    dtil_i = float(R.dtil_of_eta(eta_i))
    r_L = V_parent / dV
    K_ = (B / (2 * np.pi)) ** 2 * np.exp(lnD)
    return float(np.log(9.0 * K_ * lam_i / (r_L * dtil_i)) + np.log(M ** 4 / V_parent) - B), B


chain = []
for (e1, l1, e2, l2, split) in ((0.30, 1.0, 0.30, 1.0, 0.6), (0.32, 1.0, 0.28, 2.0, 0.6), (0.30, 1.0, 0.33, 0.5, 0.5), (0.27, 3.0, 0.27, 3.0, 0.5)):
    d1, d2 = split * K.V_F, (1.0 - split) * K.V_F
    ln1, B1 = ln_rate_chain(l1, e1, d1, K.V_F)
    ln2, B2 = ln_rate_chain(l2, e2, d2, d2)                       # generation 2: parent energy d2 (the daughter's vacuum energy), H2 = sqrt(d2 / 3) / M
    H2_over_H1 = float(np.sqrt(d2 / K.V_F))
    chain.append(dict(eta1=e1, lam1=l1, eta2=e2, lam2=l2, dV1_over_rho=split, dV2_over_rho=1 - split, B1=B1, B2=B2, ln_gamma1_over_H4=ln1, ln_gamma2_over_H2_4=ln2,
                      log10_t1_yr=float(R.log10_decay_time_years(ln1)), log10_t2_yr=float(np.log10(K.T_HUBBLE_YR / H2_over_H1) - np.log10((4 * np.pi / 3)) - ln2 / np.log(10.0)),
                      total_released_over_rho=float(split + (1 - split)), T_max_total_meV=float(K.t_reh_max(d1 + d2) * 1e12)))
out["D_chain"] = chain
print("D chain", [(c["eta1"], round(c["log10_t1_yr"]), round(c["log10_t2_yr"]), round(c["T_max_total_meV"], 2)) for c in chain], flush=True)

# D2: entropy per comoving curvature volume produced in the single-field and AdS runs, against the horizon entropy of the parent and the bound for the bubble volume
S_dS = 8.0 * np.pi ** 2 * M ** 2 / (K.H_L ** 2)                   # de Sitter horizon entropy of the parent, 8 pi^2 M^2 / H^2
R_b_over_RH = [p for p in json.load(open(os.path.join(HERE, "falsification_map.json"), encoding="utf-8"))["physical"] if p["lam"] == 1.0 and p["eta"] == 0.30][0]["bubble_radius_over_hubble"]
S_bound_bubble = 8.0 * np.pi ** 2 * M ** 2 * (R_b_over_RH / K.H_L) ** 2   # area/4G of a sphere of the wall radius
sm = simulate(pot1, g_phys, (m_sim, 0.0), x0, t_end=400.0, n_out=2000)
s_den = (2.0 * np.pi ** 2 / 45.0) * GSTAR * temperature_tilde(sm["rho_r"]) ** 3 * sm["a"] ** 3          # entropy per comoving curvature volume (f^-3 units): constant after reheating
S_c = float(s_den[-1])
# entropy inside the bubble-wall volume at the time of the temperature maximum: s V with V = (4 pi / 3) R_b^3 in physical units at that moment (T_max in meV, R_b in meV^-1)
Tmax_mev = [r for r in rowsA if r["gamma_over_m"] == 1.0][0]["T_max_meV"]
R_b_mev = R_b_over_RH / (K.H_L * 1e12)
S_in_bubble = float((2.0 * np.pi ** 2 / 45.0) * GSTAR * Tmax_mev ** 3 * (4.0 * np.pi / 3.0) * R_b_mev ** 3)
out["D_entropy"] = dict(S_comoving_curvature_volume=S_c, S_constant_after_reheating=bool(np.max(np.abs(s_den[-500:] / s_den[-1] - 1)) < 1e-2), S_dS_parent=float(S_dS), S_bound_wall_volume=float(S_bound_bubble),
                        S_in_wall_volume_at_Tmax=S_in_bubble, ratio_to_bound=float(S_in_bubble / S_bound_bubble), R_b_over_RH=float(R_b_over_RH))
print("D entropy", out["D_entropy"], flush=True)
json.dump(out, open(os.path.join(HERE, "interior_sim.json"), "w"), indent=1, default=float)
print("saved interior_sim.json [%.0fs]" % (time.time() - T_START))
