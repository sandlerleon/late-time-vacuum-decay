# -*- coding: utf-8 -*-
"""Section 7.5: is the vacuum interface selective? Transmission and reflection of fields through the wall, and the pressure they exert on it.

Planar wall in its rest frame, field profile x(z) (units of f; z in 1/(sqrt(lam) f)). A test scalar psi with L_int = -(1/2) g^2 phi^2 psi^2 has
        m_psi^2(z) = m_0^2 + g^2 f^2 x(z)^2       (units lam f^2:  m^2 = m0^2 + gh^2 x^2,  gh = g / sqrt(lam))
and its modes obey  [ -d^2/dz^2 + m^2(z) ] u = k^2 u,  k^2 = omega^2 - k_par^2. With a transmitted wave e^{i k_+ z} on the exterior (false-vacuum) side
the solution is integrated back across the wall and decomposed into A e^{i k_- z} + B e^{-i k_- z} on the interior side; the flux-normalized
coefficients are T = (k_+ / k_-) / |A|^2 and R = |B/A|^2, and T + R = 1 is checked. Below the exterior threshold (k^2 < m_+^2) T = 0.

Validation (interface_validation.py): the sharp step, T = 4 k_- k_+ / (k_- + k_+)^2, and the Poeschl-Teller well of the degenerate kink x = tanh(z/2)
(eta = theta = 0), for which m^2 = gh^2 - gh^2 sech^2(z/2) and
        T = sinh^2(2 pi q) / [ sinh^2(2 pi q) + cos^2( (pi/2) sqrt(1 + 16 gh^2) ) ],     q^2 = k^2 - gh^2,
reflectionless when gh^2 = l (l + 1) / 4.

The coupled dark matter is far heavier than the wall scale (m / sqrt(lam) f ~ 1e9), so its transmission is a step: a particle at rest in the cosmic
frame crosses a wall moving with Lorentz factor gamma if gamma A_f > A_t, i.e. gamma > gamma_th = A(x_t) / A(x_f).

Pressure: reflected momentum flux of a thermal psi bath on a static wall, P = Integral d^3p/(2 pi)^3 f(E) (p_z/E) 2 p_z R(p_z); for an
ultra-relativistic wall P -> Delta m^2 T^2 / 24 per bosonic degree of freedom (Boedeker and Moore 2009); dark matter at most rho_* (A_t - A_f).

    python interface_transport.py          (writes interface_transport.json, ~1 min)
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp, quad

import constants as K
import matter_coupling as MC
import vacuum_phase_map as VP

HERE = os.path.dirname(os.path.abspath(__file__))


class Profile:
    def __init__(self, z, x):
        self.z, self.x = np.asarray(z, float), np.asarray(x, float)

    def __call__(self, z):
        return np.interp(z, self.z, self.x)


def kink_profile(L=40.0, n=8001):
    z = np.linspace(-L, L, n)
    return Profile(z, np.tanh(z / 2.0))


def wall_profile_from_bounce(prof, pad=25.0):
    """Planar wall from an O(4) bounce profile: z = r - R_w; interior side extended at the central value, exterior side at x_f."""
    r, x = np.array(prof["r"]), np.array(prof["x"])
    z = r - prof["R_w"]
    zl = np.linspace(z[0] - pad, z[0], 200)[:-1]
    zr = np.linspace(z[-1], z[-1] + pad, 200)[1:]
    return Profile(np.concatenate([zl, z, zr]), np.concatenate([np.full(zl.size, x[0]), x, np.full(zr.size, prof["x_f"])]))


def transmission(m2, k2, zL, zR, rtol=1e-10):
    """Flux-normalized (T, R) for -u'' + m2(z) u = k2 u. Incident from z < zL is irrelevant by symmetry of T for real potentials."""
    kp2, km2 = k2 - m2(zR), k2 - m2(zL)
    if kp2 <= 0 or km2 <= 0:
        return 0.0, 1.0
    kp, km = np.sqrt(kp2), np.sqrt(km2)

    def rhs(z, y):
        u = y[0] + 1j * y[1]; up = y[2] + 1j * y[3]
        upp = (m2(z) - k2) * u
        return [up.real, up.imag, upp.real, upp.imag]
    u0 = np.exp(1j * kp * zR); up0 = 1j * kp * u0
    sol = solve_ivp(rhs, (zR, zL), [u0.real, u0.imag, up0.real, up0.imag], rtol=rtol, atol=1e-12, method="DOP853")
    u = sol.y[0, -1] + 1j * sol.y[1, -1]; up = sol.y[2, -1] + 1j * sol.y[3, -1]
    A = 0.5 * (u + up / (1j * km)) * np.exp(-1j * km * zL)
    B = 0.5 * (u - up / (1j * km)) * np.exp(1j * km * zL)
    T = (kp / km) / abs(A) ** 2
    R = abs(B / A) ** 2
    return float(T), float(R)


def pt_exact(gh, k2):
    q2 = k2 - gh ** 2
    if q2 <= 0:
        return 0.0
    s = np.sinh(2.0 * np.pi * np.sqrt(q2)) ** 2
    return float(s / (s + np.cos(0.5 * np.pi * np.sqrt(1.0 + 16.0 * gh ** 2)) ** 2))


def reflection_grid(m2, zL, zR, pmax, nk=160):
    mR2 = m2(zR)
    pz = np.linspace(1e-3, pmax, nk)
    Rv = np.array([transmission(m2, p * p + mR2, zL, zR)[1] for p in pz])
    return pz, Rv


def thermal_pressure_static(m2, zR, Tb, pz, Rv):
    """Reflected momentum flux of a Bose bath (temperature Tb, units sqrt(lam) f) incident from the exterior on a static wall, and the incident
    momentum flux. The transverse integral is done in closed form: Integral d^2p_t f(E)/E = 2 pi Tb (-ln(1 - exp(-E_0/Tb))), E_0 = sqrt(p_z^2 + m^2)."""
    mR2 = m2(zR)
    E0 = np.sqrt(pz ** 2 + mR2)
    w = Tb * (-np.log(-np.expm1(-E0 / Tb))) / (4.0 * np.pi ** 2)
    P = float(np.trapezoid(w * 2.0 * pz ** 2 * Rv, pz))
    Pinc = float(np.trapezoid(w * pz ** 2, pz))
    return P, Pinc


if __name__ == "__main__":
    WJ = json.load(open(os.path.join(HERE, "vacuum_wall.json"), encoding="utf-8"))
    PS, TW = WJ["physical"], WJ["trigger_wall"]
    out = {}
    # ---- validation of the solver on the degenerate kink (Poeschl-Teller) and on a sharp step
    kp = kink_profile()
    val = []
    for gh in (0.5, np.sqrt(0.5), 1.0, np.sqrt(1.5)):
        m2 = lambda z, gh=gh: gh ** 2 * kp(z) ** 2
        for q in (0.05, 0.2, 0.5, 1.0):
            k2 = gh ** 2 + q * q
            T, R = transmission(m2, k2, -40.0, 40.0)
            val.append(dict(gh=float(gh), q=q, T=T, T_exact=pt_exact(gh, k2), unitarity=T + R))
    out["validation_poeschl_teller"] = val
    out["max_PT_error"] = max(abs(v["T"] - v["T_exact"]) for v in val)
    out["max_unitarity_error"] = max(abs(v["unitarity"] - 1) for v in val)
    # ---- the physical wall at the trigger (baseline eta, theta_trigger)
    import vacuum_wall as W
    prof = W.bounce_profile(WJ["base"]["eta"], TW["theta"])
    wp = wall_profile_from_bounce(prof)
    zL, zR = wp.z[0], wp.z[-1]
    out["wall"] = dict(x_interior=float(wp.x[0]), x_exterior=float(wp.x[-1]), zL=float(zL), zR=float(zR))
    species = {}
    for gh in (0.25, 0.5, 1.0, 2.0):
        m2 = lambda z, gh=gh: gh ** 2 * wp(z) ** 2
        mL2, mR2 = m2(zL), m2(zR)
        thr = max(mL2, mR2)
        ks = np.linspace(0.0, 4.0, 81)
        Ts = [transmission(m2, thr + k * k + 1e-9, zL, zR)[0] for k in ks]
        species["%.2f" % gh] = dict(m_in=float(np.sqrt(mL2)), m_out=float(np.sqrt(mR2)), k_above_threshold=ks.tolist(), T=Ts,
                                    k_half=float(ks[np.argmax(np.array(Ts) >= 0.5)]) if max(Ts) >= 0.5 else None)
    out["species"] = species
    # ---- pressure of a thermal psi bath on the static wall, and the ultra-relativistic (Boedeker-Moore) limit, relative to the driving pressure
    dVeff = TW["dV"]                                                       # units lam f^4
    press = []
    for gh in (0.5, 1.0, 2.0):
        m2 = lambda z, gh=gh: gh ** 2 * wp(z) ** 2
        dm2 = m2(zL) - m2(zR)
        pz, Rv = reflection_grid(m2, zL, zR, 12.0 + 3.0 * gh)
        for Tb in (0.05, 0.2, 1.0):                                         # units sqrt(lam) f  (f = 3 meV: 1 unit = 4.4 meV)
            P, Pinc = thermal_pressure_static(m2, zR, Tb, pz, Rv)
            press.append(dict(gh=gh, T=Tb, P_static_over_dV=P / dVeff, P_incident_over_dV=Pinc / dVeff,
                              P_BM_over_dV=float(max(dm2, 0.0) * Tb ** 2 / 24.0 / dVeff)))
    out["pressure"] = press
    T_cmb_trig = K.T_CMB_GEV / WJ["a_trigger"] / (np.sqrt(PS["lam"]) * PS["f_meV"] * 1e-12)
    out["T_cmb_at_trigger_units"] = float(T_cmb_trig)
    # ---- dark matter: mass ratio across the wall, threshold Lorentz factor, filtered fraction
    a_tr = WJ["a_trigger"]
    rho_star = K.RHO_CRIT * MC.OM_DM / a_tr ** 3                           # GeV^4 (A = 1 at x = 1)
    f2_over_M2 = TW["theta"] * PS["E0_GeV4"] / rho_star                   # f^2 / M^2 = theta lam f^4 / rho_*
    A = lambda x: 1.0 + (x - 1.0) ** 2 * f2_over_M2 / 2.0
    A_t, A_f = A(TW["x_t"]), A(TW["x_f"])
    gamma_th = A_t / A_f
    R0_mm = TW["R_w_mm"]
    H_tr = np.sqrt(K.V_F / 3.0) / K.M_PL * W.hubble_over_HL(a_tr)          # GeV
    R_H_mm = 1.0 / H_tr * K.HBAR_GEV_S * 2.99792458e11
    f_filt = (gamma_th * R0_mm / R_H_mm) ** 3
    P_dm_max = rho_star * (A_t - A_f)                                      # GeV^4
    dV_vac = TW["dV"] * PS["E0_GeV4"] + P_dm_max                          # vacuum part of the driving pressure (dV_eff = dV_vac - rho_* dA)
    out["dark_matter"] = dict(A_true=float(A_t), A_false=float(A_f), delta_ln_m=float(np.log(A_t / A_f)), gamma_threshold=float(gamma_th),
                              R0_mm=R0_mm, R_threshold_mm=float(gamma_th * R0_mm), hubble_radius_mm=float(R_H_mm), filtered_fraction=float(f_filt),
                              P_dm_max_over_dV_vac=float(P_dm_max / dV_vac), runaway=bool(P_dm_max < dV_vac))
    json.dump(out, open(os.path.join(HERE, "interface_transport.json"), "w"), indent=1)
    print("PT max error %.2e, unitarity %.2e" % (out["max_PT_error"], out["max_unitarity_error"]))
    print("wall", out["wall"])
    print({k: (round(v["m_in"], 3), round(v["m_out"], 3), v["k_half"]) for k, v in species.items()})
    for p_ in press:
        print(p_)
    print("T_cmb(trigger) =", T_cmb_trig, "units")
    print("DM:", out["dark_matter"])
