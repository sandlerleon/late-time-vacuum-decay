# -*- coding: utf-8 -*-
"""Section 7.1: does matter dilution create a metastable vacuum, i.e. two coexisting minima separated by a barrier?

Two matter-coupled potentials are mapped (both of the form V_eff(phi, rho_*) = V_0(phi) + rho_* A(phi), rho_* the conserved Einstein-frame density):

(1) the symmetron form of the held vacuum (Sections 6.2-6.15, matter_coupling.py):
        V_eff = V_c + (rho_*/M^2 - mu^2) phi^2/2 + lam phi^4/4 + eps phi
    The matter-held point phi = 0 loses stability at rho_crit = mu^2 M^2 without a barrier ever separating it from the broken minima: the transition
    is a classical (spinodal) roll, not tunneling. With a tilt eps the occupied minimum is followed continuously into the deeper broken minimum.
    The only interfaces are domain walls between the two broken minima, which form if the sign of the roll is not set coherently. The release is
    coherent if the classical seed (the restored value phi_rest or the tilt) exceeds the quantum fluctuation of the field on the Hubble scale.

(2) the tunneling branch (Section 6.1, density_bounce.py), in units x = phi/f, potential / lam f^4:
        p(x; eta, theta) = (x^2 - 1)^2/8 + eta x/2 + theta (x - 1)^2/2,      theta = rho_*/(lam f^2 M^2)
    Two minima coexist on a density interval; the matter-held minimum near x = 1 is the true vacuum above the degeneracy density theta_deg and
    metastable below it, and for eta > eta_c it disappears below theta_low (classical roll). The map gives, for each (eta, theta): the number of
    minima, which one is lower, the barrier height, and the energy difference.

Also: the logarithmic coupling estimate of coupling.py (delta V = rho_m Delta ln m) against the exact Einstein-frame matter term rho_* Delta A, and the
smoothing of the symmetron release by matter inhomogeneity (linear sigma(R) on the scale of the field's Compton wavelength, BBKS transfer function).

    python vacuum_phase_map.py          (writes vacuum_phase_map.json, ~10 s)
"""
import json
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import constants as K
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
ETA_C = 2.0 / (3.0 * np.sqrt(3.0))          # x^3 - x + eta = 0 has three real roots for |eta| < eta_c


# ------------------------------------------------------------------------------------------------ tunneling branch
def p_tb(x, eta, theta):
    return (x * x - 1.0) ** 2 / 8.0 + eta * x / 2.0 + theta * (x - 1.0) ** 2 / 2.0


def dp_tb(x, eta, theta):
    return x * (x * x - 1.0) / 2.0 + eta / 2.0 + theta * (x - 1.0)


def d2p_tb(x, eta, theta):
    return (3.0 * x * x - 1.0) / 2.0 + theta


def stationary_tb(eta, theta):
    """Real stationary points: x^3 + (2 theta - 1) x + (eta - 2 theta) = 0, sorted."""
    r = np.roots([1.0, 0.0, 2.0 * theta - 1.0, eta - 2.0 * theta])
    return np.sort(r[np.abs(r.imag) < 1e-9].real)


def classify_tb(eta, theta):
    """(n_minima, x_true, x_top, x_held, dV = p(held) - p(true) [>0: held state metastable], barrier = p(top) - p(held))."""
    st = stationary_tb(eta, theta)
    mins = [x for x in st if d2p_tb(x, eta, theta) > 0]
    if len(mins) == 1:
        return 1, mins[0], None, mins[0], 0.0, 0.0
    x_t, x_top, x_h = st                                  # the matter-held minimum is the one nearest x = 1 (the largest root)
    return 2, x_t, x_top, x_h, float(p_tb(x_h, eta, theta) - p_tb(x_t, eta, theta)), float(p_tb(x_top, eta, theta) - p_tb(x_h, eta, theta))


def window_tb(eta):
    """theta_low, theta_high: two minima exist for theta_low < theta < theta_high (theta_low = 0 for eta < eta_c), from the cubic discriminant."""
    def disc(th):
        pq, qq = 2.0 * th - 1.0, eta - 2.0 * th
        return -(4.0 * pq ** 3 + 27.0 * qq ** 2)          # > 0: three real roots
    th = np.linspace(0.0, 2.0, 20001)
    d = disc(th)
    ok = d > 0
    if not ok.any():
        return None, None
    k = np.where(ok)[0]
    lo = 0.0 if k[0] == 0 else brentq(disc, th[k[0] - 1], th[k[0]], xtol=1e-14)
    hi = brentq(disc, th[k[-1]], th[k[-1] + 1], xtol=1e-14)
    return float(lo), float(hi)


def theta_deg_tb(eta):
    lo, hi = window_tb(eta)
    if lo is None:
        return None
    g = lambda th: classify_tb(eta, th)[4]
    th = np.linspace(lo + 1e-9, hi - 1e-9, 4001)
    gv = np.array([g(t) for t in th])
    idx = np.where(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0)[0]
    return float(brentq(g, th[idx[0]], th[idx[0] + 1], xtol=1e-14)) if idx.size else None


def phase_map_tb(n_eta=121, n_th=241, eta_max=0.6, th_max=0.6):
    eta = np.linspace(0.0, eta_max, n_eta)
    th = np.linspace(0.0, th_max, n_th)
    code = np.zeros((n_eta, n_th), dtype=int)    # 0: one minimum (matter-held, x~1, stable); 1: two minima, held is true; 2: two minima, held metastable;
    for i, e in enumerate(eta):                  # 3: one minimum away from x~1 (the held state does not exist: classical roll)
        for j, t in enumerate(th):
            n, x_t, _, x_h, dV, _ = classify_tb(e, t)
            if n == 2:
                code[i, j] = 2 if dV > 0 else 1
            else:
                code[i, j] = 0 if x_h > 0 else 3
    lines = []
    for e in np.linspace(0.0, eta_max, 61):
        lo, hi = window_tb(e)
        lines.append(dict(eta=float(e), theta_low=lo, theta_high=hi, theta_deg=theta_deg_tb(e) if lo is not None else None))
    return dict(eta=eta.tolist(), theta=th.tolist(), code=code.tolist(), lines=lines)


def profile_tb(eta, thetas):
    """Along theta for one eta: minima, energy difference, barrier height (units lam f^4)."""
    rows = []
    for t in thetas:
        n, x_t, x_top, x_h, dV, bar = classify_tb(eta, t)
        rows.append(dict(theta=float(t), n_min=n, x_true=float(x_t), x_top=None if x_top is None else float(x_top), x_held=float(x_h), dV=dV, barrier=bar))
    return rows


# ------------------------------------------------------------------------------------------------ symmetron
def symmetron_map(eps_hat=(0.0, 1e-3, 1e-2), n=401):
    """Stationary points of v(y) = (r - 1) y^2/2 + y^4/4 + eps_hat y,  y = phi/v, r = rho_*/rho_crit, units mu^2 v^2 (lam v^2 = mu^2).
    Tracks the occupied minimum from high density (r = 3) down to r = 0 and records whether it ever becomes metastable (a lower minimum separated by a
    barrier while the occupied one still exists)."""
    out = {}
    for e in eps_hat:
        r = np.linspace(3.0, 0.0, n)
        y_occ = None
        rows, metastable = [], False
        for rr in r:
            roots = np.roots([1.0, 0.0, rr - 1.0, e])
            real = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
            mins = [y for y in real if 3 * y * y + rr - 1.0 > 0]
            v = lambda y: (rr - 1.0) * y * y / 2.0 + y ** 4 / 4.0 + e * y
            if y_occ is None:
                y_occ = mins[0]
            else:
                y_occ = min(mins, key=lambda y: abs(y - y_occ))     # follow the occupied minimum continuously
            vmin = min(v(y) for y in mins)
            meta = len(mins) == 2 and v(y_occ) > vmin + 1e-15
            metastable |= meta
            rows.append(dict(r=float(rr), n_min=len(mins), y_occupied=float(y_occ), occupied_is_global=bool(not meta)))
        out["%g" % e] = dict(rows=rows[::20], occupied_ever_metastable=bool(metastable),
                             spinodal_r=1.0 if e == 0 else float(1.0 - 3.0 * (e / 2.0) ** (2.0 / 3.0)))
    return out


def sigma_R_bbks(R_mpc, h=0.6766, om=0.3111, ob=0.0490, ns=0.9665, sigma8=0.8102):
    """Linear rms density contrast in a top-hat sphere of radius R (Mpc, not Mpc/h), BBKS transfer function with the Sugiyama shape parameter."""
    gam = om * h * np.exp(-ob * (1.0 + np.sqrt(2.0 * h) / om))

    def T(k):                                   # k in h/Mpc
        q = k / gam
        return np.log(1.0 + 2.34 * q) / (2.34 * q) * (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25

    def W(x):
        return 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3

    def s2(Rh):
        f = lambda lnk: np.exp(lnk) ** (3.0 + ns) * T(np.exp(lnk)) ** 2 * W(np.exp(lnk) * Rh) ** 2
        return quad(f, np.log(1e-5), np.log(50.0), limit=400)[0] / (2.0 * np.pi ** 2)
    norm = sigma8 ** 2 / s2(8.0)
    return float(np.sqrt(norm * s2(R_mpc * h)))


def symmetron_coherence():
    """Is the release of the held vacuum coherent? Compare the classical seed with the Hubble-scale quantum fluctuation, and estimate the smoothing of
    the release by density fluctuations on the Compton scale of the field."""
    CM = json.load(open(os.path.join(HERE, "cycle_map.json"), encoding="utf-8"))
    CR = json.load(open(os.path.join(HERE, "continuous_cycles.json"), encoding="utf-8"))
    mu, a_c, q = CR["params"]["mu"], CR["params"]["a_c"], CR["params"]["q"]
    m = MC.Model(a_c=a_c, mu=mu, q=q)
    phi_rest = [p_["phi_rest"] for p_ in CM["held"]["points"]]               # in units of v
    H0_mpl = K.H0_GEV / K.M_PL                                               # H_0 in M_Pl units
    H_L = np.sqrt(MC.OM_L)                                                   # Hubble rate of the held vacuum, units H_0
    dphi_q = H_L * H0_mpl / (2.0 * np.pi) / m.v                              # Hubble-scale quantum fluctuation, units v
    # tilt that makes the roll coherent: displacement eps t^2/2 over one Hubble time at the onset of the instability exceeds dphi_q
    eps_min = dphi_q * m.v * H_L ** 2                                        # units H_0^2 M_Pl  (field equation phi'' = -eps in H_0 units)
    eps_hat_min = eps_min / (mu ** 2 * m.v)                                  # relative to the natural scale mu^2 v
    # domain walls if the sign is incoherent: tension and thickness at density r = rho_*/rho_crit (V = lam/4 (phi^2 - phi_b^2)^2 + const)
    walls = []
    for r in (0.9, 0.5, 0.0):
        pb = m.v * np.sqrt(1.0 - r)
        sig = 2.0 * np.sqrt(2.0) / 3.0 * np.sqrt(m.lam) * pb ** 3             # H_0 M_Pl^2
        thick = 2.0 / (np.sqrt(2.0 * m.lam) * pb)                             # 2 / m_b, units 1/H_0
        walls.append(dict(r=r, sigma_H0MPl2=float(sig), thickness_over_H0inv=float(thick),
                          wall_energy_per_Hubble_volume_over_rho_crit=float(sig * H_L / (3.0 * H_L ** 2))))
    # release-time spread from density fluctuations on the Compton scale 1/mu of the field at release
    R_mpc = K.MPC_KM and (299792.458 / K.H0_KMS_MPC) / mu                    # c / (mu H_0) in Mpc
    sR = sigma_R_bbks(R_mpc)
    a_rel_spread = sR / 3.0                                                  # delta a / a at fixed rho_*
    growth = mu * a_rel_spread / H_L                                         # e-folds of tachyonic growth between early and late patches (rate ~ mu)
    return dict(phi_rest_over_v=dict(min=float(min(phi_rest)), median=float(np.median(phi_rest)), max=float(max(phi_rest))),
                dphi_quantum_over_v=float(dphi_q), seed_over_fluctuation_median=float(np.median(phi_rest) / dphi_q),
                eps_min_H0sq_MPl=float(eps_min), eps_hat_min=float(eps_hat_min), domain_walls=walls,
                compton_scale_Mpc=float(R_mpc), sigma_R_linear_today=sR, release_spread_delta_a_over_a=float(a_rel_spread),
                growth_efolds_between_patches=float(growth), field_ratio_between_patches=float(np.exp(growth)))


# ------------------------------------------------------------------------------------------------ coupling check
def coupling_check():
    """coupling.py uses delta(DeltaV) = rho_m Delta ln m. The Einstein-frame action gives delta(DeltaV) = rho_* Delta A exactly (A(phi_ref) = 1), i.e.
    rho_m (e^{Delta ln m} - 1) with rho_m = rho_* at the reference vacuum. Ratio exact / log = (e^d - 1)/d."""
    CJ = json.load(open(os.path.join(HERE, "coupling.json"), encoding="utf-8"))
    rows = []
    for r in CJ["rows"]:
        for c in r["cases"]:
            d = c["delta_ln_m"]
            rows.append(dict(eta=r["eta"], case=c["label"], delta_ln_m=d, exact_over_log=float(np.expm1(d) / d)))
    return rows


if __name__ == "__main__":
    out = {"eta_c": ETA_C}
    out["tunneling_map"] = phase_map_tb()
    out["tunneling_profiles"] = {"%.2f" % e: dict(window=window_tb(e), theta_deg=theta_deg_tb(e), rows=profile_tb(e, np.linspace(0.0, 0.5, 51)))
                                 for e in (0.20, 0.30, 0.36, 0.45)}
    out["symmetron"] = symmetron_map()
    out["symmetron_coherence"] = symmetron_coherence()
    out["coupling_check"] = coupling_check()
    json.dump(out, open(os.path.join(HERE, "vacuum_phase_map.json"), "w"), indent=1)
    for e, d in out["tunneling_profiles"].items():
        print("eta %s window %s theta_deg %s" % (e, d["window"], d["theta_deg"]))
    print("symmetron: occupied minimum ever metastable:", {k: v["occupied_ever_metastable"] for k, v in out["symmetron"].items()})
    print("coherence:", {k: v for k, v in out["symmetron_coherence"].items() if k != "domain_walls"})
    print("walls:", out["symmetron_coherence"]["domain_walls"])
    print("coupling check:", out["coupling_check"])
