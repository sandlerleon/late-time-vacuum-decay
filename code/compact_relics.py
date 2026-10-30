# -*- coding: utf-8 -*-
"""Compact-object relics: the black holes of a cycle in the contraction and at the bounce.

Mass function n(M, t) (comoving number per log-mass bin), evolved through the contracting branch of a cycle computed by continuous_cycles.py:
    dn/dt + 3H n + d(Mdot n)/dM = C_merge + S_form
  accretion   Mdot = 27 pi G^2 M^2 rho_r / c^3 (capture of radiation), capped at the Hubble-horizon mass M_H = 4 pi M_Pl^2/H_rho, H_rho = (rho_th/3)^(1/2) where |H| is smaller (at the bounce): in these units
              dM/dt = 3.64e-23 M^2 rho_r  (M in solar masses, t in 1/H0, rho_r in H0^2 M_Pl^2), solved along characteristics: 1/M = 1/M0 - 3.64e-23 J
  mergers     binned Smoluchowski coagulation with the capture cross-section sigma = pi (R_1 + R_2)^2 (1 + c^2/v^2), peculiar velocity
              v = min(c, v_0 a_0/a), v_0 = 300 km/s, conserving mass
  formation   S_form: the present-day population (stellar remnants, Omega = 1e-4 around 10 M_sun; supermassive, Omega = 7.6e-6 around 10^8 M_sun,
              log-normal widths 0.5 and 1 dex, exponential cut-offs above 100 and 5e9 M_sun), formed before today; experiments:
                E1  none (no black holes): the homogeneous cycle;
                E2  fixed fraction: the present population, no accretion or mergers;
                E3  evolving: the present population with accretion (capped at the horizon) and mergers.
The diagnostic that decides whether a homogeneous bounce is possible is the fraction of space inside black-hole horizons,
    f_V = sum_i n_i (4 pi/3) R_s,i^3,  R_s = 2GM/c^2;
where f_V approaches 1 (percolation) the universe is no longer a near-homogeneous fluid with dilute black holes. The black holes that reach the
bounce are relics of the next cycle (dust that does not take part in the re-thermalization); their abundance per unit entropy is diluted by the
entropy growth of the cycle, S_n/S_n+1.

    python compact_relics.py     (writes compact_relics.json)
"""
import json
import os
import time

import numpy as np

import continuous_cycles as CC

HERE = os.path.dirname(os.path.abspath(__file__))
K_ACC = 3.64e-23                 # dM/dt = K_ACC M^2 rho_r
MH_COEF = 4.64e22                # M_H = MH_COEF / H  (solar masses, H in H0)
FV_COEF = 4.64e-46               # f_V today = FV_COEF Omega mu^2 for a population of mass mu M_sun (per unit Omega)
RS_OVER_HUBBLE = 2.708e-22 / (4 * np.pi)   # R_s H0 / c per solar mass:  R_s = M/(4 pi M_Pl^2)
V0_C = 1e-3                      # 300 km/s


def population(lgM, omega=(1e-4, 7.6e-6), centre=(1.0, 8.0), width=(0.5, 1.0), cutoff=(100.0, 5e9)):
    """Omega per log-mass bin (summing to the given Omegas)."""
    om = np.zeros_like(lgM)
    for o, c, w, mc in zip(omega, centre, width, cutoff):
        g = np.exp(-0.5 * ((lgM - c) / w) ** 2 - 10.0 ** lgM / mc)          # log-normal with an exponential high-mass cutoff
        om += o * g / g.sum()
    return om


def evolve(hist, experiment="E3", n_bins=120, lg_range=(-1.0, 13.0), merge=True):
    """hist: rows [ln a, t, H, T, ...] of the contracting branch (from today's epoch on), in order of time."""
    lg = np.linspace(lg_range[0], lg_range[1], n_bins)
    M = 10.0 ** lg
    if experiment == "E1":
        return dict(experiment="E1", f_V_bounce=0.0, f_V_max=0.0)
    om = population(lg)
    # comoving number per bin (in units of rho_c0 / M_sun per unit comoving volume): N_i = Omega_i / M_i
    N = om / M
    la = np.array([h[0] for h in hist]); t = np.array([h[1] for h in hist]); H = np.array([h[2] for h in hist]); T = np.array([h[3] for h in hist])
    J = np.array([h[11] for h in hist])
    rho_th = np.array([h[4] for h in hist])                      # integral of rho_r dt (exact, from the integration)
    a0 = 1.0
    out = dict(experiment=experiment, T=[], f_V=[], M_max=[], merge_prob=[], frac_capped=[])
    f_max = 0.0
    merged = 0.0
    for k in range(1, la.size):
        dt = t[k] - t[k - 1]
        dt = max(dt, 0.0)                                    # near the bounce t (stored per cycle) no longer resolves the steps; J is exact
        a = np.exp(la[k])
        if experiment == "E3":
            # accretion along characteristics, capped at the horizon mass
            MH = MH_COEF / max(abs(H[k]), np.sqrt(rho_th[k] / 3.0))
            dJ = J[k] - J[k - 1]
            inv = 1.0 / M - K_ACC * dJ
            Mn = np.where(inv > 1.0 / MH, 1.0 / np.maximum(inv, 1e-300), np.maximum(M, MH))
            M = np.minimum(np.maximum(Mn, M), np.maximum(MH, M))
            if merge:
                # binned coagulation: rate per pair (comoving) = sigma v / a^3 (physical density n = N rho_c0 / (M_sun a^3))
                v = min(1.0, V0_C * a0 / a)
                Rs = RS_OVER_HUBBLE * M                        # in c/H0
                # physical number density in (H0/c)^3: n = N rho_c0 / (M_sun a^3), rho_c0/M_sun in (H0/c)^3 = 3 M_Pl^2 H0^2 / M_sun * (c/H0)^3 = 3 M_Pl^2/(H0 M_sun)
                n_phys = N * (3.0 / 2.708e-22) / a ** 3
                R2 = (Rs[:, None] + Rs[None, :]) ** 2
                sig = np.pi * R2 * (1.0 + 1.0 / v ** 2)
                rate = sig * v * dt                            # per target per unit number density
                P = np.minimum(rate * n_phys[None, :], 1.0)    # probability that a black hole of bin i merges with one of bin j
                # mass fraction absorbed in this step (a black hole of bin j is absorbed by targets i with probability sum_i N_i P_ij / N_j)
                absorbed = np.minimum((N[:, None] * P).sum(axis=0) / np.maximum(N, 1e-300), 1.0)
                frac = float(np.sum(absorbed * N * M) / max(np.sum(N * M), 1e-300))
                merged += frac
                if frac > 1e-15:
                    dN = np.zeros_like(N)
                    for i in range(n_bins):
                        for j in range(i, n_bins):
                            if P[i, j] < 1e-14:
                                continue
                            pairs = min(N[i] * P[i, j] * (0.5 if i == j else 1.0), N[i], N[j])
                            m_new = M[i] + M[j]
                            kk = min(int(np.searchsorted(M, m_new)), n_bins - 1)
                            dN[i] -= pairs; dN[j] -= pairs
                            dN[kk] += pairs * m_new / M[kk]
                    N = np.maximum(N + dN, 0.0)
        Rs = RS_OVER_HUBBLE * M
        n_phys = N * (3.0 / 2.708e-22) / a ** 3
        fV = float(np.sum(n_phys * 4.0 * np.pi / 3.0 * Rs ** 3))
        f_max = max(f_max, fV)
        if k % max(1, la.size // 200) == 0 or k == la.size - 1:
            out["T"].append(float(T[k])); out["f_V"].append(fV); w = N * M / max(np.sum(N * M), 1e-300)
            out["M_max"].append(float(M[np.searchsorted(np.cumsum(w), 0.999)]))      # 99.9th mass percentile
            out["merge_prob"].append(merged)
            out["frac_capped"].append(float(np.sum(N * M * (M >= MH_COEF / max(abs(H[k]), np.sqrt(rho_th[k] / 3.0)))) / max(np.sum(N * M), 1e-300)))
    out["f_V_bounce"] = out["f_V"][-1]
    out["f_V_max"] = f_max
    out["mass_density_bounce_over_today"] = float(np.sum(N * M) / np.sum(om))
    out["merged_mass_fraction"] = merged
    out["T_bounce"] = float(T[-1])
    return out


def percolation_temperature(omega, mu, G_entropy=1.0):
    """T at which f_V = 1 in a contraction with aT constant (radiation era), for a population (Omega, mu M_sun) without growth.
    f_V = FV_COEF Omega mu^2 a^-3 and a = (T0/T) G_entropy (entropy produced since today enlarges a at fixed T)."""
    fv0 = FV_COEF * omega * mu ** 2
    a_p = fv0 ** (1.0 / 3.0)
    return CC.T0 / a_p * G_entropy


if __name__ == "__main__":
    t0 = time.time()
    P = CC.Params()
    r = CC.run(P, n_cycles=0)
    c0 = r["cycles"][0]
    hist = [h for h in c0["hist"] if h[2] < 0]                 # contracting branch of the present cycle
    hist.sort(key=lambda h: h[1])
    out = dict(cycle=dict(T_bounce=c0["T_bounce"], ln_a_min=c0["ln_a_min"], ln_S_turn=c0["ln_S_turn"], ln_S_bounce=c0["ln_S_bounce"]))
    for e in ("E1", "E2", "E3"):
        o = evolve(hist, e)
        out[e] = o
        print(e, {k: v for k, v in o.items() if not isinstance(v, list)}, "[%.0fs]" % (time.time() - t0), flush=True)
    # analytic percolation temperatures (physical units, MeV), with the entropy growth of the present cycle up to the bounce
    Gs = float(np.exp((c0["ln_S_bounce"] - 3 * np.log(CC.T0)) / 3.0))      # (S_bounce / S_today)^(1/3), S_today = T0^3 (a = 1)
    out["percolation"] = {}
    for lab, om, mu in (("stellar 10 Msun", 1e-4, 10.0), ("SMBH 1e8 Msun", 7.6e-6, 1e8), ("SMBH 1e9 Msun", 7.6e-6, 1e9), ("SMBH 1e10 Msun", 7.6e-6, 1e10)):
        Tp = percolation_temperature(om, mu, Gs)
        out["percolation"][lab] = dict(T_units=Tp, T_MeV=Tp * CC.T_UNIT_EV * 1e-6)
    out["entropy_growth_to_bounce_cuberoot"] = Gs
    out["T_bounce_MeV"] = c0["T_bounce"] * CC.T_UNIT_EV * 1e-6
    # percolation temperature of the evolved population, extrapolated from the bounce with f_V ~ a^-3 ~ T^3 (accretion would only raise f_V further)
    out["T_perc_E3_MeV"] = out["T_bounce_MeV"] * out["E3"]["f_V_bounce"] ** (-1.0 / 3.0) if out["E3"]["f_V_bounce"] > 0 else None
    out["M_runaway_Msun"] = 1.0 / (K_ACC * hist[-1][11]) if hist[-1][11] > 0 else None    # J since the start of the cycle
    print(out["percolation"], "G_S", Gs, "T_b MeV", out["T_bounce_MeV"], "M_runaway", out["M_runaway_Msun"])
    json.dump(out, open(os.path.join(HERE, "compact_relics.json"), "w"), indent=1)
    print("[%.0fs]" % (time.time() - t0))
