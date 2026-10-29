# -*- coding: utf-8 -*-
"""Thermal analysis (Steps 1-4): exponents against temperature, rates against redshift, survival with a time-dependent rate, and prior-mass maps.

Two regimes are distinguished and never mixed:
  decoupled   : the scalar does not interact with the CMB; T_gamma is only a clock, and the decay is the zero-temperature O(4) channel of Section 4;
  equilibrated: the scalar is in thermal equilibrium with the CMB bath below the redshift z_coup (thermal coefficient c): the exponent is that of the dominant saddle,
                O(3) if S_3/T < B_4 and O(4) otherwise, with the thermally rescaled potential. This is an upper bound on the thermal channel (equilibrium assumed).

    python thermal_analysis.py          (writes thermal_results.json; about 5 minutes)
"""
import json
import os
import time

import numpy as np
from scipy.optimize import brentq

import constants as K
import monte_carlo as M
import rates as R
import survival as S
import thermal as TH

HERE = os.path.dirname(os.path.abspath(__file__))
SV = json.load(open(os.path.join(HERE, "survival.json"), encoding="utf-8"))
N95 = SV["N_95"]
LN_N95 = float(np.log(N95))
out = {}
t0 = time.time()

# ---------------------------------------------------------------------------------------- (1) vacuum structure and critical temperature
lam, eta = 1.0, 0.30
f_mev = float(TH.f_scale_gev(lam, eta) * 1e12)
out["benchmark"] = dict(lam=lam, eta=eta, f_meV=f_mev, T_cmb_meV=TH.T0_MEV, theta_c=float(TH.theta_critical(eta)), eta_c=TH.ETA_C,
                        theta_today_per_c=float(TH.theta_of_T(1.0, lam, eta, TH.T0_MEV)),
                        m_false_meV=float(np.sqrt(abs(3 * __import__("cdl_lib").Potential(1.0, eta).x_f ** 2 - 1) / 2) * f_mev))
# temperature at which the false minimum disappears: theta(T_c) = theta_c  ->  T_c = f sqrt(lam theta_c / c)
out["T_c_meV"] = {("%g" % c): float(f_mev * np.sqrt(lam * TH.theta_critical(eta) / c)) for c in (0.001, 0.01, 0.1, 1.0, 10.0)}
out["z_c"] = {k: float(v / TH.T0_MEV - 1.0) for k, v in out["T_c_meV"].items()}

# ---------------------------------------------------------------------------------------- (3) exponents against temperature, crossover temperature
Tg = np.logspace(np.log10(0.05), np.log10(60.0), 220)
curves = {}
for c in (0.0, 0.1, 1.0, -1.0):
    B4, S3T, th, ex = TH.exponents(lam, eta, c, Tg)
    lr, ch = TH.ln_rate_over_hl4(lam, eta, c, Tg)
    curves[("%g" % c)] = dict(B4=[None if not np.isfinite(x) else float(x) for x in B4], S3_over_T=[None if not np.isfinite(x) else float(x) for x in S3T],
                              theta=th.tolist(), ln_rate=[None if not np.isfinite(x) else float(x) for x in lr], channel=ch.tolist())
out["curves"] = dict(T_meV=Tg.tolist(), by_c=curves)
cross = {}
for c in (0.0, 0.01, 0.1, 1.0, -0.1, -1.0):
    g = lambda T: float(TH.exponents(lam, eta, c, T)[1] - TH.exponents(lam, eta, c, T)[0])
    try:
        Tx = brentq(g, 0.02, 2.0, xtol=1e-10) if np.isfinite(g(0.02)) and np.isfinite(g(2.0)) else None
    except ValueError:
        Tx = None
    cross[("%g" % c)] = None if Tx is None else float(Tx)
out["crossover_T_meV"] = cross                                  # S_3/T = B_4 : the O(3) saddle dominates above, the O(4) below
out["crossover_z"] = {k: (None if v is None else float(v / TH.T0_MEV - 1.0)) for k, v in cross.items()}
# horizon temperature, the 'clock' temperature at which the CMB equals it, and the exponent S_3/T_dS (hopeless)
T_dS_mev = float(TH.HL_MEV / (2.0 * np.pi))
out["T_dS_meV"] = T_dS_mev
out["T_dS_over_T_cmb"] = T_dS_mev / TH.T0_MEV
out["S3_over_T_dS"] = float(f_mev * TH.s3_of_eta(eta) / T_dS_mev)
out["e_folds_until_T_cmb_equals_T_dS"] = float(np.log(TH.T0_MEV / T_dS_mev))
print("benchmark, crossover", cross, "T_dS", T_dS_mev, "[%.0fs]" % (time.time() - t0), flush=True)

# ---------------------------------------------------------------------------------------- rates against redshift for the benchmark
zg = np.concatenate([np.linspace(0, 12, 121), np.linspace(12.5, 60, 20)])
rows = {}
for label, c in (("decoupled", None), ("c=0", 0.0), ("c=0.1", 0.1), ("c=1", 1.0), ("c=-1", -1.0)):
    if c is None:
        lr = np.full(zg.shape, float(R.ln_gamma_over_h4(lam, eta)[0]))
    else:
        lr, _ = TH.ln_rate_over_hl4(lam, eta, c, TH.T0_MEV * (1 + zg))
    rows[label] = [None if not np.isfinite(x) else float(x) for x in np.atleast_1d(lr)]
out["rate_vs_z"] = dict(z=zg.tolist(), ln_rate=rows)

# ---------------------------------------------------------------------------------------- (4) survival with a time-dependent rate: benchmark
lnV4 = float(np.log(SV["V4_HL4"]))
surv = {}
for label, c, zc in (("decoupled", None, None), ("c=0, z<=1", 0.0, 1.0), ("c=0, z<=3", 0.0, 3.0), ("c=0, z<=10", 0.0, 10.0), ("c=0.1, z<=1", 0.1, 1.0), ("c=0.1, z<=3", 0.1, 3.0),
                     ("c=1, z<=1", 1.0, 1.0), ("c=-1, z<=1100", -1.0, 1100.0), ("c=-0.1, z<=1100", -0.1, 1100.0)):
    if c is None:
        lnI = float(R.ln_gamma_over_h4(lam, eta)[0]) + lnV4
        ok = True
    else:
        lnI_, ok_ = TH.ln_I(np.array([lam]), np.array([eta]), np.array([c]), z_max=zc)
        lnI, ok = float(lnI_[0]), bool(ok_[0])
    surv[label] = dict(ln_I=lnI, false_vacuum_at_all_z=ok, excluded=bool(lnI > LN_N95 or not ok))
out["benchmark_survival"] = surv
print("benchmark survival", {k: round(v["ln_I"], 1) for k, v in surv.items()}, "[%.0fs]" % (time.time() - t0), flush=True)
# consistency: constant-rate limit of the time-dependent integral reproduces survival.json
lr0 = np.array([R.ln_gamma_over_h4(lam, 0.32)[0]])
z_, w_ = TH.light_cone_grid(1100.0, 600)
out["light_cone_check"] = dict(V4_HL4_grid=float(np.sum(w_) * K.OMEGA_L ** 2), V4_HL4_survival_json=SV["V4_HL4"])

# ---------------------------------------------------------------------------------------- (4) prior-mass maps (same priors and seed as Section 4.5)
N = 200_000
lamD, etaD, lnD, rl = M.draw(N, MC_SEED := 20261030)
rng = np.random.default_rng(MC_SEED + 11)
lnDT = rng.normal(0.0, 1.5 * np.log(10.0), N)
lg0, B0, t0log = M.evaluate(lamD, etaD, lnD, rl)                  # decoupled: zero-temperature quantum channel
x0 = np.exp(np.clip(lg0, -700, 50))
base_excl = x0 >= M.SURVIVAL_X
fut_t = t0log                                                      # log10 years of the future (zero-temperature) decay time
res = {"N": N, "decoupled": dict(excluded=float(base_excl.mean()), inconsistent=0.0, allowed=float((~base_excl).mean()),
                                  decay_by_1e20=float(((~base_excl) & (fut_t <= 20)).mean()), no_decay_1e100=float(((~base_excl) & (fut_t > 100)).mean()))}
for label, c, zc in (("c=0 (equilibrated), z<=1", 0.0, 1.0), ("c=0, z<=3", 0.0, 3.0), ("c=0, z<=10", 0.0, 10.0), ("c=0.1, z<=1", 0.1, 1.0), ("c=0.1, z<=3", 0.1, 3.0),
                     ("c=1, z<=1", 1.0, 1.0), ("c=-0.1, z<=1100", -0.1, 1100.0), ("c=-1, z<=1100", -1.0, 1100.0)):
    lnI, ok = TH.ln_I(lamD, etaD, np.full(N, c), lnD, lnDT, rl, z_max=zc, n=300)
    incons = ~ok
    excl = ok & (lnI > LN_N95)
    allowed = ok & ~excl
    # decays also by the quantum (zero-temperature) channel in the future: the thermal channel switches off as T falls as 1/a
    res[label] = dict(c=c, z_coup=zc, inconsistent=float(incons.mean()), excluded=float(excl.mean()), allowed=float(allowed.mean()),
                      decay_by_1e20=float((allowed & (fut_t <= 20)).mean()), no_decay_1e100=float((allowed & (fut_t > 100)).mean()),
                      allowed_but_decoupled_excluded=float((allowed & base_excl).mean()), excluded_but_decoupled_allowed=float(((excl | incons) & ~base_excl).mean()))
    print(label, {k: round(v, 4) for k, v in res[label].items() if isinstance(v, float)}, "[%.0fs]" % (time.time() - t0), flush=True)
out["prior_maps"] = res

# a (eta, c) map of the survival exponent for lam = 1, z_coup = 3 (equilibrated)
etas = np.linspace(0.05, 0.38, 34)
cs = np.array([-1.0, -0.3, -0.1, -0.03, 0.0, 0.03, 0.1, 0.3, 1.0])
E, Cc = np.meshgrid(etas, cs)
lnI_map = np.zeros(E.shape)
ok_map = np.zeros(E.shape, dtype=bool)
for i in range(len(cs)):
    zc = 1100.0 if cs[i] < 0 else 3.0
    l_, o_ = TH.ln_I(np.ones(len(etas)), etas, np.full(len(etas), cs[i]), z_max=zc, n=300)
    lnI_map[i], ok_map[i] = l_, o_
out["map_eta_c"] = dict(eta=etas.tolist(), c=cs.tolist(), ln_I=[[None if not np.isfinite(x) else float(x) for x in r] for r in lnI_map], ok=ok_map.tolist(), ln_N95=LN_N95,
                        z_coup="3 for c >= 0, 1100 for c < 0")
json.dump(out, open(os.path.join(HERE, "thermal_results.json"), "w"), indent=1)
print("saved thermal_results.json [%.0fs]" % (time.time() - t0))
