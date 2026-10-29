# -*- coding: utf-8 -*-
"""Monte Carlo over the potential parameters, replacing the dimensionless toy of the first draft.

Each draw is a valid potential of cdl_lib (a false minimum exists by construction), its flat-space bounce action is interpolated
from the validated numerical table (bounce_tables.json), gravity is negligible (g ~ 1e-60, see falsification_map.py), and the
nucleation rate per Hubble four-volume follows from rates.py with the measured dark-energy density.

Priors (documented assumptions, varied in the sensitivity runs):
    log10 lam ~ U(-2, log10 4 pi)        quartic coupling, upper limit from perturbative unitarity
    eta       ~ U(0.03, 0.38)            tilt / barrier ratio; the false minimum exists for eta < 0.3849
    ln D      ~ N(0, (1.5 ln 10)^2)      one-loop determinant factor, K = (B / 2 pi)^2 D
    r_L       = 1 (Minkowski true vacuum) in the baseline; log-uniform on [1, 10] in the sensitivity run

    python monte_carlo.py                (writes monte_carlo.json)
"""
import json
import os

import numpy as np

import constants as K
import rates as R

HERE = os.path.dirname(os.path.abspath(__file__))
SURV = json.load(open(os.path.join(HERE, "survival.json"), encoding="utf-8"))
ETA_LO, ETA_HI = 0.03, 0.38
LAM_LO, LAM_HI = -2.0, np.log10(4.0 * np.pi)
SURVIVAL_X = SURV["gamma_over_hl4_max_95"]   # Gamma/H^4 at which the expected number of nucleations in the past light cone is -ln 0.05 (95% exclusion)
EUCLID_X = 3.0 / (8.0 * np.pi ** 2)           # the first-draft criterion (Euclidean four-sphere volume), kept only for the comparison in the text
PERC_X = (0.1, 1.0, 10.0)                     # rates (in H^-4) taken as "of order unity", i.e. able to percolate
HORIZONS_YR = [1e10, 1e12, 1e20, 1e50, 1e100]


def draw(n, seed, sigma_log10_D=1.5, rl_range=None, lam_range=(LAM_LO, LAM_HI), eta_range=(ETA_LO, ETA_HI)):
    rng = np.random.default_rng(seed)
    lam = 10.0 ** rng.uniform(lam_range[0], lam_range[1], n)
    eta = rng.uniform(eta_range[0], eta_range[1], n)
    lnD = rng.normal(0.0, sigma_log10_D * np.log(10.0), n)
    rl = np.ones(n) if rl_range is None else 10.0 ** rng.uniform(0.0, np.log10(rl_range), n)
    return lam, eta, lnD, rl


def evaluate(lam, eta, lnD, rl):
    lg, B = R.ln_gamma_over_h4(lam, eta, lnD, rl)
    return lg, B, R.log10_decay_time_years(lg)          # third entry: log10 of the decay time in years


def summarize(lg, B, lt):
    n = lg.size
    x = np.exp(np.clip(lg, -700.0, 50.0))
    t = lt                                               # log10(years)
    out = dict(n=int(n),
               frac_inconsistent_with_survival=float((x >= SURVIVAL_X).mean()),
               frac_B_below={("%g" % b_): float((B < b_).mean()) for b_ in (10.0, 30.0, 100.0)},
               frac_B_below_30_and_not_excluded=float(((B < 30.0) & (x < SURVIVAL_X)).mean()),
               frac_excluded_by_old_euclidean_criterion=float((x >= EUCLID_X).mean()),
               frac_percolating_and_surviving={("%g" % p): float(((x >= p) & (x < SURVIVAL_X)).mean()) for p in PERC_X},
               frac_decay_by={("%g" % h): float(((x < SURVIVAL_X) & (t <= np.log10(h))).mean()) for h in HORIZONS_YR},
               frac_never_within_1e100=float(((x < SURVIVAL_X) & (t > 100.0)).mean()),
               B_quantiles={q: float(np.quantile(B, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
               log10_t_quantiles={q: float(np.quantile(np.maximum(t, 0.0), q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
    return out


def batch_ci(n_total, seed, nb=20, **kw):
    """Batch-mean standard errors of the regime fractions."""
    per = n_total // nb
    fr = {k: [] for k in ("surv", "1e10", "1e12", "1e20", "1e50", "1e100", "never", "perc")}
    for b in range(nb):
        lam, eta, lnD, rl = draw(per, seed + 1000 + b, **kw)
        lg, B, t = evaluate(lam, eta, lnD, rl)
        s = summarize(lg, B, t)
        fr["surv"].append(s["frac_inconsistent_with_survival"])
        fr["perc"].append(s["frac_percolating_and_surviving"]["1"])
        for h in ("1e+10", "1e+12", "1e+20", "1e+50", "1e+100"):
            fr[h.replace("e+", "e")].append(s["frac_decay_by"][h])
        fr["never"].append(s["frac_never_within_1e100"])
    return {k: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(nb))) for k, v in fr.items()}


if __name__ == "__main__":
    SEED = 20261030
    N = 1_000_000
    res = {"seed": SEED, "N": N, "priors": dict(log10_lam=[LAM_LO, float(LAM_HI)], eta=[ETA_LO, ETA_HI], sigma_log10_D=1.5, r_L=1.0),
           "survival": dict(x_max=SURVIVAL_X, x_old_euclidean=EUCLID_X, V4_HL4=SURV["V4_HL4"])}
    lam, eta, lnD, rl = draw(N, SEED)
    lg, B, t = evaluate(lam, eta, lnD, rl)
    res["baseline"] = summarize(lg, B, t)
    res["baseline_batch"] = batch_ci(N, SEED)
    # the width of the reset window in B and its prior mass
    Bc = np.array([float(R.b_critical(l, e, d)) for l, e, d in zip(lam[:20000], eta[:20000], lnD[:20000])])
    res["B_critical_quantiles"] = {q: float(np.quantile(Bc, q)) for q in (0.05, 0.5, 0.95)}
    # convergence: the regime fractions as N grows (independent seeds)
    conv = []
    for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        l2, e2, d2, r2 = draw(n, SEED + 7 * n)
        s = summarize(*evaluate(l2, e2, d2, r2))
        conv.append(dict(n=n, surv=s["frac_inconsistent_with_survival"], d1e20=s["frac_decay_by"]["1e+20"], never=s["frac_never_within_1e100"],
                         B_median=s["B_quantiles"][0.5]))
    res["convergence"] = conv
    # sensitivity to the priors and to the prefactor
    sens = {}
    for name, kw in [("sigma_log10_D = 0", dict(sigma_log10_D=0.0)), ("sigma_log10_D = 3", dict(sigma_log10_D=3.0)), ("sigma_log10_D = 6", dict(sigma_log10_D=6.0)),
                     ("r_L ~ logU(1, 10)", dict(rl_range=10.0)), ("lam <= 1", dict(lam_range=(LAM_LO, 0.0))),
                     ("eta in [0.10, 0.38]", dict(eta_range=(0.10, 0.38))), ("log10 lam in [-4, 2]", dict(lam_range=(-4.0, 2.0)))]:
        l2, e2, d2, r2 = draw(N, SEED + 5, **kw)
        sens[name] = summarize(*evaluate(l2, e2, d2, r2))
    res["sensitivity"] = sens
    # decay-time CDF on a grid for the figure
    grid = np.arange(0, 201, 2.0)
    good = np.exp(np.clip(lg, -700.0, 50.0)) < SURVIVAL_X
    res["cdf"] = dict(log10_years=grid.tolist(), frac=[float((good & (t <= g_)).mean()) for g_ in grid])
    res["frac_survival_ok"] = float(good.mean())
    json.dump(res, open(os.path.join(HERE, "monte_carlo.json"), "w"), indent=1)
    b = res["baseline"]
    print("percolating+surviving", b["frac_percolating_and_surviving"], " old Euclidean criterion would exclude", b["frac_excluded_by_old_euclidean_criterion"])
    print("N=%d  inconsistent with survival %.4f ; decay by 1e12 %.4f, 1e20 %.4f, 1e50 %.4f, 1e100 %.4f ; never %.4f"
          % (N, b["frac_inconsistent_with_survival"], b["frac_decay_by"]["1e+12"], b["frac_decay_by"]["1e+20"], b["frac_decay_by"]["1e+50"],
             b["frac_decay_by"]["1e+100"], b["frac_never_within_1e100"]))
    print("B quantiles", b["B_quantiles"]); print("B_crit quantiles", res["B_critical_quantiles"])
    print("batch", res["baseline_batch"])
    for c in conv: print("conv", c)
    for k, v in sens.items(): print(k, round(v["frac_inconsistent_with_survival"], 4), round(v["frac_decay_by"]["1e+20"], 4), round(v["frac_never_within_1e100"], 4))
