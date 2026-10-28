# -*- coding: utf-8 -*-
"""From the dimensionless bounce action to cosmological decay rates.

The potential of cdl_lib is  V(phi) = f^4 * v(phi / f; lam, eta, v0)  with the energy scale fixed by the measured dark energy:
the false-vacuum energy is V_f = rho_Lambda, so that f^4 = V_f / (lam * dtil(eta) * r_L), r_L = V_f / DeltaV >= 1
(r_L = 1: decay to a Minkowski vacuum). Then

    B = b(eta) / lam                                   (flat-space action; gravity corrects it by a relative O(g), g ~ 1e-60)
    Gamma / H^4 = 9 K (M_Pl^4 / V_f) lam e^{-B} / (r_L dtil)          K = (B / 2 pi)^2 * D   (one-loop prefactor estimate)

and the mean decay time of a Hubble patch is  t_dec = H^-1 / ((4 pi / 3) Gamma / H^4)  when Gamma / H^4 << 1.
"""
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

import constants as K
import cdl_lib as C

HERE = os.path.dirname(os.path.abspath(__file__))
_T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
_ETA = np.array([r["eta"] for r in _T["flat"]])
_LNB = np.log([r["b"] for r in _T["flat"]])
_ORDER = np.argsort(_ETA)
_ETA, _LNB = _ETA[_ORDER], _LNB[_ORDER]
_SPL = PchipInterpolator(_ETA, _LNB)
_DT = PchipInterpolator(_ETA, np.log([r["dtil"] for r in np.array(_T["flat"], dtype=object)[_ORDER]]))
ETA_MIN, ETA_MAX = float(_ETA[0]), float(_ETA[-1])
ETA_C = C.eta_crit()


def b_of_eta(eta):
    """Flat-space bounce action for lam = 1 (interpolated table, validated off-grid in verify.py)."""
    eta = np.asarray(eta, dtype=float)
    return np.exp(_SPL(np.clip(eta, ETA_MIN, ETA_MAX)))


def dtil_of_eta(eta):
    return np.exp(_DT(np.clip(np.asarray(eta, dtype=float), ETA_MIN, ETA_MAX)))


def ln_gamma_over_h4(lam, eta, ln_D=0.0, r_L=1.0):
    B = b_of_eta(eta) / lam
    K_ = (B / (2.0 * np.pi)) ** 2 * np.exp(ln_D)
    return np.log(9.0 * K_ * lam / (r_L * dtil_of_eta(eta))) + K.LN_M4_OVER_VF - B, B


def log10_decay_time_years(ln_gamma_h4):
    """log10 of the mean time (years) for a Hubble patch to nucleate a bubble: t = H^-1 / ((4 pi / 3) Gamma / H^4). Works in the log
    domain because Gamma/H^4 spans hundreds of orders of magnitude."""
    return (np.log(K.T_HUBBLE_YR) - np.log(4.0 * np.pi / 3.0) - np.asarray(ln_gamma_h4, dtype=float)) / np.log(10.0)


def decay_time_years(ln_gamma_h4):
    """The same in years (may overflow to inf for absurdly small rates; use log10_decay_time_years for those)."""
    with np.errstate(over="ignore"):
        return 10.0 ** np.minimum(log10_decay_time_years(ln_gamma_h4), 300.0)


def b_critical(lam, eta, ln_D=0.0, r_L=1.0):
    """The action at which Gamma/H^4 = 1: B_c = ln(M^4/V_f) + ln(9 K lam / (r_L dtil)); solved by fixed-point iteration."""
    Bc = K.LN_M4_OVER_VF
    for _ in range(40):
        Bc = K.LN_M4_OVER_VF + np.log(9.0 * (Bc / (2 * np.pi)) ** 2 * np.exp(ln_D) * lam / (r_L * dtil_of_eta(eta)))
    return Bc
