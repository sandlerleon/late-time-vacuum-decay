# -*- coding: utf-8 -*-
"""Figures 6 and 7 (finite-temperature analysis).   python make_thermal_figures.py   (needs thermal_tables.json, thermal_results.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as K
import survival as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
TT = json.load(open(os.path.join(HERE, "thermal_tables.json"), encoding="utf-8"))
TR = json.load(open(os.path.join(HERE, "thermal_results.json"), encoding="utf-8"))
SV = json.load(open(os.path.join(HERE, "survival.json"), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"
nz = lambda v: np.array([np.nan if x is None else x for x in v], dtype=float)


def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    a = ax[0]
    vs = [r for r in TT["vacuum_structure"] if r["eta"] == 0.30 and r["x_t"] is not None]
    th = np.array([r["theta"] for r in vs])
    for key, lab, col in (("x_t", "true minimum", C1), ("x_top", "barrier top", GR), ("x_f", "false minimum", C2)):
        a.plot(th, [r[key] for r in vs], color=col, lw=1.7, label=lab)
    thc = TR["benchmark"]["theta_c"]
    a.axvline(thc, color="k", ls="--", lw=0.9)
    a.text(thc + 0.004, -1.25, "$\\theta_c=%.3f$\nfalse minimum\ndisappears" % thc, fontsize=7)
    a.axvline(TR["benchmark"]["theta_today_per_c"], color=C3, ls=":", lw=1.0)
    a.text(TR["benchmark"]["theta_today_per_c"] + 0.004, 0.95, "today,\n$c=1$", fontsize=6.8, color=C3)
    a.set_xlabel("thermal strength  $\\theta = cT^2/(\\lambda f^2)$"); a.set_ylabel("extrema  $x=\\varphi/f$")
    a.set_title("(a) Vacuum structure, $\\eta=0.30$", fontsize=8.6, fontweight="bold"); a.legend(fontsize=6.8, loc="lower left")
    a.set_xlim(-0.3, 0.12)
    b = ax[1]
    T = np.array(TR["curves"]["T_meV"])
    cur = TR["curves"]["by_c"]
    b.loglog(T, nz(cur["0"]["B4"]), color=C1, lw=1.8, label="$B_4$ (quantum, O(4))")
    b.loglog(T, nz(cur["0"]["S3_over_T"]), color=C2, lw=1.8, label="$S_3/T$ (thermal, O(3)), $c=0$")
    b.loglog(T, nz(cur["-1"]["S3_over_T"]), color=C2, lw=1.0, ls="--", label="$S_3/T$, $c=-1$")
    b.loglog(T, nz(cur["0.1"]["S3_over_T"]), color=C2, lw=1.0, ls=":", label="$S_3/T$, $c=+0.1$")
    Tx = TR["crossover_T_meV"]["0"]
    b.axvline(Tx, color=GR, ls="--", lw=0.9); b.text(Tx * 0.93, 3.5e4, "$T_\\times=%.2f$ meV\nO(4) below,\nO(3) above" % Tx, fontsize=6.8, ha="right")
    b.axvline(K.T_CMB_GEV * 1e12, color=C3, ls="-.", lw=1.0); b.text(K.T_CMB_GEV * 1e12 * 1.03, 1.3e3, "CMB\ntoday", fontsize=6.8, color=C3)
    b.axvline(TR["T_c_meV"]["0.1"], color="k", ls=":", lw=0.9); b.text(TR["T_c_meV"]["0.1"] * 1.04, 600, "$T_c$ for\n$c=0.1$", fontsize=6.8)
    b.set_xlabel("bath temperature  $T$  (meV)"); b.set_ylabel("exponent")
    b.set_ylim(1, 3e5); b.set_xlim(0.05, 40)
    b.set_title("(b) Quantum and thermal exponents", fontsize=8.6, fontweight="bold"); b.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_thermal_exponents.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig6_thermal_exponents.png")


def fig7():
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    a = ax[0]
    z = np.array(TR["rate_vs_z"]["z"])
    sel = z <= 6
    R_ = TR["rate_vs_z"]["ln_rate"]
    a.plot(z[sel], nz(R_["decoupled"])[sel], color=C1, lw=1.9, label="decoupled (O(4) only)")
    a.plot(z[sel], nz(R_["c=0"])[sel], color=C2, lw=1.7, label="equilibrated, $c=0$")
    a.plot(z[sel], nz(R_["c=0.1"])[sel], color=C2, lw=1.0, ls=":", label="equilibrated, $c=+0.1$")
    a.plot(z[sel], nz(R_["c=-1"])[sel], color=C4, lw=1.2, ls="--", label="equilibrated, $c=-1$")
    a.axhline(np.log(SV["gamma_over_hl4_max_95"]), color="k", ls="--", lw=0.9)
    a.text(3.0, np.log(SV["gamma_over_hl4_max_95"]) + 8, "survival bound", fontsize=6.8)
    a.set_ylim(-200, 260)
    a.set_xlabel("redshift $z$ (time runs to the right)"); a.set_ylabel("$\\ln(\\Gamma/H_\\Lambda^4)$")
    a.invert_xaxis()
    a.set_title("(a) Rate history, $\\lambda=1$, $\\eta=0.30$", fontsize=8.6, fontweight="bold"); a.legend(fontsize=6.5, loc="upper left")
    b = ax[1]
    pm = TR["prior_maps"]
    labs = [("decoupled", "decoupled"), ("c=0 (equilibrated), z<=1", "$c=0$\n$z\\leq1$"), ("c=0, z<=3", "$c=0$\n$z\\leq3$"), ("c=0, z<=10", "$c=0$\n$z\\leq10$"),
            ("c=0.1, z<=3", "$c=0.1$\n$z\\leq3$"), ("c=1, z<=1", "$c=1$\n$z\\leq1$"), ("c=-1, z<=1100", "$c=-1$\n$z\\leq1100$")]
    x = np.arange(len(labs))
    inc = np.array([pm[k]["inconsistent"] for k, _ in labs]); exc = np.array([pm[k]["excluded"] for k, _ in labs])
    dec = np.array([pm[k]["decay_by_1e20"] for k, _ in labs]); alw = np.array([pm[k]["allowed"] for k, _ in labs]) - dec
    b.bar(x, exc, color=C2, label="above survival bound")
    b.bar(x, inc, bottom=exc, color=GR, label="no false minimum in the past")
    b.bar(x, alw, bottom=exc + inc, color="#DDDDDD", label="allowed, no decay by $10^{20}$ yr")
    b.bar(x, dec, bottom=exc + inc + alw, color=C3, label="allowed, decays by $10^{20}$ yr")
    b.set_xticks(x); b.set_xticklabels([l for _, l in labs], fontsize=6.2); b.set_ylabel("fraction of the prior"); b.set_ylim(0, 1.02)
    b.set_title("(b) Prior mass by thermal scenario", fontsize=8.6, fontweight="bold"); b.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_thermal_survival.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig7_thermal_survival.png")


if __name__ == "__main__":
    fig6(); fig7()
