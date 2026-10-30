# -*- coding: utf-8 -*-
"""Figures 11 and 12: oscillating cycles (Option B).   python make_osc_figures.py   (needs oscillation.json, oscillation_validation.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
O = json.load(open(os.path.join(HERE, "oscillation.json"), encoding="utf-8"))
OV = json.load(open(os.path.join(HERE, "oscillation_validation.json"), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.3, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"


def fig11():
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    a = ax[0]
    tr = O["trace"]
    t = np.array(tr["t"]); H = np.array(tr["H"])
    P = O["periodic"]["period_mean"]
    sel = t < 3.2 * P
    a.plot(t[sel] / P, H[sel] / np.max(np.abs(H)), color=C1, lw=1.2, label="simulation")
    # sinusoid of the same period with zeros at the bounces
    tb0 = t[sel][np.argmax(H[sel])]
    a.plot(t[sel] / P, np.sin(2 * np.pi * (t[sel] - tb0) / P + np.pi / 2), color=C2, lw=1.0, ls="--", label="sinusoid, same period")
    a.set_xlabel("time / period"); a.set_ylabel("$H / H_{max}$"); a.set_ylim(-1.1, 1.1)
    a.set_title("(a) Radiation-only cycles", fontsize=8.5, fontweight="bold"); a.legend(fontsize=6.4, loc="lower left")
    b = ax[1]
    cols = {0.0: "k", 1e-4: C2, 3e-4: C4, 1e-3: C3, 1e-2: C1}
    for r in O["dissipation"]:
        am = np.array(r["a_max"])
        b.semilogy(np.arange(len(am)), am, "o-", ms=2.5, lw=1.1, color=cols.get(r["upsilon"], GR), label="$\\Upsilon P$ = %.1f" % r["upsilon_times_period"])
    b.set_xlabel("cycle"); b.set_ylabel("turnaround radius $a_{max}$")
    b.set_title("(b) Hysteresis and its damping", fontsize=8.5, fontweight="bold"); b.legend(fontsize=6.2)
    c = ax[2]
    fl = O["stability"]["scalar_floquet"]
    c.semilogy([f["m_over_H_ads"] for f in fl], [f["energy_multiplier_per_cycle"] for f in fl], "s-", color=C2, lw=1.2)
    c.axhline(1.0, color="k", ls="--", lw=0.8)
    c.set_xlabel("$m / H_{AdS}$"); c.set_ylabel("scalar energy multiplier per cycle")
    c.set_title("(c) Instability of the periodic orbit", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_oscillating_cycles.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig11_oscillating_cycles.png")


def fig12():
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    a = ax[0]
    for label, col in (("phi0=1e-3, no dissipation", C2), ("phi0=1e-3, Upsilon P=0.92", C4), ("phi0=1e-8, no dissipation", C1)):
        ct = OV["hysteresis"][label]
        W = np.array([c["W"] for c in ct]) / 3.0
        da = np.array([c["delta_a_max"] for c in ct])
        nxt = np.array([c["a_max_next"] for c in ct])
        ok = (np.abs(W) > 0) & (np.abs(da) > 1e-3 * np.array([c["a_max"] for c in ct])) & (nxt < 2e4)   # resolved, curvature-dominated turnarounds
        a.loglog(np.abs(W[ok]), np.abs(da[ok]), "o", ms=3, color=col, label=label.replace("phi0", "$\\varphi_0$").replace("Upsilon", "$\\Upsilon$"))
    lim = [1e-1, 1e5]
    a.loglog(lim, lim, "k--", lw=0.8)
    a.set_xlim(1e-1, 1e5); a.set_ylim(1e-1, 1e5)
    a.set_xlabel("$|W_n| / 3$"); a.set_ylabel("$|\\Delta a_{max,n}|$")
    a.set_title("(a) Hysteresis law $\\Delta a_{max} = W/3$", fontsize=8.5, fontweight="bold"); a.legend(fontsize=5.6, loc="upper left")
    b = ax[1]
    cmap = {"stable": C3, "growing": C4, "growing (runaway)": C2, "shrinking": C1, "irregular": GR, "terminating (turnaround lost)": "k"}
    for g in OV["phase_diagram"]:
        off = -0.07 if g["phi0"] < 1e-6 else 0.07
        b.scatter(np.log10(g["m"]) + off, g["upsilon_P"] if g["upsilon_P"] > 0 else 0.1, color=cmap.get(g["cls"], GR), marker="o" if g["phi0"] < 1e-6 else "s", s=26)
    for k_, v in cmap.items():
        if any(g["cls"] == k_ for g in OV["phase_diagram"]):
            b.scatter([], [], color=v, label=k_)
    b.set_yscale("log"); b.set_xlabel("$\\log_{10} m$"); b.set_ylabel("$\\Upsilon P$  (0 plotted at 0.1)")
    b.set_title("(b) 100-cycle phase diagram", fontsize=8.5, fontweight="bold"); b.legend(fontsize=5.8, loc="lower right")
    c = ax[2]
    T = OV["transitions"]
    for key, lab, col in (("A", "A: no transition", "k"), ("C1", "C1: energy to radiation", C3), ("C2", "C2: field displaced, $\\Upsilon P$=3", C4), ("C2_UP1", "C2, $\\Upsilon P$=1", C2)):
        am = np.array(T[key]["a_max"])
        c.semilogy(np.arange(len(am)), am, "o-", ms=2.2, lw=1.0, color=col, label=lab)
    c.axvline(3, color=GR, ls=":", lw=0.8)
    c.set_xlabel("cycle (transition at the 3rd turnaround)"); c.set_ylabel("$a_{max}$")
    c.set_title("(c) Vacuum transition during the cycles", fontsize=8.5, fontweight="bold"); c.legend(fontsize=5.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig12_hysteresis_stability.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig12_hysteresis_stability.png")


if __name__ == "__main__":
    fig11(); fig12()
