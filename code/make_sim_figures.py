# -*- coding: utf-8 -*-
"""Figure 10: the simulated daughter region.   python make_sim_figures.py   (needs interior_sim.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
S = json.load(open(os.path.join(HERE, "interior_sim.json"), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"


def fig10():
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    a = ax[0]
    rows = {r["kappa"]: r for r in S["B_reservoir_ads"]["rows"]}
    t = np.logspace(-21.5, 4, 400)                                   # seconds
    for key, lab, col in ((1e35, "reservoir, $|V_t|=10^{35}\\rho_\\Lambda$", C2), (1e30, "$|V_t|=10^{30}\\rho_\\Lambda$", C4)):
        r = rows[key]
        T = np.where(t >= r["t_peak_s"], r["Tt_product_sim"] * K.HBAR_GEV_S / t * 1e3, np.nan)       # MeV
        T = np.where(t <= r["t_crunch_s"], T, np.nan)
        a.loglog(t, T, color=col, lw=1.8, label=lab)
        a.plot([r["t_peak_s"]], [r["T_max_MeV"]], "o", color=col, ms=4)
        a.axvline(r["t_crunch_s"], color=col, ls="--", lw=0.9)
    a.fill_between([1.0, 200.0], [0.1, 0.1], [1.0, 1.0], color=C3, alpha=0.25, label="nucleosynthesis window")
    a.set_xlim(1e-22, 1e4); a.set_ylim(1e-30, 1e1)
    a.set_xlabel("time since nucleation (s)"); a.set_ylabel("radiation temperature (MeV)")
    a.set_title("(a) Reservoir hidden by an AdS vacuum", fontsize=8.6, fontweight="bold"); a.legend(fontsize=6.6, loc="lower left")
    b = ax[1]
    rc = S["C_hilltop"]["rows"]
    w = [r["w_over_M"] for r in rc]
    b.plot(w, [r["N_flatness"] for r in rc], "o-", color=C1, lw=1.5, label="flatness $N=-\\ln\\Omega_{k,min}/2$ (simulation)")
    b.plot(w, [r["slow_roll_estimate"] for r in rc], "s--", color=GR, lw=1.0, ms=3, label="slow-roll estimate $\\frac{w^2}{4M^2}\\ln(w/\\chi_0)$")
    b.axhline(60, color="k", ls="--", lw=0.9); b.text(1.1, 64, "$N=60$", fontsize=7)
    b.set_xlabel("vev  $w / M_{Pl}$"); b.set_ylabel("e-folds")
    b2 = b.twinx(); b2.grid(False)
    b2.plot(w, [r["T_max_meV"] for r in rc], "^-", color=C2, lw=1.2, ms=4, label="$T_{max}$ (meV)")
    b2.set_ylim(0, 2.6); b2.set_ylabel("$T_{max}$ (meV)", color=C2)
    b2.axhline(K.t_reh_max(K.V_F) * 1e12, color=C2, ls=":", lw=0.9); b2.text(5.6, 2.28, "bound 2.2 meV", fontsize=6.8, color=C2)
    h1, l1 = b.get_legend_handles_labels()
    b.legend(h1, l1, fontsize=6.4, loc="upper left")
    b.set_title("(b) Flattening at the dark-energy scale", fontsize=8.6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_daughter_simulations.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig10_daughter_simulations.png")


if __name__ == "__main__":
    fig10()
