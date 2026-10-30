# -*- coding: utf-8 -*-
"""Figures 17-21: continuous cycles without resets, causal controls, contraction stability, bounce validation, relics and the failure map.
    python make_cyc_figures.py   (needs continuous_cycles.json, cycle_scan.json, causal_controls.json, contraction_stability.json,
                                  bounce_validation.json, compact_relics.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import causal_controls as CCt
import continuous_cycles as CC
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.3, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"
T_ = lambda a, s: a.set_title(s, fontsize=8.5, fontweight="bold")
MEV = CC.T_UNIT_EV * 1e-6
COLS = ["lna", "t", "H", "T", "rho_th", "rho_m", "K", "V", "rs", "phi", "x", "J", "w"]


def H(c):
    h = np.array(c["hist"])
    return {k: h[:, i] for i, k in enumerate(COLS)}


def expanding(h, until_release=False):
    """Index mask of the expanding branch: from the start of the cycle to the first point with H < 0 (optionally only up to the release)."""
    n = h["H"].size
    neg = np.where(h["H"] < 0)[0]
    end = neg[0] if neg.size else n
    if until_release:
        restored = np.where(h["phi"][:end] < 0.01)[0]
        if restored.size:
            rel = np.where(h["phi"][restored[0]:end] > 0.1)[0]
            end = restored[0] + rel[0] if rel.size else end
    m = np.zeros(n, dtype=bool)
    m[:end] = True
    return m


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig17():
    R = J("continuous_cycles.json")
    cs = R["cycles"]
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.0))
    a = ax[0, 0]
    for k, (c, ls) in enumerate(((cs[0], ":"), (cs[1], "-"))):
        h = H(c)
        sel = expanding(h)
        T = h["T"][sel]
        pos = h["rho_th"][sel] + h["K"][sel] + np.maximum(h["V"][sel], 0) + h["rs"][sel]
        rad = T ** 4 / pos
        mat = np.where(h["x"][sel] > 10, h["rho_m"][sel] / pos, 0)
        vac = np.maximum(h["V"][sel], 0) / pos
        lab = (lambda s: s + (" (cycle 1)" if k else " (today on)"))
        a.semilogx(T * MEV, rad, color=C2, ls=ls, lw=1.2, label=lab("radiation") if k else None)
        a.semilogx(T * MEV, mat, color=C1, ls=ls, lw=1.2, label=lab("matter") if k else None)
        a.semilogx(T * MEV, vac, color=C3, ls=ls, lw=1.2, label=lab("vacuum") if k else None)
    a.invert_xaxis(); a.set_xlabel("bath temperature (MeV), expanding branch"); a.set_ylabel("fraction of the density")
    T_(a, "(a) A full thermal history in every cycle"); a.legend(fontsize=6.3, loc="center right")
    b = ax[0, 1]
    for c, col in ((cs[0], GR), (cs[1], C1), (cs[2], C4)):
        h = H(c)
        b.semilogy(h["t"] - h["t"][0] + (0 if c["n"] == 0 else 0), np.maximum(h["phi"], 1e-80), color=col, lw=1.1, label="cycle %d" % c["n"])
    b.set_xlabel(r"time within the cycle ($1/H_0$)"); b.set_ylabel(r"$|\varphi|/v$"); b.set_ylim(1e-80, 10)
    T_(b, "(b) Frozen at the bounce, restored, held, released"); b.legend(fontsize=6.3)
    c_ = ax[1, 0]
    h = H(cs[1])
    sel = expanding(h, until_release=True) & (h["x"] > 0.05)
    Y = h["rho_m"][sel] / h["T"][sel] ** 3 / CC.KAPPA
    c_.loglog(h["x"][sel], Y, color=C1, lw=1.2, label=r"$\rho_{dm}/T^3$ (cycle 1)")
    xs = np.logspace(-1, 3, 300)
    yeq = np.array([CC.neq_factor(x) for x in xs]) * 0.1 * xs * CC.Params(sigv=1.0).m / CC.KAPPA
    c_.loglog(xs, np.maximum(yeq, 1e-30), color=GR, ls=":", lw=1.0, label="equilibrium ($\\varphi=0$)")
    c_.axhline(1.0, color=C2, ls="--", lw=0.9, label="present value")
    c_.set_ylim(1e-3, 1e6); c_.set_xlabel(r"$x = m_{eff}/T$"); c_.set_ylabel(r"$(\rho_{dm}/T^3)/\kappa$")
    T_(c_, "(c) Dark matter re-thermalized and frozen out"); c_.legend(fontsize=6.3)
    d = ax[1, 1]
    n = np.array([c["n"] for c in cs if c["n"] >= 1 and "held_accelerated_efolds" in c])
    held = np.array([c["held_accelerated_efolds"] for c in cs if c["n"] >= 1 and "held_accelerated_efolds" in c])
    Yl = np.array([c.get("Y_held_over_kappa", np.nan) for c in cs if c["n"] >= 1 and "held_accelerated_efolds" in c])
    lg = np.array([c.get("lnG", np.nan) for c in cs if c["n"] >= 1 and "held_accelerated_efolds" in c])
    d.plot(n, held, "o", ms=2.2, color=C3, label="held e-folds")
    d.plot(n, Yl, "s", ms=2.0, color=C1, label=r"$Y/\kappa$")
    d.plot(n, lg, "^", ms=2.0, color=C2, label=r"$\ln G_n$")
    d.set_xlabel("cycle"); d.set_ylim(0, 9.5)
    T_(d, "(d) %d cycles without resets" % len(n)); d.legend(fontsize=6.3, ncol=3, loc="upper center")
    save(fig, "fig17_continuous_cycles.png")


def fig18():
    CJ = J("causal_controls.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.1))
    a = ax[0]
    m = MC.Model(2.0, 30.0, 1.0)
    for kind, col, lab in (("U", C4, "U: no coupling"), ("F", C3, "F: coupled, matter frozen"), ("D", C1, "D: coupled, matter dilutes")):
        r = CCt.evolve(m, kind, t_max=40.0, dense=True)
        s = r["sol"]
        tt = np.linspace(0, s.t[-1], 1500)
        a.plot(tt * MC.H0_GYR, s.sol(tt)[1], color=col, lw=1.3, label=lab)
    for q, col, lab in ((None, GR, r"P: $\Lambda$CDM"), (1.0, C2, "N: negative vacuum")):
        r = CCt.constant_vacuum(q, t_max=40.0, dense=True)
        s = r["sol"]
        tt = np.linspace(0, s.t[-1], 1500)
        a.plot(tt * MC.H0_GYR, s.sol(tt)[1], color=col, lw=1.0, ls="--", label=lab)
    a.axhline(0, color="k", lw=0.6); a.set_ylim(-1.5, 1.6); a.set_xlim(0, 400)
    a.set_xlabel("time from today (Gyr)"); a.set_ylabel(r"$H/H_0$")
    T_(a, "(a) Interventions ($a_c=2$, $\\mu=30H_0$, $q=1$)"); a.legend(fontsize=6.2, loc="lower left")
    b = ax[1]
    rows = CJ["rows"]
    A, M = CCt.A_C, CCt.MU
    for iq, q in enumerate(CCt.Q):
        for i, a_c in enumerate(A):
            for j, mu in enumerate(M):
                r = [x for x in rows if x["a_c"] == a_c and x["mu"] == mu and x["q"] == q][0]
                col = C1 if r["dilution_causes_recollapse"] else (C3 if not r["D"]["turn"] else C2)
                b.add_patch(plt.Rectangle((j + 0.05 + 5.4 * (iq % 2), i + 0.05 + 5.4 * (iq // 2)), 0.9, 0.9, color=col, lw=0))
        b.text(2.5 + 5.4 * (iq % 2), 5.08 + 5.4 * (iq // 2), "q = %g" % q, ha="center", va="bottom", fontsize=6.5)
    b.set_xlim(-0.2, 10.6); b.set_ylim(-0.2, 11.2); b.set_aspect("equal"); b.grid(False)
    b.set_xticks([j + 0.5 for j in range(5)] + [5.4 + j + 0.5 for j in range(5)]); b.set_xticklabels(["%g" % mu for mu in M] * 2, fontsize=5.5)
    b.set_yticks([i + 0.5 for i in range(5)] + [5.4 + i + 0.5 for i in range(5)]); b.set_yticklabels(["%g" % x for x in A] * 2, fontsize=5.5)
    b.set_xlabel(r"$\mu/H_0$"); b.set_ylabel(r"$a_c$")
    b.legend(handles=[Patch(color=C1, label="dilution causes recollapse (D turns, F expands)"), Patch(color=C3, label="release, expansion continues"),
                      Patch(color=C2, label="other")], fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    T_(b, "(b) Outcome map over the grid (F never turns)")
    save(fig, "fig19_causal_controls.png")


def fig19():
    S = J("contraction_stability.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.1))
    a = ax[0]
    for row, col in zip(S["scan"], (GR, C1, C3, C2)):
        a.loglog(row["s"], np.maximum(row["bounce_shear_fraction"], 1e-40), color=col, lw=1.3, label=row["case"] if row["ek"] is None else "ekpyrotic, " + row["case"])
    a.axhline(0.1, color="k", ls=":", lw=0.8); a.text(1e-39, 0.012, "control limit 0.1", fontsize=6.5)
    a.set_xlabel(r"initial anisotropy $\rho_\sigma/\rho_m$ at the turnaround"); a.set_ylabel(r"$\rho_\sigma/\rho$ at the bounce"); a.set_ylim(1e-40, 3)
    T_(a, "(a) Shear at the bounce"); a.legend(fontsize=6.3, loc="lower right")
    b = ax[1]
    oe = S["onset_exit"]
    for re, col in zip(sorted({r["rho_end_over_rho_c"] for r in oe}), (C1, C3, C2)):
        rr = sorted([r for r in oe if r["rho_end_over_rho_c"] == re], key=lambda r: r["f0"])
        b.loglog([r["f0"] for r in rr], [max(r["s_max"], 1e-40) for r in rr], "o-", color=col, ms=3, lw=1.1, label=r"exit at $\rho_{end}=10^{%d}\,\rho_c$" % round(np.log10(re)))
    noek = [r for r in S["scan"] if r["ek"] is None][0]["s_max"]
    b.axhline(noek, color=GR, ls="--", lw=0.9); b.text(1e-19, noek * 3, "no ekpyrosis", fontsize=6.5, color=GR)
    b.set_xlabel(r"ekpyrotic fraction $f_0$ at the turnaround ($c=5$)"); b.set_ylabel(r"largest controlled anisotropy $s_{max}$")
    T_(b, "(b) Tolerance set by the ekpyrotic onset"); b.legend(fontsize=6.3)
    save(fig, "fig20_contraction_stability.png")


def fig20():
    BV = J("bounce_validation.json")
    R = J("continuous_cycles.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.1))
    a = ax[0]
    h0, h1 = H(R["cycles"][0]), H(R["cycles"][1])
    rc = R["params"]["rho_c"]
    for h, sgn in ((h0, -1), (h1, 1)):
        rho = h["rho_th"] + h["K"] + h["V"] + h["rs"]
        p = h["w"] * rho
        sel = rho > 1e-6 * rc
        H2 = rho * (1 - rho / rc) / 3
        dH = -0.5 * (rho + p) * (1 - 2 * rho / rc)
        Rr = 6 * (dH + 2 * H2) / rc
        Kk = 12 * ((dH + H2) ** 2 + H2 ** 2) / rc ** 2
        x = sgn * np.abs(h["lna"][sel] - R["cycles"][0]["ln_a_min"])      # contraction (cycle 0) to the left of the bounce
        a.plot(x, Rr[sel], color=C1, lw=1.2, label=r"$R/\rho_c$" if sgn < 0 else None)
        a.plot(x, Kk[sel], color=C2, lw=1.2, label=r"$K/\rho_c^2$" if sgn < 0 else None)
        a.plot(x, rho[sel] / rc, color=GR, lw=0.9, ls=":", label=r"$\rho/\rho_c$" if sgn < 0 else None)
    a.axhline(4.0, color=C1, ls="--", lw=0.6); a.axhline(16 / 3, color=C2, ls="--", lw=0.6)
    a.set_xlim(-2.5, 2.5); a.set_xlabel(r"$\ln(a/a_{bounce})$"); a.set_ylabel("curvature invariants")
    T_(a, "(a) Finite invariants through the bounce"); a.legend(fontsize=6.3)
    b = ax[1]
    for row, col, mk in zip(BV["C"], (C1, C2), ("o", "s")):
        cs = [c for c in row["cycles"] if c.get("T_bounce")]
        b.semilogy([c["n"] for c in cs], [c["T_bounce"] * MEV for c in cs], mk + "-", color=col, ms=3.5, lw=1.1,
                   label=("effective LQC (density bound)" if row["bounce"] == "lqc" else r"ghost stiff fluid ($\rho_g=-g/a^6$)"))
    b.axhline(5.0, color=GR, ls=":", lw=0.9); b.text(0.05, 6.5, "BBN 5 MeV", fontsize=6.5, color=GR)
    b.set_xlabel("cycle"); b.set_ylabel("bounce temperature (MeV)")
    T_(b, "(b) Test C: an independent bounce"); b.legend(fontsize=6.3)
    save(fig, "fig21_bounce_validation.png")


def fig21():
    RL = J("compact_relics.json")
    SC = J("cycle_scan.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a = ax[0]
    for e, col, lab in (("E2", C1, "E2 fixed population"), ("E3", C2, "E3 accretion + mergers")):
        o = RL[e]
        a.loglog(np.array(o["T"]) * MEV, np.maximum(o["f_V"], 1e-60), color=col, lw=1.3, label=lab)
    a.axhline(1.0, color="k", ls=":", lw=0.8); a.text(1e-2, 0.03, "percolation", fontsize=6.5)
    a.axvline(RL["T_bounce_MeV"], color=GR, ls="--", lw=0.9); a.text(RL["T_bounce_MeV"] * 1.3, 1e-40, "bounce", fontsize=6.5, rotation=90, color=GR)
    a.set_xlabel("bath temperature in the contraction (MeV)"); a.set_ylabel("fraction of space inside horizons $f_V$"); a.set_ylim(1e-40, 10)
    T_(a, "(a) Black holes of the present cycle"); a.legend(fontsize=6.3, loc="center left")
    b = ax[1]
    cmap = {"working": C3, "F1": C2, "F2": C4, "F3": C1, "F4": "#6B4E8A", "F5": GR, "F6": "#222222"}
    ms_ = sorted({r["params"]["m_dm"] for r in SC["rows"] if set(r["params"]) == {"m_dm", "rho_c"}})
    rcs = sorted({r["params"]["rho_c"] for r in SC["rows"] if set(r["params"]) == {"m_dm", "rho_c"}})
    for r in SC["rows"]:
        if set(r["params"]) != {"m_dm", "rho_c"}:
            continue
        i, j = ms_.index(r["params"]["m_dm"]), rcs.index(r["params"]["rho_c"])
        key = "working" if r["classification"] == "working" else r["classification"][:2]
        b.add_patch(plt.Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, color=cmap.get(key, "k"), lw=0))
        b.text(j + 0.5, i + 0.5, "ok" if key == "working" else key, ha="center", va="center", fontsize=6.5, color="white")
    b.set_xlim(0, len(rcs)); b.set_ylim(0, len(ms_)); b.set_aspect("equal"); b.grid(False)
    b.set_xticks([j + 0.5 for j in range(len(rcs))]); b.set_xticklabels(["%.2g" % (r ** 0.25 * MEV) for r in rcs], fontsize=6.5)
    b.set_yticks([i + 0.5 for i in range(len(ms_))]); b.set_yticklabels(["%.2g" % (m * MEV) for m in ms_], fontsize=6.5)
    b.set_xlabel(r"bounce temperature $\rho_c^{1/4}$ (MeV)"); b.set_ylabel("dark-matter mass (MeV)")
    b.legend(handles=[Patch(color=cmap[k], label=l) for k, l in (("working", "working"), ("F1", "F1 not re-thermalized"), ("F2", "F2 not restored"),
                                                                    ("F6", "F6 numerical"))], fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    T_(b, "(b) Where the cycles work (%d cycles each)" % SC["n_cycles"])
    save(fig, "fig22_relics_failure_map.png")


def fig22():
    CM = J("cycle_map.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a = ax[0]
    for key, col, lab in (("5500000000.0", C1, "m = 10 MeV (robust)"), ("55000000000.0", C2, "m = 100 MeV (marginal)")):
        rows = [r for r in CM["map"].get(key, {}).get("rows", []) if r.get("phi_out") is not None]
        if not rows:
            continue
        x = [r["phi_in"] for r in rows]; y = [r["phi_out"] for r in rows]
        a.plot(x, y, "o-", color=col, ms=3.5, lw=1.1, label=lab)
        bad = [r for r in rows if r["classification"] != "working"]
        a.plot([r["phi_in"] for r in bad], [r["phi_out"] for r in bad], "x", color="k", ms=6)
    a.plot([0, 1.05], [0, 1.05], color=GR, ls=":", lw=0.9, label="identity")
    a.set_xlim(0, 1.05); a.set_ylim(0, 1.1); a.set_xlabel(r"$\varphi_b/v$ at a bounce"); a.set_ylabel(r"$\varphi_b/v$ at the next bounce")
    T_(a, "(a) Bounce-to-bounce return map"); a.legend(fontsize=6.3, loc="upper left")
    b = ax[1]
    pts = CM.get("held", {}).get("points", [])
    if pts:
        b.plot([p["predicted"] for p in pts], [p["held"] for p in pts], "o", color=C3, ms=2.5)
        lo = min(min(p["predicted"] for p in pts), min(p["held"] for p in pts)) - 0.1
        hi = max(max(p["predicted"] for p in pts), max(p["held"] for p in pts)) + 0.1
        b.plot([lo, hi], [lo, hi], color=GR, ls=":", lw=0.9)
        b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    b.set_xlabel("held e-folds, closed form"); b.set_ylabel("held e-folds, integration")
    T_(b, "(b) The held epoch in closed form (%d cycles)" % len(pts))
    save(fig, "fig18_cycle_map.png")


if __name__ == "__main__":
    for f in (fig17, fig18, fig19, fig20, fig21, fig22):
        try:
            f()
            print(f.__name__, "ok")
        except FileNotFoundError as e:
            print(f.__name__, "missing data:", e)
