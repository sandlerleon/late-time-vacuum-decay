# -*- coding: utf-8 -*-
"""Figures 13-16: the matter-held vacuum, its release, the tunneling branch and the hot cycles.
    python make_dil_figures.py     (needs turnaround_scan.json, dilution_thermal.json, density_bounce.json, hot_cycles.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dilution_background as D
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.3, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"
T_ = lambda a, s: a.set_title(s, fontsize=8.5, fontweight="bold")


def fig13():
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    m = MC.Model(2.0, 100.0, 1.0)
    a = ax[0]
    ph = np.linspace(-1.3, 1.3, 400) * m.v
    for r, col in ((2.0, C1), (1.0, GR), (0.5, C4), (0.0, C2)):
        rs = r * m.rho_crit
        Ve = m.V0(ph) + rs * (m.A(ph) - 1.0)
        a.plot(ph / m.v, Ve / MC.VC, color=col, lw=1.3, label=r"$\rho_*/\rho_{crit}$ = %g" % r)
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel(r"$\varphi / v$"); a.set_ylabel(r"$V_{eff} - \rho_*$  (units of $V_c$)")
    T_(a, "(a) The vacuum held by matter"); a.legend(fontsize=6.3)
    b = ax[1]
    c = ax[2]
    for lab, kw, col in (("coupled (C)", dict(coupled=True), C1), ("uncoupled control (B)", dict(coupled=False), C2)):
        mm = MC.Model(2.0, 100.0, 1.0, **kw)
        r = D.evolve(mm, 1e-60 * mm.v, a_i=1e-3, t_max=45.0, rtol=1e-11, dense=True)
        sol = r["sol"]
        tt = np.linspace(sol.t[0], sol.t[-1], 4000)
        yy = sol.sol(tt)
        t0 = r["t_today_from_start"]
        b.plot((tt - t0) * MC.H0_GYR, yy[1], color=col, lw=1.3, label=lab)
        c.semilogx(np.exp(yy[0]), np.abs(yy[2]) / mm.v + 1e-70, color=col, lw=1.2, label=lab)
    lc = D.lcdm_future(45.0)
    tl = np.linspace(-13.8, 120, 400)
    b.axhline(np.sqrt(MC.VC / 3.0), color=GR, ls=":", lw=0.9, label=r"$\Lambda$CDM, $H \to H_\Lambda$")
    b.axhline(0, color="k", lw=0.6)
    b.set_xlim(-14, 110); b.set_ylim(-2, 3)
    b.set_xlabel("time from today (Gyr)"); b.set_ylabel(r"$H / H_0$")
    T_(b, "(b) Expansion, turnaround, recollapse"); b.legend(fontsize=6.2, loc="upper right")
    c.set_yscale("log"); c.set_ylim(1e-62, 10)
    c.axvline(1.0, color=GR, ls=":", lw=0.9); c.text(1.05, 1e-55, "today", fontsize=6.5, color=GR)
    c.axvline(2.0, color=C3, ls="--", lw=0.9); c.text(2.1, 1e-45, "$a_c$", fontsize=7, color=C3)
    c.set_xlabel("scale factor $a$"); c.set_ylabel(r"$|\varphi| / v$")
    T_(c, "(c) Held until $a_c$, then released"); c.legend(fontsize=6.2, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_matter_held_vacuum.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig13")


def fig14():
    S, TH = J("turnaround_scan.json"), J("dilution_thermal.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    a = ax[0]
    xs = np.linspace(0, 2.5, 200)
    a.fill_between(xs, np.maximum(1 - xs, 0), 2.5, color="#E6EEF5", label="symmetric point held")
    a.plot(xs, np.maximum(1 - xs, 0), "k-", lw=0.9)
    for key, lab, col in (("A_matter", "A: matter only", C1), ("B_temperature", "B: temperature only", C4), ("C_both", "C: both", C3)):
        tr = TH["models"][key]["track"]
        xx, yy_ = np.array(tr["x"]), np.array(tr["y"])
        keep = (xx <= 2.6) & (yy_ <= 2.6)
        a.plot(xx[keep], yy_[keep], color=col, lw=2.2 if key != "C_both" else 1.5, label=lab, clip_on=False, zorder=4)
        ar = {"A_matter": 2.0, "B_temperature": 2.0, "C_both": TH["a_release_C_analytic"]}[key]       # analytic release, x + y = 1
        xr = (2.0 / ar) ** 3 if key != "B_temperature" else 0.0
        yr = (2.0 / ar) ** 2 if key != "A_matter" else 0.0
        a.plot(xr, yr, "o", color=col, ms=5, zorder=5, clip_on=False)
    a.text(0.08, 0.12, "released", fontsize=6.5)
    a.set_xlim(-0.04, 2.5); a.set_ylim(-0.04, 2.5)
    a.set_xlabel(r"$\rho_* / \rho_{crit}$"); a.set_ylabel(r"$(T / T_c)^2$")
    T_(a, "(a) Release: matter, temperature, both"); a.legend(fontsize=6.0, loc="upper right")
    b = ax[1]
    rows = S["rows"]
    mus = sorted({r["mu"] for r in rows})
    for a_c, col in zip(sorted({r["a_c"] for r in rows}), (C1, C3, C4, C2, GR)):
        ys = [[r for r in rows if r["coupled"] and r["a_c"] == a_c and r["mu"] == mu and r["q"] == 1.0][0]["a_turn"] for mu in mus]
        b.loglog(mus, ys, "o-", color=col, ms=3.5, lw=1.2, label="coupled, $a_c$ = %g" % a_c)
    yb = []
    for mu in mus:
        r = [r for r in rows if (not r["coupled"]) and r["mu"] == mu and r["q"] == 1.0][0]
        yb.append(r["a_turn"] if r["cls"] != "rolled before today" else np.nan)
    b.loglog(mus, yb, "s--", color="k", ms=3.5, lw=1.0, label="uncoupled control")
    b.axhline(1, color=GR, ls=":", lw=0.8)
    b.annotate("control rolled\nbefore today", xy=(300, 1.0), xytext=(60, 3e-1), fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=0.6))
    b.set_ylim(1e-1, 1e28)
    b.set_xlabel(r"$\mu / H_0$"); b.set_ylabel("scale factor at turnaround")
    T_(b, "(b) Turnaround, coupled vs control"); b.legend(fontsize=5.6, loc="upper right")
    c = ax[2]
    sd = S["seeds"]
    for coupled, col, lab in ((True, C1, "coupled"), (False, C2, "uncoupled control")):
        rr = [s for s in sd if s["coupled"] == coupled]
        c.semilogx([s["seed"] for s in rr], [s["a_roll"] for s in rr], "o-", color=col, label=lab + ", release", ms=3.5)
    c.axhline(1, color=GR, ls=":", lw=0.8); c.axhline(2, color=C3, ls="--", lw=0.8)
    c.set_xlabel(r"seed $\varphi_0 / v$ at $z$ = 1100"); c.set_ylabel(r"scale factor of the release, $|\varphi| = v/2$")
    T_(c, r"(c) Seed dependence ($\mu$ = 100 $H_0$)"); c.legend(fontsize=6.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig14_release_scan.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig14")


def fig15():
    DB = J("density_bounce.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    a = ax[0]
    for (eta, tab), col in zip(DB["b_table"].items(), (C1, C3, C2)):
        rows = tab["rows"]
        a.semilogy([r["theta_over_deg"] for r in rows], [r["b"] / rows[0]["b"] for r in rows], "o-", ms=3, color=col, label=r"$\eta$ = %s" % eta)
    a.set_xlabel(r"matter density $\theta / \theta_{deg}$"); a.set_ylabel(r"$B(\theta) / B(0)$")
    T_(a, "(a) Matter raises the bounce action"); a.legend(fontsize=6.3)
    b = ax[1]
    H = DB["histories"]
    for B0, col in zip((150.0, 200.0, 250.0, 287.0, 400.0), (C2, C4, C3, C1, GR)):
        rr = [h for h in H if h["eta"] == 0.30 and h["B0"] == B0]
        xs = [h["theta0_over_deg"] for h in rr]
        b.semilogx(xs, [h["lnI_past"] if np.isfinite(h["lnI_past"]) else -2000 for h in rr], "o-", ms=3, color=col, label="$B_0$ = %g" % B0)
        b.axhline(rr[0]["lnI_without_coupling"], color=col, ls=":", lw=0.8)
    b.axhline(0, color="k", lw=0.8)
    b.set_ylim(-1000, 200)
    b.set_xlabel(r"density today $\theta_0 / \theta_{deg}$"); b.set_ylabel(r"$\ln I$ (past light cone)")
    T_(b, "(b) Survival with the coupling (dotted: without)"); b.legend(fontsize=5.8, loc="lower left")
    c = ax[2]
    for eta, mk in ((0.20, "o"), (0.30, "s"), (0.36, "^")):
        for f0, col in ((0.1, C1), (0.3, C2)):
            rr = [h for h in H if h["eta"] == eta and h["theta0_over_deg"] == f0 and h["survives"] and h["a_star"]]
            c.plot([h["B0"] for h in rr], [h["a_star"] for h in rr], marker=mk, color=col, lw=1.0, ms=3.5,
                   label=r"$\eta$=%.2f, $\theta_0$=%.1f$\theta_{deg}$" % (eta, f0))
    c.set_xlabel(r"vacuum action $B_0$"); c.set_ylabel(r"$a_*$: one nucleation per Hubble volume")
    T_(c, "(c) Dilution-triggered tunneling"); c.legend(fontsize=5.2, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig15_tunneling_branch.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig15")


def fig16():
    HC = J("hot_cycles.json")
    base = HC["runs"][0]
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    a = ax[0]
    for c, ls in zip(base["cycles"][1:4], ("-", "--", ":")):
        h = c.get("history")
        if not h:
            continue
        held = np.array(h["phi"]) < 0.5
        k = int(np.argmin(held)) if (~held).any() else held.size       # up to the release
        k = min(k, int(np.argmax(np.array(h["T"]) < c["T_release"])) if (np.array(h["T"]) < c["T_release"]).any() else k)
        T = np.array(h["T"])[:k]
        a.semilogx(T, np.array(h["rad"])[:k], color=C2, ls=ls, lw=1.2)
        a.semilogx(T, np.array(h["mat"])[:k], color=C1, ls=ls, lw=1.2)
        a.semilogx(T, np.array(h["vac"])[:k], color=C3, ls=ls, lw=1.2)
        a.axvline(c["T_release"], color=GR, ls=ls, lw=0.7)
    h0 = base["cycles"][0].get("history")
    a.plot([], [], color=C2, label="radiation"); a.plot([], [], color=C1, label="matter"); a.plot([], [], color=C3, label="vacuum")
    a.plot([], [], color="k", ls="-", label="cycle 1"); a.plot([], [], color="k", ls="--", label="cycle 2"); a.plot([], [], color="k", ls=":", label="cycle 3")
    a.invert_xaxis()
    a.text(c["T_release"] * 1.3, 0.45, "release", fontsize=6.3, color=GR, rotation=90)
    a.set_xlabel("radiation temperature (code units)"); a.set_ylabel("fraction of the energy density")
    T_(a, "(a) The same thermal history each cycle"); a.legend(fontsize=5.6, loc="center left")
    b = ax[1]
    cs = [c for c in base["cycles"] if "a_max" in c]
    n = np.arange(len(cs))
    b.semilogy(n, [c["a_max"] for c in cs], "o-", color=C1, label=r"$a_{max}$ (turnaround)")
    b.semilogy(n[1:], [c["a_min"] for c in cs[1:] if "a_min" in c][:len(n) - 1] if False else [c.get("a_min", np.nan) for c in cs][:len(n)][1:] if False else [c.get("a_min", np.nan) for c in cs][:-1] if False else [c.get("a_min", np.nan) for c in cs], "s-", color=C2, label=r"$a_{min}$ (bounce)") if False else None
    amin = [c.get("a_min", np.nan) for c in cs]
    b.semilogy(n, amin, "s-", color=C2, label=r"$a_{min}$ (following bounce)")
    S = [c["S_turn"] for c in cs]
    b2 = b.twinx()
    b2.semilogy(n, S, "^:", color=C4, label="radiation entropy at turnaround")
    b2.set_ylabel("entropy (code units)", color=C4); b2.tick_params(axis="y", colors=C4); b2.grid(False)
    b.set_xlabel("cycle"); b.set_ylabel("scale factor")
    G = [c.get("G") for c in cs if c.get("G")]
    T_(b, "(b) Growth $G_n$ = %.1f = $(S_{n+1}/S_n)^{1/3}$" % np.median(G[1:]) if len(G) > 1 else "(b) Growth")
    b.legend(fontsize=5.8, loc="upper left")
    c = ax[2]
    runs = HC["runs"]
    pick = [(0, "full model", C1), (1, "no freeze-out reset", C2), (2, r"no thermal dissipation ($C_T$ = 0)", C4)]
    for i, lab, col in pick:
        r = runs[i]
        c.plot([cc["n"] for cc in r["cycles"]], [cc.get("held_accelerated_efolds", 0) for cc in r["cycles"]], "o-", color=col, ms=3.5, label=lab)
    c.axhline(np.log(1 / 0.6), color=GR, ls=":", lw=0.8); c.text(0.1, np.log(1 / 0.6) + 0.1, "our universe so far", fontsize=6, color=GR)
    c.set_xlabel("cycle"); c.set_ylabel(r"e-folds of acceleration with $\varphi$ held")
    T_(c, "(c) A dark-energy epoch in every cycle?"); c.legend(fontsize=5.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig16_hot_cycles.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig16")


if __name__ == "__main__":
    fig13(); fig14(); fig15(); fig16()
