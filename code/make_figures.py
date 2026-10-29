# -*- coding: utf-8 -*-
"""Figures 1-6 of the paper from the stored results.   python make_figures.py"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import cdl_lib as C
import constants as K
import rates as R

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)
T = json.load(open(os.path.join(HERE, "bounce_tables.json"), encoding="utf-8"))
F = json.load(open(os.path.join(HERE, "falsification_map.json"), encoding="utf-8"))
MC = json.load(open(os.path.join(HERE, "monte_carlo.json"), encoding="utf-8"))
IN = json.load(open(os.path.join(HERE, "interior_evolution.json"), encoding="utf-8"))
SV = json.load(open(os.path.join(HERE, "survival.json"), encoding="utf-8"))

plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    lam, eta = 1.0, 0.30
    pot = C.Potential(lam, eta)
    flat = C.flat_bounce(lam, eta, with_profile=True)
    res = C.grav_bounce(lam, eta, 0.1)
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
    x = np.linspace(-1.5, 1.4, 600)
    a = ax[0]
    a.plot(x, pot.v(x), color=C1, lw=1.8)
    for xx, lab in [(pot.x_t, "true"), (pot.x_top, "top"), (pot.x_f, "false")]:
        a.plot([xx], [pot.v(xx)], "o", color=C2, ms=4)
    a.annotate("$V_f$", (pot.x_f, pot.v(pot.x_f)), textcoords="offset points", xytext=(-22, -16))
    a.annotate("$V_t$", (pot.x_t, pot.v(pot.x_t)), textcoords="offset points", xytext=(-4, 8))
    a.annotate("", xy=(0.95, pot.v_f), xytext=(0.95, 0.0), arrowprops=dict(arrowstyle="<->", color=GR, lw=0.9))
    a.text(1.0, pot.v_f / 2, "$\\Delta V$", fontsize=8, color=GR)
    a.set_xlabel("$\\varphi / f$"); a.set_ylabel("$V / f^4$")
    a.set_title("(a) Potential, $\\lambda$ = 1, $\\eta$ = 0.30", fontsize=8.6, fontweight="bold")
    b = ax[1]
    b.plot(flat["r"], flat["x"], color=C1, lw=1.8, label="flat space")
    b.axhline(pot.x_f, color=GR, ls=":", lw=0.8); b.axhline(pot.x_t, color=GR, ls=":", lw=0.8)
    sol = res["sol"]
    s = np.linspace(sol.t[0], sol.t[-1], 2000)
    xs, ps, rhos, rps = sol.sol(s)
    b.plot(rhos[: np.argmax(rhos) + 1], xs[: np.argmax(rhos) + 1], color=C2, lw=1.4, ls="--", label="with gravity, $g$ = 0.1")
    b.set_xlim(0, 12); b.set_xlabel("radius $\\rho$ ($1/f$)"); b.set_ylabel("$\\varphi / f$")
    b.set_title("(b) Bounce profile", fontsize=8.6, fontweight="bold"); b.legend(fontsize=7, loc="lower right")
    c = ax[2]
    etas = np.array([r["eta"] for r in T["flat"]])
    bb = np.array([r["b"] for r in T["flat"]])
    th = np.array([r["thin"] for r in T["flat"]])
    c.semilogy(etas, bb, "o-", color=C1, ms=3, lw=1.3, label="numerical bounce $b(\\eta)$")
    c.semilogy(etas, th, "--", color=C2, lw=1.2, label="thin-wall formula")
    c.axvline(C.eta_crit(), color=GR, ls=":", lw=0.9); c.text(C.eta_crit() - 0.004, 1.5e5, "no false\nvacuum", ha="right", fontsize=6.8, color=GR)
    c.set_xlabel("tilt / barrier  $\\eta$"); c.set_ylabel("action $b = B\\lambda$")
    c.set_title("(c) Action against thin-wall", fontsize=8.6, fontweight="bold"); c.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig1_potential_bounce.png")


# ---------------------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    G = T["grid"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
    a = ax[0]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, 7))
    for col, eta in zip(cols, sorted({r["eta"] for r in G})):
        rows = sorted([r for r in G if r["eta"] == eta], key=lambda r: r["g"])
        g = [r["g"] for r in rows if r["found"]]; ra = [r["ratio"] for r in rows if r["found"]]
        a.semilogx(g, ra, "o-", ms=3, lw=1.2, color=col, label="$\\eta$ = %.2f" % eta)
        nf = [r for r in rows if not r["found"]]
        if nf:
            a.plot([nf[0]["g"]], [0.0], "x", color=col, ms=6)
    a.set_xlabel("gravity parameter  $g = f^2 / M_{Pl}^2$"); a.set_ylabel("$B(g) / B_{flat}$")
    a.set_ylim(-0.03, 1.03); a.legend(fontsize=6.4, ncol=2, loc="lower left")
    a.set_title("(a) Gravitational suppression", fontsize=8.6, fontweight="bold")
    a.text(0.55, 0.45, "× : no oscillating\nbounce found", transform=a.transAxes, fontsize=6.8, color=GR)
    b = ax[1]
    gg = np.logspace(-62, 1.0, 200)
    for eta, col in [(0.12, cols[0]), (0.2, cols[2]), (0.3, cols[4]), (0.37, cols[6])]:
        p_ = C.Potential(1.0, eta)
        b.loglog(gg, abs(p_.m2_top) / (4.0 * gg * p_.v_f / 3.0), "-", color=col, lw=1.3, label="$\\eta$ = %.2f" % eta)
        rows = sorted([r for r in G if r["eta"] == eta], key=lambda r: r["g"])
        b.plot([r["g"] for r in rows if r["found"]], [r["hm_ratio"] for r in rows if r["found"]], "o", color=col, ms=3)
        b.plot([r["g"] for r in rows if not r["found"]], [r["hm_ratio"] for r in rows if not r["found"]], "x", color=col, ms=5)
    b.axhline(1.0, color="k", ls="--", lw=0.9)
    b.axvspan(5e-61, 3e-60, color=C2, alpha=0.35)
    b.text(3e-30, 2e-1, "Hawking-Moss regime below the dashed line (no CDL bounce)", fontsize=6.8, color=GR)
    b.text(2e-60, 1e12, "physical:\n$g \\sim 10^{-60}$,\nratio $\\sim 10^{60}$", fontsize=7, color=C2)
    b.set_xlabel("gravity parameter  $g$"); b.set_ylabel("$|V''_{top}| / 4H^2$")
    b.set_xlim(1e-62, 10); b.set_ylim(1e-1, 1e64)
    b.set_title("(b) Existence criterion", fontsize=8.6, fontweight="bold"); b.legend(fontsize=6.6, loc="upper right")
    fig.tight_layout()
    save(fig, "fig2_gravity_hawking_moss.png")


# ---------------------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    from matplotlib.patches import Patch
    etas = np.array(F["grid"]["eta"]); ll = np.array(F["grid"]["log10_lam"])
    lt = np.array(F["grid"]["log10_t_yr"])
    E, L = np.meshgrid(etas, ll)
    t_surv = float(np.log10(SV["t_dec_min_yr"]))
    levels = [-60, t_surv, 12, 20, 50, 100, 400]
    cols = ["#E8BFBF", "#F2DFA7", "#CFE3CF", "#9FC49F", "#B8CFE0", "#E6E6E6"]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.contourf(E, L, np.clip(lt, -59, 399), levels=levels, colors=cols)
    for name, lab, ls in [("survival_bound", "", "-"), ("1e12yr", "", "--"), ("1e20yr", "", "--"), ("1e50yr", "", ":"), ("1e100yr", "", ":")]:
        pts = [(p["eta"], p["log10_lam"]) for p in F["contours"][name] if p["eta"] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "k", ls=ls, lw=0.9)
    ax.axhline(np.log10(4 * np.pi), color="k", lw=0.9, ls="-.")
    ax.text(0.033, np.log10(4 * np.pi) + 0.05, "$\\lambda = 4\\pi$ (perturbative unitarity; upper edge of the prior)", fontsize=6.6)
    ax.axvspan(C.eta_crit(), 0.40, color="#444444", alpha=0.45)
    ax.text(0.393, -1.0, "no false vacuum  ($\\eta > 0.385$)", rotation=90, ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(0.03, 0.40); ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("tilt / barrier  $\\eta = \\epsilon / \\lambda$"); ax.set_ylabel("$\\log_{10}\\ \\lambda$")
    ax.set_title("Mean decay time of a Hubble patch for $V_f = \\rho_\\Lambda$", fontsize=8.8, fontweight="bold")
    ts = SV["t_dec_min_yr"]
    ax.legend(handles=[Patch(color=cols[0], label="$< %.1f\\times10^{%d}$ yr: excluded by survival (95%%)" % (ts / 10 ** int(np.floor(np.log10(ts))), int(np.floor(np.log10(ts))))),
                       Patch(color=cols[1], label="survival bound to $10^{12}$ yr"),
                       Patch(color=cols[2], label="$10^{12}$ to $10^{20}$ yr"), Patch(color=cols[3], label="$10^{20}$ to $10^{50}$ yr"),
                       Patch(color=cols[4], label="$10^{50}$ to $10^{100}$ yr"), Patch(color=cols[5], label="$> 10^{100}$ yr (no decay in practice)")],
              fontsize=6.6, loc="lower left", framealpha=0.95)
    ax.grid(False)
    save(fig, "fig4_falsification_diagram.png")


# ---------------------------------------------------------------------------------------------------------- Fig. 4
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9))
    a = ax[0]
    a.plot(MC["cdf"]["log10_years"], MC["cdf"]["frac"], color=C1, lw=1.8)
    a.axvline(np.log10(K.T_HUBBLE_YR), color=GR, ls=":", lw=0.8)
    a.set_xlabel("$\\log_{10}$ decay time (yr)"); a.set_ylabel("fraction of prior, survival-consistent")
    a.set_title("(a) Decay-time distribution", fontsize=8.6, fontweight="bold")
    b = ax[1]
    labs = ["excluded by\nsurvival", "decay by\n$10^{12}$ yr", "by\n$10^{20}$ yr", "by\n$10^{100}$ yr", "never"]
    base = MC["baseline"]
    vals = [base["frac_inconsistent_with_survival"], base["frac_decay_by"]["1e+12"], base["frac_decay_by"]["1e+20"], base["frac_decay_by"]["1e+100"], base["frac_never_within_1e100"]]
    ses = [MC["baseline_batch"][k]["se"] for k in ("surv", "1e12", "1e20", "1e100", "never")]
    b.bar(range(5), vals, yerr=[3 * s for s in ses], color=[C2, C3, C3, C3, GR], capsize=2)
    b.set_xticks(range(5)); b.set_xticklabels(labs, fontsize=5.8); b.set_yscale("log"); b.set_ylim(3e-4, 1.5)
    b.set_ylabel("fraction of prior")
    b.set_title("(b) Prior mass by regime", fontsize=8.6, fontweight="bold")
    c = ax[2]
    ns = [x["n"] for x in MC["convergence"]]
    c.semilogx(ns, [x["surv"] for x in MC["convergence"]], "o-", color=C2, label="survival-excluded")
    c.semilogx(ns, [x["d1e20"] for x in MC["convergence"]], "s-", color=C3, label="decay by $10^{20}$ yr")
    c.semilogx(ns, [x["never"] for x in MC["convergence"]], "^-", color=GR, label="no decay within $10^{100}$ yr")
    c.set_xlabel("number of draws"); c.set_ylabel("fraction"); c.set_yscale("log")
    c.set_title("(c) Convergence", fontsize=8.6, fontweight="bold"); c.legend(fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig5_monte_carlo.png")


# ---------------------------------------------------------------------------------------------------------- Fig. 5
def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9))
    r = F["reheat"]
    a = ax[0]
    dV = np.array(r["dV_gev4"]); Tg = np.array(r["T_gev"]) * 1e12
    a.loglog(dV, Tg, color=C1, lw=1.8)
    a.plot([K.V_F], [r["T_max_meV_at_Vf"]], "o", color=C2); a.annotate("$\\rho_\\Lambda$: %.1f meV" % r["T_max_meV_at_Vf"], (K.V_F, r["T_max_meV_at_Vf"]), textcoords="offset points", xytext=(6, -12), fontsize=7)
    a.axhline(K.T_BBN_GEV * 1e12, color=GR, ls="--", lw=0.9); a.text(dV[0] * 1.5, K.T_BBN_GEV * 1e12 * 1.5, "T = 1 MeV (nucleosynthesis)", fontsize=6.8)
    a.axhline(K.T_CMB_GEV * 1e12, color=GR, ls=":", lw=0.9); a.text(1e-25, K.T_CMB_GEV * 1e12 * 0.35, "CMB today", fontsize=6.8)
    a.axvline(((np.pi ** 2 * K.GSTAR_BBN / 30.0) * K.T_BBN_GEV ** 4), color=C2, ls="--", lw=0.9)
    a.text(((np.pi ** 2 * K.GSTAR_BBN / 30.0) * K.T_BBN_GEV ** 4) * 0.6, 3e-2, "energy density\nneeded for 1 MeV", rotation=90, ha="right", fontsize=6.8, color=C2)
    a.set_xlabel("released energy density $\\epsilon\\,\\Delta V$ (GeV$^4$)"); a.set_ylabel("$T_{reh}^{max}$ (meV)")
    a.set_title("(a) Reheating bound", fontsize=8.6, fontweight="bold")
    b = ax[1]
    tr = IN["trace"]
    b.loglog(tr["tau"], np.maximum(tr["T_meV"], 1e-60), color=C1, lw=1.8)
    b.axhline(K.T_BBN_GEV * 1e12, color=GR, ls="--", lw=0.9); b.text(2e-31, K.T_BBN_GEV * 1e12 * 1.5, "1 MeV", fontsize=6.8)
    b.axhline(K.T_CMB_GEV * 1e12, color=GR, ls=":", lw=0.9); b.text(2e-31, K.T_CMB_GEV * 1e12 * 1.5, "CMB today", fontsize=6.8)
    b.axvline(1.0, color=GR, ls=":", lw=0.8); b.text(1.3, 1e-26, "$1/H_\\Lambda$", fontsize=6.8)
    b.set_ylim(1e-40, 1e12); b.set_xlabel("time since nucleation  $\\tau = H_\\Lambda t$"); b.set_ylabel("radiation temperature (meV)")
    b.set_title("(b) Interior after reheating", fontsize=8.6, fontweight="bold")
    c = ax[2]
    eb = IN["energy_budget"]
    c.plot([e["eta"] for e in eb], [e["eps_in"] for e in eb], "o-", color=C2, ms=3, lw=1.3)
    c.set_yscale("log"); c.set_xlabel("tilt / barrier  $\\eta$"); c.set_ylabel("interior energy fraction  $\\epsilon_{in}$")
    c.set_title("(c) Energy carried by the interior", fontsize=8.6, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig8_reheating_interior.png")


# ---------------------------------------------------------------------------------------------------------- Fig. 6
def fig6():
    import interior_evolution as I
    fig, a = plt.subplots(figsize=(4.4, 3.2))
    cu = SV["cumulative"]
    z = np.array(cu["z"]); v4 = np.array(cu["V4_HL4"])
    sel = z <= 30
    a.semilogx(np.maximum(z[sel], 1e-3), v4[sel], color=C1, lw=1.8, label="past light cone, observed expansion history")
    a.axhline(SV["V4_HL4"], color=C1, ls=":", lw=0.8)
    a.axhline(SV["euclid_HL4"], color=C2, ls="--", lw=1.4, label="Euclidean four-sphere $8\\pi^2/3$")
    a.set_yscale("log"); a.set_ylim(3e-3, 80)
    a.text(1.2e-3, SV["V4_HL4"] * 1.35, "$V_4 = %.3f$" % SV["V4_HL4"], fontsize=7.5, color=C1)
    a.text(1.2e-3, SV["euclid_HL4"] * 0.45, "$%.1f$: %.0f times larger" % (SV["euclid_HL4"], SV["ratio_euclid_to_lorentzian"]), fontsize=7.5, color=C2)
    a.set_xlabel("redshift $z$ (events at redshift below $z$ are included)"); a.set_ylabel("four-volume  ($H_\\Lambda^{-4}$)")
    a.set_title("Survival: four-volume of the past light cone", fontsize=8.4, fontweight="bold"); a.legend(fontsize=6.4, loc="lower right", bbox_to_anchor=(1.0, 0.0), framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig3_survival_cone.png")
    fig, b = plt.subplots(figsize=(4.4, 3.2))
    tau0 = IN["scales"]["tau0"]
    tau = np.logspace(np.log10(tau0), 4.0, 600)
    m_dim = IN["scales"]["m_over_H"]
    for eps, lab, col in ((1.0, "$\\epsilon = 1$ (all of $\\Delta V$)", C1), (IN["eps_in_ref"], "$\\epsilon = %.3f$ (benchmark, $\\eta=0.30$)" % IN["eps_in_ref"], C2)):
        rr = I.rho_r_milne(tau, eps, m_dim, tau0)
        rphi = eps * (tau0 / tau) ** 3 * np.exp(-m_dim * (tau - tau0))
        b.loglog(tau, tau ** 2 * (rr + rphi), color=col, lw=1.6, label=lab)
    b.axhline(1.0, color="k", ls="--", lw=0.9); b.text(tau0 * 3, 1e1, "$Y = 1$: flat-like radiation or matter epoch", fontsize=6.8)
    b.set_xlabel("time since nucleation  $\\tau = H_\\Lambda t$"); b.set_ylabel("curvature parameter  $Y = a^2\\rho / 3M_{Pl}^2$")
    b.set_ylim(1e-150, 1e6); b.set_title("Interior: curvature dominated at all times", fontsize=8.4, fontweight="bold"); b.legend(fontsize=6.4, loc="center right")
    fig.tight_layout()
    save(fig, "fig9_curvature_interior.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    import make_thermal_figures as _mt
    _mt.fig6(); _mt.fig7()
    import make_sim_figures as _ms
    _ms.fig10()
