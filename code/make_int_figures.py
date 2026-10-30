# -*- coding: utf-8 -*-
"""Figures 23-27: the density-dependent vacuum interface (Section 7).
    python make_int_figures.py   (needs vacuum_phase_map.json, vacuum_wall.json, interface_transport.json, bubble_gravity.json, bubble_to_cycle.json)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import interface_transport as IT
import vacuum_phase_map as VP

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
plt.rcParams.update({"font.size": 8.3, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.default": "regular"})
C1, C2, C3, C4, GR = "#3E6B8A", "#B5544E", "#5C8A5C", "#8A7A3E", "#777777"
T_ = lambda a, s: a.set_title(s, fontsize=8.5, fontweight="bold")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig23():
    PM = J("vacuum_phase_map.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a = ax[0]
    x = np.linspace(-1.6, 1.6, 400)
    eta = 0.30
    for th, c in zip((0.0, 0.075, 0.15, 0.20, 0.26), (C1, C3, C4, C2, GR)):
        p = VP.p_tb(x, eta, th)
        a.plot(x, p - p.min(), color=c, lw=1.2, label=r"$\theta/\theta_{deg}$ = %.2g" % (th / 0.15))
    a.set_ylim(-0.02, 0.6)
    a.set_xlabel(r"$x=\varphi/f$"); a.set_ylabel(r"$p - p_{min}$  (units $\lambda f^4$)")
    a.legend(fontsize=6.5, loc="upper center")
    T_(a, r"(a) Effective potential, $\eta$ = 0.30")
    a = ax[1]
    tm = PM["tunneling_map"]
    code = np.array(tm["code"])
    cmap = ListedColormap(["#DDE6EE", "#C9DCC9", "#E9C9C6", "#EADFC2"])
    a.pcolormesh(tm["theta"], tm["eta"], code, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest")
    L = tm["lines"]
    e = np.array([l["eta"] for l in L])
    a.plot([l["theta_deg"] if l["theta_deg"] is not None else np.nan for l in L], e, "k-", lw=1, label=r"$\theta_{deg}=\eta/2$")
    a.plot([l["theta_high"] if l["theta_high"] is not None else np.nan for l in L], e, "k--", lw=1, label=r"$\theta_{high}$ (true min. disappears)")
    a.plot([l["theta_low"] if l["theta_low"] else np.nan for l in L], e, "k:", lw=1, label=r"$\theta_{low}$ (held min. disappears)")
    a.axhline(VP.ETA_C, color=GR, lw=0.6)
    a.set_xlabel(r"density $\theta=\rho_*/(\lambda f^2M^2)$"); a.set_ylabel(r"tilt $\eta$")
    a.legend(handles=[Patch(color="#DDE6EE", label="one minimum (held)"), Patch(color="#C9DCC9", label="two minima, held = true"),
                      Patch(color="#E9C9C6", label="two minima, held metastable"), Patch(color="#EADFC2", label="held state absent")]
             + a.get_legend_handles_labels()[0], fontsize=5.6, loc="lower right", framealpha=0.9)
    T_(a, "(b) Phase diagram of the tunneling branch")
    save(fig, "fig23_vacuum_landscape.png")


def fig24():
    WJ = J("vacuum_wall.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))
    cols = {"0.20": C1, "0.30": C2, "0.36": C3}
    for e, c in cols.items():
        rows = [r for r in WJ["walls"][e]["rows"] if not r.get("failed")]
        f = [r["theta_over_deg"] for r in rows]
        ax[0].plot(f, [r["sigma_bounce"] for r in rows], "o-", ms=2.5, color=c, label=r"$\sigma$, $\eta$ = %s" % e)
        ax[0].plot(f, [r["thickness"] / 10 for r in rows], "s--", ms=2, color=c, alpha=0.6)
        ax[1].semilogy(f, [r["b"] for r in rows], "o-", ms=2.5, color=c, label=r"numerical, $\eta$ = %s" % e)
        ax[1].semilogy(f, [r["B_tw"] for r in rows], ":", color=c)
    ax[0].set_xlabel(r"$\theta/\theta_{deg}$"); ax[0].set_ylabel(r"tension $\sigma$ ($\sqrt{\lambda}f^3$); thickness/10 (dashed)")
    ax[0].legend(fontsize=6.5); T_(ax[0], "(a) Wall tension and thickness")
    ax[1].set_xlabel(r"$\theta/\theta_{deg}$"); ax[1].set_ylabel(r"$b=\lambda B$"); ax[1].legend(fontsize=6.5, title="dotted: thin wall", title_fontsize=6.5)
    T_(ax[1], "(b) Bounce action against density")
    a = ax[2]
    hist = WJ["nucleation_history"]
    aa = np.array([h["a"] for h in hist]); lg = np.array([h["ln_gamma_h4"] if h["ln_gamma_h4"] is not None else np.nan for h in hist])
    a.plot(aa, lg, color=C1)
    a.axhline(WJ["ln_completion"], color=C2, ls="--", lw=0.9, label=r"completion, $\Gamma/H^4=9/4\pi$")
    a.axvline(1.0, color=GR, lw=0.7); a.text(1.02, -200, "today", fontsize=6.5, color=GR)
    if WJ["a_trigger"]:
        a.axvline(WJ["a_trigger"], color=C3, lw=0.9, label="trigger a = %.2f" % WJ["a_trigger"])
    a.set_xscale("log"); a.set_ylim(-400, 60); a.set_xlabel("scale factor a"); a.set_ylabel(r"ln($\Gamma/H^4$)")
    a.legend(fontsize=6.3, loc="lower right"); T_(a, r"(c) Nucleation as matter dilutes (baseline)")
    save(fig, "fig24_interface_properties.png")


def fig25():
    TJ = J("interface_transport.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a = ax[0]
    for (g, d), c in zip(TJ["species"].items(), (C1, C3, C4, C2)):
        a.plot(d["k_above_threshold"], d["T"], color=c, label=r"$g/\sqrt{\lambda}$ = %s  ($m_{in}/m_{out}$ = %.2f)" % (g, d["m_in"] / d["m_out"]))
    v = TJ["validation_poeschl_teller"]
    a.set_xlabel(r"momentum above the interior threshold, $\sqrt{k^2-m_{in}^2}$ ($\sqrt{\lambda}f$)"); a.set_ylabel("transmission T")
    a.set_ylim(0, 1.05); a.legend(fontsize=6.2, loc="lower right"); T_(a, "(a) Test scalar through the trigger wall")
    a = ax[1]
    kp = IT.kink_profile()
    q = np.linspace(0.01, 1.0, 60)
    for gh, c in ((0.5, C1), (np.sqrt(0.5), C3), (1.0, C2)):
        a.plot(q, [IT.pt_exact(gh, gh ** 2 + qq ** 2) for qq in q], color=c, lw=1.0, label=r"exact, $g/\sqrt{\lambda}$ = %.3g" % gh)
        pts = [r for r in v if abs(r["gh"] - gh) < 1e-9]
        a.plot([r["q"] for r in pts], [r["T"] for r in pts], "o", color=c, ms=3)
    a.set_xlabel(r"momentum above threshold q"); a.set_ylabel("T"); a.set_ylim(0, 1.05)
    a.legend(fontsize=6.3, title="dots: numerical", title_fontsize=6.3); T_(a, "(b) Validation: Poeschl-Teller kink")
    save(fig, "fig25_selective_transmission.png")


def fig26():
    GJ, CJ = J("bubble_gravity.json"), J("bubble_to_cycle.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))
    a = ax[0]
    w = GJ["wall"]
    a.loglog(w["R_over_RH"], w["Gamma"], color=C1, label=r"$\Gamma_-$ (junction)")
    a.loglog(w["R_over_RH"], w["flat_Gamma"], ":", color=C2, label=r"flat space $R/R_0$")
    a2 = a.twinx()
    a2.loglog(w["R_over_RH"], w["grav"], color=C3, lw=0.9)
    a2.set_ylabel(r"self-gravity $\sigma R/2\Gamma$", color=C3, fontsize=7); a2.grid(False)
    a.set_xlabel(r"wall radius $R\,H$"); a.set_ylabel(r"Lorentz factor $\Gamma$"); a.legend(fontsize=6.5, loc="upper left")
    T_(a, "(a) Wall: runaway, negligible self-gravity")
    for k, (key, ttl) in enumerate((("percolated", "(b) Percolated transition"), ("isolated", "(c) Isolated bubble interior"))):
        a = ax[1 + k]
        tr = GJ[key]["traj"]
        for seg in sorted(set(p["segment"] for p in tr)):
            pass
        t = np.array([p["tau"] for p in tr]); la = np.array([p["lna"] for p in tr])
        a.plot(t, la, color=C1, lw=1.0, label="ln a")
        ev = GJ[key]["events"]
        for e in ev:
            col = {"nucleation": C2, "turnaround": GR, "restoration": C3, "bounce": C4}.get(e["event"])
            if col:
                a.axvline(e["tau"], color=col, lw=0.6, ls="--")
        a.set_xlabel(r"proper time ($H_0^{-1}$)"); a.set_ylabel("ln a")
        a.legend(handles=[plt.Line2D([], [], color=C1, label="ln a"), plt.Line2D([], [], color=C2, ls="--", label="nucleation"),
                          plt.Line2D([], [], color=GR, ls="--", label="turnaround"), plt.Line2D([], [], color=C3, ls="--", label="restoration"),
                          plt.Line2D([], [], color=C4, ls="--", label="bounce")], fontsize=5.8, loc="lower left")
        T_(a, ttl)
    save(fig, "fig26_daughter_evolution.png")


def fig27():
    CJ = J("bubble_to_cycle.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    a = ax[0]
    m = CJ["map"]
    code = np.array(m["code"])
    cmap = ListedColormap(["#E9C9C6", "#EADFC2", "#C9DCC9"])
    a.pcolormesh(m["B0"], m["f0"], code, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest")
    w = CJ["D"]["window_B0"]
    if w:
        a.axvspan(w[0], w[1], color=C2, alpha=0.5)
    a.set_yscale("log"); a.set_xlabel(r"vacuum action $B_0$ (zero density)"); a.set_ylabel(r"$\theta_0/\theta_{deg}$ (coupling today)")
    a.legend(handles=[Patch(color="#E9C9C6", label="already decayed (past light cone)"), Patch(color="#EADFC2", label="no completed transition"),
                      Patch(color="#C9DCC9", label="nucleation, recollapse, bounce, reheating"),
                      Patch(color=C2, alpha=0.5, label="window without coupling (D)")], fontsize=5.8, loc="lower right", framealpha=0.9)
    T_(a, r"(a) Viability, $\eta$ = 0.30, q = 1")
    a = ax[1]
    xq = np.linspace(-2, 2, 200); yq = np.linspace(-0.6, 2.0, 200)
    X, Y = np.meshgrid(xq, yq)
    gth = J("interface_transport.json")["dark_matter"]["A_true"]
    Z = np.where(Y <= 0, 0, np.where(X < np.log10(0.2 * gth), 1, 2))
    a.pcolormesh(xq, yq, Z, cmap=ListedColormap(["#EADFC2", "#D8D0E8", "#C9DCC9"]), vmin=-0.5, vmax=2.5, shading="nearest")
    a.set_xlabel(r"log$_{10}(T_b/m_{dm})$"); a.set_ylabel(r"depth of the true vacuum q  ($V_t=-qV_f$)")
    a.legend(handles=[Patch(color="#EADFC2", label="no recollapse (q <= 0)"), Patch(color="#D8D0E8", label="B, C viable; C* fails (no re-thermalization)"),
                      Patch(color="#C9DCC9", label="B, C and C* viable")], fontsize=5.8, loc="upper left", framealpha=0.9)
    T_(a, "(b) Contraction and re-thermalization")
    save(fig, "fig27_viability_map.png")


if __name__ == "__main__":
    for f in (fig23, fig24, fig25, fig26, fig27):
        f()
        print(f.__name__, "done")
