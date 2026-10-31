# -*- coding: utf-8 -*-
"""Sections 7.4 and 7.6: gravity of the vacuum interface, and the evolution of the daughter region it encloses.

(1) The wall. A thin spherical shell of tension sigma and areal radius R separating an interior of density rho_in from an exterior of density rho_out
obeys the Israel junction condition in the generalized-Lorentz-factor form (8 pi G = 1, Misner-Sharp mass 2GM/R = rho R^2/3 on each side):
        Gamma_- - Gamma_+ = sigma R / 2,        Gamma_+- = sqrt(1 + U^2 - rho_+- R^2 / 3),        U = dR/dtau,
so that Gamma_- = Delta_rho R / (3 sigma) + sigma R / 4 and U^2 = Gamma_-^2 - 1 + rho_in R^2 / 3, Delta_rho = rho_out - rho_in. In flat space this is
Gamma = R / R_0 with R_0 = 3 sigma / Delta_rho (the wall runs away). The gravitational parameter sigma R / (2 Gamma_-) -> 3 sigma^2 / (2 Delta_rho) measures
the effect of the wall's own gravity. The densities are taken at the nucleation epoch (the wall crosses the Hubble radius in a time short compared with
the change of rho only for R << 1/H; beyond, the wall moves on light rays to excellent accuracy since Gamma >> 1).

(2) The daughter region. The interior is solved as a Friedmann region with the effective (loop-quantum) bounce of Section 6.8:
        H^2 = (rho/3)(1 - rho/rho_c) + K/a^2,        dH/dtau = -(rho + P)/2 (1 - 2 rho/rho_c) - K/a^2,
with radiation (comoving entropy), baryons, dark matter rho_* of field-dependent mass A(x) and the scalar held in one of its minima (m_phi ~ meV >> H, so
the field follows its minimum adiabatically). rho = rho_r + rho_b + rho_* + U(x, theta), rho + P = 4/3 rho_r + rho_b + rho_* A(x), with
U(x, theta) = E0 [p(x; eta, theta) - p(x_t0; eta, 0)] - q V_f containing the coupling energy rho_*(A - 1) through theta = c rho_*.
Events: nucleation (false -> true), the disappearance of the true minimum during contraction at theta_high (true -> held minimum: restoration; the
energy difference is assumed dissipated into radiation), and the bounce. Two cases:
  isolated bubble   the interior keeps the Hubble rate of the parent at nucleation while its density drops by Delta V_eff, so K/a^2 = Delta V_eff / 3 > 0
                    (an open region); the released energy is carried by the wall;
  percolated        bubbles fill space within ~ one Hubble time (Gamma/H^4 > 9/(4 pi)); the walls collide and their energy is assumed thermalized,
                    rho_r += Delta V_eff, K = 0 (the same history in every region).

    python bubble_gravity.py          (writes bubble_gravity.json, ~1 min)
"""
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import constants as K
import matter_coupling as MC
import vacuum_phase_map as VP

HERE = os.path.dirname(os.path.abspath(__file__))
H0_MPL = K.H0_GEV / K.M_PL


# ------------------------------------------------------------------------------------------------ junction
def junction(R, sigma, rho_in, rho_out):
    """Generalized Lorentz factors and areal velocity of the wall (units H_0, M_Pl = 1)."""
    drho = rho_out - rho_in
    Gm = drho * R / (3.0 * sigma) + sigma * R / 4.0
    Gp = Gm - sigma * R / 2.0
    U2 = Gm ** 2 - 1.0 + rho_in * R ** 2 / 3.0
    M_wall = 4.0 * np.pi * sigma * R ** 2 * 0.5 * (Gm + Gp)                    # Misner-Sharp mass jump carried by the wall
    return dict(Gamma_in=Gm, Gamma_out=Gp, U2=U2, grav=sigma * R / (2.0 * Gm), M_wall_over_deficit=M_wall / (4.0 * np.pi / 3.0 * drho * R ** 3))


# ------------------------------------------------------------------------------------------------ daughter region
class Daughter:
    def __init__(self, eta, theta0, q, theta_trig, rho_c=1e44, regenerate=True, lnrate=None, wall_w=None, bounce_form="lqc"):
        self.eta, self.q, self.rho_c = eta, q, rho_c
        _, x_t0, _, x_f0, d0, _ = VP.classify_tb(eta, 0.0)
        self.x_t0, self.E0 = x_t0, (1.0 + q) * MC.VC / d0
        self.c_theta = theta0 / MC.RHO_DM0                                     # theta = c rho_*
        self.theta_trig = theta_trig
        self.theta_high = VP.window_tb(eta)[1]
        self.regenerate = regenerate
        self.lnrate = lnrate                    # theta -> ln(Gamma / H_Lambda^4); if given, nucleation is where ln(Gamma/H^4) first exceeds ln(9/(4 pi))
        self.wall_w = wall_w                    # None: wall energy thermalized at collision; else a separate fluid with w = wall_w until the bounce
        self.bounce_form = bounce_form          # 'lqc': H^2 = rho/3 (1 - rho/rho_c);  'quadratic': H^2 = rho/3 (1 - (rho/rho_c)^2)
        self.Sx = 0.0

    def x_min(self, state, theta):
        st = VP.stationary_tb(self.eta, theta)
        mins = [x for x in st if VP.d2p_tb(x, self.eta, theta) > 0]
        if state == "true":
            return min(mins) if len(mins) == 2 else None
        return max(mins)

    def U(self, x, theta):
        return self.E0 * (VP.p_tb(x, self.eta, theta) - VP.p_tb(self.x_t0, self.eta, 0.0)) - self.q * MC.VC

    def parts(self, N, Sr, nd, state):
        a = np.exp(N)
        rs = nd / a ** 3
        th = self.c_theta * rs * self.cf                                      # only dark matter inside the true-vacuum region couples (cf)
        x = self.x_min(state, th)
        if x is None:
            x = self.x_min("false", th)
        rr = Sr / a ** 4
        rx = self.Sx / a ** (3.0 * (1.0 + self.wall_w)) if self.wall_w is not None else 0.0
        rb = MC.RHO_B0 / a ** 3 * self.bscale
        U = self.U(x, th)
        rho = rr + rb + rs + U + rx
        rpp = 4.0 / 3.0 * rr + rb + rs + self.E0 * th * (x - 1.0) ** 2 / 2.0 + (1.0 + (self.wall_w or 0.0)) * rx
        return rho, rpp, x, th, rr

    def F(self, N, Sr, nd, state, Kc):
        """H^2 from the Friedmann constraint (effective bounce, curvature K)."""
        rho = self.parts(N, Sr, nd, state)[0]
        f = rho / self.rho_c if self.bounce_form == "lqc" else (rho / self.rho_c) ** 2
        return rho / 3.0 * (1.0 - f) + Kc * np.exp(-2.0 * N)

    def segment(self, N1, N2, Sr, nd, state, Kc, sign, n=240):
        """Proper time and samples along a monotonic segment N1 -> N2 (H = sign sqrt(F)); 1/sqrt singularities at turning points are integrable."""
        Ns = np.linspace(N1, N2, n)
        Fs = np.array([max(self.F(x, Sr, nd, state, Kc), 0.0) for x in Ns])
        # tau = Integral dN / |H|. At a turning point F -> 0 linearly, so the end pieces are integrated analytically: Integral_0^d dN / sqrt(F' N) =
        # 2 d / sqrt(F(d)); the interior is integrated with quad.
        from scipy.integrate import quad
        L = abs(N2 - N1)
        d = 1e-7 * L
        lo, hi = min(N1, N2), max(N1, N2)
        f = lambda x: 1.0 / np.sqrt(max(self.F(x, Sr, nd, state, Kc), 1e-300))
        Fmax = max(Fs.max(), 1e-300)
        tau = 0.0
        a_, b_ = lo, hi
        if self.F(lo, Sr, nd, state, Kc) < 1e-6 * Fmax:
            tau += 2.0 * d / np.sqrt(max(self.F(lo + d, Sr, nd, state, Kc), 1e-300)); a_ = lo + d
        if self.F(hi, Sr, nd, state, Kc) < 1e-6 * Fmax:
            tau += 2.0 * d / np.sqrt(max(self.F(hi - d, Sr, nd, state, Kc), 1e-300)); b_ = hi - d
        pts = np.linspace(a_, b_, 41)
        tau += sum(quad(f, pts[i], pts[i + 1], limit=200, epsrel=1e-10)[0] for i in range(40))
        tau = float(tau)
        # proper time at every sample (for the figure): subinterval quadratures, analytic pieces where F -> 0 at an end, scaled to the total
        cum = [0.0]
        for i in range(n - 1):
            x0, x1 = Ns[i], Ns[i + 1]
            F0, F1 = Fs[i], Fs[i + 1]
            if min(F0, F1) < 1e-6 * Fmax:
                cum.append(cum[-1] + 2.0 * abs(x1 - x0) / np.sqrt(max(F0, F1, 1e-300)))
            else:
                cum.append(cum[-1] + quad(f, min(x0, x1), max(x0, x1), limit=100)[0])
        cum = np.array(cum)
        self._tl = cum * (tau / cum[-1]) if cum[-1] > 0 else cum
        return tau, Ns, sign * np.sqrt(Fs)

    def run(self, mode="percolated", n_cycles=2, filter_frac=0.0, regen_allowed=True):
        """Start today (a = 1, field held in the false vacuum): nucleation, turnaround, restoration, bounce, and the next nucleation, found as roots in
        N = ln a; H from the constraint; proper time by quadrature."""
        from scipy.optimize import brentq
        """filter_frac: fraction of the dark matter kept out of the true-vacuum region by the wall (it remains as gravitating matter in
        false-vacuum remnants and no longer couples to the field); regen_allowed: the bounce is hot enough (T_b > m A / 5) to re-thermalize it."""
        self.bscale, self.cf, self.Sx = 1.0, 1.0, 0.0
        Sr, nd, state, Kc = MC.RHO_R0, MC.RHO_DM0, "false", 0.0
        s_ratio0 = nd / Sr ** 0.75
        N, tau = 0.0, 0.0
        events, traj = [], []

        def rec(name, N, extra=None):
            rho, _, x, th, rr = self.parts(N, Sr, nd, state)
            e = dict(event=name, tau=float(tau), lna=float(N), rho=float(rho), theta=float(th), x=float(x), rho_r=float(rr), S_comoving=float(Sr ** 0.75),
                     H2_constraint=float(self.F(N, Sr, nd, state, Kc)))
            e.update(extra or {})
            events.append(e)
            return e

        def add(N1, N2, sign, label):
            nonlocal tau
            dt, Ns, Hs = self.segment(N1, N2, Sr, nd, state, Kc, sign)
            tl = tau + self._tl
            for t_, N_, H_ in zip(tl, Ns, Hs):
                rho, _, x, th, rr = self.parts(N_, Sr, nd, state)
                traj.append(dict(tau=float(t_), lna=float(N_), H=float(H_), rho=float(rho), rho_r=float(rr), x=float(x), segment=label))
            tau += dt

        for cyc in range(n_cycles):
            if state == "true":
                # the field never returned to the held minimum: expansion in the true vacuum to the next turnaround, no held epoch
                rec("no held epoch: the field stays in the true vacuum", N)
                break
            # expansion in the false vacuum until the transition completes
            N_nuc = np.log(self.c_theta * nd / self.theta_trig) / 3.0
            if self.lnrate is not None:
                HL = np.sqrt(MC.VC / 3.0)

                def gap(x):
                    th_ = self.c_theta * nd * self.cf * np.exp(-3 * x)
                    return self.lnrate(th_) - 4.0 * np.log(np.sqrt(max(self.F(x, Sr, nd, state, Kc), 1e-300)) / HL) - np.log(9.0 / (4.0 * np.pi))
                Ng = np.linspace(N + 1e-6, N + 40.0, 8001)
                gv = np.array([gap(x) for x in Ng])
                k = np.argmax(gv >= 0)
                if gv[k] < 0:
                    rec("no completed transition", N); break
                N_nuc = brentq(gap, Ng[k - 1], Ng[k], xtol=1e-12) if k > 0 else Ng[0]
            if min(self.F(x, Sr, nd, state, Kc) for x in np.linspace(N, N_nuc, 200)[1:]) <= 0:
                rec("no nucleation before turnaround", N); break
            add(N, N_nuc, +1, "held (false vacuum)")
            N = N_nuc
            x_f = self.parts(N, Sr, nd, state)[2]
            th = self.c_theta * nd * np.exp(-3 * N)
            dU = self.U(x_f, th) - self.U(self.x_min("true", th), th)
            e = rec("nucleation", N, dict(dV_eff=float(dU), cycle=cyc))
            H_before = np.sqrt(self.F(N, Sr, nd, state, Kc))
            state = "true"
            self.cf = 1.0 - filter_frac
            if mode == "percolated" and self.wall_w is not None:
                self.Sx += dU * np.exp(3.0 * (1.0 + self.wall_w) * N)        # unthermalized wall/collision products
            elif mode == "percolated":
                Sr += dU * np.exp(4.0 * N)
            else:
                Kc += dU / 3.0 * np.exp(2.0 * N)                            # nested bubbles: the deficits add
            e["S_after"] = float(Sr ** 0.75)
            e["H_jump"] = float(np.sqrt(max(self.F(N, Sr, nd, state, Kc), 0)) - H_before)
            # turnaround
            Ng = np.linspace(N, N + 12.0, 6001)
            Fg = np.array([self.F(x, Sr, nd, state, Kc) for x in Ng])
            k = np.argmax(Fg <= 0)
            if Fg[k] > 0:
                rec("no turnaround", Ng[-1]); break
            N_ta = brentq(lambda x: self.F(x, Sr, nd, state, Kc), Ng[k - 1], Ng[k], xtol=1e-14)
            add(N, N_ta, +1, "true vacuum, expanding")
            N = N_ta
            rec("turnaround", N, dict(a_max_over_a_nuc=float(np.exp(N_ta - N_nuc))))
            def restore(N, label):
                nonlocal Sr, state
                th = self.c_theta * nd * self.cf * np.exp(-3 * N)
                x_old = self.parts(N, Sr, nd, state)[2]
                state = "false"
                x_new = self.parts(N, Sr, nd, state)[2]
                dE = self.U(x_old, th) - self.U(x_new, th)
                Sr += dE * np.exp(4.0 * N)
                rec(label, N, dict(dE_released=float(dE), x_from=float(x_old), x_to=float(x_new)))
            # contraction in the true vacuum until the true minimum disappears (theta = theta_high); impossible if no dark matter couples (cf = 0)
            if self.cf > 0:
                N_res = np.log(self.c_theta * nd * self.cf / (self.theta_high * (1 - 1e-9))) / 3.0
                add(N, N_res, -1, "true vacuum, contracting")
                N = N_res
                restore(N, "restoration")
            # contraction to the bounce: F = 0 at rho ~ rho_c
            N_lo = 0.25 * np.log(Sr / (2.0 * self.rho_c)) - 3.0
            Ng = np.linspace(N, N_lo, 20001)[1:]
            Fg = np.array([self.F(x, Sr, nd, state, Kc) for x in Ng])
            k = np.argmax(Fg <= 0)
            N_b = brentq(lambda x: self.F(x, Sr, nd, state, Kc), Ng[k - 1], Ng[k], xtol=1e-15)
            add(N, N_b, -1, "held, contracting" if state == "false" else "true vacuum, contracting")
            N = N_b
            ex = {}
            if self.Sx > 0:                                                   # collision products thermalize in the hot bounce
                ex["wall_fluid_fraction_at_bounce"] = float(self.Sx / np.exp(3.0 * (1.0 + self.wall_w) * N) / self.parts(N, Sr, nd, state)[0])
                Sr += self.Sx / np.exp(3.0 * (1.0 + self.wall_w) * N) * np.exp(4.0 * N)
                self.Sx = 0.0
            rr = Sr * np.exp(-4 * N)
            ex.update(T_b_MeV=float((rr * K.RHO_CRIT / 3.0) ** 0.25 * 1e3), radiation_fraction=float(rr / self.parts(N, Sr, nd, state)[0]))
            if self.regenerate and regen_allowed:
                nd_new = s_ratio0 * Sr ** 0.75
                ex["dm_regenerated_factor"] = float(nd_new / nd)
                nd, self.cf = nd_new, 1.0                                      # thermal dark matter fills the region again (remnants dissolve)
            rec("bounce", N, ex)
            if state == "true" and self.cf > 0 and self.c_theta * nd * self.cf * np.exp(-3 * N) > self.theta_high:
                restore(N, "restoration at the bounce")
        return dict(events=events, traj=traj, K=float(Kc))


if __name__ == "__main__":
    t0 = time.time()
    WJ = json.load(open(os.path.join(HERE, "vacuum_wall.json"), encoding="utf-8"))
    PS, TW, base = WJ["physical"], WJ["trigger_wall"], WJ["base"]
    out = {}
    # ---- wall: junction along R from R_0 to the Hubble radius, at the nucleation epoch
    a_tr = WJ["a_trigger"]
    sigma = TW["sigma_meV3"] * 1e-36 / (K.M_PL ** 2 * K.H0_GEV)               # H_0 M_Pl^2
    dVe = TW["dV"] * PS["E0_GeV4"] / (K.M_PL ** 2 * K.H0_GEV ** 2)          # H_0^2 M_Pl^2
    rho_out = MC.RHO_R0 / a_tr ** 4 + (MC.RHO_B0 + MC.RHO_DM0) / a_tr ** 3 + MC.VC
    rho_in = rho_out - dVe
    R0 = 3.0 * sigma / dVe
    H_tr = np.sqrt(rho_out / 3.0)
    Rs = np.logspace(np.log10(R0), np.log10(1.0 / H_tr), 300)
    J = [junction(R, sigma, rho_in, rho_out) for R in Rs]
    out["wall"] = dict(sigma_H0MPl2=float(sigma), dV_eff=float(dVe), R0_H0inv=float(R0), R0_mm=float(R0 / K.H0_GEV * K.HBAR_GEV_S * 2.99792458e11),
                       rho_in=float(rho_in), rho_out=float(rho_out), H=float(H_tr), R_over_RH=(Rs * H_tr).tolist(),
                       Gamma=[j["Gamma_in"] for j in J], flat_Gamma=(Rs / R0).tolist(), grav=[j["grav"] for j in J],
                       closure=[j["M_wall_over_deficit"] for j in J], Gamma_at_RH=float(J[-1]["Gamma_in"]), max_grav=float(max(j["grav"] for j in J)),
                       max_rel_dev_from_flat=float(max(abs(j["Gamma_in"] / (R / R0) - 1) for j, R in zip(J, Rs))))
    print("wall: R0 %.3e /H0 (%.3f mm), Gamma(R_H) %.3e, gravity parameter %.2e, max dev from flat %.2e" %
          (R0, out["wall"]["R0_mm"], out["wall"]["Gamma_at_RH"], out["wall"]["max_grav"], out["wall"]["max_rel_dev_from_flat"]))
    # ---- daughter region: isolated bubble and percolated transition (baseline), two cycles
    th_deg = WJ["walls"]["0.30"]["theta_deg"]
    D = Daughter(base["eta"], base["f0"] * th_deg, base["q"], TW["theta"])
    for mode in ("percolated", "isolated"):
        r = D.run(mode, n_cycles=3)
        for e in r['events']:
            print(mode, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
        out[mode] = r
        print("%s done [%.0fs]" % (mode, time.time() - t0))
    json.dump(out, open(os.path.join(HERE, "bubble_gravity.json"), "w"), indent=1)
    print("saved [%.0fs]" % (time.time() - t0))
