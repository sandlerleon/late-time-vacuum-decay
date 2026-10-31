# -*- coding: utf-8 -*-
"""Continuous hot cycles: the cycle of hot_cycles.py with every prescribed step replaced by dynamics.

No resets. The dark matter is a gas of particles of field-dependent mass m_eff = m A(phi) in kinetic equilibrium with the radiation bath; its comoving
number N = n a^3 obeys the Boltzmann equation, and its energy and pressure follow from the Maxwell-Boltzmann distribution at the bath temperature:
    dN/dt = -<sigma v> (N^2 - N_eq^2) / a^3,     n_eq = c_n (x^2 K_2(x)/2) T^3,  x = m_eff/T
    eps(T, m_eff) = m_eff K_1(x)/K_2(x) + 3T  (energy per particle),   pressure per particle T
The bath (radiation rho_r = T^4 in these units, plus the dark-matter gas) has rho_th = T^4 + n eps and p_th = T^4/3 + n T. It exchanges energy with
the scalar through the dissipation Upsilon = C_T T^3 and through the field dependence of the particle masses (each particle's energy changes by
(m_eff/E) dm_eff at fixed momentum):
    phi'' + (3H + Upsilon) phi' + V_0'(phi) + F = 0,       F = n <m_eff/E> m A'(phi) = n (K_1/K_2) m A'(phi)
    d rho_th/dt = -3H (rho_th + p_th) + Upsilon phi'^2 + F phi'
so that the total stress-energy is conserved by construction. The temperature is a state variable, from d rho_th/dt = (4T^3 + n d eps/dT) dT/dt
+ eps dn/dt + n (d eps/d m_eff) m A' phi'. The same coupling gives the hot-phase thermal mass (relativistic gas, F ~ n m_eff m A'/(2T)) and the
late-time hold (non-relativistic gas, F = rho_* A'), so the restoration of the vacuum and the freeze-out of the dark matter both emerge from the
equations. All matter is the coupled dark matter (baryons are not followed).

Gravity (bounce='lqc'): effective loop-quantum-cosmology equations for k = 0, H^2 = rho(1 - rho/rho_c)/3, dH/dt = -(rho + p)(1 - 2rho/rho_c)/2,
with H continuous through both turning points. bounce='ghost' replaces this by general relativity with a negative-energy stiff fluid,
rho_g = -g/a^6 (an independent bounce mechanism, Test C of the bounce validation). Optional Bianchi I shear: rho_sigma = Sigma/a^6 (stiff, comoving
Sigma conserved) in the Friedmann equation.

Units: time 1/H0, densities H0^2 M_Pl^2 (reduced Planck mass), fields M_Pl; T^4 = rho_r, so T = 1 corresponds to 1.82 meV (T0 = 0.129 is 2.725 K).
The dissipation Upsilon = C_T T^3 is C_T T^3/f^2 with f = (H0 M_Pl^3)^(1/4) = 2.1 TeV. The bounce density must exceed the dark-matter freeze-out
scale so that the dark matter is in equilibrium at the bounce.

Numerical treatment (stated, not physics): LSODA in chunks of at most one doubling (halving) of the scale factor, with the clock restarted in each
chunk (the system is autonomous; this keeps the resolution near a high-density bounce) and absolute tolerances set relative to each variable at the
start of the chunk (the field is damped by hundreds of e-folds in the hot phase); chunk lengths are also bounded by the field's damping time. H is
projected onto the constraint at the start of each chunk away from turning points (a conserved constraint violation would act as a cosmological
constant; every correction is recorded). At each bounce the scale factor is renormalized (a change of units; physical densities unchanged) so that
100 cycles fit in floating point; the cumulative growth is kept.

    python continuous_cycles.py [n_cycles]       (writes continuous_cycles.json and continuous_cycles.csv)
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kve

import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
RHO_M0 = MC.RHO_DM0 + MC.RHO_B0          # all matter is the coupled dark matter here
T0 = MC.RHO_R0 ** 0.25
KAPPA = RHO_M0 / T0 ** 3                  # rho_m / T^3 of the present universe
T_UNIT_EV = 2.35e-4 / T0                  # 1 temperature unit in eV (T0 = 2.725 K)
H0_OVER_MPL = 1.4385e-33 / 2.435e27       # H0 / M_Pl (reduced), both in eV, h = 0.6766


def gas_terms(x):
    """For the Maxwell-Boltzmann gas: K1/K2, g = x K1/K2 (eps = T (g + 3)), g'(x) and the heat capacity per particle d eps/dT = 3 + g - x g'."""
    if x > 100.0:
        # asymptotic series K1/K2 = 1 - 3/(2x) + 15/(8x^2) - 15/(8x^3) + 135/(128x^4) (written out so that nothing cancels; error < 1e-9)
        u = 1.0 / x
        kr = 1.0 - 1.5 * u + 1.875 * u ** 2 - 1.875 * u ** 3 + 135.0 / 128.0 * u ** 4
        gp = 1.0 - 1.875 * u ** 2 + 3.75 * u ** 3 - 405.0 / 128.0 * u ** 4
        cv = 1.5 + 3.75 * u - 5.625 * u ** 2 + (135.0 / 128.0 * 4.0) * u ** 3
        return kr, x * kr, gp, cv
    k0, k1, k2, k3 = kve(0, x), kve(1, x), kve(2, x), kve(3, x)
    kr = k1 / k2
    d1 = -0.5 * (k0 + k2)
    d2 = -0.5 * (k1 + k3)
    gp = kr + x * (d1 * k2 - k1 * d2) / (k2 * k2)
    return kr, x * kr, gp, 3.0 + x * kr - x * gp


def neq_factor(x):
    """x^2 K_2(x)/2: 1 for a relativistic gas, Boltzmann suppressed for x >> 1."""
    return 0.5 * x * x * kve(2, x) * np.exp(-x) if 0 < x < 700 else (1.0 if x <= 0 else 0.0)


def calibrate_sigv(m, c_n, target=KAPPA):
    """<sigma v> that gives the present matter-to-entropy ratio rho_m/T^3 = KAPPA by freeze-out in a radiation background."""
    def yield_for(sv):
        def rhs(lnT, y):                                  # Y = n/T^3 vs ln T (radiation era, H = T^2/sqrt(3), T ~ 1/a)
            T = np.exp(lnT)
            Yeq = c_n * neq_factor(m / T)
            return [sv * T * np.sqrt(3.0) * (y[0] * y[0] - Yeq * Yeq)]
        sol = solve_ivp(rhs, (np.log(10.0 * m), np.log(1e-3 * m)), [c_n], method="LSODA", rtol=1e-9, atol=1e-40)
        return sol.y[0, -1] * m
    lo, hi = -60.0, 10.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if yield_for(10 ** mid) > target:
            lo = mid
        else:
            hi = mid
    sv = 10 ** (0.5 * (lo + hi))
    return sv, yield_for(sv) / target


def attractor_scale(m_dm):
    """<sigma v> relative to its radiation-background calibration, fixed so that the attractor cycle reproduces the present dark-matter density in
    the held epoch (sigv_calibration.json, written by 'python continuous_cycles.py calibrate'); 1 if no calibration exists for this mass."""
    f = os.path.join(HERE, "sigv_calibration.json")
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        for row in d.get("rows", []):
            if abs(row["m_dm"] / m_dm - 1) < 1e-9:
                return row["scale"]
    return 1.0


class Params:
    def __init__(self, a_c=2.0, mu=30.0, q=1.0, m_dm=5.5e9, c_n=0.1, sigv=None, C_T=1.0, rho_c=1e44, bounce="lqc", ghost_g=None,
                 shear0=0.0, seed=1e-30, sigv_scale=None):
        self.mod = MC.Model(a_c, mu, q)
        # all matter couples here: rho_crit from the total matter density (a_c keeps its meaning)
        self.mod.rho_crit = RHO_M0 / a_c ** 3
        self.mod.M2 = self.mod.rho_crit / mu ** 2
        self.a_c, self.mu, self.q, self.m, self.c_n, self.C_T, self.rho_c = a_c, mu, q, m_dm, c_n, C_T, rho_c
        self.bounce, self.shear0, self.seed = bounce, shear0, seed
        self.sigv_radiation = None
        if sigv is None:
            sigv, ratio = calibrate_sigv(m_dm, c_n)
            self.sigv_radiation = sigv
            if sigv_scale is None:
                sigv_scale = attractor_scale(m_dm)
            sigv = sigv * sigv_scale
        self.sigv_scale = sigv_scale
        self.sigv = sigv
        self.ghost_g = ghost_g


def make(P, aux):
    m, mod = P.m, P.mod
    lqc = P.bounce == "lqc"

    def parts(y):
        la, H, p, pi, T, N = y[:6]
        a3 = np.exp(3.0 * la)
        n = max(N, 0.0) / a3
        me = m * mod.A(p)
        x = me / T
        kr, g, gp, cv = gas_terms(x)
        eps = T * (g + 3.0)
        rho_th = T ** 4 + n * eps
        p_th = T ** 4 / 3.0 + n * T
        r6 = np.exp(-6.0 * la)
        rs = aux["Sigma"] * r6
        rg = -aux["g"] * r6 if not lqc else 0.0
        V = mod.V0(p)
        rho = rho_th + 0.5 * pi * pi + V + rs + rg
        prs = p_th + 0.5 * pi * pi - V + rs + rg
        return dict(n=n, me=me, x=x, kr=kr, eps=eps, deps_dT=cv, deps_dm=gp, rho_th=rho_th, p_th=p_th, rho=rho, p=prs, a3=a3,
                    rs=rs, rg=rg, V=V, K=0.5 * pi * pi)

    def slaved_pi(y, c):
        """Overdamped limit (Upsilon >> m_eff, H): the field moves at its terminal velocity."""
        U = P.C_T * y[4] ** 3
        F = c["n"] * c["kr"] * m * mod.dA(y[2])
        return -(mod.dV0(y[2]) + F) / (3.0 * y[1] + U)

    def rhs(t, y):
        c = parts(y)
        if aux.get("cons"):                         # constrained formulation: H from the Friedmann constraint, not from y[1]
            y = np.array(y, dtype=float)
            y[1] = aux["s"] * np.sqrt(max(constraint(P, c), 0.0))
        if aux["od"]:
            y = np.array(y, dtype=float)
            y[3] = slaved_pi(y, c)
            c = parts(y)
        la, H, p, pi, T, N, J, W = y
        n = c["n"]
        U = P.C_T * T ** 3
        dAp = mod.dA(p)
        F = n * c["kr"] * m * dAp
        if lqc:
            dH = -0.5 * (c["rho"] + c["p"]) * (1.0 - 2.0 * c["rho"] / P.rho_c)
        else:
            dH = -0.5 * (c["rho"] + c["p"])
        dpi = -(3.0 * H + U) * pi - mod.dV0(p) - F
        Neq = P.c_n * neq_factor(c["x"]) * T ** 3 * c["a3"]
        dN = -P.sigv * (N * N - Neq * Neq) / c["a3"]
        dn = dN / c["a3"] - 3.0 * H * n
        drho_th = -3.0 * H * (c["rho_th"] + c["p_th"]) + U * pi * pi + F * pi
        dT = (drho_th - c["eps"] * dn - n * c["deps_dm"] * m * dAp * pi) / (4.0 * T ** 3 + n * c["deps_dT"])
        if aux["od"]:
            dpi = 0.0
        return [H, dH, pi, dpi, dT, dN, T ** 4, 3.0 * H * c["p"] * c["a3"]]
    rhs.slaved_pi = slaved_pi
    return rhs, parts


class _Budget(Exception):
    pass


class _Failed:
    success = False
    message = "evaluation budget exceeded"


def _solve(rhs, dt, y, method, rtol, atol, ms, budget=400000, events=None):
    """solve_ivp with a cap on right-hand-side evaluations (a stalled chunk is reported as a failure and retried by the caller)."""
    cnt = [0]

    def f(t, yy):
        cnt[0] += 1
        if cnt[0] > budget:
            raise _Budget()
        return rhs(t, yy)
    try:
        return solve_ivp(f, (0.0, dt), y, method=method, rtol=rtol, atol=atol, max_step=ms, events=events)
    except (_Budget, ValueError, np.linalg.LinAlgError):
        return _Failed()


def constraint(P, c):
    return c["rho"] * (1.0 - c["rho"] / P.rho_c) / 3.0 if P.bounce == "lqc" else c["rho"] / 3.0


def run(P, n_cycles=6, rtol=1e-8, verbose=False, wall=None, keep_history=True, floor=False, trace=None, y0=None, aux0=None,
        od_ratio=1e3, proj_every=1, gravity="projected", near=0.05):
    """floor=True: the field is not allowed below the de Sitter fluctuation amplitude H/(2 pi) (physical units) once the vacuum is tachyonic.
    The default is the deterministic classical evolution with no floor.
    y0, aux0: start from a given state just after a bounce (cycle_map.py) instead of today.
    od_ratio: the overdamped reduction is used where Upsilon > od_ratio max(m_eff, H, mu); proj_every: project H onto the constraint at every
    proj_every-th chunk (numerical-admissibility tests, numerics_admissibility.py).
    gravity='constrained': an independent formulation of the gravitational equations (independent_solver.py). Away from the turning points H is not
    a variable but is computed from the constraint, H = s sqrt(rho(1 - rho/rho_c)/3), so the constraint holds identically and nothing is projected;
    where H^2 falls below (near x natural rate)^2, i.e. close to a turnaround or bounce, H is evolved with the Raychaudhuri equation (the square root
    has no information about the sign change there). The constraint mismatch accumulated across each such bridge is recorded in 'projections'."""
    aux = dict(Sigma=0.0, g=0.0, od=False)
    rhs, parts = make(P, aux)
    od_chunks = 0
    retries = 0
    mod = P.mod
    N0 = RHO_M0 / P.m
    p0 = P.seed * mod.v
    y = np.array([0.0, 0.0, p0, 0.0, T0, N0, 0.0, 0.0])
    aux["Sigma"] = P.shear0 * 3.0                         # Omega_sigma today = shear0
    if P.bounce == "ghost":
        aux["g"] = P.ghost_g if P.ghost_g is not None else 0.0      # set at the first turnaround (see below) unless given
    c = parts(y)
    y[1] = np.sqrt(constraint(P, c))
    start = "today"
    if y0 is not None:
        y = np.array(y0, dtype=float)
        if aux0:
            aux.update({k: v for k, v in aux0.items() if k in ("Sigma", "g")})
        start = "bounce"
    lnG = 0.0
    cycles = []
    cur = dict(n=0, start=start, hist=[])
    chunk_no = 0
    proj = dict(n=0, max_rel=0.0)
    max_res = 0.0
    prev_H = y[1]
    t_cyc = 0.0
    held_N = acc_N = 0.0
    t_start = time.time()
    status = "completed"
    phase = "expanding"
    J_last_bounce = 0.0
    wall = wall or 600 + 120 * n_cycles
    while len(cycles) < n_cycles + 1:
        c = parts(y)
        rho = c["rho"]
        hc2 = constraint(P, c)
        far = (P.bounce != "lqc" or rho < 0.4 * P.rho_c) and rho > 0 and hc2 > 0 and abs(y[1]) > 1e-3 * np.sqrt(hc2)
        if P.bounce == "ghost":
            far = far and abs(c["rg"]) < 0.25 * (rho - c["rg"])
        chunk_no += 1
        cons_events = None
        if gravity == "constrained":
            scale0 = np.sqrt(max(abs(c["rho_th"]) + c["K"] + abs(c["V"]) + c["rs"], 1e-12) / 3.0)
            ok = rho < 0.3 * P.rho_c and hc2 > 0                      # below the bounce event at 0.4 rho_c (hysteresis)
            if aux.get("cons"):
                aux["cons"] = ok and hc2 > (1.5 * near * scale0) ** 2      # a chunk that ended at the event threshold hands over to the bridge
            else:
                aux["cons"] = ok and hc2 > (2 * near * scale0) ** 2
                if aux["cons"]:                       # leaving a Raychaudhuri bridge: record the constraint mismatch accumulated across it
                    proj["n"] += 1
                    proj["max_rel"] = max(proj["max_rel"], abs(abs(y[1]) / np.sqrt(hc2) - 1))
            if aux["cons"]:
                aux["s"] = 1.0 if y[1] >= 0 else -1.0
                y[1] = aux["s"] * np.sqrt(hc2)

                def ev(t, yy):                        # H^2 falls below (near x the current natural rate)^2: hand over to the Raychaudhuri bridge
                    cc = parts(yy)
                    return constraint(P, cc) - near ** 2 * (abs(cc["rho_th"]) + cc["K"] + abs(cc["V"]) + cc["rs"]) / 3.0

                def ev_bounce(t, yy):                 # approach of the bounce density
                    return 0.4 * P.rho_c - parts(yy)["rho"]
                ev.terminal, ev.direction = True, -1
                ev_bounce.terminal, ev_bounce.direction = True, -1
                cons_events = [ev, ev_bounce]
        elif far and chunk_no % proj_every == 0:
            proj["n"] += 1
            proj["max_rel"] = max(proj["max_rel"], abs(abs(y[1]) / np.sqrt(hc2) - 1))
            y[1] = np.sign(y[1]) * np.sqrt(hc2)
        scale = np.sqrt(max(abs(c["rho_th"]) + c["K"] + abs(c["V"]) + c["rs"] + abs(c["rg"]), 1e-12) / 3.0)
        dt = min(np.log(2.0) / max(abs(y[1]), 1e-300), 0.5 / scale)
        # field: effective mass about phi = 0 and damping; bound the step by the period and the chunk by a few damping times
        m2 = abs(c["n"] * c["kr"] * P.m * mod.dA(1.0) * mod.A(y[2]) - P.mu ** 2) + P.mu ** 2
        meff = np.sqrt(m2)
        U = P.C_T * y[4] ** 3
        damp = m2 / U if U > 2 * meff else 0.5 * U
        dt = min(dt, 5.0 / max(damp, 1e-300))
        ms = 0.6 / meff if meff > 20 * abs(y[1]) else np.inf
        # overdamped reduction where the dissipation exceeds every other rate by 10^3 (error O(m_eff^2/Upsilon^2, H/Upsilon))
        aux["od"] = bool(U > od_ratio * max(meff, scale, P.mu))
        if aux["od"]:
            od_chunks += 1
            ms = np.inf
            y[3] = rhs.slaved_pi(y, c)
        amp = np.sqrt(y[2] ** 2 + (y[3] / meff) ** 2)
        atol = [1e-13, 1e-10 * scale, 1e-9 * amp + 1e-300, 1e-9 * amp * meff + 1e-300, 1e-12 * y[4], 1e-10 * max(y[5], 1e-300),
                1e-9 * max(abs(y[6]), y[4] ** 4 * dt), 1e-9 * max(abs(y[7]), 3.0 * abs(y[1] * c["p"]) * c["a3"] * dt, 1e-300)]
        if trace is not None:
            trace.append((phase, y.copy(), dt, ms, c["x"], meff, U))
            if verbose == 2 or (verbose and len(trace) % 100 == 0):
                print("   chunk %d %s lna %.3f H %.3g phi/v %.3g T %.4g x %.3g meff/H %.3g [%.0fs]" % (len(trace), phase, y[0] + lnG, y[1], y[2] / mod.v, y[4],
                      c["x"], meff / max(abs(y[1]), 1e-300), time.time() - t_start), flush=True)
        if not np.all(np.isfinite(y)):
            status = "non-finite state"
            break
        # explicit DOP853 first on a small evaluation budget (most chunks are not stiff); LSODA, Radau and shorter chunks as fall-backs
        sol = _solve(rhs, dt, y, "DOP853", rtol, atol, ms, budget=30000, events=cons_events)
        if not sol.success:
            sol = _solve(rhs, dt, y, "LSODA", rtol, atol, ms, events=cons_events)
        tries = 0
        while not sol.success and tries < 8:                  # LSODA occasionally fails or stalls: retry with Radau, then shorter chunks
            tries += 1
            dt_try = dt * 0.5 ** ((tries + 1) // 2)
            sol = _solve(rhs, dt_try, y, "Radau" if tries % 2 else "LSODA", rtol, atol, ms, events=cons_events)
            retries += 1
        if not sol.success:
            status = "integration failed: " + sol.message
            break
        Y = sol.y
        if aux.get("cons"):
            for i in range(Y.shape[1]):
                Y[1, i] = aux["s"] * np.sqrt(max(constraint(P, parts(Y[:, i])), 0.0))
        if aux["od"]:
            for i in range(Y.shape[1]):
                Y[3, i] = rhs.slaved_pi(Y[:, i], parts(Y[:, i]))
        n_pts = Y.shape[1]
        Hs = Y[1]
        cut = None
        for i in range(1, n_pts):
            hp = prev_H if i == 1 else Hs[i - 1]
            if hp > 0 >= Hs[i] and phase == "expanding" and parts(Y[:, i])["rho"] < 0.5 * P.rho_c:
                cut, kind = i, "turn"
                break
            if hp < 0 <= Hs[i] and phase == "contracting":
                cut, kind = i, "bounce"
                break
        last = cut if cut is not None else n_pts - 1
        idx = np.unique(np.concatenate([np.linspace(0, last, min(25, last + 1)).astype(int), [last]]))
        prev = None
        for i in idx:
            yi = Y[:, i]
            ci = parts(yi)
            comp = ci["rho_th"] + ci["K"] + abs(ci["V"]) + ci["rs"] + abs(ci["rg"])
            res = abs(3 * yi[1] ** 2 - 3 * constraint(P, ci)) / max(comp * (1 + (ci["rho"] / P.rho_c if P.bounce == "lqc" else 0)), MC.VC)
            max_res = max(max_res, res)
            pos = ci["rho_th"] + ci["K"] + max(ci["V"], 0) + ci["rs"]
            H = yi[1]
            if prev is not None and H > 0 and phase == "expanding":
                dtt = sol.t[i] - sol.t[prev]
                ddot = (-0.5 * (ci["rho"] + ci["p"]) * ((1 - 2 * ci["rho"] / P.rho_c) if P.bounce == "lqc" else 1.0)) + H * H
                if ddot > 0:
                    acc_N += H * dtt
                    if abs(yi[2]) < 0.1 * mod.v:
                        held_N += H * dtt
            if phase == "expanding":
                mfrac = ci["n"] * ci["me"] / pos if ci["x"] > 10 else 0.0
                cur["max_matter_fraction"] = max(cur.get("max_matter_fraction", 0.0), mfrac)
                cur["max_vacuum_fraction"] = max(cur.get("max_vacuum_fraction", 0.0), max(ci["V"], 0) / pos)
                if "T_release" not in cur and abs(yi[2]) > 0.5 * mod.v and H > 0:
                    cur["T_release"] = float(yi[4])
                if cur["start"] == "bounce" and "Y_fo" not in cur and ci["x"] > 300 and H > 0:
                    cur["Y_fo"] = float(ci["n"] * ci["me"] / yi[4] ** 3)             # just after freeze-out (the field may still be frozen)
                    cur["Y_fo_over_kappa"] = cur["Y_fo"] / KAPPA
                if cur["start"] == "bounce" and ci["x"] > 300 and H > 0 and abs(yi[2]) < 0.01 * mod.v and "T_release" not in cur:
                    cur["Y_held_over_kappa"] = float(ci["n"] * ci["me"] / yi[4] ** 3) / KAPPA     # field restored: the value compared with today
            cur["max_shear_fraction"] = max(cur.get("max_shear_fraction", 0.0), ci["rs"] / pos)
            if keep_history:
                cur["hist"].append([float(yi[0] + lnG), float(t_cyc + sol.t[i]), float(yi[1]), float(yi[4]), float(ci["rho_th"]), float(ci["n"] * ci["me"]),
                                    float(ci["K"]), float(ci["V"]), float(ci["rs"]), float(abs(yi[2]) / mod.v), float(ci["x"]), float(yi[6] - J_last_bounce),
                                    float((ci["p"] / ci["rho"]) if ci["rho"] != 0 else 0.0)])
            prev = i
        t_cyc += sol.t[last]
        if floor and phase == "expanding":
            yl = Y[:, last]
            cl = parts(yl)
            if cl["n"] * cl["kr"] * P.m / mod.M2 < P.mu ** 2:                     # tachyonic at phi = 0
                phq = abs(yl[1]) * H0_OVER_MPL / (2 * np.pi)
                if abs(yl[2]) < phq:
                    Y[2, last] = np.sign(yl[2] if yl[2] != 0 else 1.0) * phq
                    cur["floor_applied"] = True
        if cut is not None:
            yi = Y[:, cut]
            ci = parts(yi)
            pos = ci["rho_th"] + ci["K"] + max(ci["V"], 0) + ci["rs"]
            if kind == "turn":
                cur.update(ln_a_max=float(yi[0] + lnG), T_turn=float(yi[4]), ln_S_turn=float(3 * (yi[0] + lnG) + 3 * np.log(yi[4])),
                           held_accelerated_efolds=held_N, accelerated_efolds=acc_N, phi_turn_over_v=float(yi[2] / mod.v),
                           shear_fraction_turn=float(ci["rs"] / pos), W_expand=float(yi[7]))
                if P.bounce == "ghost" and P.ghost_g is None and not cycles:
                    # the ghost's one constant: chosen at the first turnaround so that the first radiation bounce happens at rho_c, as for the LQC bounce
                    # (rho_r a^4 = rho_g a^6 at rho_r = rho_c); it is fixed from then on, so later bounces show how the ghost bounce responds to the entropy
                    ra4 = yi[4] ** 4 * np.exp(4 * yi[0])
                    aux["g"] = ra4 ** 1.5 / np.sqrt(P.rho_c)
                phase = "contracting"
            else:
                cur.update(ln_a_min=float(yi[0] + lnG), T_bounce=float(yi[4]), bounce_radiation_fraction=float(yi[4] ** 4 / pos),
                           bounce_matter_fraction=float(ci["n"] * ci["eps"] / pos), bounce_kinetic_fraction=float(ci["K"] / pos),
                           bounce_shear_fraction=float(ci["rs"] / pos), bounce_x=float(ci["x"]), bounce_rho=float(ci["rho"]),
                           ln_S_bounce=float(3 * (yi[0] + lnG) + 3 * np.log(yi[4])), phi_bounce_over_v=float(abs(yi[2]) / mod.v),
                           W_cycle=float(yi[7]), t_cycle=float(t_cyc))
                cycles.append(cur)
                if verbose:
                    print("  cycle %d: %s [%.0fs]" % (cur["n"], {k: (("%.4g" % v) if isinstance(v, float) else v) for k, v in cur.items() if k != "hist"},
                                                    time.time() - t_start), flush=True)
                cur = dict(n=len(cycles), start="bounce", hist=[])
                held_N = acc_N = 0.0
                t_cyc = 0.0
                phase = "expanding"
                shift = float(Y[0, cut])
                lnG += shift
                y = Y[:, cut].copy()
                y[0] -= shift
                y[5] *= np.exp(-3 * shift)
                y[7] = 0.0
                J_last_bounce = y[6]
                aux["Sigma"] *= np.exp(-6 * shift)
                aux["g"] *= np.exp(-6 * shift)
                prev_H = y[1]
                continue
            prev_H = Y[1, cut]
            y = Y[:, cut].copy()
            continue
        prev_H = Y[1, last]
        y = Y[:, last].copy()
        if t_cyc > 1e4:
            status = "no turnaround" if phase == "expanding" else "no bounce"
            cur["end"] = status
            cycles.append(cur)
            break
        if time.time() - t_start > wall:
            status = "wall-clock limit"
            break
    for k in range(len(cycles) - 1):
        a, b = cycles[k], cycles[k + 1]
        if "ln_a_max" in a and "ln_a_max" in b:
            a["lnG"] = b["ln_a_max"] - a["ln_a_max"]
            a["lnG_entropy"] = (b["ln_S_turn"] - a["ln_S_turn"]) / 3.0
    return dict(params=dict(a_c=P.a_c, mu=P.mu, q=P.q, m_dm=P.m, c_n=P.c_n, sigv=P.sigv, sigv_scale=P.sigv_scale, sigv_radiation=P.sigv_radiation, C_T=P.C_T, rho_c=P.rho_c, bounce=P.bounce, shear0=P.shear0,
                            seed=P.seed, floor=floor, kappa_target=KAPPA),
                cycles=cycles, status=status, max_residual=float(max_res), projections=proj, overdamped_chunks=od_chunks, retries=retries, seconds=time.time() - t_start, y_end=[float(v) for v in y],
                aux_end=dict(aux))


def summary(r):
    print(r["params"], r["status"], "res %.1e" % r["max_residual"], r["projections"], "%.0fs" % r["seconds"], flush=True)
    for c in r["cycles"]:
        print("  n%d %s" % (c["n"], {k: (("%.4g" % v) if isinstance(v, float) else v) for k, v in c.items() if k not in ("n", "hist")}))


def classify(r):
    """Outcome of a run from its cycles n >= 1 (the cycles that start at a bounce)."""
    cs = [c for c in r["cycles"] if c["n"] >= 1]
    if r["status"] != "completed":
        return "F6 " + r["status"]
    if not cs:
        return "F5 no bounce"
    if any(c.get("bounce_shear_fraction", 0) > 0.1 for c in r["cycles"]):
        return "F4 shear-dominated bounce"
    if any(c.get("bounce_kinetic_fraction", 0) > 0.5 for c in r["cycles"]):
        return "F3 kinetic-dominated bounce"
    if any(c.get("Y_fo_over_kappa", c.get("Y_late_over_kappa", 0)) < 0.1 for c in cs):
        return "F1 dark matter not regenerated (bounce too cold)"
    if any(c.get("held_accelerated_efolds", 0) < 0.5 for c in cs):
        return "F2 vacuum not restored (no held epoch)"
    return "working"


def calibrate_attractor(m_dm=5.5e9, n_cycles=3, iterations=6, seed_points=(), **kw):
    """Fix <sigma v> by the held-epoch abundance of the attractor (mean over cycles >= 2): secant iteration on ln Y_held(ln scale), since
    Y_held is not proportional to 1/<sigma v> (the freeze-out happens while the dark matter is heavy and the restoration heats the bath)."""
    pts = [tuple(p) for p in seed_points]
    hist = [dict(scale=a, Y_held_over_kappa=b, status="seed") for a, b in pts]

    def measure(scale):
        P = Params(m_dm=m_dm, sigv_scale=scale, **kw)
        r = run(P, n_cycles=n_cycles, keep_history=False)
        ys = [c["Y_held_over_kappa"] for c in r["cycles"] if c["n"] >= 2 and c.get("Y_held_over_kappa")]
        y = float(np.mean(ys)) if ys else float("nan")
        hist.append(dict(scale=scale, Y_held_over_kappa=y, status=r["status"], per_cycle=ys))
        print("  calibration: scale %.6g -> Y_held/kappa %.5f %s" % (scale, y, ys), flush=True)
        return y
    if not pts:
        pts.append((1.0, measure(1.0)))
    for it in range(iterations):
        if len(pts) >= 2:
            (s0, y0), (s1, y1) = pts[-2], pts[-1]
            b = (np.log(y1) - np.log(y0)) / (np.log(s1) - np.log(s0))
            ln_s = np.log(s1) - np.log(y1) / b
        else:
            ln_s = np.log(pts[-1][0] * pts[-1][1])
        s_new = float(np.exp(ln_s))
        y = measure(s_new)
        pts.append((s_new, y))
        if not np.isfinite(y) or abs(y - 1) < 0.02:
            break
    best = min(pts, key=lambda p: abs(np.log(p[1])) if np.isfinite(p[1]) else 1e9)
    return dict(m_dm=m_dm, scale=best[0], Y_held_over_kappa=best[1], iterations=hist)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "calibrate":
    rows = [calibrate_attractor(5.5e9, seed_points=[(1.0, 0.14499), (0.14499, 0.35548)])]       # the first two points of an earlier run
    json.dump(dict(note="<sigma v> = scale x radiation-background calibration; scale fixed by the held-epoch abundance of the attractor cycle", rows=rows),
              open(os.path.join(HERE, "sigv_calibration.json"), "w"), indent=1)
elif __name__ == "__main__":
    # baseline: 100 MeV dark matter, bounce at ~180 MeV, Omega_sigma = 1e-30 today (passive Bianchi I shear), 100 cycles
    n_c = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    P = Params(shear0=1e-30)
    print("sigma v:", P.sigv, flush=True)
    r = run(P, n_cycles=n_c, verbose=True, wall=6 * 3600)
    r["classification"] = classify(r)
    summary(r)
    for c in r["cycles"]:
        if c["n"] > 2 and c["n"] < len(r["cycles"]) - 1:
            c["hist"] = c["hist"][::8]                          # keep the full history of the first cycles and the last one
    json.dump(r, open(os.path.join(HERE, "continuous_cycles.json"), "w"), indent=0)
    keys = sorted({k for c in r["cycles"] for k in c if k != "hist"})
    with open(os.path.join(HERE, "continuous_cycles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for c in r["cycles"]:
            w.writerow({k: c.get(k) for k in keys})
