# -*- coding: utf-8 -*-
"""Matter dilution, cooling, or both: which can hold and then release the symmetric vacuum?

Three triggers with the same scalar (mu = 100 H_0, q = 1) and identical initial data at z = 1100 (phi = 10^-60 v):
    A  matter only      rho_* A(phi), critical scale factor a_c = 2
    B  temperature only a thermal mass c T^2 phi^2 / 2 from a bath at T = T_0/a (saturating form, dilution_background.VT), critical a_T = 2
    C  both             a_c = a_T = 2
The symmetric point is held while rho_*/rho_crit + (T/T_c)^2 > 1: the trajectory of each model in the plane (rho_*/rho_crit, (T/T_c)^2) crosses
the line x + y = 1 at the release; A and B release at a = 2, C at the root of (2/a)^3 + (2/a)^2 = 1.

Physical viability of the thermal trigger. In physical units sqrt(c) = mu a_T / T_0 (T_0 = 2.35 x 10^-4 eV, mu in units of
H_0 = 1.44 x 10^-33 eV): a release at a_T ~ 1-10 needs sqrt(c) ~ 10^-29 mu a_T. A thermal mass of that origin needs the scalar to interact with
the bath, at a rate Gamma ~ c^2 T (for a coupling g ~ sqrt(c)); equilibrium today needs Gamma > H_0, i.e. sqrt(c) > (H_0/T_0)^(1/4) ~ 5 x 10^-8,
and a scalar coupled that strongly is held until a_T = sqrt(c) T_0 / mu > 10^20 / mu. A late thermal release and an equilibrium bath exclude each
other by about 20 orders of magnitude in the coupling. (The CMB photons couple to a neutral scalar only through higher-dimension operators,
whose thermal mass falls as T^4 / Lambda^2 and is smaller still.)

    python dilution_thermal.py         (writes dilution_thermal.json)
"""
import json
import os

import numpy as np

import dilution_background as D
import matter_coupling as MC

HERE = os.path.dirname(os.path.abspath(__file__))
H0_EV = 1.4368e-33 * (67.36 / 67.36)       # H_0 = 67.36 km/s/Mpc in eV
T0_EV = 2.3487e-4                          # CMB temperature today in eV


def run():
    out = {"models": {}}
    cases = {"A_matter": dict(coupled=True), "B_temperature": dict(coupled=False, a_T=2.0), "C_both": dict(coupled=True, a_T=2.0),
             "uncoupled_control": dict(coupled=False)}
    for k, kw in cases.items():
        m = MC.Model(2.0, 100.0, 1.0, **kw)
        r = D.evolve(m, 1e-60 * m.v, a_i=1e-3, t_max=45.0, rtol=1e-11)
        sol = r.pop("sol")
        a = np.exp(sol.y[0])
        x = (MC.RHO_DM0 / a ** 3) / m.rho_crit if m.coupled else 0.0 * a
        y = (2.0 / a) ** 2 if m.a_T else 0.0 * a
        k_rel = int(np.argmax(x + y < 1.0)) if ((x + y) < 1.0).any() else None
        out["models"][k] = dict(a_roll=r["a_roll"], a_turn=r["a_turn"], t_turn=r["t_turn"], max_residual=r["max_residual"],
                                a_release=(float(a[k_rel]) if k_rel is not None else None),
                                track=dict(a=a[::max(1, a.size // 300)].tolist(), x=np.asarray(x)[::max(1, a.size // 300)].tolist(),
                                           y=np.asarray(y)[::max(1, a.size // 300)].tolist()))
        print(k, {kk: out["models"][k][kk] for kk in ("a_release", "a_roll", "a_turn", "max_residual")})
    # analytic release of the combined model: (2/a)^3 + (2/a)^2 = 1
    from scipy.optimize import brentq
    out["a_release_C_analytic"] = float(brentq(lambda a: (2 / a) ** 3 + (2 / a) ** 2 - 1.0, 2.0, 10.0))
    # viability of the thermal trigger
    via = []
    for mu in (10.0, 100.0, 1000.0):
        for aT in (2.0, 10.0):
            sqrt_c = mu * H0_EV * aT / T0_EV
            via.append(dict(mu=mu, a_T=aT, sqrt_c_needed=sqrt_c, gamma_over_H0=sqrt_c ** 4 * T0_EV / H0_EV))
    g_eq = (H0_EV / T0_EV) ** 0.25
    out["thermal_viability"] = dict(rows=via, sqrt_c_for_equilibrium_today=g_eq,
                                    a_T_if_equilibrated={str(mu): g_eq * T0_EV / (mu * H0_EV) for mu in (10.0, 100.0, 1000.0)})
    print(out["thermal_viability"])
    return out


if __name__ == "__main__":
    o = run()
    json.dump(o, open(os.path.join(HERE, "dilution_thermal.json"), "w"), indent=1)
