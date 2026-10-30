# -*- coding: utf-8 -*-
"""Matter-coupled scalar: the dilution-triggered vacuum transition (Section 6).

Action (Einstein frame, reduced Planck units):
    S = Int d^4x sqrt(-g) [ R/2 - (1/2)(d phi)^2 - V_0(phi) ] + S_dm[ A^2(phi) g, psi_dm ] + S_b[g] + S_r[g]
with a conformal coupling of the scalar to dark matter only (baryons and radiation are uncoupled). For pressureless dark matter the
conserved density rho_* = rho_*0 a^-3 enters the scalar equation through

    V_eff(phi, a) = V_0(phi) + rho_*(a) A(phi),       physical dark-matter density  rho_dm = A(phi) rho_*

    V_0(phi) = V_c - mu^2 phi^2 / 2 + lam phi^4 / 4 + eps phi ,       A(phi) = 1 + phi^2 / (2 M^2)

so that m_eff^2(phi = 0) = rho_* / M^2 - mu^2: the symmetric point is held by matter while rho_* > rho_crit = mu^2 M^2 and becomes a maximum
once rho_* falls below it, at a_crit = (rho_*0 / mu^2 M^2)^(1/3). The broken minima sit at phi^2 = v^2 (1 - rho_*/rho_crit), v^2 = mu^2/lam, with
V_0(v) = V_c - mu^2 v^2 / 4. The model is the symmetron form (Hinterbichler and Khoury 2010) with a dark-matter coupling and a negative broken vacuum.

Units: time in 1/H_0, densities in H_0^2 M_Pl^2 (critical density today = 3), fields in M_Pl. The dark-energy density today is V_c = 3 Omega_Lambda
(phi = 0 today). A model is fixed by
    a_c   critical scale factor (a_c > 1: the symmetric point is still stable today)
    mu    tachyonic mass scale in units of H_0
    q     depth of the broken vacuum, V_0(v) = -q V_c (q > 0: anti-de Sitter; q < 0: a lower de Sitter vacuum)
    eps   tilt (selects the branch), usually 0, with an initial displacement phi_0 instead
    a_T   optional thermal branch: a thermal mass c T^2 phi^2 / 2 from a bath at temperature T (T = T_0/a for an adiabatic bath), normalized so
          that it alone would hold the symmetric point until a = a_T, i.e. c T_0^2 = mu^2 a_T^2 (bath energy rho_r = T^4 in these units)

Thermal branch (consistent thermodynamics): bath pressure P = T^4/3 - c T^2 phi^2/2, entropy density s = dP/dT, energy density rho = T s - P;
the comoving entropy s a^3 is conserved (plus Upsilon pi^2/T when the scalar dissipates into the bath) and the scalar feels -dP/dphi = c T^2 phi.
The symmetric point is stable while rho_*/rho_crit + (T/T_c)^2 > 1, with T_c = mu/sqrt(c).
"""
import numpy as np

import constants as K

H0_GYR = 977.792 / 67.36                     # 1/H_0 in Gyr (h = 0.6736)
OM_R = 9.1e-5                                # radiation (photons + massless neutrinos)
OM_B = 0.0493                                # baryons (uncoupled)
OM_L = K.OMEGA_L
OM_DM = 1.0 - OM_L - OM_B - OM_R             # coupled dark matter (flat)
VC = 3.0 * OM_L
RHO_DM0 = 3.0 * OM_DM
RHO_B0 = 3.0 * OM_B
RHO_R0 = 3.0 * OM_R


class Model:
    def __init__(self, a_c=2.0, mu=10.0, q=1.0, eps=0.0, coupled=True, a_T=None):
        self.a_c, self.mu, self.q, self.eps, self.coupled = float(a_c), float(mu), float(q), float(eps), bool(coupled)
        self.a_T = None if a_T is None else float(a_T)
        self.T0 = RHO_R0 ** 0.25
        self.c_T = 0.0 if a_T is None else self.mu ** 2 * self.a_T ** 2 / self.T0 ** 2
        self.rho_crit = RHO_DM0 / self.a_c ** 3
        self.M2 = self.rho_crit / self.mu ** 2                      # rho_crit = mu^2 M^2
        self.v2 = 4.0 * (1.0 + self.q) * VC / self.mu ** 2          # V_0(v) = V_c - mu^2 v^2/4 = -q V_c
        self.v = np.sqrt(self.v2)
        self.lam = self.mu ** 2 / self.v2

    # ---------------------------------------------------------------- potentials
    def V0(self, p):
        return VC - 0.5 * self.mu ** 2 * p ** 2 + 0.25 * self.lam * p ** 4 + self.eps * p

    def dV0(self, p):
        return -self.mu ** 2 * p + self.lam * p ** 3 + self.eps

    def A(self, p):
        return 1.0 + 0.5 * p ** 2 / self.M2 if self.coupled else 1.0 + 0.0 * p

    def dA(self, p):
        return p / self.M2 if self.coupled else 0.0 * p

    def rho_star(self, a):
        return RHO_DM0 / a ** 3

    def V_eff(self, p, a):
        return self.V0(p) + self.rho_star(a) * self.A(p)

    def m2_eff0(self, a):
        """Effective squared mass at phi = 0."""
        return (self.rho_star(a) / self.M2 if self.coupled else 0.0) + self.c_T * (self.T0 / a) ** 2 - self.mu ** 2

    def V_min(self):
        return VC - 0.25 * self.mu ** 2 * self.v2

    def coupling_beta(self, p):
        """Strength of the dark-matter coupling, beta = M_Pl d ln A / d phi (zero at the symmetric point)."""
        return self.dA(p) / self.A(p)

    def describe(self):
        return dict(a_c=self.a_c, mu=self.mu, q=self.q, eps=self.eps, coupled=self.coupled, a_T=self.a_T, c_T=self.c_T, M=float(np.sqrt(self.M2)), v=float(self.v),
                    lam=float(self.lam), rho_crit=float(self.rho_crit), V_min=float(self.V_min()),
                    m_eff_today=float(np.sqrt(max(self.m2_eff0(1.0), 0.0))), A_at_v=float(self.A(self.v)))
