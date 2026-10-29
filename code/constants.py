# -*- coding: utf-8 -*-
"""Physical constants and the observed dark-energy scale used to translate the dimensionless bounce results into cosmological
quantities. All energies in GeV (natural units, hbar = c = 1) unless stated.

Cosmological parameters: Planck 2018 TT,TE,EE+lowE+lensing+BAO (Planck Collaboration 2020, A&A 641, A6):
H0 = 67.66 km/s/Mpc, Omega_Lambda = 0.6889. The reduced Planck mass is M_Pl = (8 pi G)^(-1/2).
"""
import numpy as np

M_PL = 2.435e18                 # GeV, reduced Planck mass
HBAR_GEV_S = 6.582119569e-25    # GeV s
YEAR_S = 3.15576e7              # s
MPC_KM = 3.0856775814913673e19  # km per Mpc
H0_KMS_MPC = 67.66
OMEGA_L = 0.6889
T_CMB_K = 2.7255
K_TO_GEV = 8.617333262e-14      # GeV per kelvin

H0_GEV = H0_KMS_MPC / MPC_KM * HBAR_GEV_S                 # H0 in GeV
RHO_CRIT = 3.0 * M_PL ** 2 * H0_GEV ** 2                  # GeV^4
V_F = OMEGA_L * RHO_CRIT                                  # present dark-energy density (the false-vacuum energy), GeV^4
H_L = np.sqrt(V_F / 3.0) / M_PL                           # asymptotic de Sitter Hubble rate, GeV
T_HUBBLE_YR = HBAR_GEV_S / H_L / YEAR_S                   # 1/H_Lambda in years
M4_OVER_VF = M_PL ** 4 / V_F                              # (M_Pl^4 / V_f), about 1.4e120
LN_M4_OVER_VF = float(np.log(M4_OVER_VF))
T_CMB_GEV = T_CMB_K * K_TO_GEV
T_BBN_GEV = 1e-3                                          # GeV: the weak-interaction freeze-out / nucleosynthesis scale, ~1 MeV
GSTAR_BBN = 10.75                                         # photons, e+-, three neutrino species at T ~ 1 MeV
GSTAR_LOW = 3.36                                          # photons + three neutrino species at T << MeV (g_*)


def t_reh_max(delta_v_gev4, eps_th=1.0, gstar=GSTAR_LOW):
    """Upper bound on the reheating temperature if the released energy density eps_th * dV thermalizes into g_* relativistic species."""
    return (30.0 * eps_th * delta_v_gev4 / (np.pi ** 2 * gstar)) ** 0.25


if __name__ == "__main__":
    print("rho_Lambda = %.3e GeV^4  (%.3e eV^4)  quarter-root %.3f meV" % (V_F, V_F * 1e36, V_F ** 0.25 * 1e12))
    print("H_Lambda = %.3e GeV ; 1/H_Lambda = %.2f Gyr ; M^4/V_f = %.3e ; ln = %.2f" % (H_L, T_HUBBLE_YR / 1e9, M4_OVER_VF, LN_M4_OVER_VF))
    print("T_reh,max(eps=1, dV=V_f) = %.3e GeV = %.2f meV = %.1f K ; T_CMB = %.3f meV ; T_BBN/T_reh = %.2e"
          % (t_reh_max(V_F), t_reh_max(V_F) * 1e12, t_reh_max(V_F) / K_TO_GEV, T_CMB_GEV * 1e12, T_BBN_GEV / t_reh_max(V_F)))
