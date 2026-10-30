# Daughter Universes and Oscillating Cycles from a Metastable Dark-Energy Vacuum

Leon Sandler, Independent Researcher — sandler.leon@gmail.com — ORCID 0009-0007-4584-808X

Code, data and manuscript for *"Daughter Universes and Oscillating Cycles from a Metastable Dark-Energy Vacuum: Quantum and Thermal Decay,
Survival, and Reheating Bounds,"* prepared for **The European Physical Journal C**. (Earlier titles: "Daughter-Universe Nucleation from a Metastable Dark-Energy Vacuum: …", "Late-Time Vacuum Decay as a Cosmological Reset: …"
and "Late-Time Vacuum Decay and Cosmological Reheating: …".)

## What the paper does

Can the decay of a metastable late-time vacuum nucleate a daughter region that becomes a hot expanding universe, and does cosmological cooling make the
transition more likely? The paper treats both as calculations. A scalar potential with an explicit metastable minimum is specified; the O(4)
Coleman–De Luccia bounce is solved numerically in flat space and with Einstein gravity; the action is converted to a nucleation rate with the **measured
dark-energy density** as the false-vacuum energy; survival of the present vacuum is evaluated on the **Lorentzian past light cone** of the observed expansion
history; a **thermal mass** from a coupling to the CMB is added (the potential rescales exactly, the O(3) thermal and O(4) quantum exponents are compared, and the survival
integral is computed with a time-dependent rate); and the interior of a nucleated bubble is followed to see whether it can become a hot universe. A daughter
region is not a cyclic cosmology, and the paper claims only to test this one reset mechanism (Option A in the paper; Option B, a genuine bounce, is not modelled).

## Results the code reproduces

| Result | Where |
|---|---|
| With all scales set by the dark energy g = f²/M_Pl² ≈ 10⁻⁶⁰; gravitational corrections are ~10⁻⁵⁹ and the Hawking–Moss regime is irrelevant | `falsification_map.py`, `bounce_tables.py` |
| Numerical error budget of the action (≤ 3×10⁻⁸ at the production settings) and the negative-mode structure (one negative eigenvalue, l = 1 zero modes, no node) | `convergence.py` |
| Quantum decay controlled by one action B = b(η)/λ; Γ/H⁴ = 1 at B_c ≈ 287 | `rates.py`, `falsification_map.py` |
| Past-light-cone four-volume V₄ = 0.059 H_Λ⁻⁴, 445× below the Euclidean four-sphere; 95% survival bound Γ/H_Λ⁴ < 51 | `survival.py` |
| Thermal mass rescales the potential exactly (η_eff = η/(1−2θ)^{3/2}); O(3) thermal saddle dominates above T× = 0.18 meV for the benchmark; the rate falls monotonically as the universe cools | `thermal.py`, `thermal_analysis.py` |
| Over the stated priors (a sample space, not probabilities): 14.8% above the survival bound for a decoupled scalar, 24–49% if it is in equilibrium with the CMB, and no late-time decay window left | `monte_carlo.py`, `thermal_analysis.py` |
| Reheating T ≤ 2.2 meV (4.6×10⁸ below nucleosynthesis); the open interior stays curvature dominated for any sub-Planckian energy and any number of scalar fields, so a hot epoch at 1 s needs a parent scale at least 41 orders of magnitude above M_Pl | `interior_evolution.py`, `scope.py` |
| Lorentzian simulations of the daughter region (two fields, dissipation, gravity): closed-form interior reproduced to 7%; a reservoir hidden by a negative offset leaves an AdS vacuum that crunches after 5.5 s when it reaches 0.7 MeV; hilltop flattening at the dark-energy scale needs a vev above 4 M_Pl and still reheats below 1.5 meV; repeated transitions obey ΣΔV ≤ ρ_Λ; entropy accounting | `interior_sim.py` |
| Option B (same region oscillating), effective LQC bounce + AdS turnaround: radiation-only cycles periodic over 1000+ bounces; a scalar excitation is amplified ×6–400 per cycle (cosmological hysteresis; validated with 3 integrators); turnaround radius obeys Δa_max = W/3 (W = ∮p dV); 100-cycle phase diagram: stable only for ΥP ≳ ln μ; a vacuum transition during the cycles destabilises them; no cycle has positive vacuum energy | `oscillation.py`, `oscillation_validation.py` |
| A matter trigger needs Δln m ≳ 10⁻³ between the vacua: linear β ~ 10²⁶–10²⁹, or a quadratic scale of tens of meV | `coupling.py` |

The cyclic-reset hypothesis, in the form tested, is not supported; the daughter region is cold and empty. The paper separates internal consistency conditions from observational tests.

## Contents

```
code/       cdl_lib.py              potential, flat-space and gravitational bounce solvers, actions, analytic references
            constants.py            dark-energy density, Hubble rate, reheating bound (Planck 2018)
            bounce_tables.py        b(eta), gravitational suppression grid, Hawking-Moss criterion
            convergence.py          numerical error budget, gravity coefficient, negative-mode spectrum
            survival.py             past-light-cone four-volume and the survival bound
            thermal.py              thermal potential, O(3)/O(4) bounces, time-dependent survival integral; writes thermal_tables.json
            thermal_analysis.py     exponents against temperature, rate history, prior maps (thermal_results.json)
            rates.py                action -> Gamma/H^4, critical action, decay time
            falsification_map.py    the (eta, lambda) falsification diagram, physical g, reheating bound
            monte_carlo.py          prior-sensitivity Monte Carlo with convergence and sensitivity runs
            interior_evolution.py   energy budget, open-universe solution with reheating, AdS crunch
            scope.py                scope of the reheating bound: curvature parameter, envelope, e-folds, AdS, reservoirs
            coupling.py             requirement on a matter coupling (general, linear, quadratic)
            interior_sim.py         Lorentzian simulations of the daughter region (Steps 6-7): two fields, AdS reservoirs, hilltop flattening, repeated transitions, entropy
            oscillation.py          Option B cycles: periodicity, scaling, scalar Floquet multipliers, hysteresis, dissipation
            oscillation_validation.py   integrators, bounce continuity, W = oint p dV law, 100/1000-cycle phase diagram, second bounce prescription, vacuum transitions (A/B/C)
            oscillation_config.json, oscillation_cycles.csv   configuration and per-cycle results
            verify.py               26 executable checks
            make_figures.py, make_thermal_figures.py, make_sim_figures.py, make_osc_figures.py   Figs 1-12
            omml.py, render.py      LaTeX -> native Word equations; Word and LaTeX renderers
            build_paper.py, thermal_blocks.py, sim_blocks.py, osc_blocks.py, build_supporting.py   build the manuscript, supplement and cover letter from the stored results
*.json      stored results that the manuscript reads
figures/    Figs 1-12 (300 dpi)
manuscript/ manuscript (Word and LaTeX/PDF), supplement, cover letter
```

## Reproducing

```bash
pip install -r requirements.txt
cd code
python bounce_tables.py        # ~3 min
python survival.py             # seconds
python monte_carlo.py          # ~20 s (needs survival.json)
python falsification_map.py    # (needs survival.json)
python interior_evolution.py
python convergence.py          # ~4 min
python scope.py
python coupling.py
python thermal.py              # ~40 s (writes thermal_tables.json)
python thermal_analysis.py     # ~6 min (needs thermal_tables.json)
python interior_sim.py         # ~25 s (needs bounce_tables.json, interior_evolution.json)
python oscillation.py          # ~40 s
python oscillation_validation.py   # ~25 min
python verify.py               # ~25 s; 26 checks
python make_figures.py
python build_paper.py          # refuses to run unless verify.py passed
python build_supporting.py
```

Seeds are fixed (Monte Carlo: 20261030), so the numbers reproduce exactly.

## Revision history

* **v1.4.0** — Option B (same region oscillating): effective loop-quantum-cosmology bounce with the AdS turnaround, sinusoid analysis, periodic radiation-only cycles, scalar amplification (cosmological
  hysteresis) validated with three integrators and convergence tests, the law Δa_max = W/3, 100-1000-cycle phase diagram with the boundary ΥP ≈ ln μ, a second bounce prescription, vacuum transitions
  during the cycles (Models A/B/C), Section 5.6, Table 14, Figs 11-12, 2 new checks (26), new title.
* **v1.3.0** — Steps 6-7 simulations. Lorentzian homogeneous evolution of the daughter region with one and two fields, dissipation and gravity (constraint residual ≤ 1e-8), the AdS-compensated
  reservoir and its crunch, hilltop flattening, chain of repeated transitions with the energy sum rule, entropy accounting, criteria for a viable daughter universe (Table 13), Section 5.5, Fig. 10, 3 new checks (24 in total).
* **v1.2.0** — finite-temperature extension and a new title. Thermal mass with exact rescaling of the potential, O(3) action table, O(3)/O(4) crossover, survival with a
  time-dependent rate on the past light cone, thermal prior maps (decoupled vs equilibrated scalar), two-field energy-budget argument, daughter-region versus cyclic
  distinction, consistency conditions C7 and test O6, 3 new checks (21 in total), Figs 6–7 (old Figs 6–7 are now 8–9).
* **v1.1.0** — revised after review. Survival evaluated on the Lorentzian past light cone (the Euclidean four-sphere of v1.0.0 overstated V₄ by 445×); error budget and negative-mode analysis; prefactor
  as a systematic; scope of the reheating bound; coupling-independent matter-trigger requirement; consistency conditions vs observational tests.
* **v1.0.0** — first release.

## License

Code MIT (`LICENSE`); manuscript text and figures CC BY 4.0 (`manuscript/LICENSE`).
