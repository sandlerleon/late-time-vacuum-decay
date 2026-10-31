# Matter-Dilution-Triggered Recollapse and Conditional Nonsingular Cycles in a Metastable Dark-Energy Model

Leon Sandler, Independent Researcher — sandler.leon@gmail.com — ORCID 0009-0007-4584-808X

Code, data and manuscript for the paper of the same title, prepared for **The European Physical Journal C** (Regular Article, Theoretical Physics).
Manuscript: `manuscript/EPJC_Dilution_Triggered_Cycles.{docx,tex,pdf}`, supplement `manuscript/EPJC_Dilution_Triggered_Cycles_Supplement.docx`,
cover letter `manuscript/EPJC_Cover_Letter.docx`. Archived on Zenodo: paper 10.5281/zenodo.23717616, code 10.5281/zenodo.23717618 (all versions).

## What the paper does

A scalar coupled to dark matter through the symmetron form A(φ) = 1 + φ²/2M² holds a vacuum with the observed dark-energy density while the dark-matter
density is above a critical value, and releases it by a classical (spinodal) roll into a negative vacuum once the expansion has diluted the matter.
While the field is held, the **homogeneous background** reproduces ΛCDM to numerical accuracy; perturbative and environmental deviations (structure
formation, voids) are not tested. After the release the expansion turns around. With an **assumed** effective bounce (loop-quantum-cosmology form at a
density far below the Planck density — a phenomenological prescription, not a derivation), thermal dissipation Υ = C_T T³ and dark matter that obeys its own
Boltzmann equation with a field-dependent mass, 100 consecutive cycles are integrated with no variable reset, and the repeating cycle is an attracting fixed
point of the bounce-to-bounce map.

The results are **conditional background solutions**. The cycle does not produce the observed primordial perturbations (n_s ≈ 3), the bounce is assumed,
and the dynamics are homogeneous.

## Results and the scripts that produce them

| Result | Paper | Scripts |
|---|---|---|
| Matter-held vacuum, exact relation V_c − V₀(v) = (A(v)−1)ρ_crit/2, release and turnaround; uncoupled control rolls before today for large μ | 2, 3.1 | `matter_coupling.py`, `dilution_background.py`, `turnaround_scan.py` |
| Spinodal release, coherence, domain-wall energy, required tilt | 2.2 | `vacuum_phase_map.py` |
| Cooling cannot replace dilution; interventions on the turnaround | 3.2, 3.3 | `dilution_thermal.py`, `causal_controls.py` |
| Cold bounces without dissipation; stability of repeated cycles (ΥP ≳ ln μ, ΔA_max = W/3) | 4.1, 4.2 | `cyclic_extension.py`, `oscillation.py`, `oscillation_validation.py` |
| Anisotropy and ekpyrotic tolerance; bounce validation (ρ_c, invariants, ghost fluid) | 4.3, 4.4 | `contraction_stability.py`, `bounce_validation.py` |
| Black holes through the bounce and the sensitivity of the percolation bound (overlap threshold, population, accretion, clustering) | 4.5 | `compact_relics.py`, `relic_sensitivity.py` |
| 100 continuous cycles without resets; 100 MeV run; prescribed-step comparison | 5.1 | `continuous_cycles.py`, `hot_cycles.py` |
| Attracting fixed point and closed-form held epoch | 5.2 | `cycle_map.py` |
| Numerical diagnostics (residual vs. projection) and an **independent constraint-preserving solver** | 6.1 | `numerics_admissibility.py`, `independent_solver.py` |
| Failure map; joint viability of the held branch | 6.2, 6.3 | `cycle_scan.py`, `viability_mc.py` |
| Observational consequences, perturbation spectrum | 7.1 | `predictions.py` |
| Executable checks V27–V33 (the manuscript is built only if they pass) | 6.4 | `verify.py` |

## Reproduction

Python 3.11 with NumPy, SciPy, Matplotlib and python-docx (`requirements.txt`); LaTeX (pdflatex) for the PDF. From `code/`:

```
python matter_coupling.py && python dilution_background.py && python turnaround_scan.py
python dilution_thermal.py && python causal_controls.py && python dilution_validation.py
python oscillation.py && python oscillation_validation.py && python cyclic_extension.py && python hot_cycles.py
python continuous_cycles.py calibrate && python continuous_cycles.py          # several hours (100 cycles)
python cycle_map.py && python numerics_admissibility.py && python independent_solver.py && python cycle_scan.py
python contraction_stability.py && python bounce_validation.py && python compact_relics.py && python relic_sensitivity.py
python predictions.py && python vacuum_phase_map.py && python viability_mc.py
python make_dil_figures.py && python make_osc_figures.py && python make_cyc_figures.py
python verify.py && python paper1_build.py && python paper1_supplement.py
cd ../manuscript && pdflatex EPJC_Dilution_Triggered_Cycles.tex && pdflatex EPJC_Dilution_Triggered_Cycles.tex
```

Every number in the manuscript is read from the JSON files these scripts write. `numerics_admissibility.py reuse` keeps the stored no-projection run,
which takes about an hour to fail.

## Provenance and the companion paper

This repository began as a broader manuscript (versions up to v1.8.2, Zenodo 10.5281/zenodo.23823727 / 23823733) that also contained an analysis of
late-time vacuum decay by tunneling and of density-dependent vacuum walls. Following a review, the work was split. **This repository now contains only
the EPJ C paper**; the combined manuscript and the scripts and figures used only by it were removed in v2.0.0 and remain available at tag `v1.8.2`. A few
JSON tables from that analysis (`bounce_tables.json`, `vacuum_wall.json`, `survival.json`, `thermal_tables.json`, `density_bounce.json`) are kept because
`viability_mc.py` and `verify.py` read them; they are not used by this paper's results. The vacuum-wall analysis continues in a separate repository,
`sandlerleon/matter-coupled-vacuum-bubbles`, for a companion manuscript in preparation for Physics of the Dark Universe; neither paper relies on the
results of the other. (The tag `v1.8.1` was created by mistake and is identical to `v1.8.0`.)

## License

MIT (code and data); the manuscript text is © the author.
