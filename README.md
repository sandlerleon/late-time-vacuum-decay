# Late-Time Vacuum Decay as a Cosmological Reset

Leon Sandler, Independent Researcher — sandler.leon@gmail.com — ORCID 0009-0007-4584-808X

Code, data and manuscript for *"Late-Time Vacuum Decay as a Cosmological Reset: Coleman–De Luccia Tunneling, Reheating Constraints, and
Consistency Boundaries,"* prepared for **The European Physical Journal C**.

## What the paper does

Can the decay of a metastable late-time vacuum reset the cosmological history? The paper treats this as a calculation. A scalar potential with an
explicit metastable minimum is specified; the O(4) Coleman–De Luccia bounce is solved numerically in flat space and with Einstein gravity; the
action is converted to a nucleation rate with the **measured dark-energy density** as the false-vacuum energy; survival of the present vacuum is
evaluated on the **Lorentzian past light cone** of the observed expansion history; and the interior of a nucleated bubble is followed to see whether it
can become a hot expanding universe. A bubble is not a cyclic cosmology, and the paper claims only to test this one reset mechanism.

## Results the code reproduces

| Result | Where |
|---|---|
| With all scales set by the dark energy g = f²/M_Pl² ≈ 10⁻⁶⁰; gravitational corrections are ~10⁻⁵⁹ and the Hawking–Moss regime is irrelevant | `falsification_map.py`, `bounce_tables.py` |
| Numerical error budget of the action (≤ 3×10⁻⁸ at the production settings) and the negative-mode structure (one negative eigenvalue, l = 1 zero modes, no node) | `convergence.py` |
| The decay is a flat-space process controlled by one action B = b(η)/λ; Γ/H⁴ = 1 at B_c ≈ 287 | `rates.py`, `falsification_map.py` |
| Past-light-cone four-volume V₄ = 0.059 H_Λ⁻⁴, 445× below the Euclidean four-sphere; 95% survival bound Γ/H_Λ⁴ < 51 | `survival.py` |
| Over the stated priors (a sample space, not probabilities): 14.8% above the survival bound, 0.77% decay within 10²⁰ yr, 80.4% show no decay within 10¹⁰⁰ yr | `monte_carlo.py` |
| Reheating T ≤ 2.2 meV (4.6×10⁸ below nucleosynthesis); the open interior stays curvature dominated for any sub-Planckian energy, so a hot epoch at 1 s needs a parent scale at least 41 orders of magnitude above M_Pl | `interior_evolution.py`, `scope.py` |
| A matter trigger needs Δln m ≳ 10⁻³ between the vacua: linear β ~ 10²⁶–10²⁹, or a quadratic scale of tens of meV | `coupling.py` |

The cyclic-reset hypothesis, in the form tested, is not supported; the paper separates internal consistency conditions from observational tests.

## Contents

```
code/       cdl_lib.py              potential, flat-space and gravitational bounce solvers, actions, analytic references
            constants.py            dark-energy density, Hubble rate, reheating bound (Planck 2018)
            bounce_tables.py        b(eta), gravitational suppression grid, Hawking-Moss criterion
            convergence.py          numerical error budget, gravity coefficient, negative-mode spectrum
            survival.py             past-light-cone four-volume and the survival bound
            rates.py                action -> Gamma/H^4, critical action, decay time
            falsification_map.py    the (eta, lambda) falsification diagram, physical g, reheating bound
            monte_carlo.py          prior-sensitivity Monte Carlo with convergence and sensitivity runs
            interior_evolution.py   energy budget, open-universe solution with reheating, AdS crunch
            scope.py                scope of the reheating bound: curvature parameter, envelope, e-folds, AdS, reservoirs
            coupling.py             requirement on a matter coupling (general, linear, quadratic)
            verify.py               18 executable checks
            make_figures.py         Figs 1-7
            omml.py, render.py      LaTeX -> native Word equations; Word and LaTeX renderers
            build_paper.py, build_supporting.py   build the manuscript, supplement and cover letter from the stored results
*.json      stored results that the manuscript reads
figures/    Figs 1-7 (300 dpi)
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
python verify.py               # ~15 s; 18 checks
python make_figures.py
python build_paper.py          # refuses to run unless verify.py passed
python build_supporting.py
```

Seeds are fixed (Monte Carlo: 20261030), so the numbers reproduce exactly.

## Revision history

* **v1.1.0** — revised after review. The survival of the present vacuum is now evaluated on the Lorentzian past light cone (the Euclidean
  four-sphere volume of v1.0.0 overstated V₄ by a factor 445; the excluded prior mass changes from 15.0% to 14.8%). Added: numerical error budget and
  negative-mode analysis, prefactor treated as a systematic uncertainty (spreads of 0, 3 and 6 decades), scope of the reheating bound (the energy
  bound is no longer presented as universal; geometric argument, anti-de Sitter case, extra reservoirs), a coupling-independent requirement on matter
  triggers (linear, quadratic, screened), consistency conditions separated from observational tests, a literature comparison, a retitled paper, and 5
  new executable checks (18 in total).
* **v1.0.0** — first release.

## License

Code MIT (`LICENSE`); manuscript text and figures CC BY 4.0 (`manuscript/LICENSE`).
