# Late-Time Vacuum Decay and Cosmological Reheating

Leon Sandler, Independent Researcher — sandler.leon@gmail.com — ORCID 0009-0007-4584-808X

Code, data and manuscript for *"Late-Time Vacuum Decay and Cosmological Reheating: Gravitational Tunneling, Stochastic Sensitivity, and
Falsification Boundaries,"* prepared for **The European Physical Journal C**.

## What the paper does

Can the decay of a metastable late-time vacuum seed a new cosmological cycle? The paper treats this as a calculation. A scalar potential with an
explicit metastable minimum is specified; the O(4) Coleman–De Luccia bounce is solved numerically in flat space and with Einstein gravity; the
action is converted to a nucleation rate per Hubble four-volume with the **measured dark-energy density** as the false-vacuum energy; and the
interior of a nucleated bubble is followed to see whether it can become a hot expanding universe.

## Results the code reproduces

| Result | Where |
|---|---|
| With all scales set by the dark energy the gravitational parameter is g = f²/M_Pl² ≈ 10⁻⁶⁰; gravitational corrections to the action are ~10⁻⁵⁹ and the Hawking–Moss regime is irrelevant | `falsification_map.py`, `bounce_tables.py` |
| The decay is a flat-space process controlled by one action B = b(η)/λ; nucleation within a Hubble four-volume needs B ≲ 287 | `rates.py`, `falsification_map.py` |
| The mean decay time runs from 10¹¹ yr to beyond 10¹⁰⁰ yr as B increases by ~205 (15% of the prior is excluded by the survival of the present vacuum, 80% never decays; the window of decay within 10²⁰ yr holds 0.56%) | `monte_carlo.py` |
| After a nucleation the interior is a curvature-dominated open universe with at most the dark-energy density: T_reh ≤ 2.2 meV, 4.6×10⁸ below nucleosynthesis | `interior_evolution.py` |
| A trigger by matter dilution would need a scalar–matter coupling ~10²⁸–10³⁰ times larger than allowed | `build_paper.py` (Section 6) |

The cyclic-reset hypothesis, in the form tested, is not supported; the paper states the falsification conditions explicitly.

## Contents

```
code/       cdl_lib.py              potential, flat-space and gravitational bounce solvers, actions, analytic references
            constants.py            dark-energy density, Hubble rate, reheating bound (Planck 2018)
            bounce_tables.py        b(eta), gravitational suppression grid, Hawking-Moss criterion
            rates.py                action -> Gamma/H^4, critical action, decay time
            falsification_map.py    the (eta, lambda) falsification diagram, physical g, reheating bound
            monte_carlo.py          prior-sensitivity Monte Carlo with convergence and sensitivity runs
            interior_evolution.py   energy budget, open-universe solution with reheating, AdS crunch
            verify.py               13 executable checks
            make_figures.py         Figs 1-5
            omml.py, render.py      LaTeX -> native Word equations; Word and LaTeX renderers
            build_paper.py, build_supporting.py   build the manuscript, supplement and cover letter from the stored results
*.json      stored results that the manuscript reads
figures/    Figs 1-5 (300 dpi)
manuscript/ manuscript (Word and LaTeX/PDF), supplement, cover letter
```

## Reproducing

```bash
pip install -r requirements.txt
cd code
python bounce_tables.py        # ~4 min
python monte_carlo.py          # ~1 min
python falsification_map.py
python interior_evolution.py
python verify.py               # ~3 min; 13 checks
python make_figures.py
python build_paper.py          # refuses to run unless verify.py passed
python build_supporting.py
```

Seeds are fixed (Monte Carlo: 20261030), so the numbers reproduce exactly.

## License

Code MIT (`LICENSE`); manuscript text and figures CC BY 4.0 (`manuscript/LICENSE`).
