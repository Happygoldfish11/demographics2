# Methodology

## What this estimator does

Given some combination of a person's **first name**, **surname**, **address**, and (optionally) **employer**, this tool produces a probability distribution over six U.S. Census race / ethnicity categories:

| Key        | Label                                                     |
|------------|-----------------------------------------------------------|
| `white`    | White (non-Hispanic)                                       |
| `black`    | Black or African American (non-Hispanic)                   |
| `api`      | Asian or Pacific Islander (non-Hispanic)                   |
| `native`   | American Indian / Alaska Native (non-Hispanic)             |
| `multiple` | Two or more races (non-Hispanic)                           |
| `hispanic` | Hispanic or Latino (any race)                              |

The probabilities sum to 1. The output is **not a classification** — it is an estimate of how likely each category is given the evidence.

## The math

Let `r` denote race, `s` surname, `f` first name, `g` geography (a Census ZIP Code Tabulation Area, or ZCTA), and `e` employer.

We assume conditional independence of each evidence component given race:

```
P(s, f, g, e | r) = P(s | r) · P(f | r) · P(g | r) · P(e | r)
```

Applying Bayes' theorem and the marginalisation identity `P(x | r) = P(r | x) P(x) / P(r)`, the four-component posterior is:

```
                    P(r | s) · P(f | r) · P(g | r) · P(e | r)
P(r | s,f,g,e) = ────────────────────────────────────────────────
                  Σᵣ P(r | s) · P(f | r) · P(g | r) · P(e | r)
```

(Constant factors from `P(s) / P(r)` and similar cancel in the normalisation.) Using the surname *posterior* `P(r | s)` rather than the surname likelihood is convenient because the Census Bureau publishes that quantity directly — we don't have to invert through `P(r)` ourselves.

When fewer inputs are available, the unused term is replaced by a uniform `1/k` likelihood, which leaves the remaining estimate unchanged. Concretely:

- **Surname only**: `P(r | s)` taken directly from the Census surname file.
- **BISG**: `P(r | s, g) ∝ P(r | s) · P(g | r)` — the original Elliott et al. (2009) estimator.
- **BIFSG**: `P(r | s, f, g) ∝ P(r | s) · P(f | r) · P(g | r)` — Voicu (2018).
- **BIFSG + employer**: same, multiplied by `P(e | r) = P(r | e) · P(e) / P(r)`. The `P(e)` term cancels in the normalisation, so we only need to divide by national race shares.

The implementation lives in `bisg_estimator/core.py` and is ~30 lines of NumPy.

## Reference data

| Component             | Source                                                                 | Coverage         |
|-----------------------|------------------------------------------------------------------------|------------------|
| `P(r | s)`            | U.S. Census Bureau, 2010 Surname File (Word et al. 2008)               | ~162,000 surnames |
| `P(f | r)`            | Tzioumis (2018), "Demographic aspects of first names", *Sci. Data*     | ~4,250 first names |
| `P(g | r)`            | Census 2010 ZCTA × race tabulations                                     | ~33,000 ZCTAs    |
| Address → ZCTA        | U.S. Census Bureau Geocoder API (`geocoding.geo.census.gov`)            | All U.S. addresses |
| `P(r | e)` (optional) | Public corporate diversity reports, EEOC EEO-1 industry aggregates      | Bundled demo: 13 employers |

Data is loaded through the [`surgeo`](https://pypi.org/project/surgeo/) package, which packages and ships the Census tables in their canonical form. The Bayesian math, geocoding layer, employer extension, and UI are all original to this codebase.

## Validation

The BIFSG output of this implementation matches `surgeo.BIFSGModel` **bit-for-bit** (max absolute difference `< 1e-15` on randomised inputs). See `tests/test_pipeline.py::test_bifsg_matches_surgeo`.

## What the "confidence" number means

We report `1 − H(p) / log₂(k)` where `H(p)` is the Shannon entropy of the posterior in bits and `k = 6` is the number of categories. This is **not** a frequentist confidence interval — it's a single-number summary of how peaked the distribution is.

- `confidence ≈ 1.0` → the estimator is putting almost all its mass on one race.
- `confidence ≈ 0.0` → the estimator returns a near-uniform distribution; treat it as effectively no information.

For applications that need calibrated uncertainty (e.g., legal proceedings, public-health analyses), use the full distribution, not just the top label.

## What the "most-likely" race means

It is **the mode of the posterior**. Reporting `most_likely` for a single individual is generally a bad idea: BIFSG is designed for *aggregate* analyses, where individual misclassifications average out. See `LIMITATIONS.md`.

## References

- Elliott, M.N., Fremont, A., Morrison, P.A., Pantoja, P., Lurie, N. (2008). "A new method for estimating race/ethnicity and associated disparities where administrative records lack self-reported race/ethnicity." *Health Services and Outcomes Research Methodology*, 8, 36–55.
- Elliott, M.N., Morrison, P.A., Fremont, A., McCaffrey, D.F., Pantoja, P., Lurie, N. (2009). "Using the Census Bureau's surname list to improve estimates of race/ethnicity and associated disparities." *Health Services and Outcomes Research Methodology*, 9(2), 69–83.
- Voicu, I. (2018). "Using First Name Information to Improve Race and Ethnicity Classification." *Statistics and Public Policy*, 5(1), 1–13.
- Word, D.L., Coleman, C.D., Nunziata, R., Kominski, R. (2008). "Demographic Aspects of Surnames from Census 2000." U.S. Census Bureau technical paper.
- Tzioumis, K. (2018). "Demographic aspects of first names." *Scientific Data*, 5(180025).
