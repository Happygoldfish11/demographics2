# BISG Estimator

A production-grade Python implementation of **Bayesian Improved First-name
Surname Geocoding (BIFSG)** — the state-of-the-art probabilistic method
for inferring race/ethnicity from a name and address — packaged as a
Streamlit web app, a clean Python API, and a comprehensive test suite.

> **Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before doing anything
> with this code.** BIFSG is a population-level statistical tool. Using it
> to make decisions about specific individuals is unsupported by the
> underlying science, ethically fraught, and in many U.S. contexts
> (employment, lending, housing) outright illegal.

---

## What it does

Given some subset of:

- **first name** (e.g. *Maria*)
- **surname** (e.g. *Garcia*)
- **address** (geocoded → ZCTA via the U.S. Census Geocoder API)
- **employer** (matched against published EEO-1 / diversity reports)

…it returns a probability distribution over the six U.S. Census
race/ethnicity categories (white, Black, API, AmInd/AlNat, two-or-more,
Hispanic) using the BIFSG formula:

```
P(r | s, f, g) ∝ P(r | s) · P(f | r) · P(g | r)
```

with an optional fourth factor `P(e | r)` from employer diversity data.
The math is implemented from scratch and **cross-validated bit-for-bit
against the canonical [`surgeo`](https://github.com/theonaunheim/surgeo)
reference library** (`atol = 0`, exact floating-point match).

## Reference data

| Component       | Source                                                      | Size       |
| --------------- | ----------------------------------------------------------- | ---------- |
| `P(r \| s)`     | U.S. Census Bureau — 2010 surname file                      | ~162,000 surnames |
| `P(f \| r)`     | Tzioumis (2018) "Demographic aspects of first names"        | ~4,250 first names |
| `P(g \| r)`     | U.S. Census Bureau — ZCTA × race                            | ~33,000 ZCTAs |
| `P(r)`          | U.S. Census Bureau — 2020 national race shares              | 6 categories |
| `P(r \| e)`     | Bundled demo employer table + user-supplied overrides       | 13 demo employers |

The Census-derived tables are loaded from the `surgeo` package (which we
treat purely as a static data source — none of its estimation code runs at
inference time). The employer table is editorial and intentionally small;
production users should provide their own.

## Install

```bash
git clone <this repo>
cd bisg-estimator
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

The app has four tabs:

1. **Estimate** — single-record entry form with a stage-by-stage
   visualization showing how each input updates the posterior
   (national → +surname → +first name → +geography → +employer).
2. **Batch** — upload a CSV with `first_name`, `surname`, `address`,
   `employer` columns; download enriched results as CSV/JSON.
3. **Methodology** — the math, derived inline, with citations.
4. **Limitations** — when this tool is and isn't appropriate.

## Use the Python API

```python
from bisg_estimator import estimate

result = estimate(
    first_name="Maria",
    surname="Garcia",
    address="100 Main St, Los Angeles, CA 90012",
    employer="Google",
)

print(result.final.as_dict())
# {'white': 0.108, 'black': 0.012, 'api': 0.044, 'native': 0.003,
#  'multiple': 0.018, 'hispanic': 0.815}

print(result.final.most_likely)        # ('hispanic', 0.815)
print(result.final.normalised_entropy) # 0.41 — moderate concentration
```

`EstimationResult` also exposes intermediate stages (`bifsg`, `bisg`,
`surname_only`, `national`), the geocoding outcome, the employer evidence,
and a `inputs_used` provenance log explaining what happened with each
input.

See [`examples/batch_processing.py`](examples/batch_processing.py) for a
batch example.

## Run the test suite

```bash
./run_tests.sh
# or:
python -m pytest tests/ -v
```

101 tests covering the Bayesian core, reference-data loading, geocoding
(with mocked HTTP), employer enrichment, the end-to-end pipeline, and
13 surgeo cross-validation cases requiring exact floating-point agreement.

## Project layout

```
bisg-estimator/
├── app.py                      # Streamlit UI
├── bisg_estimator/
│   ├── core.py                 # Bayesian math (BIFSG, BISG, update, temper)
│   ├── data.py                 # Reference-table loading + normalization
│   ├── geocoding.py            # Census Geocoder client + ZIP fallback
│   ├── employer.py             # Employer diversity-table lookup
│   └── pipeline.py             # End-to-end orchestration
├── tests/                      # 101 tests, including surgeo cross-validation
├── docs/
│   ├── METHODOLOGY.md          # Math, derivations, references
│   └── LIMITATIONS.md          # Ethics & appropriate use
├── examples/
│   └── batch_processing.py     # Programmatic batch example
├── requirements.txt
├── run_tests.sh
└── README.md
```

## References

- Elliott, M. N. et al. (2008). *A new method for estimating race/ethnicity
  and associated disparities where administrative records lack
  self-reported race/ethnicity*. Health Services Research, 43(5).
- Elliott, M. N. et al. (2009). *Using the Census Bureau's surname list
  to improve estimates of race/ethnicity and associated disparities*.
  Health Services & Outcomes Research Methodology, 9(2).
- Voicu, I. (2018). *Using First Name Information to Improve Race and
  Ethnicity Classification*. Statistics and Public Policy, 5(1).
- Word, D. L. et al. (2008). *Demographic Aspects of Surnames from
  Census 2000*. U.S. Census Bureau.
- Tzioumis, K. (2018). *Demographic aspects of first names*. Scientific
  Data, 5, 180025.
- CFPB (2014). *Using publicly available information to proxy for
  unidentified race and ethnicity: A methodology and assessment*.

## License

This is a demonstration project. The underlying Census data is public
domain; the Tzioumis first-name dataset is CC0; the `surgeo` package is
MIT-licensed. The original code in this repository is provided as-is for
research and educational use, subject to the warnings in `docs/LIMITATIONS.md`.
