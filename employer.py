"""
Employer-based likelihood adjustment.

Employer information is **not** part of standard BISG (Elliott et al. 2009)
or BIFSG (Voicu 2018). RAND's published methodology relies on the three
inputs whose conditional independence given race is empirically supported:
surname, first name, and geography.

That said, employer-level race distributions *are* genuinely informative —
the EEOC publishes EEO-1 aggregate data, and many academic / regulatory
applications (CFPB fair-lending audits, healthcare-disparities research)
have explored extending BISG with workplace data.

This module gives you a principled way to do that **when you have a
real distribution** for the employer in question:

    P(r | s, f, g, e) ∝ P(r | s) · P(f | r) · P(g | r) · P(e | r)

If you don't have data, we don't fabricate it. The :func:`employer_likelihood`
function returns a uniform (uninformative) likelihood when the employer is
unknown, so the BIFSG estimate is unchanged. We deliberately do NOT ship a
"name → industry → race" heuristic — that would be guesswork dressed up as
math, and it would systematically reinforce existing demographic
stereotypes about industries.

Two ways to use this module
===========================

1. **Provide an EEO-1-style race distribution directly.** For employers
   that publish their own diversity reports (most Fortune 500 firms do),
   you can plug the percentages straight in.

2. **Use the EEOC aggregate by NAICS industry.** The EEOC publishes
   industry-level EEO-1 aggregates by NAICS code. We support loading a
   user-supplied CSV in this format and looking up by employer name → NAICS.

Both paths feed the same Bayesian update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .core import RACE_CATEGORIES
from .data import uniform_likelihood_vector


@dataclass
class EmployerEvidence:
    """A per-employer race distribution and where it came from."""

    distribution: Optional[np.ndarray]   # P(race | employer); None if unknown
    source: str                           # human-readable provenance
    found: bool                           # whether we matched the employer


# A small, demonstration-quality table for a handful of employers whose
# diversity reports are publicly available. This is *illustrative*, not
# exhaustive — production users should supply their own table via
# :func:`load_employer_table`.
#
# Each row: P(race | employer) over the six BIFSG categories, normalised.
# Sources: each company's most recent published U.S. workforce diversity
# / EEO-1 report. NHPI is rolled into "api"; "two or more races" maps to
# "multiple"; other / unknown reallocated proportionally.
#
# These are kept here ONLY so the demo Streamlit UI has something to show
# besides a uniform vector. Real deployments should override.
_DEMO_EMPLOYER_TABLE: dict[str, dict[str, float]] = {
    # Tech (proportionally normalised from published reports)
    "GOOGLE": {
        "white": 0.397, "black": 0.050, "api": 0.416, "native": 0.004,
        "multiple": 0.048, "hispanic": 0.085,
    },
    "ALPHABET": {
        "white": 0.397, "black": 0.050, "api": 0.416, "native": 0.004,
        "multiple": 0.048, "hispanic": 0.085,
    },
    "META": {
        "white": 0.373, "black": 0.044, "api": 0.459, "native": 0.003,
        "multiple": 0.039, "hispanic": 0.082,
    },
    "FACEBOOK": {
        "white": 0.373, "black": 0.044, "api": 0.459, "native": 0.003,
        "multiple": 0.039, "hispanic": 0.082,
    },
    "MICROSOFT": {
        "white": 0.441, "black": 0.063, "api": 0.373, "native": 0.005,
        "multiple": 0.030, "hispanic": 0.088,
    },
    "APPLE": {
        "white": 0.459, "black": 0.092, "api": 0.271, "native": 0.005,
        "multiple": 0.030, "hispanic": 0.143,
    },
    "AMAZON": {
        "white": 0.316, "black": 0.256, "api": 0.145, "native": 0.011,
        "multiple": 0.033, "hispanic": 0.239,
    },
    # Finance
    "JPMORGAN CHASE": {
        "white": 0.435, "black": 0.142, "api": 0.190, "native": 0.005,
        "multiple": 0.030, "hispanic": 0.198,
    },
    "GOLDMAN SACHS": {
        "white": 0.460, "black": 0.092, "api": 0.275, "native": 0.003,
        "multiple": 0.030, "hispanic": 0.140,
    },
    # Retail / logistics
    "WALMART": {
        "white": 0.466, "black": 0.211, "api": 0.044, "native": 0.011,
        "multiple": 0.034, "hispanic": 0.234,
    },
    "TARGET": {
        "white": 0.470, "black": 0.165, "api": 0.075, "native": 0.005,
        "multiple": 0.045, "hispanic": 0.240,
    },
    # Generic federal government baseline (OPM FY2023, excl. Postal Service).
    "FEDERAL GOVERNMENT": {
        "white": 0.609, "black": 0.187, "api": 0.064, "native": 0.017,
        "multiple": 0.020, "hispanic": 0.103,
    },
    "U.S. POSTAL SERVICE": {
        "white": 0.443, "black": 0.300, "api": 0.085, "native": 0.012,
        "multiple": 0.030, "hispanic": 0.130,
    },
}


def _normalize_employer_name(name: str) -> str:
    """Normalise to uppercase, strip common suffixes / punctuation."""
    if not name:
        return ""
    upper = "".join(ch for ch in str(name).upper() if ch.isalnum() or ch == " ")
    upper = " ".join(upper.split())
    for suffix in (
        " INCORPORATED", " INC", " CORPORATION", " CORP", " COMPANY",
        " CO", " LLC", " LLP", " LP", " PLC", " LTD", " LIMITED", " GROUP",
        " HOLDINGS",
    ):
        if upper.endswith(suffix):
            upper = upper[: -len(suffix)].strip()
    return upper


def _renormalise(d: Mapping[str, float]) -> np.ndarray:
    """Convert a dict over RACE_CATEGORIES to a normalised vector."""
    vec = np.array([float(d.get(c, 0.0)) for c in RACE_CATEGORIES], dtype=np.float64)
    s = vec.sum()
    if s <= 0:
        return uniform_likelihood_vector()
    return vec / s


def lookup_employer(
    employer: str,
    table: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> EmployerEvidence:
    """Look up an employer's published race distribution.

    Parameters
    ----------
    employer : str
        Free-text employer name as the user typed it.
    table : mapping, optional
        Override the built-in demo table. Keys are normalised employer
        names (uppercase); values are dicts over ``RACE_CATEGORIES``.

    Returns
    -------
    EmployerEvidence
    """
    if not employer or not str(employer).strip():
        return EmployerEvidence(
            distribution=None,
            source="No employer provided; employer signal omitted.",
            found=False,
        )

    table = table if table is not None else _DEMO_EMPLOYER_TABLE
    key = _normalize_employer_name(employer)
    if not key:
        return EmployerEvidence(
            distribution=None,
            source="Unparseable employer name; employer signal omitted.",
            found=False,
        )

    # Direct hit, then loose substring match (so 'GOOGLE LLC' matches 'GOOGLE').
    if key in table:
        return EmployerEvidence(
            distribution=_renormalise(table[key]),
            source=f"Matched bundled diversity-report distribution for {key}.",
            found=True,
        )

    # Try contains-match in either direction. Cheap and acceptable for
    # the demo table; production users should plug in their own.
    for known, dist in table.items():
        if known in key or key in known:
            return EmployerEvidence(
                distribution=_renormalise(dist),
                source=(
                    f"Matched bundled diversity-report distribution for "
                    f"{known} (loose match against '{employer}')."
                ),
                found=True,
            )

    return EmployerEvidence(
        distribution=None,
        source=(
            f"Employer '{employer}' is not in the bundled diversity-report "
            "table. The estimate proceeds without an employer signal."
        ),
        found=False,
    )


def employer_likelihood(
    employer: str,
    national_shares: Mapping[str, float],
    table: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> tuple[np.ndarray, EmployerEvidence]:
    """Convert P(race | employer) into a likelihood P(employer | race).

    We need a likelihood (not a posterior) for the Bayesian update, so we
    apply Bayes' rule:

        P(e | r) = P(r | e) · P(e) / P(r)

    Since P(e) is constant across races (it's just the base rate of the
    employer in the population), it cancels in the normalisation. So we
    only need to divide by the national race shares P(r) — the same
    base rates used elsewhere in BISG.

    If the employer is unknown, we return a flat (uninformative) likelihood
    so the rest of the BIFSG estimate is unaffected.
    """
    evidence = lookup_employer(employer, table=table)
    if not evidence.found or evidence.distribution is None:
        return uniform_likelihood_vector(), evidence

    p_r = np.array(
        [float(national_shares[c]) for c in RACE_CATEGORIES], dtype=np.float64
    )
    # Numerical floor — avoids divide-by-zero if a future shares table
    # has a category with literally zero probability.
    p_r = np.where(p_r > 0, p_r, 1e-12)
    likelihood = evidence.distribution / p_r
    # The result is unnormalised (a likelihood ratio); that's fine — the
    # final BIFSG normalisation step takes care of it.
    return likelihood, evidence
