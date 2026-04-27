"""
Core Bayesian math for BISG / BIFSG.

This module implements Bayesian Improved Surname Geocoding (BISG) and its
Bayesian Improved First-name Surname Geocoding (BIFSG) extension, both
developed and validated by RAND Corporation researchers (Elliott et al. 2008,
2009; Voicu 2018).

The math, in plain English
==========================

We want to estimate P(race | observed evidence) for a person, where the
"evidence" is some combination of: surname, first name, and geographic
location (typically a Census ZIP Code Tabulation Area, or ZCTA).

From Bayes' theorem, conditional independence of surname / first name /
geography given race (the standard BIFSG assumption — see Voicu 2018), and
the marginalisation identity
    P(x | r) = P(r | x) P(x) / P(r),
we get:

    BISG :   P(r | s, g) ∝ P(r | s) · P(g | r)

    BIFSG:   P(r | f, s, g) ∝ P(r | s) · P(f | r) · P(g | r)

In each case we then normalise across races so the probabilities sum to 1.

Why this form?
    - The Census publishes P(r | s) directly from the 2010 Census
      surname file (Word et al. 2008).
    - Census tract / ZCTA tables give us P(g | r) easily.
    - Tzioumis (2018) gives us P(f | r) for ~4,250 first names.
    - We never actually need P(r) on its own — it cancels in the
      normalisation. (You can prove this with one line of algebra.)

Sources
=======
Elliott, M.N. et al. (2008). "A new method for estimating race/ethnicity
    and associated disparities..." Health Services and Outcomes Research
    Methodology, 8, 36–55.
Elliott, M.N. et al. (2009). "Using the Census Bureau's surname list to
    improve estimates of race/ethnicity and associated disparities."
    Health Services and Outcomes Research Methodology, 9(2), 69–83.
Voicu, I. (2018). "Using First Name Information to Improve Race and
    Ethnicity Classification." Statistics and Public Policy, 5(1), 1–13.
Word, D.L. et al. (2008). "Demographic Aspects of Surnames from Census
    2000." U.S. Census Bureau technical paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# The six race / ethnicity categories used by the Census Bureau surname file
# (Word et al. 2008) and by all downstream RAND BISG/BIFSG implementations.
# Order is fixed so we can index into vectors / arrays consistently.
RACE_CATEGORIES: tuple[str, ...] = (
    "white",
    "black",
    "api",       # Asian or Pacific Islander
    "native",    # American Indian / Alaska Native
    "multiple",  # Two or more races
    "hispanic",  # Hispanic or Latino (any race)
)

# Human-readable labels for display.
RACE_LABELS: Mapping[str, str] = {
    "white": "White (non-Hispanic)",
    "black": "Black or African American (non-Hispanic)",
    "api": "Asian or Pacific Islander (non-Hispanic)",
    "native": "American Indian / Alaska Native (non-Hispanic)",
    "multiple": "Two or more races (non-Hispanic)",
    "hispanic": "Hispanic or Latino (any race)",
}


@dataclass(frozen=True)
class RaceProbabilities:
    """An immutable, normalised probability distribution over RACE_CATEGORIES.

    Acts as both a container and a small value-object: stores a vector aligned
    with ``RACE_CATEGORIES`` and exposes the most useful derived quantities
    (top race, entropy as a confidence proxy, Herfindahl index, etc.)."""

    probabilities: np.ndarray  # shape (6,), sums to 1.0
    categories: tuple[str, ...] = field(default=RACE_CATEGORIES)

    def __post_init__(self) -> None:
        if self.probabilities.shape != (len(self.categories),):
            raise ValueError(
                f"probabilities must have shape ({len(self.categories)},), "
                f"got {self.probabilities.shape}"
            )
        total = float(self.probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"probabilities must sum to 1.0 (within 1e-6); got {total:.6f}"
            )
        if (self.probabilities < -1e-9).any():
            raise ValueError("probabilities must be non-negative")

    # ---- accessors -------------------------------------------------------

    def as_dict(self) -> dict[str, float]:
        return {c: float(p) for c, p in zip(self.categories, self.probabilities)}

    def top(self, n: int = 1) -> list[tuple[str, float]]:
        order = np.argsort(-self.probabilities)
        return [
            (self.categories[i], float(self.probabilities[i]))
            for i in order[:n]
        ]

    @property
    def most_likely(self) -> tuple[str, float]:
        return self.top(1)[0]

    @property
    def entropy_bits(self) -> float:
        """Shannon entropy in bits. Higher = more uncertainty.

        Range: 0 (all mass on one race) to log2(6) ≈ 2.585 (uniform).
        """
        p = self.probabilities
        nz = p[p > 0]
        return float(-np.sum(nz * np.log2(nz)))

    @property
    def normalised_entropy(self) -> float:
        """Entropy scaled to [0, 1] for easy display."""
        return self.entropy_bits / np.log2(len(self.categories))

    @property
    def herfindahl(self) -> float:
        """Sum of squared probabilities. 1/H is the "effective number" of
        races. Useful as a concentration / confidence measure."""
        return float(np.sum(self.probabilities ** 2))


# ---------------------------------------------------------------------------
# Core Bayesian update routines
# ---------------------------------------------------------------------------


def _as_vector(d: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce a per-race input (dict or sequence) into a 6-vector ordered
    by RACE_CATEGORIES."""
    if isinstance(d, Mapping):
        return np.array([float(d[c]) for c in RACE_CATEGORIES], dtype=np.float64)
    arr = np.asarray(d, dtype=np.float64)
    if arr.shape != (len(RACE_CATEGORIES),):
        raise ValueError(
            f"expected length-{len(RACE_CATEGORIES)} vector, got shape {arr.shape}"
        )
    return arr


def _normalise(v: np.ndarray) -> np.ndarray:
    """Normalise a non-negative vector to sum to 1. If it sums to zero,
    fall back to a uniform distribution (the caller has no information)."""
    s = v.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(v, 1.0 / len(v))
    return v / s


def bisg(
    p_race_given_surname: Mapping[str, float] | Sequence[float],
    p_geo_given_race: Mapping[str, float] | Sequence[float],
) -> RaceProbabilities:
    """Standard BISG estimate.

    P(r | s, g) ∝ P(r | s) · P(g | r)

    Parameters
    ----------
    p_race_given_surname : dict or sequence
        For each race in ``RACE_CATEGORIES``, the conditional probability
        of that race given the surname. Should sum to ~1.
    p_geo_given_race : dict or sequence
        For each race, the conditional probability of the person's
        geographic unit (e.g. ZCTA) given that race. Need not sum to 1.

    Returns
    -------
    RaceProbabilities
    """
    rs = _as_vector(p_race_given_surname)
    gr = _as_vector(p_geo_given_race)
    posterior = _normalise(rs * gr)
    return RaceProbabilities(posterior)


def bifsg(
    p_race_given_surname: Mapping[str, float] | Sequence[float],
    p_first_name_given_race: Mapping[str, float] | Sequence[float],
    p_geo_given_race: Mapping[str, float] | Sequence[float],
) -> RaceProbabilities:
    """BIFSG estimate (Voicu 2018).

    P(r | f, s, g) ∝ P(r | s) · P(f | r) · P(g | r)

    Parameters mirror :func:`bisg`, with ``p_first_name_given_race``
    being the likelihood of the first name conditional on each race.
    """
    rs = _as_vector(p_race_given_surname)
    fr = _as_vector(p_first_name_given_race)
    gr = _as_vector(p_geo_given_race)
    posterior = _normalise(rs * fr * gr)
    return RaceProbabilities(posterior)


def surname_only(
    p_race_given_surname: Mapping[str, float] | Sequence[float],
) -> RaceProbabilities:
    """Trivial baseline: just the Census surname-only distribution."""
    return RaceProbabilities(_normalise(_as_vector(p_race_given_surname)))


# ---------------------------------------------------------------------------
# Optional likelihood adjustments (for advanced users / sensitivity analysis)
# ---------------------------------------------------------------------------


def update(
    prior: RaceProbabilities,
    likelihood: Mapping[str, float] | Sequence[float],
) -> RaceProbabilities:
    """Generic Bayesian update: posterior ∝ prior · likelihood, then normalise.

    Useful for layering in additional evidence (e.g. an EEO-1-derived
    employer race distribution) on top of an existing estimate. The caller
    is responsible for ensuring the likelihood is genuinely conditionally
    independent of the existing evidence given race — otherwise the
    posterior will be biased."""
    lik = _as_vector(likelihood)
    return RaceProbabilities(_normalise(prior.probabilities * lik))


def temper(
    probs: RaceProbabilities, alpha: float
) -> RaceProbabilities:
    """Sharpen (alpha > 1) or flatten (alpha < 1) a distribution.

    Pure utility — not part of standard BISG. Useful for sensitivity
    analyses ("what if the surname signal were half as informative?")."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return RaceProbabilities(_normalise(probs.probabilities ** alpha))
