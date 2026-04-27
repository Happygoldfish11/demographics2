"""
Data loading for BISG / BIFSG.

Backs the estimator with the three reference tables required by RAND's
methodology:

1. ``P(race | surname)``     — from the U.S. Census Bureau's 2010 surname
                                 file (Word et al. 2008).
2. ``P(first_name | race)``  — from Tzioumis (2018), "Demographic aspects
                                 of first names" (Scientific Data).
3. ``P(zcta | race)``        — derived from the 2010 Decennial Census
                                 ZCTA × race tabulations.

Rather than redistribute these large CSVs ourselves, we rely on the
``surgeo`` package (MIT-licensed, pip-installable) which bundles them in
the exact form RAND's methodology requires. We expose them through a
narrow, documented interface so the rest of this codebase has no
dependency on surgeo's internal API.

If ``surgeo`` is unavailable, :class:`ReferenceData` falls back to user-
supplied CSV paths, allowing the estimator to run from a clean Census
download. See ``docs/DATA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .core import RACE_CATEGORIES


# Default values used when a name / location is *not found* in the reference
# tables. Two policies, and we expose both:
#
#   "uniform"  — assume the missing entry tells us nothing. Equivalent to a
#                completely flat distribution. Keeps the estimator running
#                but discards all signal from that input.
#
#   "national" — fall back to U.S. national race shares (Census 2020
#                redistricting summary). This is what RAND recommends for
#                geocoding misses (Elliott et al. 2009 §2.2).
#
# 2020 P2 redistricting shares for the six BIFSG categories, normalised.
# Source: U.S. Census Bureau, 2020 Census Redistricting Data (P.L. 94-171).
_NATIONAL_SHARES_2020 = {
    "white":    0.5784,
    "black":    0.1213,
    "api":      0.0625,   # Asian + NHPI
    "native":   0.0070,
    "multiple": 0.1057,
    "hispanic": 0.1851,
}
# Renormalise (the Census categories above slightly exceed 1.0 due to NHPI
# being grouped with API but counted separately in some tables).
_total = sum(_NATIONAL_SHARES_2020.values())
NATIONAL_SHARES = {k: v / _total for k, v in _NATIONAL_SHARES_2020.items()}


@dataclass
class LookupResult:
    """Result of a name / location lookup against a reference table.

    Attributes
    ----------
    probabilities : np.ndarray | None
        Shape-(6,) vector aligned with ``RACE_CATEGORIES``, or ``None``
        if the key was not found.
    found : bool
        Whether the key was present in the reference table.
    key : str
        The normalised lookup key actually used.
    """

    probabilities: Optional[np.ndarray]
    found: bool
    key: str

    def filled(self, fallback: np.ndarray) -> np.ndarray:
        """Return the probabilities, substituting ``fallback`` if not found."""
        return self.probabilities if self.found else fallback


def _normalize_name(name: str) -> str:
    """Match the normalisation used by the Census surname file: uppercase,
    stripped, with non-alphabetic characters removed (handles 'O\\'Brien',
    'Smith-Jones', etc.).

    NB: this preserves cross-cultural names like 'Nguyen' and 'Garcia' as-is;
    we deliberately *don't* strip non-ASCII so accented surnames remain
    intact for upstream tokenisation."""
    if name is None:
        return ""
    cleaned = "".join(ch for ch in str(name) if ch.isalpha())
    return cleaned.upper().strip()


def _normalize_zcta(zcta: str) -> str:
    """Census ZCTAs are zero-padded 5-digit strings."""
    if zcta is None:
        return ""
    digits = "".join(ch for ch in str(zcta) if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(5)[:5]


class ReferenceData:
    """Wrapper around the three reference tables.

    Resolves them in this order:
      1. Explicit ``surname_csv`` / ``firstname_csv`` / ``zcta_csv`` paths.
      2. The bundled tables in the ``surgeo`` package.

    The resulting frames are indexed by (uppercased) name or 5-digit ZCTA
    and have one column per race in ``RACE_CATEGORIES``.
    """

    def __init__(
        self,
        surname_csv: Optional[Path] = None,
        firstname_csv: Optional[Path] = None,
        zcta_csv: Optional[Path] = None,
    ):
        self._surname_df = self._load_table(surname_csv, "surname")
        self._firstname_df = self._load_table(firstname_csv, "firstname")
        self._zcta_df = self._load_table(zcta_csv, "zcta")

    # ---- table-loading machinery ----------------------------------------

    @staticmethod
    def _load_table(csv_path: Optional[Path], kind: str) -> pd.DataFrame:
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            return ReferenceData._coerce_table(df, kind)
        # Fall back to surgeo's bundled tables.
        return ReferenceData._load_from_surgeo(kind)

    @staticmethod
    def _load_from_surgeo(kind: str) -> pd.DataFrame:
        try:
            from surgeo import BIFSGModel
        except ImportError as exc:  # pragma: no cover - exercised in docs
            raise RuntimeError(
                "surgeo is not installed and no CSV path was provided. "
                "Install with `pip install surgeo`, or pass explicit "
                "CSV paths to ReferenceData."
            ) from exc

        model = BIFSGModel()
        if kind == "surname":
            df = model._PROB_RACE_GIVEN_SURNAME.copy()
        elif kind == "firstname":
            df = model._PROB_FIRST_NAME_GIVEN_RACE.copy()
        elif kind == "zcta":
            df = model._PROB_ZCTA_GIVEN_RACE.copy()
        else:
            raise ValueError(f"unknown table kind: {kind}")
        return ReferenceData._coerce_table(df, kind)

    @staticmethod
    def _coerce_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Standardise columns / index regardless of where the table came
        from. Index is always the uppercased name (or ZCTA string); columns
        are exactly ``RACE_CATEGORIES`` in canonical order."""
        df = df.copy()
        # Some surgeo versions keep index as 'name' or 'zcta5'; that's fine.
        missing = [c for c in RACE_CATEGORIES if c not in df.columns]
        if missing:
            raise ValueError(f"reference table is missing columns: {missing}")
        df = df[list(RACE_CATEGORIES)].astype(np.float64)
        # Normalise the index.
        if kind == "zcta":
            df.index = [_normalize_zcta(str(i)) for i in df.index]
        else:
            df.index = [_normalize_name(str(i)) for i in df.index]
        # In case of duplicate keys after normalisation, keep first.
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ---- lookups ---------------------------------------------------------

    def surname(self, surname: str) -> LookupResult:
        """P(race | surname). Returns the row directly (already conditioned)."""
        key = _normalize_name(surname)
        if key in self._surname_df.index:
            return LookupResult(self._surname_df.loc[key].to_numpy(), True, key)
        return LookupResult(None, False, key)

    def first_name(self, first_name: str) -> LookupResult:
        """P(first_name | race). Returns the likelihood vector."""
        key = _normalize_name(first_name)
        if key in self._firstname_df.index:
            return LookupResult(self._firstname_df.loc[key].to_numpy(), True, key)
        return LookupResult(None, False, key)

    def zcta(self, zcta: str) -> LookupResult:
        """P(zcta | race). Returns the likelihood vector."""
        key = _normalize_zcta(zcta)
        if key in self._zcta_df.index:
            return LookupResult(self._zcta_df.loc[key].to_numpy(), True, key)
        return LookupResult(None, False, key)

    # ---- introspection ---------------------------------------------------

    @property
    def n_surnames(self) -> int:
        return len(self._surname_df)

    @property
    def n_first_names(self) -> int:
        return len(self._firstname_df)

    @property
    def n_zctas(self) -> int:
        return len(self._zcta_df)

    def coverage_summary(self) -> dict[str, int]:
        return {
            "surnames": self.n_surnames,
            "first_names": self.n_first_names,
            "zctas": self.n_zctas,
        }


@lru_cache(maxsize=1)
def default_reference_data() -> ReferenceData:
    """Process-wide singleton. Loading the reference tables takes a second
    or two; we don't want to do it more than once per Streamlit session."""
    return ReferenceData()


# ---------------------------------------------------------------------------
# Helpers used by the pipeline / Streamlit layer
# ---------------------------------------------------------------------------


def national_likelihood_vector() -> np.ndarray:
    """Return the U.S. national race-share vector (used as a uniform-ish
    fallback when no geographic information is available)."""
    return np.array([NATIONAL_SHARES[c] for c in RACE_CATEGORIES], dtype=np.float64)


def uniform_likelihood_vector() -> np.ndarray:
    """A flat 1/6 vector — used to *cancel out* a missing input."""
    n = len(RACE_CATEGORIES)
    return np.full(n, 1.0 / n, dtype=np.float64)
