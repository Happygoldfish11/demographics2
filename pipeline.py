"""
End-to-end estimation pipeline.

Stitches together: name parsing → reference-table lookups → geocoding →
employer enrichment → BIFSG → diagnostics.

The :class:`EstimationResult` carries everything we need for both the UI
display and downstream programmatic use: the final probabilities, every
intermediate distribution (so we can show *how* each input updated the
estimate), and a per-input ``inputs_used`` log explaining what we did
with each piece of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .core import (
    RACE_CATEGORIES,
    RaceProbabilities,
    bifsg,
    bisg,
    surname_only,
    update,
)
from .data import (
    NATIONAL_SHARES,
    ReferenceData,
    default_reference_data,
    national_likelihood_vector,
    uniform_likelihood_vector,
)
from .employer import EmployerEvidence, employer_likelihood
from .geocoding import GeocodeResult, geocode


@dataclass
class EstimationResult:
    """Everything you might want to know about a single estimation run."""

    final: RaceProbabilities                 # full BIFSG (+ employer if applicable)
    bifsg: RaceProbabilities                 # BIFSG without employer
    bisg: RaceProbabilities                  # BISG (no first name)
    surname_only: RaceProbabilities          # baseline: just surname
    national: RaceProbabilities              # baseline: just national shares

    geocode: GeocodeResult
    employer_evidence: EmployerEvidence

    # Provenance / diagnostics, one entry per input.
    inputs_used: list[dict] = field(default_factory=list)

    # The exact normalised inputs used in the BIFSG calculation, for the
    # waterfall / diagnostic UI.
    p_race_given_surname: np.ndarray = field(default_factory=lambda: np.array([]))
    p_first_name_given_race: np.ndarray = field(default_factory=lambda: np.array([]))
    p_geo_given_race: np.ndarray = field(default_factory=lambda: np.array([]))
    p_employer_likelihood: np.ndarray = field(default_factory=lambda: np.array([]))


def estimate(
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    address: Optional[str] = None,
    employer: Optional[str] = None,
    *,
    reference: Optional[ReferenceData] = None,
    skip_geocoding: bool = False,
    employer_table: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> EstimationResult:
    """Estimate race probabilities for a single individual.

    Parameters
    ----------
    first_name, surname : str, optional
        The person's first and last names. At minimum a surname is needed
        for a meaningful estimate; without one, the estimate falls back
        to the national distribution.
    address : str, optional
        A street address. We attempt to resolve it to a Census ZCTA via
        the Census Geocoder API; on failure we fall back to extracting
        a ZIP from the address string.
    employer : str, optional
        Employer name. Only used if a published diversity-report
        distribution is available (see :mod:`bisg_estimator.employer`).
    reference : ReferenceData, optional
        Override the reference tables. Uses the bundled defaults if None.
    skip_geocoding : bool
        If True, don't make an HTTP call — just try ZIP extraction from
        the supplied address. Useful for tests and offline batch jobs.
    employer_table : mapping, optional
        Override the built-in employer diversity table.

    Returns
    -------
    EstimationResult
    """
    ref = reference or default_reference_data()
    inputs_used: list[dict] = []
    national_vec = national_likelihood_vector()

    # --- Surname: P(race | surname) ------------------------------------
    surname_lookup = ref.surname(surname or "")
    if surname_lookup.found:
        p_race_given_surname = surname_lookup.probabilities
        inputs_used.append(
            {
                "input": "surname",
                "value": surname,
                "status": "found",
                "key": surname_lookup.key,
                "note": (
                    "Matched the Census 2010 surname file "
                    f"({ref.n_surnames:,} entries)."
                ),
            }
        )
    else:
        p_race_given_surname = national_vec
        inputs_used.append(
            {
                "input": "surname",
                "value": surname or "(not provided)",
                "status": "missing" if not surname else "not_found",
                "key": surname_lookup.key,
                "note": (
                    "Surname is not in the Census 2010 surname file "
                    "(it lists ~162,000 surnames with ≥100 occurrences). "
                    "Falling back to U.S. national race shares for this "
                    "component, which means the surname provides no signal."
                ) if surname else (
                    "No surname provided. Falling back to national shares; "
                    "the estimate will be very weakly informed."
                ),
            }
        )

    # --- First name: P(first_name | race) -----------------------------
    fn_lookup = ref.first_name(first_name or "")
    if fn_lookup.found:
        p_first_name_given_race = fn_lookup.probabilities
        inputs_used.append(
            {
                "input": "first_name",
                "value": first_name,
                "status": "found",
                "key": fn_lookup.key,
                "note": (
                    "Matched Tzioumis (2018) first-name table "
                    f"({ref.n_first_names:,} entries)."
                ),
            }
        )
    else:
        # A truly uninformative likelihood (uniform) leaves the BIFSG
        # estimate equal to the BISG estimate when the first name is
        # missing — exactly the right behaviour.
        p_first_name_given_race = uniform_likelihood_vector()
        inputs_used.append(
            {
                "input": "first_name",
                "value": first_name or "(not provided)",
                "status": "missing" if not first_name else "not_found",
                "key": fn_lookup.key,
                "note": (
                    "First name not in the Tzioumis (2018) ~4,250-name "
                    "table. The estimate falls back to BISG (surname + "
                    "geography) for this evidence."
                ) if first_name else (
                    "No first name provided; using BISG (surname + geo) only."
                ),
            }
        )

    # --- Geocoding: address → ZCTA, then P(zcta | race) ----------------
    if address and not skip_geocoding:
        gc_result = geocode(address)
    elif address and skip_geocoding:
        # Offline mode: just extract a ZIP.
        from .geocoding import extract_zip
        zip_only = extract_zip(address)
        gc_result = GeocodeResult(
            zcta=zip_only,
            method="zip_fallback" if zip_only else "none",
            notes=(
                "Skipped Census Geocoder; using ZIP from address."
                if zip_only else "No ZIP found in address."
            ),
        )
    else:
        gc_result = GeocodeResult(
            zcta=None, method="none", notes="No address provided."
        )

    if gc_result.zcta:
        zcta_lookup = ref.zcta(gc_result.zcta)
        if zcta_lookup.found:
            p_geo_given_race = zcta_lookup.probabilities
            inputs_used.append(
                {
                    "input": "geography",
                    "value": gc_result.matched_address or address,
                    "status": "found",
                    "key": gc_result.zcta,
                    "note": gc_result.notes,
                }
            )
        else:
            # Geocoding succeeded but the ZCTA isn't in our reference table
            # — uncommon but possible (e.g., a brand-new ZCTA).
            # Use a UNIFORM likelihood, not national shares: P(g|r) is a
            # likelihood in the BIFSG formula, so an uninformative geographic
            # signal should be flat across r (otherwise we double-count the
            # national prior, which is already implicit via P(r|s)).
            p_geo_given_race = uniform_likelihood_vector()
            inputs_used.append(
                {
                    "input": "geography",
                    "value": gc_result.matched_address or address,
                    "status": "zcta_not_in_table",
                    "key": gc_result.zcta,
                    "note": (
                        f"Resolved ZCTA {gc_result.zcta} but it is not in "
                        "the reference ZCTA × race table. Treating geography "
                        "as uninformative (uniform) for this estimate."
                    ),
                }
            )
    else:
        p_geo_given_race = uniform_likelihood_vector()
        inputs_used.append(
            {
                "input": "geography",
                "value": address or "(not provided)",
                "status": "missing" if not address else "geocode_failed",
                "key": "",
                "note": gc_result.notes or "No address provided.",
            }
        )

    # --- The four reference distributions for the UI / diagnostics -----
    surname_only_dist = surname_only(p_race_given_surname)
    bisg_dist = bisg(p_race_given_surname, p_geo_given_race)
    bifsg_dist = bifsg(
        p_race_given_surname, p_first_name_given_race, p_geo_given_race
    )

    national_dist = RaceProbabilities(national_vec)

    # --- Employer (optional) -------------------------------------------
    p_emp_lik, emp_evidence = employer_likelihood(
        employer or "", NATIONAL_SHARES, table=employer_table
    )
    if emp_evidence.found:
        final_dist = update(bifsg_dist, p_emp_lik)
        inputs_used.append(
            {
                "input": "employer",
                "value": employer,
                "status": "found",
                "key": "",
                "note": emp_evidence.source,
            }
        )
    else:
        final_dist = bifsg_dist
        inputs_used.append(
            {
                "input": "employer",
                "value": employer or "(not provided)",
                "status": "missing" if not employer else "not_found",
                "key": "",
                "note": emp_evidence.source,
            }
        )

    return EstimationResult(
        final=final_dist,
        bifsg=bifsg_dist,
        bisg=bisg_dist,
        surname_only=surname_only_dist,
        national=national_dist,
        geocode=gc_result,
        employer_evidence=emp_evidence,
        inputs_used=inputs_used,
        p_race_given_surname=p_race_given_surname,
        p_first_name_given_race=p_first_name_given_race,
        p_geo_given_race=p_geo_given_race,
        p_employer_likelihood=p_emp_lik,
    )


def estimate_batch(
    records: list[dict],
    *,
    reference: Optional[ReferenceData] = None,
    skip_geocoding: bool = False,
    employer_table: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> list[EstimationResult]:
    """Run :func:`estimate` over a list of dicts. Each dict may contain
    ``first_name``, ``surname``, ``address``, ``employer``."""
    ref = reference or default_reference_data()
    return [
        estimate(
            first_name=rec.get("first_name"),
            surname=rec.get("surname"),
            address=rec.get("address"),
            employer=rec.get("employer"),
            reference=ref,
            skip_geocoding=skip_geocoding,
            employer_table=employer_table,
        )
        for rec in records
    ]
