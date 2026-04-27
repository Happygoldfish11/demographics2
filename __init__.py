"""
bisg_estimator
==============

Bayesian Improved Surname Geocoding (BISG) and its first-name extension
(BIFSG), with a Streamlit UI on top.

Quick start
-----------

    from bisg_estimator.pipeline import estimate
    result = estimate(
        first_name="Maria",
        surname="Garcia",
        address="350 Fifth Ave, New York, NY 10118",
        employer="Google",
    )
    print(result.final.as_dict())
    print(result.final.most_likely)

See the README and ``docs/METHODOLOGY.md`` for the full statistical
treatment, and ``docs/LIMITATIONS.md`` for a frank discussion of what
this method can and cannot tell you.
"""

from .core import (
    RACE_CATEGORIES,
    RACE_LABELS,
    RaceProbabilities,
    bifsg,
    bisg,
    surname_only,
    update,
    temper,
)
from .data import (
    NATIONAL_SHARES,
    ReferenceData,
    default_reference_data,
    national_likelihood_vector,
    uniform_likelihood_vector,
)
from .employer import (
    EmployerEvidence,
    employer_likelihood,
    lookup_employer,
)
from .geocoding import GeocodeResult, geocode, extract_zip
from .pipeline import EstimationResult, estimate, estimate_batch

__version__ = "1.0.0"

__all__ = [
    "RACE_CATEGORIES",
    "RACE_LABELS",
    "RaceProbabilities",
    "bifsg",
    "bisg",
    "surname_only",
    "update",
    "temper",
    "NATIONAL_SHARES",
    "ReferenceData",
    "default_reference_data",
    "national_likelihood_vector",
    "uniform_likelihood_vector",
    "EmployerEvidence",
    "employer_likelihood",
    "lookup_employer",
    "GeocodeResult",
    "geocode",
    "extract_zip",
    "EstimationResult",
    "estimate",
    "estimate_batch",
    "__version__",
]
