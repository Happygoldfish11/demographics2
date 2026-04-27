"""Tests for the employer-evidence module."""

from __future__ import annotations

import numpy as np
import pytest

from bisg_estimator.core import RACE_CATEGORIES
from bisg_estimator.data import NATIONAL_SHARES, uniform_likelihood_vector
from bisg_estimator.employer import (
    _DEMO_EMPLOYER_TABLE,
    _normalize_employer_name,
    employer_likelihood,
    lookup_employer,
)


class TestEmployerNameNormalisation:
    def test_strips_legal_suffixes(self):
        assert _normalize_employer_name("Google LLC") == "GOOGLE"
        assert _normalize_employer_name("Apple Inc") == "APPLE"
        assert _normalize_employer_name("Apple Inc.") == "APPLE"
        assert _normalize_employer_name("Microsoft Corporation") == "MICROSOFT"

    def test_uppercases(self):
        assert _normalize_employer_name("google") == "GOOGLE"

    def test_handles_empty(self):
        assert _normalize_employer_name("") == ""
        assert _normalize_employer_name(None) == ""

    def test_collapses_whitespace(self):
        assert _normalize_employer_name("  J P  Morgan  ") == "J P MORGAN"


class TestLookupEmployer:
    def test_known_employer(self):
        result = lookup_employer("Google")
        assert result.found
        assert result.distribution is not None
        assert sum(result.distribution) == pytest.approx(1.0)

    def test_legal_suffix_handled(self):
        result = lookup_employer("Google LLC")
        assert result.found

    def test_loose_match(self):
        # 'Microsoft Corp' should match 'MICROSOFT' in the demo table.
        result = lookup_employer("Microsoft Corp")
        assert result.found

    def test_unknown_employer(self):
        result = lookup_employer("Acme Widgets and Things, Ltd.")
        assert not result.found
        assert result.distribution is None

    def test_empty_employer(self):
        result = lookup_employer("")
        assert not result.found
        result = lookup_employer(None)
        assert not result.found

    def test_demo_table_distributions_normalised(self):
        for name, dist in _DEMO_EMPLOYER_TABLE.items():
            assert set(dist.keys()) == set(RACE_CATEGORIES)
            total = sum(dist.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"{name} distribution does not sum to 1: {total}"
            )

    def test_user_supplied_table_overrides(self):
        custom = {
            "ACME": {
                "white": 0.1, "black": 0.1, "api": 0.1,
                "native": 0.1, "multiple": 0.1, "hispanic": 0.5,
            }
        }
        result = lookup_employer("Acme", table=custom)
        assert result.found
        # Hispanic should dominate.
        hispanic_idx = list(RACE_CATEGORIES).index("hispanic")
        assert result.distribution[hispanic_idx] > 0.4


class TestEmployerLikelihood:
    def test_unknown_employer_returns_uniform(self):
        likelihood, evidence = employer_likelihood("Unknown Employer", NATIONAL_SHARES)
        np.testing.assert_allclose(likelihood, uniform_likelihood_vector())
        assert not evidence.found

    def test_empty_returns_uniform(self):
        likelihood, evidence = employer_likelihood("", NATIONAL_SHARES)
        np.testing.assert_allclose(likelihood, uniform_likelihood_vector())
        assert not evidence.found

    def test_known_employer_inverts_through_base_rate(self):
        """For a known employer, the resulting likelihood should be
        proportional to P(r | e) / P(r) — and applying it as a Bayesian
        update to the national prior must reproduce P(r | e)."""
        from bisg_estimator import update, RaceProbabilities
        from bisg_estimator.data import national_likelihood_vector

        likelihood, evidence = employer_likelihood("Google", NATIONAL_SHARES)
        assert evidence.found

        national_prior = RaceProbabilities(national_likelihood_vector())
        posterior = update(national_prior, likelihood)
        # Posterior should match the original P(race | employer) we
        # started with (Bayes' rule turned upside-down and back).
        np.testing.assert_allclose(
            posterior.probabilities, evidence.distribution, atol=1e-9
        )
