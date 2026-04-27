"""Tests for the data-loading layer."""

from __future__ import annotations

import numpy as np
import pytest

from bisg_estimator.data import (
    NATIONAL_SHARES,
    ReferenceData,
    _normalize_name,
    _normalize_zcta,
    default_reference_data,
    national_likelihood_vector,
    uniform_likelihood_vector,
)
from bisg_estimator.core import RACE_CATEGORIES


class TestNameNormalisation:
    def test_uppercase(self):
        assert _normalize_name("smith") == "SMITH"

    def test_strips_punctuation(self):
        assert _normalize_name("O'Brien") == "OBRIEN"
        assert _normalize_name("Smith-Jones") == "SMITHJONES"

    def test_strips_whitespace(self):
        assert _normalize_name("  Smith  ") == "SMITH"

    def test_handles_none_and_empty(self):
        assert _normalize_name(None) == ""
        assert _normalize_name("") == ""


class TestZCTANormalisation:
    def test_pads_short_zips(self):
        # Census Geocoder occasionally returns zero-stripped ZCTAs; we
        # need to re-pad to 5 digits.
        assert _normalize_zcta("123") == "00123"
        assert _normalize_zcta("1") == "00001"

    def test_truncates_zip_plus_4(self):
        assert _normalize_zcta("90210-1234") == "90210"

    def test_strips_non_digits(self):
        assert _normalize_zcta("abc 90210") == "90210"

    def test_empty(self):
        assert _normalize_zcta(None) == ""
        assert _normalize_zcta("") == ""
        assert _normalize_zcta("abc") == ""


class TestNationalShares:
    def test_sums_to_one(self):
        assert sum(NATIONAL_SHARES.values()) == pytest.approx(1.0)

    def test_keys_match_categories(self):
        assert set(NATIONAL_SHARES.keys()) == set(RACE_CATEGORIES)


class TestVectors:
    def test_national_vector_aligned(self):
        vec = national_likelihood_vector()
        assert vec.shape == (6,)
        assert vec.sum() == pytest.approx(1.0)
        # The first element should be the white share.
        assert vec[0] == NATIONAL_SHARES["white"]

    def test_uniform_vector(self):
        vec = uniform_likelihood_vector()
        assert vec.shape == (6,)
        assert all(v == pytest.approx(1 / 6) for v in vec)


class TestReferenceData:
    @pytest.fixture(scope="class")
    def ref(self) -> ReferenceData:
        return default_reference_data()

    def test_default_loads(self, ref: ReferenceData):
        cov = ref.coverage_summary()
        assert cov["surnames"] > 100_000  # Census 2010 has ~162k
        assert cov["first_names"] > 4_000
        assert cov["zctas"] > 30_000

    def test_default_singleton(self):
        # @lru_cache should make this a singleton.
        a = default_reference_data()
        b = default_reference_data()
        assert a is b

    def test_known_surname_lookup(self, ref: ReferenceData):
        result = ref.surname("Smith")
        assert result.found
        assert result.key == "SMITH"
        # Smith is heavily white.
        white_idx = list(RACE_CATEGORIES).index("white")
        assert result.probabilities[white_idx] > 0.5
        # Probability vector sums to ~1 for surname (it's already a posterior).
        assert result.probabilities.sum() == pytest.approx(1.0, abs=0.01)

    def test_garcia_is_hispanic(self, ref: ReferenceData):
        result = ref.surname("Garcia")
        assert result.found
        hispanic_idx = list(RACE_CATEGORIES).index("hispanic")
        assert result.probabilities[hispanic_idx] > 0.85

    def test_washington_is_black(self, ref: ReferenceData):
        result = ref.surname("Washington")
        assert result.found
        # The Census 2010 file places Washington at >85% Black.
        black_idx = list(RACE_CATEGORIES).index("black")
        assert result.probabilities[black_idx] > 0.85

    def test_nguyen_is_api(self, ref: ReferenceData):
        result = ref.surname("Nguyen")
        assert result.found
        api_idx = list(RACE_CATEGORIES).index("api")
        assert result.probabilities[api_idx] > 0.85

    def test_unknown_surname(self, ref: ReferenceData):
        result = ref.surname("Xyzzyqwertytest")
        assert not result.found
        assert result.probabilities is None

    def test_first_name_lookup(self, ref: ReferenceData):
        result = ref.first_name("Maria")
        assert result.found
        assert result.key == "MARIA"

    def test_unknown_first_name(self, ref: ReferenceData):
        result = ref.first_name("Xqwertyzpdvfg")
        assert not result.found

    def test_zcta_lookup(self, ref: ReferenceData):
        # 10001 is Manhattan and definitely in the table.
        result = ref.zcta("10001")
        assert result.found

    def test_zcta_lookup_pads_short(self, ref: ReferenceData):
        # 0123 should be padded to 00123, which is a real Puerto Rico ZCTA prefix.
        # We don't know if that exact ZCTA exists, but the normalisation
        # should at least produce the right key.
        result = ref.zcta("123")
        assert result.key == "00123"

    def test_unknown_zcta(self, ref: ReferenceData):
        result = ref.zcta("99999")
        # 99999 isn't a real ZCTA.
        assert not result.found
