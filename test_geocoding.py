"""Tests for the geocoding module.

The Census Geocoder API requires network access and is slow; these tests
exercise the fallback / parsing logic without hitting the network. The
real API is exercised in ``tests/test_pipeline.py`` only when the
``ENABLE_NETWORK_TESTS`` env var is set.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from bisg_estimator.geocoding import (
    GeocodeResult,
    _parse_census_response,
    extract_zip,
    geocode,
)


class TestExtractZip:
    def test_basic_zip(self):
        assert extract_zip("123 Main St, Anytown, NY 10001") == "10001"

    def test_zip_plus_four(self):
        assert extract_zip("123 Main St, Anytown, NY 10001-1234") == "10001"

    def test_no_zip(self):
        assert extract_zip("123 Main St, Anytown") is None

    def test_empty(self):
        assert extract_zip("") is None
        assert extract_zip(None) is None

    def test_doesnt_grab_street_number(self):
        # The street number 12345 is six digits-adjacent, but the regex
        # boundary should still work since it's anchored with non-digits.
        addr = "12345 Main St, Anytown, NY 90210"
        assert extract_zip(addr) == "90210"

    def test_doesnt_match_partial_digits(self):
        # 'NY100012' should not yield 10001 because the trailing digit
        # breaks the boundary.
        assert extract_zip("NY100012345") is None

    def test_last_zip_wins(self):
        # In US addresses the ZIP is the *last* numeric token, so when
        # multiple 5-digit numbers appear we take the last one. This makes
        # extract_zip robust to 5-digit street numbers in addresses like
        # "12345 Main St, Anytown, NY 90210".
        addr = "Mailbox at 90210, deliver to 10001"
        assert extract_zip(addr) == "10001"


class TestParseResponse:
    def _make_payload(self, zcta="10001", state="NY", county="New York"):
        return {
            "result": {
                "addressMatches": [
                    {
                        "matchedAddress": "350 5TH AVE, NEW YORK, NY, 10118",
                        "geographies": {
                            "ZIP Code Tabulation Areas": [{"ZCTA5": zcta}],
                            "States": [{"STUSAB": state, "BASENAME": "New York"}],
                            "Counties": [{"BASENAME": county}],
                        },
                    }
                ]
            }
        }

    def test_extracts_zcta(self):
        result = _parse_census_response(self._make_payload(), "anything")
        assert result is not None
        assert result.zcta == "10001"
        assert result.method == "census"
        assert result.state == "NY"
        assert result.county == "New York"

    def test_handles_no_match(self):
        result = _parse_census_response({"result": {"addressMatches": []}}, "x")
        assert result is None

    def test_handles_malformed(self):
        result = _parse_census_response({}, "x")
        assert result is None
        result = _parse_census_response({"result": None}, "x")
        assert result is None

    def test_pads_short_zcta(self):
        # If the API ever returns an integer-looking ZCTA, we should pad.
        payload = self._make_payload(zcta="123")
        result = _parse_census_response(payload, "x")
        assert result.zcta == "00123"


class TestGeocodeFallback:
    def setup_method(self):
        # Clear LRU cache so each test gets a fresh stack.
        geocode.cache_clear()

    def test_falls_back_to_zip_when_census_fails(self):
        # Force the Census call to fail and verify the ZIP fallback fires.
        with patch(
            "bisg_estimator.geocoding._query_census_geocoder", return_value=None
        ):
            result = geocode("123 Main St, Anytown, NY 10001")
        assert result.zcta == "10001"
        assert result.method == "zip_fallback"

    def test_returns_none_when_no_zip_and_no_census(self):
        with patch(
            "bisg_estimator.geocoding._query_census_geocoder", return_value=None
        ):
            result = geocode("Just a town with no ZIP")
        assert result.zcta is None
        assert result.method == "none"

    def test_uses_census_when_available(self):
        payload = {
            "result": {
                "addressMatches": [
                    {
                        "matchedAddress": "1 INFINITE LOOP, CUPERTINO, CA, 95014",
                        "geographies": {
                            "ZIP Code Tabulation Areas": [{"ZCTA5": "95014"}],
                            "States": [{"STUSAB": "CA"}],
                            "Counties": [{"BASENAME": "Santa Clara"}],
                        },
                    }
                ]
            }
        }
        with patch(
            "bisg_estimator.geocoding._query_census_geocoder", return_value=payload
        ):
            result = geocode("1 Infinite Loop, Cupertino, CA")
        assert result.zcta == "95014"
        assert result.method == "census"
        assert result.state == "CA"

    def test_empty_address(self):
        result = geocode("")
        assert result.zcta is None
        assert result.method == "none"

    def test_caches_results(self):
        # Two calls with the same address should hit the cache and only
        # invoke the underlying query once.
        call_count = [0]

        def fake_query(address, timeout=15):
            call_count[0] += 1
            return {
                "result": {
                    "addressMatches": [
                        {
                            "matchedAddress": address,
                            "geographies": {
                                "ZIP Code Tabulation Areas": [{"ZCTA5": "10001"}],
                                "States": [{"STUSAB": "NY"}],
                                "Counties": [{"BASENAME": "New York"}],
                            },
                        }
                    ]
                }
            }

        with patch(
            "bisg_estimator.geocoding._query_census_geocoder", side_effect=fake_query
        ):
            geocode.cache_clear()
            geocode("100 Main St, NY 10001")
            geocode("100 Main St, NY 10001")  # cached, no new call
        assert call_count[0] == 1


@pytest.mark.skipif(
    not os.getenv("ENABLE_NETWORK_TESTS"),
    reason="Set ENABLE_NETWORK_TESTS=1 to exercise the live Census Geocoder.",
)
class TestLiveGeocoder:
    """Hits the live Census Geocoder. Skipped by default."""

    def test_known_address(self):
        result = geocode("350 Fifth Avenue, New York, NY 10118")
        assert result.method in ("census", "zip_fallback")
        assert result.zcta is not None
