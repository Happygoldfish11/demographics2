"""
Address → ZCTA geocoding via the U.S. Census Bureau Geocoder.

Public, no-API-key Census Geocoder endpoint:
    https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress

We hit this at runtime, parse the response, and pull out the ZCTA5 (ZIP
Code Tabulation Area), which is the geographic unit the BISG/BIFSG
reference tables are keyed on.

Resilience strategy:

    1. Try the Census Geocoder. It's authoritative and free, but can
       time out or reject malformed addresses.
    2. If that fails, fall back to extracting a 5-digit ZIP from the
       supplied address string. ZCTA ≈ ZIP for the vast majority of
       residential addresses.
    3. If even *that* fails, return ``None`` for the ZCTA — the pipeline
       will then either fall back to national race shares or skip the
       geographic component entirely (caller's choice).

The Census Geocoder can be slow (5–15 s per call). We cache successful
lookups per process to keep batch jobs reasonable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CENSUS_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
)
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_BENCHMARK = "Public_AR_Current"
DEFAULT_VINTAGE = "Census2020_Current"

# Match a 5-digit ZIP optionally followed by a 4-digit ZIP+4 extension.
# Anchored to a non-digit boundary so we don't grab the first 5 digits of
# a street number.
_ZIP_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")


@dataclass
class GeocodeResult:
    """Outcome of a geocoding attempt.

    Attributes
    ----------
    zcta : str | None
        Five-digit ZCTA / ZIP, or ``None`` if neither a Census match nor a
        ZIP extraction succeeded.
    matched_address : str | None
        The address as the Census matcher returned it (None on fallback).
    method : str
        One of ``"census"``, ``"zip_fallback"``, ``"none"``.
    state : str | None
        Two-letter state code, when available from the Census match.
    county : str | None
        County name, when available.
    notes : str
        Human-readable explanation of what happened (good for the UI).
    """

    zcta: Optional[str]
    matched_address: Optional[str] = None
    method: str = "none"
    state: Optional[str] = None
    county: Optional[str] = None
    notes: str = ""


def extract_zip(address: str) -> Optional[str]:
    """Pull the most plausible 5-digit ZIP out of an address string.

    US addresses put the ZIP at the end (e.g. "12345 Main St, Anytown, NY
    90210"), so we take the LAST 5-digit token bounded by non-digits, not the
    first. This avoids confusing a 5-digit street number with the ZIP.
    """
    if not address:
        return None
    matches = _ZIP_RE.findall(address)
    return matches[-1] if matches else None


def _query_census_geocoder(
    address: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> Optional[dict]:
    """Make the actual HTTP request. Returns the parsed JSON dict, or
    ``None`` on any failure (network, timeout, non-200, malformed JSON)."""
    params = {
        "address": address,
        "benchmark": DEFAULT_BENCHMARK,
        "vintage": DEFAULT_VINTAGE,
        "format": "json",
        "layers": "ZIP Code Tabulation Areas,Counties,States",
    }
    try:
        response = requests.get(CENSUS_GEOCODER_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Census Geocoder query failed: %s", exc)
        return None


def _parse_census_response(
    payload: dict, original_address: str
) -> Optional[GeocodeResult]:
    """Pull a usable ZCTA out of the geocoder's JSON response."""
    try:
        matches = payload["result"]["addressMatches"]
    except (KeyError, TypeError):
        return None
    if not matches:
        return None

    match = matches[0]
    matched_address = match.get("matchedAddress")
    geographies = match.get("geographies", {}) or {}

    # ZCTA layer — the actual key the Census API uses can vary slightly
    # by vintage; we accept any key whose name mentions ZIP/ZCTA.
    zcta = None
    for key, entries in geographies.items():
        if "zip" in key.lower() or "zcta" in key.lower():
            if entries:
                zcta = entries[0].get("ZCTA5") or entries[0].get("BASENAME")
                break

    state = None
    county = None
    for state_entry in geographies.get("States", []) or []:
        state = state_entry.get("STUSAB") or state_entry.get("BASENAME")
        break
    for county_entry in geographies.get("Counties", []) or []:
        county = county_entry.get("BASENAME")
        break

    if not zcta:
        return None

    # ZCTA from the Census API is already a 5-digit string, but normalise
    # defensively in case an older vintage returns ints.
    zcta = str(zcta).zfill(5)[:5]
    return GeocodeResult(
        zcta=zcta,
        matched_address=matched_address,
        method="census",
        state=state,
        county=county,
        notes=f"Resolved via Census Geocoder ({DEFAULT_VINTAGE}).",
    )


@lru_cache(maxsize=1024)
def geocode(
    address: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> GeocodeResult:
    """Resolve an address to a ZCTA.

    Tries the Census Geocoder first; on failure, falls back to extracting
    a ZIP from the raw address string. Always returns a :class:`GeocodeResult`
    — check ``result.zcta`` for a hit.
    """
    address = (address or "").strip()
    if not address:
        return GeocodeResult(
            zcta=None, method="none", notes="Empty address."
        )

    payload = _query_census_geocoder(address, timeout=timeout)
    if payload is not None:
        parsed = _parse_census_response(payload, address)
        if parsed is not None:
            return parsed

    fallback_zip = extract_zip(address)
    if fallback_zip:
        return GeocodeResult(
            zcta=fallback_zip,
            method="zip_fallback",
            notes=(
                "Census Geocoder did not return a match; "
                f"using ZIP {fallback_zip} parsed from the address as a "
                "ZCTA proxy. Accuracy may be reduced for ZIPs that span "
                "multiple ZCTAs or for non-residential ZIPs."
            ),
        )

    return GeocodeResult(
        zcta=None,
        method="none",
        notes=(
            "Could not resolve address to a Census tract or ZIP. "
            "The estimate will fall back to the national race distribution "
            "for the geographic component."
        ),
    )
