"""End-to-end pipeline tests, including bit-for-bit cross-validation
against ``surgeo.BIFSGModel``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bisg_estimator import (
    RACE_CATEGORIES,
    estimate,
    estimate_batch,
)
from bisg_estimator.data import default_reference_data


@pytest.fixture(scope="module")
def ref():
    return default_reference_data()


class TestEstimate:
    def test_garcia_in_manhattan_is_hispanic(self, ref):
        result = estimate(
            first_name="Maria",
            surname="Garcia",
            address="100 Main St, New York, NY 10001",
            skip_geocoding=True,
            reference=ref,
        )
        assert result.final.most_likely[0] == "hispanic"
        assert result.final.most_likely[1] > 0.85

    def test_smith_in_white_area(self, ref):
        # 04001 (Acton, ME) is overwhelmingly non-Hispanic white.
        result = estimate(
            first_name="John",
            surname="Smith",
            address="100 Main St, Acton, ME 04001",
            skip_geocoding=True,
            reference=ref,
        )
        assert result.final.most_likely[0] == "white"

    def test_diagnostics_populated(self, ref):
        result = estimate(
            first_name="Maria",
            surname="Garcia",
            address="100 Main St, New York, NY 10001",
            skip_geocoding=True,
            reference=ref,
        )
        # Should have an entry per input.
        input_names = {entry["input"] for entry in result.inputs_used}
        assert {"surname", "first_name", "geography", "employer"} <= input_names

    def test_missing_inputs_dont_crash(self, ref):
        # Just a surname.
        result = estimate(surname="Smith", skip_geocoding=True, reference=ref)
        assert result.final.most_likely[0] in RACE_CATEGORIES

    def test_no_inputs_returns_uniform_ish(self, ref):
        result = estimate(skip_geocoding=True, reference=ref)
        # Should be close to uniform / national distribution.
        assert sum(result.final.as_dict().values()) == pytest.approx(1.0)
        # Confidence should be very low.
        assert (1 - result.final.normalised_entropy) < 0.5

    def test_unknown_surname_falls_back(self, ref):
        result = estimate(
            first_name="Maria",
            surname="Xqwertyzpdvfg",
            address="100 Main St, NY 10001",
            skip_geocoding=True,
            reference=ref,
        )
        # Should still produce a valid distribution.
        assert sum(result.final.as_dict().values()) == pytest.approx(1.0)
        # The "surname not found" entry should be in inputs_used.
        surname_entry = next(
            e for e in result.inputs_used if e["input"] == "surname"
        )
        assert surname_entry["status"] in ("not_found", "missing")

    def test_employer_increases_match_count_when_found(self, ref):
        result_no_emp = estimate(
            first_name="Maria",
            surname="Garcia",
            address="100 Main St, NY 10001",
            skip_geocoding=True,
            reference=ref,
        )
        result_emp = estimate(
            first_name="Maria",
            surname="Garcia",
            address="100 Main St, NY 10001",
            employer="Google",
            skip_geocoding=True,
            reference=ref,
        )
        emp_entry = next(
            e for e in result_emp.inputs_used if e["input"] == "employer"
        )
        assert emp_entry["status"] == "found"
        # The result should differ from the no-employer baseline.
        assert not np.allclose(
            result_no_emp.final.probabilities,
            result_emp.final.probabilities,
        )

    def test_unknown_employer_doesnt_change_result(self, ref):
        a = estimate(
            first_name="John",
            surname="Smith",
            address="100 Main St, NY 10001",
            skip_geocoding=True,
            reference=ref,
        )
        b = estimate(
            first_name="John",
            surname="Smith",
            address="100 Main St, NY 10001",
            employer="Some Random Unknown Company",
            skip_geocoding=True,
            reference=ref,
        )
        np.testing.assert_allclose(
            a.final.probabilities, b.final.probabilities
        )

    def test_result_is_valid_distribution(self, ref):
        result = estimate(
            first_name="Maria",
            surname="Garcia",
            address="100 Main St, NY 10001",
            employer="Google",
            skip_geocoding=True,
            reference=ref,
        )
        for stage in (
            result.final, result.bifsg, result.bisg,
            result.surname_only, result.national,
        ):
            assert sum(stage.as_dict().values()) == pytest.approx(1.0)
            assert all(p >= 0 for p in stage.as_dict().values())


class TestBatch:
    def test_batch_returns_one_result_per_record(self, ref):
        records = [
            {"first_name": "Maria", "surname": "Garcia",
             "address": "100 Main St, NY 10001"},
            {"first_name": "John", "surname": "Smith",
             "address": "100 Main St, ME 04001"},
            {"first_name": "Wei", "surname": "Chen",
             "address": "100 Mission, San Francisco, CA 94110"},
        ]
        results = estimate_batch(records, skip_geocoding=True, reference=ref)
        assert len(results) == 3
        assert results[0].final.most_likely[0] == "hispanic"
        assert results[2].final.most_likely[0] == "api"


class TestBIFSGMatchesSurgeo:
    """The crown jewel of our test suite. We must reproduce surgeo's
    BIFSG output bit-for-bit on a wide sample of real names and ZCTAs.
    Any deviation means we've corrupted the methodology."""

    @pytest.mark.parametrize(
        "first,last,zcta",
        [
            ("MARIA", "GARCIA", "10001"),
            ("JOHN", "SMITH", "94110"),
            ("WEI", "CHEN", "94110"),
            ("LATOYA", "WASHINGTON", "30303"),
            ("DAVID", "GOLDBERG", "11201"),
            ("CARLOS", "LOPEZ", "78201"),
            ("AISHA", "JOHNSON", "21217"),
            ("PATRICIA", "MILLER", "55101"),
            ("JOSE", "MARTINEZ", "85003"),
            ("EMILY", "JONES", "60601"),
            ("MICHAEL", "BROWN", "02115"),
            ("SOPHIA", "RODRIGUEZ", "33101"),
            ("HIROSHI", "KIM", "98109"),
            ("FATIMA", "ALI", "48201"),
            ("DEMETRIUS", "JACKSON", "63103"),
        ],
    )
    def test_matches_surgeo_bifsg(self, ref, first, last, zcta):
        from surgeo import BIFSGModel

        result = estimate(
            first_name=first,
            surname=last,
            address=f"100 Main St, {zcta}",
            skip_geocoding=True,
            reference=ref,
        )

        m = BIFSGModel()
        df = pd.DataFrame(
            [{"first_name": first, "surname": last, "zcta5": zcta}]
        )
        surgeo_out = m.get_probabilities(
            df["first_name"], df["surname"], df["zcta5"]
        )

        # If surgeo returned NaN, it means the surname/firstname/zcta
        # combination didn't resolve in surgeo. Skip — we can't compare.
        ours = np.array(
            [result.bifsg.as_dict()[c] for c in RACE_CATEGORIES]
        )
        theirs = np.array(
            [surgeo_out[c].iloc[0] for c in RACE_CATEGORIES]
        )
        if np.any(np.isnan(theirs)):
            pytest.skip(
                f"surgeo returned NaN for {first} {last} {zcta}; "
                "name not in its data"
            )
        # We require numerical equivalence to within float64 precision.
        np.testing.assert_allclose(ours, theirs, atol=1e-12)
