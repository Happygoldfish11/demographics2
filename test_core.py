"""Tests for the core Bayesian math module."""

from __future__ import annotations

import numpy as np
import pytest

from bisg_estimator.core import (
    RACE_CATEGORIES,
    RaceProbabilities,
    bifsg,
    bisg,
    surname_only,
    temper,
    update,
)


class TestRaceProbabilities:
    def test_constructs_from_valid_vector(self):
        vec = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1])
        rp = RaceProbabilities(vec)
        assert rp.most_likely == ("white", 0.5)

    def test_rejects_unnormalised(self):
        vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="must sum to 1"):
            RaceProbabilities(vec)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            RaceProbabilities(np.array([0.5, 0.5]))

    def test_rejects_negative_values(self):
        vec = np.array([1.1, -0.1, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-negative"):
            RaceProbabilities(vec)

    def test_top_returns_sorted(self):
        rp = RaceProbabilities(np.array([0.1, 0.4, 0.05, 0.05, 0.1, 0.3]))
        top3 = rp.top(3)
        assert top3[0] == ("black", 0.4)
        assert top3[1][0] == "hispanic"
        assert all(top3[i][1] >= top3[i + 1][1] for i in range(2))

    def test_as_dict_aligned_with_categories(self):
        rp = RaceProbabilities(np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1]))
        d = rp.as_dict()
        assert list(d.keys()) == list(RACE_CATEGORIES)

    def test_entropy_zero_for_certain(self):
        rp = RaceProbabilities(np.array([1.0, 0, 0, 0, 0, 0]))
        assert rp.entropy_bits == pytest.approx(0.0)
        assert rp.normalised_entropy == pytest.approx(0.0)

    def test_entropy_max_for_uniform(self):
        rp = RaceProbabilities(np.full(6, 1 / 6))
        assert rp.entropy_bits == pytest.approx(np.log2(6))
        assert rp.normalised_entropy == pytest.approx(1.0)

    def test_herfindahl_one_for_certain(self):
        rp = RaceProbabilities(np.array([1.0, 0, 0, 0, 0, 0]))
        assert rp.herfindahl == pytest.approx(1.0)

    def test_herfindahl_minimum_for_uniform(self):
        rp = RaceProbabilities(np.full(6, 1 / 6))
        assert rp.herfindahl == pytest.approx(1 / 6)


class TestBISG:
    def test_uniform_likelihood_recovers_prior(self):
        prior = {"white": 0.5, "black": 0.2, "api": 0.1, "native": 0.05,
                 "multiple": 0.05, "hispanic": 0.1}
        uniform = {c: 1.0 for c in RACE_CATEGORIES}
        result = bisg(prior, uniform)
        for c, v in prior.items():
            assert result.as_dict()[c] == pytest.approx(v)

    def test_concentrates_when_likelihood_concentrates(self):
        # A surname mostly Hispanic + a strongly white-skewed area should
        # still pull the posterior toward white substantially.
        surname = {"white": 0.1, "black": 0.05, "api": 0.05, "native": 0.0,
                   "multiple": 0.05, "hispanic": 0.75}
        # A geography that only has white residents, basically.
        geo_likelihood = {"white": 1.0, "black": 0.001, "api": 0.001,
                          "native": 0.001, "multiple": 0.001, "hispanic": 0.001}
        result = bisg(surname, geo_likelihood)
        assert result.most_likely[0] == "white"

    def test_normalises_to_one(self):
        surname = {"white": 0.5, "black": 0.5, "api": 0.0,
                   "native": 0.0, "multiple": 0.0, "hispanic": 0.0}
        geo = {"white": 0.5, "black": 1.0, "api": 0.5,
               "native": 0.5, "multiple": 0.5, "hispanic": 0.5}
        result = bisg(surname, geo)
        assert sum(result.as_dict().values()) == pytest.approx(1.0)

    def test_zero_likelihood_eliminates_race(self):
        surname = {"white": 0.5, "black": 0.5, "api": 0.0,
                   "native": 0.0, "multiple": 0.0, "hispanic": 0.0}
        geo = {"white": 0.0, "black": 1.0, "api": 0.0,
               "native": 0.0, "multiple": 0.0, "hispanic": 0.0}
        result = bisg(surname, geo)
        assert result.as_dict()["black"] == pytest.approx(1.0)
        assert result.as_dict()["white"] == pytest.approx(0.0)

    def test_handles_zero_total(self):
        # If posterior is zero everywhere (degenerate inputs), we should
        # gracefully return a uniform distribution rather than nan.
        zero = {c: 0.0 for c in RACE_CATEGORIES}
        result = bisg(zero, zero)
        for c in RACE_CATEGORIES:
            assert result.as_dict()[c] == pytest.approx(1 / 6)


class TestBIFSG:
    def test_collapses_to_bisg_with_uniform_first_name(self):
        surname = {"white": 0.4, "black": 0.2, "api": 0.05,
                   "native": 0.05, "multiple": 0.1, "hispanic": 0.2}
        geo = {"white": 0.5, "black": 0.2, "api": 0.05,
               "native": 0.05, "multiple": 0.1, "hispanic": 0.1}
        uniform_first = {c: 1 / 6 for c in RACE_CATEGORIES}
        a = bisg(surname, geo)
        b = bifsg(surname, uniform_first, geo)
        for c in RACE_CATEGORIES:
            assert a.as_dict()[c] == pytest.approx(b.as_dict()[c])

    def test_first_name_shifts_estimate(self):
        # Surname is balanced; geo is balanced; first name is strongly
        # suggestive of a race -> posterior should follow the first name.
        surname = {c: 1 / 6 for c in RACE_CATEGORIES}
        geo = {c: 1 / 6 for c in RACE_CATEGORIES}
        first = {"white": 0.001, "black": 0.001, "api": 0.001,
                 "native": 0.001, "multiple": 0.001, "hispanic": 0.5}
        result = bifsg(surname, first, geo)
        assert result.most_likely[0] == "hispanic"

    def test_independence_of_argument_order(self):
        """All three arguments enter symmetrically as a product."""
        surname = {"white": 0.5, "black": 0.2, "api": 0.1, "native": 0.05,
                   "multiple": 0.05, "hispanic": 0.1}
        first = {"white": 0.3, "black": 0.3, "api": 0.1, "native": 0.05,
                 "multiple": 0.05, "hispanic": 0.2}
        geo = {"white": 0.6, "black": 0.1, "api": 0.05, "native": 0.05,
               "multiple": 0.1, "hispanic": 0.1}
        # Permuting which factor we call "surname" vs "first_name" vs "geo"
        # should not change the result, since they multiply.
        a = bifsg(surname, first, geo)
        b = bifsg(geo, first, surname)  # different "labels", same product
        c = bifsg(first, surname, geo)
        for cat in RACE_CATEGORIES:
            assert a.as_dict()[cat] == pytest.approx(b.as_dict()[cat])
            assert a.as_dict()[cat] == pytest.approx(c.as_dict()[cat])


class TestSurnameOnly:
    def test_returns_normalised_input(self):
        raw = {"white": 0.5, "black": 0.2, "api": 0.1, "native": 0.05,
               "multiple": 0.05, "hispanic": 0.1}
        result = surname_only(raw)
        for c, v in raw.items():
            assert result.as_dict()[c] == pytest.approx(v)


class TestUpdate:
    def test_update_with_uniform_likelihood_unchanged(self):
        prior = RaceProbabilities(np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1]))
        uniform = {c: 1.0 for c in RACE_CATEGORIES}
        result = update(prior, uniform)
        np.testing.assert_allclose(prior.probabilities, result.probabilities)


class TestTemper:
    def test_alpha_one_unchanged(self):
        rp = RaceProbabilities(np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1]))
        result = temper(rp, 1.0)
        np.testing.assert_allclose(rp.probabilities, result.probabilities)

    def test_alpha_zero_invalid(self):
        rp = RaceProbabilities(np.full(6, 1 / 6))
        with pytest.raises(ValueError, match="positive"):
            temper(rp, 0)

    def test_alpha_large_concentrates(self):
        # Tempering with high alpha should push mass to the largest term.
        rp = RaceProbabilities(np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05]))
        sharpened = temper(rp, 10)
        assert sharpened.most_likely[1] > rp.most_likely[1]


class TestVectorCoercion:
    def test_dict_and_sequence_equivalent(self):
        # Same inputs as dict and as ordered sequence should produce
        # identical posteriors.
        d = {"white": 0.4, "black": 0.2, "api": 0.1, "native": 0.05,
             "multiple": 0.05, "hispanic": 0.2}
        seq = [d[c] for c in RACE_CATEGORIES]
        a = bisg(d, d)
        b = bisg(seq, seq)
        np.testing.assert_allclose(a.probabilities, b.probabilities)
