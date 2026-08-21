from __future__ import annotations

import numpy as np
import pytest

from app.orchestration.entropy import (
    confidence_from_belief,
    information_gain,
    kl_divergence,
    max_entropy,
    normalized_entropy,
    shannon_entropy,
)


def test_uniform_belief_is_maximum_entropy():
    uniform = np.full(8, 1.0 / 8)
    assert shannon_entropy(uniform) == pytest.approx(3.0)
    assert normalized_entropy(uniform) == pytest.approx(1.0)
    assert max_entropy(8) == pytest.approx(3.0)


def test_peaked_belief_has_low_entropy():
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert shannon_entropy(peaked) < 0.3
    assert normalized_entropy(peaked) < 0.15


def test_information_gain_is_entropy_difference():
    before = shannon_entropy(np.full(8, 1.0 / 8))
    after = shannon_entropy(np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]))
    assert information_gain(before, after) == pytest.approx(before - after)
    assert information_gain(before, after) > 0


def test_information_gain_can_be_negative_when_ambiguity_grows():
    before = shannon_entropy(np.array([0.9, 0.05, 0.05]))
    after = shannon_entropy(np.array([0.4, 0.3, 0.3]))
    assert information_gain(before, after) < 0


def test_kl_divergence_is_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_confidence_requires_evidence_not_just_a_peak():
    thin = np.array([2.0, 1.0, 1.0, 1.0])
    thick = np.array([40.0, 1.0, 1.0, 1.0])
    assert confidence_from_belief(thin) < confidence_from_belief(thick)
    assert 0.0 <= confidence_from_belief(thin) <= 1.0
