"""Unit tests for credit_risk.monitoring.drift module."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.monitoring.drift import (
    DriftSeverity,
    calculate_psi,
    classify_psi,
    detect_feature_drift,
)


@pytest.mark.unit
class TestCalculatePSI:
    def test_identical_distributions_zero_psi(self):
        """Identical samples should have PSI ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = calculate_psi(data, data.copy())
        assert psi < 0.01

    def test_shifted_distribution_high_psi(self):
        """Mean-shifted distribution should have significant PSI."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        current = rng.normal(2, 1, 5000)   # mean shifted by 2
        psi = calculate_psi(ref, current)
        assert psi > 0.25

    def test_handles_empty_input(self):
        """Empty arrays should return 0 (no info to compare)."""
        assert calculate_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert calculate_psi(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_handles_nan(self):
        """NaN values should be filtered out before computation."""
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        cur = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
        psi = calculate_psi(ref, cur)
        assert psi >= 0  # doesn't crash

    def test_psi_is_non_negative(self):
        """PSI is always ≥ 0 by definition."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            ref = rng.normal(0, 1, 500)
            cur = rng.normal(rng.normal(0, 0.5), 1, 500)
            psi = calculate_psi(ref, cur)
            assert psi >= 0


@pytest.mark.unit
class TestClassifyPSI:
    @pytest.mark.parametrize("psi,expected", [
        (0.01, DriftSeverity.NONE),
        (0.05, DriftSeverity.NONE),
        (0.09, DriftSeverity.NONE),
        (0.10, DriftSeverity.MODERATE),
        (0.15, DriftSeverity.MODERATE),
        (0.24, DriftSeverity.MODERATE),
        (0.25, DriftSeverity.SIGNIFICANT),
        (0.50, DriftSeverity.SIGNIFICANT),
        (1.00, DriftSeverity.SIGNIFICANT),
    ])
    def test_classification_thresholds(self, psi, expected):
        """PSI thresholds map correctly per industry standard."""
        assert classify_psi(psi) == expected

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        assert classify_psi(0.05, warn_threshold=0.02, alert_threshold=0.10) == DriftSeverity.MODERATE
        assert classify_psi(0.15, warn_threshold=0.02, alert_threshold=0.10) == DriftSeverity.SIGNIFICANT


@pytest.mark.unit
class TestDetectFeatureDrift:
    def test_returns_feature_drift_result(self):
        """detect_feature_drift returns a populated FeatureDriftResult."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        result = detect_feature_drift("test_feature", ref, cur)

        assert result.feature == "test_feature"
        assert isinstance(result.psi, float)
        assert isinstance(result.ks_pvalue, float)
        assert result.sample_size_reference == 1000
        assert result.sample_size_current == 1000

    def test_drifted_distribution_flagged(self):
        """Strongly drifted distributions are flagged moderate/significant."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(3, 1, 2000)  # extreme shift
        result = detect_feature_drift("shifted", ref, cur)

        assert result.severity in {DriftSeverity.MODERATE, DriftSeverity.SIGNIFICANT}
        assert result.psi > 0.10
        assert result.ks_pvalue < 0.05

    def test_means_computed(self):
        """Reference and current means are populated."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = detect_feature_drift("f", ref, cur)

        assert result.reference_mean == 3.0
        assert result.current_mean == 30.0
