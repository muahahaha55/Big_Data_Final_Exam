"""Unit tests for credit_risk.risk.capital — Basel II IRB retail capital.

These tests pin down the formula without starting Spark. The scalar functions
under test are formula-for-formula twins of the Spark Column versions, so a
regression in the maths shows up here in milliseconds rather than after a
pipeline run.

Reference values for Φ and Φ⁻¹ come from scipy; the Basel figures are
recomputed independently inside the test from the published formula.
"""

from __future__ import annotations

import math

import pytest

from credit_risk.risk.capital import (
    CapitalConfig,
    irb_k,
    norm_cdf,
    norm_ppf,
    other_retail_correlation,
    rule_of_thumb_cross_check,
)

CFG = CapitalConfig()


# ──────────────────────────────────────────────────────────────
# Φ — normal CDF
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNormCdf:
    def test_matches_erf_identity(self):
        """A&S 26.2.17 should stay within its 7.5e-8 absolute error bound."""
        for x in [-4.0, -2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 4.0, 6.0]:
            exact = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            assert abs(norm_cdf(x) - exact) < 7.5e-8, f"x={x}"

    def test_known_values(self):
        assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-7)
        assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-7)
        assert norm_cdf(3.090232306167814) == pytest.approx(0.999, abs=1e-7)

    def test_symmetry(self):
        for x in [0.3, 1.1, 2.7]:
            assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-9)

    def test_monotone(self):
        xs = [-3.0, -1.0, 0.0, 0.5, 2.0]
        vals = [norm_cdf(x) for x in xs]
        assert vals == sorted(vals)


# ──────────────────────────────────────────────────────────────
# Φ⁻¹ — inverse normal CDF
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNormPpf:
    def test_known_values(self):
        """Acklam's algorithm: max relative error 1.15e-9."""
        assert norm_ppf(0.999) == pytest.approx(3.090232306167814, rel=1.15e-9)
        assert norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1.15e-9)
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
        assert norm_ppf(0.025) == pytest.approx(-1.959963984540054, rel=1.15e-9)

    def test_matches_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        for p in [1e-6, 1e-4, 0.0003, 0.01, 0.02424, 0.02426, 0.1, 0.5,
                  0.9, 0.97574, 0.97576, 0.999, 0.9999]:
            assert norm_ppf(p) == pytest.approx(scipy_stats.norm.ppf(p), rel=1e-8), f"p={p}"

    def test_all_three_branches_are_continuous(self):
        """No jump where Acklam switches between its tail and central branches."""
        for edge in [0.02425, 1.0 - 0.02425]:
            lo = norm_ppf(edge - 1e-9)
            hi = norm_ppf(edge + 1e-9)
            assert abs(hi - lo) < 1e-6, f"discontinuity at {edge}"

    def test_round_trip_with_cdf(self):
        for p in [0.001, 0.01, 0.2, 0.5, 0.8, 0.99]:
            assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-7)

    def test_rejects_out_of_domain(self):
        for bad in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError):
                norm_ppf(bad)


# ──────────────────────────────────────────────────────────────
# Asset correlation R
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAssetCorrelation:
    def test_bounds(self):
        """Basel II §328: R runs from 0.16 at PD→0 down to 0.03 at PD→1."""
        assert other_retail_correlation(1e-9, CFG) == pytest.approx(0.16, abs=1e-6)
        assert other_retail_correlation(1.0, CFG) == pytest.approx(0.03, abs=1e-6)

    def test_decreasing_in_pd(self):
        vals = [other_retail_correlation(p, CFG) for p in [0.001, 0.01, 0.05, 0.2, 0.5]]
        assert vals == sorted(vals, reverse=True)

    def test_matches_published_formula(self):
        """Recompute §328 independently."""
        for p in [0.0047, 0.0478, 0.1382, 0.2569, 0.4549]:
            w = (1 - math.exp(-35 * p)) / (1 - math.exp(-35))
            expected = 0.03 * w + 0.16 * (1 - w)
            assert other_retail_correlation(p, CFG) == pytest.approx(expected, rel=1e-12)


# ──────────────────────────────────────────────────────────────
# Capital requirement K
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIrbK:
    def test_matches_published_formula_with_scipy(self):
        """K recomputed from Basel II §329 using scipy, for both retail classes."""
        scipy_stats = pytest.importorskip("scipy.stats")
        z = scipy_stats.norm.ppf(0.999)
        lgd = 0.45
        for p in [0.0047, 0.0478, 0.1382, 0.2569, 0.4549]:
            for r in [0.04, other_retail_correlation(p, CFG)]:
                arg = (scipy_stats.norm.ppf(p) + math.sqrt(r) * z) / math.sqrt(1 - r)
                expected = lgd * scipy_stats.norm.cdf(arg) - p * lgd
                assert irb_k(p, lgd, r, CFG) == pytest.approx(expected, abs=1e-7), f"p={p} r={r}"

    def test_increasing_in_pd(self):
        ks = [irb_k(p, 0.45, 0.04, CFG) for p in [0.001, 0.01, 0.05, 0.2, 0.45]]
        assert ks == sorted(ks)

    def test_proportional_to_lgd(self):
        """K is linear in LGD: doubling LGD doubles K."""
        k1 = irb_k(0.10, 0.45, 0.04, CFG)
        k2 = irb_k(0.10, 0.90, 0.04, CFG)
        assert k2 == pytest.approx(2 * k1, rel=1e-9)

    def test_never_negative(self):
        for p in [1e-9, 0.0003, 0.5, 0.9, 0.9999, 1.0]:
            assert irb_k(p, 0.45, 0.04, CFG) >= 0.0

    def test_pd_floor_is_applied(self):
        """Anything under the 0.03% floor is treated as the floor."""
        at_floor = irb_k(CFG.pd_floor, 0.45, 0.04, CFG)
        assert irb_k(1e-9, 0.45, 0.04, CFG) == pytest.approx(at_floor, rel=1e-12)
        assert irb_k(0.0, 0.45, 0.04, CFG) == pytest.approx(at_floor, rel=1e-12)

    def test_pd_cap_keeps_it_finite(self):
        """PD = 1 must not blow up through Φ⁻¹(1)."""
        k = irb_k(1.0, 0.45, 0.04, CFG)
        assert math.isfinite(k) and 0.0 <= k <= 0.45

    def test_concave_in_pd_at_fixed_correlation(self):
        """At fixed R, K is concave in PD (the Vasicek capital function).

        Note this does NOT carry over to the composite K(PD, R(PD)): once the
        §328 correlation is allowed to fall with PD, there is a convex stretch
        around PD ≈ 0.08–0.15 where R has flattened out near 0.03. So the sign
        of the error from averaging PD within a bucket is not guaranteed —
        capital has to be summed per obligor, which is what the pipeline does.
        """
        pds = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
        ks = [irb_k(p, 0.45, 0.04, CFG) for p in pds]
        slopes = [(ks[i + 1] - ks[i]) / (pds[i + 1] - pds[i]) for i in range(len(pds) - 1)]
        assert slopes == sorted(slopes, reverse=True), slopes

    def test_capital_to_el_ratio_strictly_decreasing(self):
        """Why no multiple of EL can stand in for capital.

        Capital / EL runs from ~11.8x at PD = 0.5% down to ~0.5x at PD = 45%.
        A single multiple (the old `UL = 3 x EL`) is only ever right for one
        slice of the book — here, obligors near PD ≈ 4%.
        """
        pds = [0.0047, 0.01, 0.0478, 0.1382, 0.2569, 0.4549]
        ratios = [
            irb_k(p, 0.45, other_retail_correlation(p, CFG), CFG) / (p * 0.45)
            for p in pds
        ]
        assert ratios == sorted(ratios, reverse=True), ratios
        assert ratios[0] > 10.0   # low PD: capital dwarfs EL
        assert ratios[-1] < 1.0   # high PD: EL dwarfs capital

    def test_zero_lgd_gives_zero_capital(self):
        assert irb_k(0.20, 0.0, 0.04, CFG) == pytest.approx(0.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────
# Rule-of-thumb cross-check block
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRuleOfThumbCrossCheck:
    def test_reproduces_the_legacy_multiples(self):
        """The old README numbers: 3 × EL and 4.5 × EL on EL = 950.72M."""
        out = rule_of_thumb_cross_check(950.72e6, 1_500e6, CFG)
        assert out["ul_million"] == pytest.approx(2852.16, abs=0.01)
        assert out["ec_million"] == pytest.approx(4278.24, abs=0.01)

    def test_reports_ratio_against_irb(self):
        out = rule_of_thumb_cross_check(1000e6, 2000e6, CFG)
        assert out["ul_over_irb_capital"] == pytest.approx(1.5)
        assert out["ec_over_irb_capital"] == pytest.approx(2.25)

    def test_is_labelled_as_an_assumption(self):
        out = rule_of_thumb_cross_check(1e6, 1e6, CFG)
        assert "not Basel" in out["note"]

    def test_handles_zero_irb_capital(self):
        out = rule_of_thumb_cross_check(1e6, 0.0, CFG)
        assert out["ul_over_irb_capital"] is None
        assert out["ec_over_irb_capital"] is None
