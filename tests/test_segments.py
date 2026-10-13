"""Unit tests for the P0.4 segment validation logic — no Spark, no pandas.

The numbers this module produces will be quoted per customer group, so the
estimators are checked against independent implementations rather than against
themselves:

    - ROC-AUC vs `scipy.stats.mannwhitneyu` (U/(n1·n2)), including ties
    - KS vs `scipy.stats.ks_2samp`, the definition `evaluate_predictions` uses
    - the Hanley-McNeil standard error vs a seeded bootstrap

The rest covers the parts that decide whether a segment claim is honest:
data-derived quantile edges, suppression of under-supported groups, the
noise bound on a dimension's spread, and the cross-check that refuses to
publish segments when the pooled AUC disagrees structurally with T1.4.

Imports `credit_risk.segments` rather than `pipelines.segment_pipeline`: the
pipeline needs pandas, the study module needs nothing.
"""

from __future__ import annotations

import math
import random

import pytest

from credit_risk.segments import (
    AUC_HARD_TOLERANCE,
    DEFAULT_MIN_POSITIVES,
    DEFAULT_MIN_ROWS,
    KIND_CATEGORICAL,
    KIND_QUANTILE,
    MISSING_LABEL,
    NOTE,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    TRANSFORMS,
    SegmentSpec,
    auc_standard_error,
    bin_bounds,
    bin_label,
    build_dimension,
    build_segment_report,
    build_setup,
    build_specs,
    compare_overall_auc,
    config_fingerprint,
    format_summary,
    group_metrics,
    ks_statistic,
    overall_metrics,
    quantile_edges,
    roc_auc,
    source_columns,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _cfg() -> dict:
    return {
        "data": {"target_column": "TARGET", "id_column": "SK_ID_CURR"},
        "segments": {
            "min_rows": 1000,
            "min_positives": 50,
            "n_bins": 5,
            "dimensions": [
                {"name": "contract_type", "column": "NAME_CONTRACT_TYPE",
                 "kind": "categorical"},
                {"name": "gender", "column": "CODE_GENDER",
                 "kind": "categorical", "sensitive": True},
                {"name": "age_group", "column": "DAYS_BIRTH",
                 "kind": "quantile", "transform": "days_to_years"},
                {"name": "income_group", "column": "AMT_INCOME_TOTAL",
                 "kind": "quantile"},
            ],
        },
    }


def _separable(n: int = 2000, seed: int = 7, strength: float = 1.0):
    """(target, score) pairs where score carries real but imperfect signal."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        target = 1 if rng.random() < 0.2 else 0
        base = rng.random()
        score = min(1.0, max(0.0, base + strength * 0.25 * target))
        rows.append((target, score))
    return rows


# ──────────────────────────────────────────────────────────────
# ROC-AUC against scipy
# ──────────────────────────────────────────────────────────────

def test_auc_matches_scipy_mannwhitney():
    scipy_stats = pytest.importorskip("scipy.stats")
    rows = _separable(1500, seed=11)
    pos = [s for t, s in rows if t == 1]
    neg = [s for t, s in rows if t == 0]
    u = scipy_stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    assert roc_auc(rows) == pytest.approx(u / (len(pos) * len(neg)), abs=1e-12)


def test_auc_matches_scipy_with_heavy_ties():
    """Mid-ranks matter: a coarse score column is mostly ties."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = random.Random(3)
    rows = [(1 if rng.random() < 0.3 else 0, round(rng.random(), 1))
            for _ in range(1200)]
    pos = [s for t, s in rows if t == 1]
    neg = [s for t, s in rows if t == 0]
    u = scipy_stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    assert roc_auc(rows) == pytest.approx(u / (len(pos) * len(neg)), abs=1e-12)


def test_auc_perfect_and_inverted():
    perfect = [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]
    assert roc_auc(perfect) == 1.0
    inverted = [(1, 0.1), (1, 0.2), (0, 0.8), (0, 0.9)]
    assert roc_auc(inverted) == 0.0


def test_auc_all_tied_is_half():
    assert roc_auc([(1, 0.5), (0, 0.5), (1, 0.5), (0, 0.5)]) == 0.5


def test_auc_none_when_one_class_missing():
    """Returning 0.5 here would look like a measurement of a coin flip."""
    assert roc_auc([(0, 0.1), (0, 0.9)]) is None
    assert roc_auc([(1, 0.1), (1, 0.9)]) is None
    assert roc_auc([]) is None


def test_auc_ignores_row_order():
    rows = _separable(400, seed=5)
    shuffled = list(rows)
    random.Random(99).shuffle(shuffled)
    assert roc_auc(rows) == pytest.approx(roc_auc(shuffled), abs=1e-12)


def test_auc_invariant_under_monotone_rescaling():
    rows = _separable(500, seed=13)
    rescaled = [(t, 3.0 * s + 7.0) for t, s in rows]
    assert roc_auc(rows) == pytest.approx(roc_auc(rescaled), abs=1e-12)


# ──────────────────────────────────────────────────────────────
# KS against scipy
# ──────────────────────────────────────────────────────────────

def test_ks_matches_scipy_ks_2samp():
    scipy_stats = pytest.importorskip("scipy.stats")
    rows = _separable(1500, seed=21)
    pos = [s for t, s in rows if t == 1]
    neg = [s for t, s in rows if t == 0]
    expected = float(scipy_stats.ks_2samp(pos, neg).statistic)
    assert ks_statistic(rows) == pytest.approx(expected, abs=1e-12)


def test_ks_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = random.Random(4)
    rows = [(1 if rng.random() < 0.25 else 0, round(rng.random(), 2))
            for _ in range(1000)]
    pos = [s for t, s in rows if t == 1]
    neg = [s for t, s in rows if t == 0]
    expected = float(scipy_stats.ks_2samp(pos, neg).statistic)
    assert ks_statistic(rows) == pytest.approx(expected, abs=1e-12)


def test_ks_perfect_separation_is_one():
    assert ks_statistic([(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]) == 1.0


def test_ks_none_when_one_class_missing():
    assert ks_statistic([(0, 0.1), (0, 0.4)]) is None


def test_ks_order_independent_within_ties():
    rows = [(1, 0.5), (0, 0.5), (1, 0.5), (0, 0.2), (1, 0.9), (0, 0.9)]
    assert ks_statistic(rows) == pytest.approx(ks_statistic(list(reversed(rows))))


# ──────────────────────────────────────────────────────────────
# Hanley-McNeil standard error
# ──────────────────────────────────────────────────────────────

def test_auc_se_agrees_with_bootstrap():
    """Seeded, so this either always passes or always fails — never flakes."""
    rows = _separable(1200, seed=31)
    auc = roc_auc(rows)
    n_pos = sum(t for t, _ in rows)
    closed_form = auc_standard_error(auc, n_pos, len(rows) - n_pos)

    rng = random.Random(1234)
    draws = []
    for _ in range(150):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        a = roc_auc(sample)
        if a is not None:
            draws.append(a)
    mean = sum(draws) / len(draws)
    boot = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))

    assert closed_form == pytest.approx(boot, rel=0.35), (
        f"Hanley-McNeil SE {closed_form:.5f} vs bootstrap {boot:.5f}"
    )


def test_auc_se_shrinks_with_sample_size():
    small = auc_standard_error(0.75, 100, 1000)
    large = auc_standard_error(0.75, 1000, 10000)
    assert large < small
    assert large == pytest.approx(small / math.sqrt(10), rel=0.05)


def test_auc_se_none_without_both_classes():
    assert auc_standard_error(0.8, 0, 100) is None
    assert auc_standard_error(0.8, 100, 0) is None
    assert auc_standard_error(None, 100, 100) is None


def test_auc_se_zero_at_perfect_separation():
    """A=1 gives Q1=Q2=1, so the variance collapses — not an error."""
    assert auc_standard_error(1.0, 50, 50) == 0.0


# ──────────────────────────────────────────────────────────────
# Quantile binning
# ──────────────────────────────────────────────────────────────

def test_quantile_edges_split_into_equal_counts():
    values = list(range(1000))
    edges = quantile_edges(values, 5)
    assert len(edges) == 4
    labels = [bin_label(v, edges) for v in values]
    counts = {l: labels.count(l) for l in set(labels)}
    assert sorted(counts) == ["q1", "q2", "q3", "q4", "q5"]
    assert max(counts.values()) - min(counts.values()) <= 1


def test_quantile_edges_come_from_the_data():
    """Not round numbers imposed in config — the point of P0.4's bands."""
    values = [20.3, 21.9, 35.1, 36.4, 58.2, 61.7]
    assert quantile_edges(values, 3) == [35.1, 58.2]


def test_quantile_edges_dedupe_on_a_modal_value():
    """A heavily repeated income collapses two cut points into one."""
    values = [1.0] * 80 + [2.0] * 10 + [3.0] * 10
    edges = quantile_edges(values, 5)
    assert edges == sorted(set(edges))
    assert len(edges) < 4


def test_quantile_edges_empty_and_all_missing():
    assert quantile_edges([], 5) == []
    assert quantile_edges([None, float("nan")], 5) == []


def test_quantile_edges_rejects_one_bin():
    with pytest.raises(ValueError, match="at least 2"):
        quantile_edges([1, 2, 3], 1)


def test_bin_label_edge_goes_to_upper_band():
    edges = [10.0, 20.0]
    assert bin_label(9.9, edges) == "q1"
    assert bin_label(10.0, edges) == "q2"
    assert bin_label(20.0, edges) == "q3"


def test_bin_label_missing_values():
    assert bin_label(None, [1.0]) == MISSING_LABEL
    assert bin_label(float("nan"), [1.0]) == MISSING_LABEL


def test_bin_bounds_open_at_the_ends():
    edges = [10.0, 20.0]
    assert bin_bounds("q1", edges) == (None, 10.0)
    assert bin_bounds("q2", edges) == (10.0, 20.0)
    assert bin_bounds("q3", edges) == (20.0, None)
    assert bin_bounds(MISSING_LABEL, edges) == (None, None)


def test_days_to_years_transform():
    """DAYS_BIRTH is negative in Home Credit; age must come out positive."""
    assert TRANSFORMS["days_to_years"](-365.25 * 40) == pytest.approx(40.0)
    assert TRANSFORMS["none"](-7.0) == -7.0


# ──────────────────────────────────────────────────────────────
# Group metrics and suppression
# ──────────────────────────────────────────────────────────────

def test_group_metrics_reports_calibration_gap():
    rows = [(1, 0.5)] * 20 + [(0, 0.5)] * 80
    m = group_metrics(rows, min_rows=0, min_positives=0)
    assert m["n"] == 100
    assert m["n_positive"] == 20
    assert m["default_rate"] == pytest.approx(0.2)
    assert m["mean_predicted_pd"] == pytest.approx(0.5)
    assert m["gap"] == pytest.approx(0.3)      # predicted minus actual


def test_group_suppressed_below_min_rows():
    rows = _separable(300, seed=41)
    m = group_metrics(rows, min_rows=1000, min_positives=50)
    assert m["evaluated"] is False
    assert m["roc_auc"] is None
    assert "min_rows=1000" in m["suppressed_reason"]
    # Counts are still reported — suppression hides the unstable AUC, not the group.
    assert m["n"] == 300 and m["default_rate"] is not None


def test_group_suppressed_below_min_positives():
    rows = [(0, 0.1)] * 1200 + [(1, 0.9)] * 5
    m = group_metrics(rows, min_rows=1000, min_positives=50)
    assert m["evaluated"] is False
    assert "min_positives=50" in m["suppressed_reason"]


def test_group_suppressed_when_every_row_defaults():
    rows = [(1, 0.4 + i / 10000) for i in range(1500)]
    m = group_metrics(rows, min_rows=1000, min_positives=50)
    assert m["evaluated"] is False
    assert "undefined" in m["suppressed_reason"]


def test_group_metrics_attaches_interval():
    rows = _separable(2000, seed=43)
    m = group_metrics(rows, min_rows=1000, min_positives=50)
    assert m["evaluated"] is True
    assert m["auc_ci95_lower"] < m["roc_auc"] < m["auc_ci95_upper"]
    assert m["auc_ci95_upper"] - m["auc_ci95_lower"] == pytest.approx(
        2 * 1.96 * m["auc_se"], abs=2e-6
    )


def test_overall_metrics_has_no_suppression_field():
    """The pooled row is never suppressed — it is the reference."""
    m = overall_metrics(_separable(100, seed=45))
    assert "suppressed_reason" not in m
    assert m["evaluated"] is True


# ──────────────────────────────────────────────────────────────
# Dimensions
# ──────────────────────────────────────────────────────────────

def _categorical_rows():
    """Two groups, both well supported; 'Revolving' is the weaker one."""
    rows = []
    for t, s in _separable(4000, seed=51, strength=1.0):
        rows.append(("Cash loans", t, s))
    for t, s in _separable(4000, seed=52, strength=0.0):   # no signal
        rows.append(("Revolving loans", t, s))
    return rows


def test_dimension_groups_and_orders_categories_by_size():
    spec = SegmentSpec(name="contract_type", column="NAME_CONTRACT_TYPE",
                       kind=KIND_CATEGORICAL)
    rows = _categorical_rows() + [(None, 0, 0.3)] * 5
    dim = build_dimension(spec, rows, min_rows=1000, min_positives=50)
    labels = [g["group"] for g in dim["groups"]]
    assert labels[-1] == MISSING_LABEL          # missing always last
    assert set(labels[:2]) == {"Cash loans", "Revolving loans"}
    assert dim["n_rows"] == len(rows)
    assert dim["n_groups_evaluated"] == 2       # the 5 missing rows are suppressed


def test_dimension_detects_a_group_with_no_signal():
    spec = SegmentSpec(name="contract_type", column="NAME_CONTRACT_TYPE")
    dim = build_dimension(spec, _categorical_rows(), min_rows=1000, min_positives=50)
    by_name = {g["group"]: g for g in dim["groups"]}
    assert by_name["Revolving loans"]["roc_auc"] < by_name["Cash loans"]["roc_auc"]
    assert dim["worst_group"] == "Revolving loans"
    assert dim["best_group"] == "Cash loans"
    assert dim["auc_spread"] > 0
    # A built-in no-signal group must clear the noise bound, or the bound is useless.
    assert dim["spread_vs_noise"] > 2
    assert dim["spread_within_noise"] is False


def test_dimension_spread_within_noise_when_groups_are_identical():
    """Same generator twice: any spread is sampling noise and must be labelled so."""
    spec = SegmentSpec(name="gender", column="CODE_GENDER", sensitive=True)
    rows = ([("F", t, s) for t, s in _separable(4000, seed=61)]
            + [("M", t, s) for t, s in _separable(4000, seed=62)])
    dim = build_dimension(spec, rows, min_rows=1000, min_positives=50)
    assert dim["spread_within_noise"] is True
    assert dim["spread_vs_noise"] < 2
    assert dim["sensitive"] is True


def test_dimension_quantile_records_edges_and_bounds():
    spec = SegmentSpec(name="age_group", column="DAYS_BIRTH",
                       kind=KIND_QUANTILE, n_bins=4, transform="days_to_years")
    rng = random.Random(71)
    rows = []
    for _ in range(8000):
        days = -365.25 * rng.uniform(21, 69)
        t = 1 if rng.random() < 0.2 else 0
        rows.append((days, t, rng.random()))
    dim = build_dimension(spec, rows, min_rows=1000, min_positives=50)

    assert dim["bin_edges"] is not None and len(dim["bin_edges"]) == 3
    assert all(20 < e < 70 for e in dim["bin_edges"])       # years, not days
    assert dim["derivation"] == "days_to_years(DAYS_BIRTH)"
    bands = [g for g in dim["groups"] if g["group"].startswith("q")]
    assert [g["group"] for g in bands] == ["q1", "q2", "q3", "q4"]
    assert bands[0]["lower"] is None and bands[-1]["upper"] is None
    assert bands[1]["lower"] == bands[0]["upper"]           # contiguous bands


def test_dimension_with_single_evaluated_group_has_no_spread():
    spec = SegmentSpec(name="contract_type", column="NAME_CONTRACT_TYPE")
    rows = ([("Cash loans", t, s) for t, s in _separable(3000, seed=81)]
            + [("Rare", t, s) for t, s in _separable(50, seed=82)])
    dim = build_dimension(spec, rows, min_rows=1000, min_positives=50)
    assert dim["n_groups_evaluated"] == 1
    assert dim["auc_spread"] is None
    assert dim["spread_vs_noise"] is None
    assert dim["spread_within_noise"] is None


def test_dimension_handles_nan_category():
    spec = SegmentSpec(name="gender", column="CODE_GENDER")
    rows = ([("F", t, s) for t, s in _separable(2000, seed=91)]
            + [(float("nan"), 0, 0.1)] * 3)
    dim = build_dimension(spec, rows, min_rows=1000, min_positives=50)
    assert [g["group"] for g in dim["groups"]][-1] == MISSING_LABEL
    assert dim["groups"][-1]["n"] == 3


# ──────────────────────────────────────────────────────────────
# Specs from config
# ──────────────────────────────────────────────────────────────

def test_build_specs_reads_all_four_dimensions():
    specs = build_specs(_cfg())
    assert [s.name for s in specs] == [
        "contract_type", "gender", "age_group", "income_group"]
    assert specs[1].sensitive is True
    assert specs[2].transform == "days_to_years"
    assert specs[3].n_bins == 5          # inherits segments.n_bins


def test_build_specs_requires_dimensions():
    with pytest.raises(ValueError, match="segments.dimensions"):
        build_specs({"segments": {}})
    with pytest.raises(ValueError, match="segments.dimensions"):
        build_specs({})


def test_build_specs_rejects_duplicate_names():
    cfg = _cfg()
    cfg["segments"]["dimensions"].append(
        {"name": "gender", "column": "CODE_GENDER", "kind": "categorical"})
    with pytest.raises(ValueError, match="duplicate segment names"):
        build_specs(cfg)


def test_spec_rejects_unknown_transform():
    """Config must not be able to name arbitrary code."""
    with pytest.raises(ValueError, match="transform"):
        SegmentSpec(name="x", column="C", transform="os.system")


def test_spec_rejects_unknown_kind_and_bad_bins():
    with pytest.raises(ValueError, match="kind"):
        SegmentSpec(name="x", column="C", kind="decile")
    with pytest.raises(ValueError, match="at least 2"):
        SegmentSpec(name="x", column="C", kind=KIND_QUANTILE, n_bins=1)


def test_spec_requires_name_and_column():
    with pytest.raises(ValueError, match="name"):
        SegmentSpec(name="", column="C")
    with pytest.raises(ValueError, match="column"):
        SegmentSpec(name="x", column="")


def test_source_columns_deduplicates():
    specs = [SegmentSpec(name="a", column="CODE_GENDER"),
             SegmentSpec(name="b", column="CODE_GENDER"),
             SegmentSpec(name="c", column="DAYS_BIRTH")]
    assert source_columns(specs) == ["CODE_GENDER", "DAYS_BIRTH"]


# ──────────────────────────────────────────────────────────────
# Setup + fingerprint
# ──────────────────────────────────────────────────────────────

def test_setup_records_thresholds_and_slices():
    setup = build_setup(_cfg(), build_specs(_cfg()))
    assert setup["min_rows"] == 1000
    assert setup["min_positives"] == 50
    assert [d["name"] for d in setup["dimensions"]] == [
        "contract_type", "gender", "age_group", "income_group"]
    assert "out-of-fold" in setup["predictions"]
    assert "pr_auc" not in setup["metrics"]
    assert setup["pr_auc_excluded"]


def test_setup_falls_back_to_defaults():
    cfg = {"segments": {"dimensions": [
        {"name": "g", "column": "CODE_GENDER", "kind": "categorical"}]}}
    setup = build_setup(cfg, build_specs(cfg))
    assert setup["min_rows"] == DEFAULT_MIN_ROWS
    assert setup["min_positives"] == DEFAULT_MIN_POSITIVES


def test_fingerprint_is_stable_and_ignores_measured_counts():
    setup_a = build_setup(_cfg(), build_specs(_cfg()), source={"oof_rows": 261652})
    setup_b = build_setup(_cfg(), build_specs(_cfg()), source={"oof_rows": 999})
    assert config_fingerprint(setup_a) == config_fingerprint(setup_b)
    assert config_fingerprint(setup_a).startswith("sha256:")


def test_fingerprint_changes_with_support_threshold():
    base = build_setup(_cfg(), build_specs(_cfg()))
    cfg = _cfg()
    cfg["segments"]["min_positives"] = 25
    changed = build_setup(cfg, build_specs(cfg))
    assert config_fingerprint(base) != config_fingerprint(changed)


def test_fingerprint_changes_with_bin_count():
    base = build_setup(_cfg(), build_specs(_cfg()))
    cfg = _cfg()
    cfg["segments"]["dimensions"][2]["n_bins"] = 10
    changed = build_setup(cfg, build_specs(cfg))
    assert config_fingerprint(base) != config_fingerprint(changed)


# ──────────────────────────────────────────────────────────────
# Cross-check against T1.4
# ──────────────────────────────────────────────────────────────

def test_compare_overall_auc_accepts_a_small_gap():
    """Pooled OOF vs the mean of five folds: not the same estimand."""
    ref = compare_overall_auc(0.7640, 0.7663)
    assert ref["delta"] == pytest.approx(-0.0023)
    assert ref["within_soft_tolerance"] is True


def test_compare_overall_auc_flags_a_medium_gap_without_raising():
    ref = compare_overall_auc(0.7463, 0.7663)
    assert ref["within_soft_tolerance"] is False
    assert ref["delta"] == pytest.approx(-0.02)


def test_compare_overall_auc_raises_on_a_structural_gap():
    """An inverted label or the wrong score column must not reach the report."""
    with pytest.raises(RuntimeError, match="structural"):
        compare_overall_auc(0.2337, 0.7663)


def test_compare_overall_auc_boundary():
    """Just inside the hard bound is recorded; just outside stops the run.

    Exactly at the bound is not asserted: float arithmetic decides it, and a
    test that pins binary round-off would claim a precision the check in the
    module does not have.
    """
    inside = compare_overall_auc(0.7663 - (AUC_HARD_TOLERANCE - 0.001), 0.7663)
    assert inside["within_soft_tolerance"] is False
    with pytest.raises(RuntimeError, match="structural"):
        compare_overall_auc(0.7663 - (AUC_HARD_TOLERANCE + 0.001), 0.7663)


def test_compare_overall_auc_without_reference():
    ref = compare_overall_auc(0.76, None)
    assert ref["delta"] is None
    assert ref["within_soft_tolerance"] is None
    assert "no reference" in ref["note"]


# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────

def _report() -> dict:
    cfg = _cfg()
    specs = build_specs(cfg)
    setup = build_setup(cfg, specs, source={"cv_folds_used": [1, 2, 3, 4, 5]})
    contract = build_dimension(specs[0], _categorical_rows(),
                               min_rows=1000, min_positives=50)
    gender_rows = ([("F", t, s) for t, s in _separable(4000, seed=61)]
                   + [("M", t, s) for t, s in _separable(4000, seed=62)])
    gender = build_dimension(specs[1], gender_rows, min_rows=1000, min_positives=50)
    pooled = [(t, s) for _, t, s in _categorical_rows()]
    pooled_auc = roc_auc(pooled)
    # The fixture's reference sits a plausible distance from its own pooled AUC;
    # hardcoding 0.7663 against synthetic scores would trip the structural check.
    return build_segment_report(
        setup=setup,
        dimensions=[contract, gender],
        overall=overall_metrics(pooled),
        reference=compare_overall_auc(pooled_auc, round(pooled_auc + 0.002, 4)),
        cv_config_fingerprint="sha256:17fb2128",
    )


def test_report_structure():
    r = _report()
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["config_fingerprint"].startswith("sha256:")
    assert r["cv_config_fingerprint"] == "sha256:17fb2128"
    assert r["n_dimensions"] == 2
    assert r["note"] == NOTE
    assert r["setup"]["source"]["cv_folds_used"] == [1, 2, 3, 4, 5]


def test_report_lists_only_dimensions_beyond_noise():
    r = _report()
    flagged = [d["name"] for d in r["dimensions_with_spread_beyond_noise"]]
    assert flagged == ["contract_type"]       # gender's two groups are identical


def test_report_verdict_is_left_empty():
    """PLAN rule 1: the wording is written from the numbers, not pre-filled."""
    assert _report()["verdict"] == ""


def test_report_filename():
    assert REPORT_FILENAME == "segment_report.json"


def test_report_is_json_serialisable():
    import json
    blob = json.dumps(_report())
    assert "sha256:" in blob


def test_format_summary_renders_groups_and_verdicts():
    out = format_summary(_report())
    assert "SEGMENT VALIDATION" in out
    assert "contract_type" in out and "Revolving loans" in out
    assert "[sensitive]" in out                 # gender is flagged in the table
    assert "beyond noise" in out and "within noise" in out


def test_format_summary_survives_an_empty_report():
    assert "SEGMENT VALIDATION" in format_summary({})
