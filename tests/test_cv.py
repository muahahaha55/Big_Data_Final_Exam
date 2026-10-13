"""Unit tests for the T1.4 cross-validation study logic — no Spark import.

Covers the parts that decide whether the confidence intervals mean anything:
deterministic stratified fold assignment, per-fold default rates, the
fingerprint/resume machinery, the t-interval formula (recomputed independently
against scipy), and the decile/ECE calibration table on synthetic data whose
answer is known in advance.

Imports `credit_risk.cv` rather than `pipelines.cv_pipeline` on purpose: the
pipeline pulls in pyspark.ml, which needs a distutils shim on Python 3.12.
"""

from __future__ import annotations

import importlib
import math

import pytest

import credit_risk.cv.study

from credit_risk.cv import (
    METRICS,
    REPORT_FILENAMES,
    SCHEMA_VERSION,
    assign_folds,
    build_calibration_report,
    build_cv_report,
    build_setup,
    check_fold_balance,
    column_audit,
    completed_folds,
    config_fingerprint,
    decile_table,
    dropped_by_prefix,
    expected_calibration_error,
    fold_balance,
    fold_ids,
    pending_folds,
    summarize_folds,
    t_critical,
    t_interval,
)


def _cfg() -> dict:
    return {
        "random_seed": 42,
        "data": {"target_column": "TARGET", "id_column": "SK_ID_CURR",
                 "split": {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}},
        "features": {"missing_threshold": 0.5},
        "models": {"gbt": {"maxIter": 100, "maxDepth": 5, "stepSize": 0.1,
                           "subsamplingRate": 0.8}},
        "spark": {"master": "local[1]", "driver_memory": "2g", "shuffle_partitions": 50},
        "cv": {"k_folds": 5, "confidence_level": 0.95, "n_deciles": 10},
    }


def _rows(n: int, positive_every: int = 10) -> list[tuple[int, int]]:
    """n ids; every `positive_every`-th is a default."""
    return [(i, 1 if i % positive_every == 0 else 0) for i in range(1, n + 1)]


def _hash(seed: int = 42):
    """A deterministic stand-in for Spark's xxhash64(id, seed)."""
    return lambda i: (hash((i, seed)) & 0xFFFFFFFF)


def _fold_rec(fold: int, roc: float, pr: float = 0.24, ks: float = 0.39, **kw) -> dict:
    return {"fold": fold, "roc_auc": roc, "pr_auc": pr, "ks_statistic": ks, **kw}


# ──────────────────────────────────────────────────────────────
# Fold ids
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFoldIds:
    def test_one_based_and_complete(self):
        assert fold_ids(5) == [1, 2, 3, 4, 5]

    def test_rejects_k_below_two(self):
        for bad in (1, 0, -3):
            with pytest.raises(ValueError, match="at least 2"):
                fold_ids(bad)


# ──────────────────────────────────────────────────────────────
# Fold assignment: deterministic + stratified
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAssignFolds:
    def test_every_id_gets_exactly_one_fold_in_range(self):
        rows = _rows(1000)
        a = assign_folds(rows, 5, _hash())
        assert len(a) == 1000
        assert set(a.values()) <= set(fold_ids(5))
        assert set(a) == {i for i, _ in rows}

    def test_deterministic_across_calls(self):
        rows = _rows(500)
        assert assign_folds(rows, 5, _hash()) == assign_folds(rows, 5, _hash())

    def test_independent_of_input_order(self):
        """Row order must not change the assignment — Spark partitioning varies."""
        rows = _rows(500)
        a = assign_folds(rows, 5, _hash())
        b = assign_folds(list(reversed(rows)), 5, _hash())
        assert a == b

    def test_different_seed_gives_different_assignment(self):
        rows = _rows(500)
        a = assign_folds(rows, 5, _hash(42))
        b = assign_folds(rows, 5, _hash(7))
        assert a != b
        assert set(a) == set(b)   # same ids, different folds

    def test_fold_sizes_differ_by_at_most_one(self):
        a = assign_folds(_rows(1000), 5, _hash())
        sizes = [sum(1 for f in a.values() if f == k) for k in fold_ids(5)]
        assert max(sizes) - min(sizes) <= 1, sizes

    def test_stratification_balances_positives_within_one(self):
        rows = _rows(1000, positive_every=10)      # exactly 100 positives
        a = assign_folds(rows, 5, _hash())
        pos = [sum(1 for i, t in rows if t == 1 and a[i] == k) for k in fold_ids(5)]
        assert sum(pos) == 100
        assert max(pos) - min(pos) <= 1, pos

    def test_stratification_holds_for_awkward_counts(self):
        """7 positives over 5 folds cannot be equal — must still be within 1."""
        rows = [(i, 1 if i <= 7 else 0) for i in range(1, 104)]
        a = assign_folds(rows, 5, _hash())
        pos = [sum(1 for i, t in rows if t == 1 and a[i] == k) for k in fold_ids(5)]
        assert sorted(pos) == [1, 1, 1, 2, 2], pos

    def test_handles_single_class(self):
        rows = [(i, 0) for i in range(1, 51)]
        a = assign_folds(rows, 5, _hash())
        sizes = [sum(1 for f in a.values() if f == k) for k in fold_ids(5)]
        assert sizes == [10, 10, 10, 10, 10]

    def test_ties_in_hash_are_broken_deterministically(self):
        rows = _rows(200)
        const = lambda i: 0          # every hash identical
        assert assign_folds(rows, 5, const) == assign_folds(list(reversed(rows)), 5, const)

    def test_rejects_k_below_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            assign_folds(_rows(10), 1, _hash())


@pytest.mark.unit
class TestFoldBalance:
    def test_counts_and_default_rate_per_fold(self):
        rows = _rows(1000, positive_every=10)
        a = assign_folds(rows, 5, _hash())
        bal = fold_balance(rows, a, 5)
        assert [b["fold"] for b in bal] == [1, 2, 3, 4, 5]
        assert sum(b["n"] for b in bal) == 1000
        assert sum(b["n_positive"] for b in bal) == 100
        for b in bal:
            assert b["default_rate"] == pytest.approx(b["n_positive"] / b["n"])

    def test_default_rate_is_near_the_base_rate_in_every_fold(self):
        rows = _rows(2000, positive_every=10)      # 10% base rate
        bal = fold_balance(rows, assign_folds(rows, 5, _hash()), 5)
        for b in bal:
            assert abs(b["default_rate"] - 0.10) < 0.002, b

    def test_check_passes_on_balanced_folds(self):
        rows = _rows(1000, positive_every=10)
        check_fold_balance(fold_balance(rows, assign_folds(rows, 5, _hash()), 5))

    def test_check_raises_on_skewed_sizes(self):
        bad = [{"fold": 1, "n": 100, "n_positive": 8, "default_rate": 0.08},
               {"fold": 2, "n": 80, "n_positive": 8, "default_rate": 0.1}]
        with pytest.raises(RuntimeError, match="fold sizes differ"):
            check_fold_balance(bad)

    def test_check_raises_when_stratification_failed(self):
        bad = [{"fold": 1, "n": 100, "n_positive": 20, "default_rate": 0.2},
               {"fold": 2, "n": 100, "n_positive": 2, "default_rate": 0.02}]
        with pytest.raises(RuntimeError, match="stratification failed"):
            check_fold_balance(bad)

    def test_check_raises_on_empty(self):
        with pytest.raises(ValueError):
            check_fold_balance([])


# ──────────────────────────────────────────────────────────────
# Setup + fingerprint
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSetupAndFingerprint:
    def test_setup_records_the_decisions_that_matter(self):
        s = build_setup(5, _cfg())
        assert s["k_folds"] == 5
        assert s["class_weighting"] is False          # D5: reproduce Production
        assert s["tune"] is False
        assert s["missing_threshold"] == 0.5
        assert s["stratified_by"] == "TARGET"
        assert "train.parquet" in s["cv_pool"] and "val.parquet" in s["cv_pool"]
        assert "test.parquet" in s["holdout_untouched"]
        assert s["hyperparams"]["maxDepth"] == 5

    def test_fingerprint_stable_and_order_independent(self):
        s = build_setup(5, _cfg())
        assert config_fingerprint(s) == config_fingerprint(s)
        assert config_fingerprint(dict(reversed(list(s.items())))) == config_fingerprint(s)

    def test_fingerprint_ignores_prose_and_measured_pool(self):
        s = build_setup(5, _cfg())
        base = config_fingerprint(s)
        s["not_comparable_to"] = "reworded"
        s["pool"] = {"n_rows": 261691, "n_master_columns": 161}
        assert config_fingerprint(s) == base

    @pytest.mark.parametrize("mutate", [
        pytest.param(lambda c: c["cv"].__setitem__("k_folds", 3), id="k_folds"),
        pytest.param(lambda c: c["models"]["gbt"].__setitem__("maxDepth", 6), id="maxDepth"),
        pytest.param(lambda c: c.__setitem__("random_seed", 7), id="seed"),
        pytest.param(lambda c: c["features"].__setitem__("missing_threshold", 0.9),
                     id="missing_threshold"),
        pytest.param(lambda c: c["spark"].__setitem__("master", "local[*]"), id="master"),
        pytest.param(lambda c: c["spark"].__setitem__("shuffle_partitions", 8),
                     id="shuffle_partitions"),
    ])
    def test_fingerprint_changes_on_comparability_critical_settings(self, mutate):
        cfg = _cfg()
        base = config_fingerprint(build_setup(int(cfg["cv"]["k_folds"]), cfg))
        mutate(cfg)
        after = config_fingerprint(build_setup(int(cfg["cv"]["k_folds"]), cfg))
        assert after != base

    def test_fingerprint_changes_if_weighting_were_turned_on(self):
        s = build_setup(5, _cfg())
        base = config_fingerprint(s)
        s["class_weighting"] = True
        assert config_fingerprint(s) != base

    def test_looks_like_a_sha256(self):
        fp = config_fingerprint(build_setup(5, _cfg()))
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 64


# ──────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestResume:
    def test_no_report_means_everything_pending(self):
        assert pending_folds(5, None) == [1, 2, 3, 4, 5]

    def test_finished_folds_skipped(self):
        report = {"folds": [_fold_rec(1, 0.75), _fold_rec(2, 0.76)]}
        assert pending_folds(5, report) == [3, 4, 5]

    def test_resume_after_crash_midway(self):
        report = {"folds": [_fold_rec(i, 0.75) for i in (1, 2, 3)]}
        assert pending_folds(5, report) == [4, 5]

    def test_complete_study_has_nothing_pending(self):
        report = {"folds": [_fold_rec(i, 0.75) for i in range(1, 6)]}
        assert pending_folds(5, report) == []

    def test_only_fold_narrows(self):
        assert pending_folds(5, None, only_fold=3) == [3]

    def test_only_fold_already_done_yields_nothing(self):
        report = {"folds": [_fold_rec(3, 0.75)]}
        assert pending_folds(5, report, only_fold=3) == []

    def test_force_reruns_everything(self):
        report = {"folds": [_fold_rec(i, 0.75) for i in range(1, 6)]}
        assert pending_folds(5, report, force=True) == [1, 2, 3, 4, 5]

    def test_completed_folds_tolerates_junk_and_strings(self):
        report = {"folds": [_fold_rec(1, 0.7), {"no_fold": 1}, None, {"fold": "4"}]}
        assert set(completed_folds(report)) == {1, 4}

    def test_out_of_order_completion(self):
        report = {"folds": [_fold_rec(5, 0.75), _fold_rec(1, 0.74)]}
        assert pending_folds(5, report) == [2, 3, 4]


# ──────────────────────────────────────────────────────────────
# t-interval — recomputed independently against scipy
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTInterval:
    def test_t_critical_matches_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        for df in (1, 2, 3, 4, 5, 9, 20):
            assert t_critical(0.95, df) == pytest.approx(stats.t.ppf(0.975, df), rel=1e-12)
            assert t_critical(0.99, df) == pytest.approx(stats.t.ppf(0.995, df), rel=1e-12)

    def test_t_critical_known_values(self):
        """The two that matter for k=5 and k=3."""
        assert t_critical(0.95, 4) == pytest.approx(2.776, abs=5e-4)
        assert t_critical(0.95, 2) == pytest.approx(4.303, abs=5e-4)

    def test_t_critical_rejects_bad_df(self):
        with pytest.raises(ValueError):
            t_critical(0.95, 0)

    def test_formula_matches_an_independent_recomputation(self):
        stats = pytest.importorskip("scipy.stats")
        vals = [0.7546, 0.7562, 0.7621, 0.7667, 0.7652]
        got = t_interval(vals)

        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))   # ddof=1
        half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)

        assert got["n"] == 5 and got["df"] == 4
        assert got["mean"] == pytest.approx(mean, abs=1e-6)
        assert got["std"] == pytest.approx(std, abs=1e-6)
        assert got["half_width"] == pytest.approx(half, abs=1e-6)
        assert got["ci95_lower"] == pytest.approx(mean - half, abs=1e-6)
        assert got["ci95_upper"] == pytest.approx(mean + half, abs=1e-6)

    def test_uses_sample_std_not_population(self):
        """ddof=1: getting this wrong narrows every interval."""
        vals = [0.70, 0.80]
        got = t_interval(vals)
        assert got["std"] == pytest.approx(0.0707107, abs=1e-6)   # not 0.05

    def test_min_max_and_mean_reported_alongside(self):
        got = t_interval([0.1, 0.2, 0.3])
        assert got["min"] == 0.1 and got["max"] == 0.3
        assert got["mean"] == pytest.approx(0.2)

    def test_interval_brackets_the_mean(self):
        got = t_interval([0.7546, 0.7562, 0.7621, 0.7667, 0.7652])
        assert got["ci95_lower"] < got["mean"] < got["ci95_upper"]

    def test_identical_values_give_zero_width(self):
        got = t_interval([0.75] * 5)
        assert got["std"] == 0.0
        assert got["ci95_lower"] == got["ci95_upper"] == pytest.approx(0.75)

    def test_single_value_has_no_interval(self):
        """One fold cannot estimate spread; 0 would read as 'perfectly stable'."""
        got = t_interval([0.75])
        assert got["n"] == 1
        assert got["std"] is None
        assert got["ci95_lower"] is None and got["ci95_upper"] is None

    def test_empty_is_all_none(self):
        got = t_interval([])
        assert got["n"] == 0 and got["mean"] is None

    def test_none_values_are_skipped(self):
        got = t_interval([0.70, None, 0.80])
        assert got["n"] == 2

    def test_k3_interval_is_wider_than_k5_for_the_same_spread(self):
        """Why D4 chose k=5: t*/sqrt(k) is 2.484 at k=3 vs 1.242 at k=5."""
        five = t_interval([0.74, 0.75, 0.76, 0.75, 0.75])
        three = t_interval([0.74, 0.75, 0.76])
        assert three["t_critical"] > five["t_critical"]


@pytest.mark.unit
class TestSummarizeFolds:
    def test_covers_all_three_metrics(self):
        folds = [_fold_rec(i, 0.75 + i / 1000, pr=0.24 + i / 1000, ks=0.39 + i / 1000)
                 for i in range(1, 6)]
        out = summarize_folds(folds)
        assert set(out) == set(METRICS)
        for m in METRICS:
            assert out[m]["n"] == 5 and out[m]["mean"] is not None

    def test_partial_study_summarizes_what_exists(self):
        out = summarize_folds([_fold_rec(1, 0.75), _fold_rec(2, 0.76)])
        assert out["roc_auc"]["n"] == 2 and out["roc_auc"]["df"] == 1


# ──────────────────────────────────────────────────────────────
# Column audit — the credit_card question
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestColumnAudit:
    def test_splits_kept_from_dropped(self):
        a = column_audit(
            all_columns=["SK_ID_CURR", "TARGET", "AMT_CREDIT", "CODE_GENDER",
                         "CC_BALANCE_MEAN", "CC_DPD_SUM"],
            numeric=["AMT_CREDIT"], categorical=["CODE_GENDER"],
            excluded=["TARGET", "SK_ID_CURR"],
        )
        assert a["n_master_columns"] == 6
        assert a["n_candidates"] == 4
        assert a["n_source_columns"] == 2
        assert a["dropped_columns"] == ["CC_BALANCE_MEAN", "CC_DPD_SUM"]
        assert a["n_dropped"] == 2

    def test_nothing_dropped(self):
        a = column_audit(["SK_ID_CURR", "TARGET", "A"], ["A"], [],
                         ["TARGET", "SK_ID_CURR"])
        assert a["dropped_columns"] == [] and a["n_dropped"] == 0

    def test_dropped_list_is_sorted_for_stable_diffs(self):
        a = column_audit(["TARGET", "Z", "A", "M"], [], [], ["TARGET"])
        assert a["dropped_columns"] == ["A", "M", "Z"]

    def test_attributes_dropped_columns_to_their_source_table(self):
        dropped = ["CC_BALANCE_MEAN", "CC_DPD_SUM", "CC_UTILIZATION_MEAN",
                   "OWN_CAR_AGE"]
        out = dropped_by_prefix(dropped, {"credit_card_balance": "CC_",
                                          "bureau": "BUREAU_"})
        assert out["credit_card_balance"]["n_dropped"] == 3
        assert out["bureau"]["n_dropped"] == 0
        assert "OWN_CAR_AGE" not in out["credit_card_balance"]["columns"]


# ──────────────────────────────────────────────────────────────
# Calibration: deciles + ECE on data with a known answer
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDecileTable:
    def test_ten_groups_of_equal_size(self):
        rows = [(i, 0, i / 1000) for i in range(1, 1001)]
        t = decile_table(rows)
        assert [d["decile"] for d in t] == list(range(1, 11))
        assert all(d["n"] == 100 for d in t)

    def test_sizes_differ_by_at_most_one_when_n_not_divisible(self):
        rows = [(i, 0, i / 1007) for i in range(1, 1008)]
        t = decile_table(rows)
        sizes = [d["n"] for d in t]
        assert sum(sizes) == 1007 and max(sizes) - min(sizes) <= 1

    def test_decile_one_is_lowest_risk(self):
        rows = [(i, 0, i / 100) for i in range(1, 101)]
        t = decile_table(rows)
        assert t[0]["pd_max"] <= t[-1]["pd_min"]
        means = [d["pd_mean_predicted"] for d in t]
        assert means == sorted(means)

    def test_perfect_calibration_gives_zero_gap(self):
        """Decile d has PD exactly d/10 and exactly that default fraction."""
        rows = []
        ident = 0
        for d in range(1, 11):
            p = d / 10.0
            n_def = int(round(p * 100))
            for j in range(100):
                ident += 1
                rows.append((ident, 1 if j < n_def else 0, p))
        t = decile_table(rows)
        for d in t:
            assert d["pd_mean_predicted"] == pytest.approx(d["actual_default_rate"], abs=1e-9)
            assert d["gap"] == pytest.approx(0.0, abs=1e-9)
        assert expected_calibration_error(t) == pytest.approx(0.0, abs=1e-9)

    def test_known_constant_overprediction(self):
        """Predict 0.30 everywhere, true rate 0.10 → gap +0.20, ECE 0.20."""
        rows = [(i, 1 if i % 10 == 0 else 0, 0.30) for i in range(1, 1001)]
        t = decile_table(rows)
        assert expected_calibration_error(t) == pytest.approx(0.20, abs=1e-6)
        total_def = sum(d["n_default"] for d in t)
        assert total_def == 100

    def test_counts_and_defaults_add_up(self):
        rows = [(i, i % 2, i / 500) for i in range(1, 501)]
        t = decile_table(rows)
        assert sum(d["n"] for d in t) == 500
        assert sum(d["n_default"] for d in t) == sum(1 for i in range(1, 501) if i % 2)

    def test_gap_is_predicted_minus_actual(self):
        rows = [(i, 1 if i <= 50 else 0, 0.9) for i in range(1, 101)]
        t = decile_table(rows, n_deciles=2)
        for d in t:
            assert d["gap"] == pytest.approx(
                d["pd_mean_predicted"] - d["actual_default_rate"], abs=1e-9)

    def test_deterministic_under_input_reordering(self):
        rows = [(i, i % 3 == 0, 0.5) for i in range(1, 201)]
        rows = [(i, int(t), p) for i, t, p in rows]
        assert decile_table(rows) == decile_table(list(reversed(rows)))

    def test_empty_input(self):
        assert decile_table([]) == []
        assert expected_calibration_error([]) is None

    def test_rejects_too_few_deciles(self):
        with pytest.raises(ValueError):
            decile_table([(1, 0, 0.5)], n_deciles=1)


@pytest.mark.unit
class TestCalibrationReport:
    def _rows(self):
        # true rate 10%, predicted 0.12 everywhere → gap +0.02
        return [(i, 1 if i % 10 == 0 else 0, 0.12) for i in range(1, 1001)]

    def test_overall_and_ece(self):
        r = build_calibration_report(self._rows(), "sha256:abc", n_folds=5)
        assert r["n_total"] == 1000 and r["n_default"] == 100
        assert r["overall"]["mean_predicted_pd"] == pytest.approx(0.12)
        assert r["overall"]["actual_default_rate"] == pytest.approx(0.10)
        assert r["overall"]["gap"] == pytest.approx(0.02)
        assert r["ece"] == pytest.approx(0.02, abs=1e-6)

    def test_verdict_left_empty_until_real_numbers_exist(self):
        r = build_calibration_report(self._rows(), "sha256:abc", n_folds=5)
        assert r["verdict"] == ""

    def test_carries_fingerprint_and_provenance(self):
        r = build_calibration_report(self._rows(), "sha256:abc", n_folds=5)
        assert r["config_fingerprint"] == "sha256:abc"
        assert r["schema_version"] == SCHEMA_VERSION
        assert "out-of-fold" in r["source"] and "5 folds" in r["source"]

    def test_reports_worst_decile(self):
        rows = [(i, 0, i / 1000) for i in range(1, 1001)]   # all non-default
        r = build_calibration_report(rows, "sha256:abc", n_folds=5)
        assert r["max_abs_gap_decile"] == 10      # highest predicted PD, zero actual
        assert r["max_abs_gap"] > 0

    def test_empty_input_does_not_crash(self):
        r = build_calibration_report([], "sha256:abc", n_folds=0)
        assert r["n_total"] == 0 and r["ece"] is None and r["deciles"] == []


# ──────────────────────────────────────────────────────────────
# cv_report assembly
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBuildCvReport:
    def test_shape_and_progress(self):
        setup = build_setup(5, _cfg())
        folds = [_fold_rec(i, 0.75) for i in (1, 2)]
        r = build_cv_report(setup, folds, [{"fold": 1, "n": 10, "n_positive": 1,
                                            "default_rate": 0.1}])
        assert r["schema_version"] == SCHEMA_VERSION
        assert r["n_folds_total"] == 5 and r["n_folds_completed"] == 2
        assert r["config_fingerprint"] == config_fingerprint(setup)
        assert len(r["fold_balance"]) == 1
        assert set(r["summary"]) == set(METRICS)

    def test_folds_sorted_by_number(self):
        setup = build_setup(5, _cfg())
        r = build_cv_report(setup, [_fold_rec(3, 0.76), _fold_rec(1, 0.74)])
        assert [f["fold"] for f in r["folds"]] == [1, 3]

    def test_note_states_the_two_statistical_caveats(self):
        r = build_cv_report(build_setup(5, _cfg()), [])
        note = r["note"].lower()
        assert "anti-conservative" in note
        assert "share training data" in note
        assert "stratified" in note
        assert "0.7663" in r["note"]

    def test_input_records_not_mutated(self):
        folds = [_fold_rec(1, 0.75)]
        build_cv_report(build_setup(5, _cfg()), folds)
        assert set(folds[0]) == {"fold", "roc_auc", "pr_auc", "ks_statistic"}


@pytest.mark.unit
def test_report_filenames_are_distinct_and_expected():
    assert REPORT_FILENAMES["cv"] == "cv_report.json"
    assert REPORT_FILENAMES["calibration"] == "calibration_report.json"
    assert len(set(REPORT_FILENAMES.values())) == 2


@pytest.mark.unit
def test_cv_study_has_no_spark_dependency():
    """The study module must stay usable without a working pyspark install.

    Checked two ways: the source mentions pyspark nowhere, and importing it
    pulls no pyspark module into sys.modules.
    """
    import pathlib
    import sys

    src = pathlib.Path(credit_risk.cv.study.__file__).read_text(encoding="utf-8")
    assert "pyspark" not in src

    before = {m for m in sys.modules if m.split(".")[0] == "pyspark"}
    importlib.reload(credit_risk.cv.study)
    after = {m for m in sys.modules if m.split(".")[0] == "pyspark"}
    assert after == before, f"reloading cv.study imported pyspark: {after - before}"
