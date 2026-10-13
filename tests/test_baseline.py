"""Unit tests for the P0.3 baseline study logic — no Spark import.

The arithmetic that matters here is the comparison: sign convention of the
deltas, where the reference numbers come from, and the guard that refuses to
compare against an ablation run on a different split. Getting the sign wrong
would invert the project's headline conclusion, so it is pinned hard.

Imports `credit_risk.baseline`, not `pipelines.baseline_pipeline`, so these
stay runnable without a working pyspark.ml.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import credit_risk.baseline.study
from credit_risk.baseline import (
    DEFAULT_VARIANTS,
    EXT_SOURCES,
    METRICS,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    VALID_MODELS,
    VariantSpec,
    attach_deltas,
    build_report,
    build_setup,
    build_variants,
    check_split_match,
    completed_variants,
    config_fingerprint,
    load_references,
    pending_variants,
    summarize,
)

# The real numbers from data/08_reporting/ablation_report.json, same split.
ABLATION_APP_ONLY = {"label": "application_only", "step": 1, "n_source_columns": 85,
                     "test_rows": 45820, "roc_auc": 0.7546, "pr_auc": 0.2356,
                     "ks_statistic": 0.3775}
ABLATION_FULL = {"label": "+pos_cash", "step": 6, "n_source_columns": 112,
                 "test_rows": 45820, "roc_auc": 0.7652, "pr_auc": 0.2505,
                 "ks_statistic": 0.3971}


def _cfg(variants=None) -> dict:
    cfg = {
        "random_seed": 42,
        "data": {"target_column": "TARGET", "id_column": "SK_ID_CURR"},
        "features": {"missing_threshold": 0.5},
        "models": {"gbt": {"maxIter": 100, "maxDepth": 5},
                   "logistic": {"maxIter": 100, "regParam": 0.01}},
        "spark": {"master": "local[1]", "driver_memory": "2g", "shuffle_partitions": 50},
    }
    if variants is not None:
        cfg["baseline"] = {"variants": variants}
    return cfg


def _refs() -> dict:
    return load_references({"steps": [ABLATION_APP_ONLY, ABLATION_FULL]})


def _rec(name: str, roc: float, pr: float = 0.23, ks: float = 0.37, **kw) -> dict:
    return {"name": name, "roc_auc": roc, "pr_auc": pr, "ks_statistic": ks, **kw}


# ──────────────────────────────────────────────────────────────
# Variant specs
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestVariantSpec:
    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="unknown model_type"):
            VariantSpec("x", "randomforest_maybe", ("A",))

    def test_rejects_empty_feature_list(self):
        with pytest.raises(ValueError, match="no features"):
            VariantSpec("x", "logistic", ())

    def test_is_immutable(self):
        with pytest.raises(Exception):
            DEFAULT_VARIANTS[0].name = "nope"

    def test_default_set_decomposes_features_vs_model(self):
        """Needs both a logistic and a GBT on the same features to separate them."""
        by_name = {v.name: v for v in DEFAULT_VARIANTS}
        assert by_name["logit_ext_all3"].features == by_name["gbt_ext_all3"].features
        assert by_name["logit_ext_all3"].model_type != by_name["gbt_ext_all3"].model_type

    def test_default_set_has_a_no_training_variant(self):
        assert any(v.model_type == "rule" for v in DEFAULT_VARIANTS)

    def test_default_set_contrasts_the_missing_threshold_choice(self):
        by_name = {v.name: v for v in DEFAULT_VARIANTS}
        assert "EXT_SOURCE_1" not in by_name["logit_ext2_ext3"].features
        assert "EXT_SOURCE_1" in by_name["logit_ext_all3"].features

    def test_all_default_models_are_valid(self):
        assert all(v.model_type in VALID_MODELS for v in DEFAULT_VARIANTS)

    def test_default_names_unique(self):
        names = [v.name for v in DEFAULT_VARIANTS]
        assert len(set(names)) == len(names)


@pytest.mark.unit
class TestBuildVariants:
    def test_falls_back_to_defaults(self):
        assert build_variants({}) == list(DEFAULT_VARIANTS)
        assert build_variants(_cfg()) == list(DEFAULT_VARIANTS)

    def test_reads_config(self):
        out = build_variants(_cfg([
            {"name": "only_ext2", "model": "logistic", "features": ["EXT_SOURCE_2"]},
        ]))
        assert len(out) == 1
        assert out[0].name == "only_ext2" and out[0].features == ("EXT_SOURCE_2",)

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="duplicate variant names"):
            build_variants(_cfg([
                {"name": "dup", "model": "rule", "features": ["EXT_SOURCE_2"]},
                {"name": "dup", "model": "gbt", "features": ["EXT_SOURCE_3"]},
            ]))

    def test_propagates_bad_model_from_config(self):
        with pytest.raises(ValueError, match="unknown model_type"):
            build_variants(_cfg([
                {"name": "x", "model": "svm", "features": ["EXT_SOURCE_2"]},
            ]))

    def test_config_matching_defaults_round_trips(self):
        cfg = _cfg([{"name": v.name, "model": v.model_type,
                     "features": list(v.features), "description": v.description}
                    for v in DEFAULT_VARIANTS])
        assert build_variants(cfg) == list(DEFAULT_VARIANTS)


# ──────────────────────────────────────────────────────────────
# References pulled from the ablation report
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLoadReferences:
    def test_picks_application_only_and_the_last_step(self):
        refs = _refs()
        assert set(refs) == {"ablation_application_only", "ablation_full"}
        assert refs["ablation_application_only"]["roc_auc"] == 0.7546
        assert refs["ablation_full"]["roc_auc"] == 0.7652
        assert refs["ablation_full"]["label"] == "+pos_cash"

    def test_last_step_is_by_step_number_not_list_order(self):
        refs = load_references({"steps": [ABLATION_FULL, ABLATION_APP_ONLY]})
        assert refs["ablation_full"]["label"] == "+pos_cash"

    def test_carries_the_columns_needed_for_the_comparison(self):
        r = _refs()["ablation_full"]
        assert r["n_source_columns"] == 112 and r["test_rows"] == 45820
        assert all(m in r for m in METRICS)

    def test_missing_report_gives_no_references(self):
        """No ablation report must mean no comparison, not an invented one."""
        assert load_references(None) == {}
        assert load_references({}) == {}
        assert load_references({"steps": []}) == {}

    def test_single_step_report_yields_only_that_one(self):
        refs = load_references({"steps": [ABLATION_APP_ONLY]})
        assert set(refs) == {"ablation_application_only"}

    def test_matches_the_real_report_on_disk(self):
        """Guard against the report's schema drifting away from this reader."""
        path = pathlib.Path("data/08_reporting/ablation_report.json")
        if not path.exists():
            pytest.skip("ablation_report.json not present")
        refs = load_references(json.loads(path.read_text(encoding="utf-8")))
        assert refs["ablation_application_only"]["roc_auc"] == 0.7546
        assert refs["ablation_full"]["roc_auc"] == 0.7652
        assert refs["ablation_full"]["test_rows"] == 45820


@pytest.mark.unit
class TestCheckSplitMatch:
    def test_passes_when_row_counts_agree(self):
        check_split_match(45820, _refs())

    def test_refuses_when_the_split_was_regenerated(self):
        with pytest.raises(RuntimeError, match="Refusing to compare"):
            check_split_match(45_900, _refs())

    def test_message_names_both_counts(self):
        with pytest.raises(RuntimeError) as e:
            check_split_match(40_000, _refs())
        assert "40000" in str(e.value) and "45820" in str(e.value)

    def test_no_references_means_nothing_to_check(self):
        check_split_match(12_345, {})

    def test_tolerates_a_reference_without_row_count(self):
        check_split_match(100, {"r": {"label": "x", "roc_auc": 0.7}})


# ──────────────────────────────────────────────────────────────
# Deltas — the sign convention is the headline
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAttachDeltas:
    def test_positive_delta_means_the_full_pipeline_is_ahead(self):
        """reference − variant. Inverting this inverts the conclusion."""
        out = attach_deltas([_rec("logit_ext_all3", 0.7400)], _refs())
        gap = out[0]["gap_to_reference"]["ablation_full"]
        assert gap["delta_roc_auc"] == pytest.approx(0.0252)   # 0.7652 − 0.7400

    def test_negative_delta_when_the_baseline_wins(self):
        out = attach_deltas([_rec("cheeky", 0.7700)], _refs())
        gap = out[0]["gap_to_reference"]["ablation_full"]
        assert gap["delta_roc_auc"] == pytest.approx(-0.0048)

    def test_gap_computed_against_every_reference(self):
        out = attach_deltas([_rec("v", 0.7400)], _refs())
        gaps = out[0]["gap_to_reference"]
        assert set(gaps) == {"ablation_application_only", "ablation_full"}
        assert gaps["ablation_application_only"]["delta_roc_auc"] == pytest.approx(0.0146)

    def test_covers_all_three_metrics(self):
        out = attach_deltas([_rec("v", 0.7400, pr=0.2300, ks=0.3700)], _refs())
        gap = out[0]["gap_to_reference"]["ablation_full"]
        assert set(gap) == {"delta_roc_auc", "delta_pr_auc", "delta_ks"}
        assert gap["delta_pr_auc"] == pytest.approx(0.0205)
        assert gap["delta_ks"] == pytest.approx(0.0271)

    def test_none_metric_gives_none_delta(self):
        rec = _rec("v", 0.74)
        rec["ks_statistic"] = None
        out = attach_deltas([rec], _refs())
        assert out[0]["gap_to_reference"]["ablation_full"]["delta_ks"] is None

    def test_no_references_gives_empty_gaps(self):
        out = attach_deltas([_rec("v", 0.74)], {})
        assert out[0]["gap_to_reference"] == {}

    def test_inputs_not_mutated(self):
        recs = [_rec("v", 0.74)]
        attach_deltas(recs, _refs())
        assert "gap_to_reference" not in recs[0]

    def test_output_sorted_by_name(self):
        out = attach_deltas([_rec("z", 0.74), _rec("a", 0.75)], _refs())
        assert [r["name"] for r in out] == ["a", "z"]


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSummarize:
    def _records(self):
        return attach_deltas([
            _rec("rule_ext_mean", 0.7210, model="rule", n_features=3),
            _rec("logit_ext2_ext3", 0.7350, model="logistic", n_features=2),
            _rec("logit_ext_all3", 0.7420, model="logistic", n_features=3),
            _rec("gbt_ext_all3", 0.7455, model="gbt", n_features=3),
        ], _refs())

    def test_picks_the_strongest_baseline(self):
        out = summarize(self._records(), _refs())
        assert out["best_baseline"]["name"] == "gbt_ext_all3"
        assert out["best_baseline"]["roc_auc"] == 0.7455

    def test_ranking_is_descending(self):
        rocs = [r["roc_auc"] for r in summarize(self._records(), _refs())["ranking_by_roc_auc"]]
        assert rocs == sorted(rocs, reverse=True)

    def test_headline_advantage_against_the_best_baseline(self):
        out = summarize(self._records(), _refs())
        adv = out["full_pipeline_advantage_over_best_baseline"]
        assert adv["delta_roc_auc"] == pytest.approx(0.0197)   # 0.7652 − 0.7455
        assert adv["reference_n_source_columns"] == 112
        assert adv["baseline_n_features"] == 3

    def test_no_advantage_block_without_a_full_reference(self):
        out = summarize(self._records(), {"ablation_application_only":
                                          _refs()["ablation_application_only"]})
        assert "full_pipeline_advantage_over_best_baseline" not in out

    def test_empty_records(self):
        assert summarize([], _refs()) == {}

    def test_skips_variants_without_a_score(self):
        recs = attach_deltas([_rec("ok", 0.74, n_features=3),
                              {"name": "failed", "roc_auc": None}], _refs())
        out = summarize(recs, _refs())
        assert out["best_baseline"]["name"] == "ok"
        assert len(out["ranking_by_roc_auc"]) == 1


# ──────────────────────────────────────────────────────────────
# Fingerprint + resume
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFingerprint:
    def test_stable_and_order_independent(self):
        s = build_setup(_cfg(), DEFAULT_VARIANTS)
        assert config_fingerprint(s) == config_fingerprint(s)
        assert config_fingerprint(dict(reversed(list(s.items())))) == config_fingerprint(s)

    def test_ignores_observed_split_counts_and_prose(self):
        s = build_setup(_cfg(), DEFAULT_VARIANTS)
        base = config_fingerprint(s)
        s["split_observed"] = {"train_rows": 261691, "test_rows": 45820}
        s["comparable_to"] = "reworded"
        assert config_fingerprint(s) == base

    def test_changes_when_the_variant_set_changes(self):
        base = config_fingerprint(build_setup(_cfg(), DEFAULT_VARIANTS))
        fewer = config_fingerprint(build_setup(_cfg(), DEFAULT_VARIANTS[:2]))
        assert fewer != base

    def test_changes_when_features_change(self):
        a = config_fingerprint(build_setup(_cfg(), [
            VariantSpec("v", "logistic", ("EXT_SOURCE_2",))]))
        b = config_fingerprint(build_setup(_cfg(), [
            VariantSpec("v", "logistic", ("EXT_SOURCE_3",))]))
        assert a != b

    @pytest.mark.parametrize("mutate", [
        pytest.param(lambda c: c.__setitem__("random_seed", 7), id="seed"),
        pytest.param(lambda c: c["models"]["gbt"].__setitem__("maxDepth", 9), id="gbt_depth"),
        pytest.param(lambda c: c["models"]["logistic"].__setitem__("regParam", 0.5),
                     id="logit_regparam"),
        pytest.param(lambda c: c["spark"].__setitem__("master", "local[*]"), id="master"),
    ])
    def test_changes_on_comparability_critical_settings(self, mutate):
        cfg = _cfg()
        base = config_fingerprint(build_setup(cfg, DEFAULT_VARIANTS))
        mutate(cfg)
        assert config_fingerprint(build_setup(cfg, DEFAULT_VARIANTS)) != base

    def test_records_that_weighting_and_tuning_stay_off(self):
        s = build_setup(_cfg(), DEFAULT_VARIANTS)
        assert s["class_weighting"] is False and s["tune"] is False


@pytest.mark.unit
class TestResume:
    def test_all_pending_initially(self):
        assert pending_variants(DEFAULT_VARIANTS, None) == list(DEFAULT_VARIANTS)

    def test_finished_variants_skipped(self):
        report = {"variants": [_rec("rule_ext_mean", 0.72), _rec("logit_ext_all3", 0.74)]}
        names = [v.name for v in pending_variants(DEFAULT_VARIANTS, report)]
        assert names == ["logit_ext2_ext3", "gbt_ext_all3"]

    def test_only_variant_narrows(self):
        got = pending_variants(DEFAULT_VARIANTS, None, only_variant="gbt_ext_all3")
        assert [v.name for v in got] == ["gbt_ext_all3"]

    def test_only_variant_already_done(self):
        report = {"variants": [_rec("gbt_ext_all3", 0.74)]}
        assert pending_variants(DEFAULT_VARIANTS, report, only_variant="gbt_ext_all3") == []

    def test_force_reruns_all(self):
        report = {"variants": [_rec(v.name, 0.74) for v in DEFAULT_VARIANTS]}
        assert len(pending_variants(DEFAULT_VARIANTS, report, force=True)) == 4

    def test_completed_variants_keyed_by_name_not_int(self):
        """Baseline ids are strings — int coercion would drop every record."""
        report = {"variants": [_rec("rule_ext_mean", 0.72)]}
        assert set(completed_variants(report)) == {"rule_ext_mean"}

    def test_completed_variants_tolerates_junk(self):
        report = {"variants": [_rec("ok", 0.7), {"no_name": 1}, None]}
        assert set(completed_variants(report)) == {"ok"}

    def test_nothing_pending_when_done(self):
        report = {"variants": [_rec(v.name, 0.74) for v in DEFAULT_VARIANTS]}
        assert pending_variants(DEFAULT_VARIANTS, report) == []


# ──────────────────────────────────────────────────────────────
# Report assembly
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBuildReport:
    def test_shape_and_progress(self):
        setup = build_setup(_cfg(), DEFAULT_VARIANTS)
        r = build_report(setup, [_rec("rule_ext_mean", 0.72, n_features=3)], _refs())
        assert r["schema_version"] == SCHEMA_VERSION
        assert r["n_variants_total"] == 4 and r["n_variants_completed"] == 1
        assert r["config_fingerprint"] == config_fingerprint(setup)
        assert set(r["references"]) == {"ablation_application_only", "ablation_full"}

    def test_note_flags_the_rule_is_not_calibrated_and_ext1_is_null_heavy(self):
        r = build_report(build_setup(_cfg(), DEFAULT_VARIANTS), [], _refs())
        note = r["note"]
        assert "not a calibrated PD" in note
        assert "EXT_SOURCE_1" in note and "null" in note
        assert "45,820" in note

    def test_partial_report_still_well_formed(self):
        setup = build_setup(_cfg(), DEFAULT_VARIANTS)
        r = build_report(setup, [_rec("a", 0.72, n_features=3),
                                 _rec("b", 0.74, n_features=3)], _refs())
        assert r["n_variants_completed"] == 2
        assert r["summary"]["best_baseline"]["name"] == "b"

    def test_report_filename(self):
        assert REPORT_FILENAME == "baseline_report.json"


@pytest.mark.unit
def test_study_module_has_no_spark_dependency():
    src = pathlib.Path(credit_risk.baseline.study.__file__).read_text(encoding="utf-8")
    assert "pyspark" not in src


@pytest.mark.unit
def test_ext_sources_constant_matches_the_dataset():
    assert EXT_SOURCES == ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
