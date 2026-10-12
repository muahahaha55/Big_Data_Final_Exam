"""Unit tests for the ablation study logic — no SparkSession, no Spark import.

Covers the four things that decide whether a multi-hour study is trustworthy:
step enumeration, the comparability fingerprint, resume/skip behaviour, and
the between-step deltas that end up quoted in the README.

Imports `credit_risk.ablation` rather than `pipelines.ablation_pipeline` on
purpose: the pipeline module pulls in pyspark.ml, so importing it here would
make these tests depend on a working Spark install (and on Python 3.12 that
needs a distutils shim). The study logic itself is pure Python.
"""

from __future__ import annotations

import json

import pytest

from credit_risk.ablation import (
    APPLICATION_ONLY,
    DEFAULT_TABLE_ORDER,
    FULL,
    REPORT_FILENAMES,
    SCHEMA_VERSION,
    StepSpec,
    atomic_write_json,
    attach_deltas,
    build_report,
    build_setup,
    build_steps,
    check_fingerprint,
    completed_steps,
    config_fingerprint,
    load_report,
    pending_steps,
    summarize,
)

TABLES = ["bureau", "previous_application", "installments", "credit_card", "pos_cash"]


def _cfg() -> dict:
    """Minimal config shaped like conf/base/config.yaml."""
    return {
        "random_seed": 42,
        "data": {"split": {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}},
        "models": {"gbt": {"maxIter": 100, "maxDepth": 5, "stepSize": 0.1,
                           "subsamplingRate": 0.8}},
        "spark": {"master": "local[1]", "driver_memory": "2g", "shuffle_partitions": 50},
    }


def _rec(step: int, label: str, roc: float, pr: float = 0.2, ks: float = 0.3,
         **kw) -> dict:
    return {"step": step, "label": label, "roc_auc": roc, "pr_auc": pr,
            "ks_statistic": ks, **kw}


# ──────────────────────────────────────────────────────────────
# 1. Step enumeration
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBuildStepsCumulative:
    def test_has_one_step_per_table_plus_baseline(self):
        steps = build_steps("cumulative", TABLES)
        assert len(steps) == len(TABLES) + 1 == 6

    def test_first_step_is_application_only_with_no_tables(self):
        first = build_steps("cumulative", TABLES)[0]
        assert first.step == 1
        assert first.label == APPLICATION_ONLY
        assert first.included_tables == ()
        assert first.added_table is None

    def test_included_tables_are_strict_prefixes_in_order(self):
        steps = build_steps("cumulative", TABLES)
        for i, s in enumerate(steps):
            assert list(s.included_tables) == TABLES[:i], s.label

    def test_each_step_adds_exactly_one_table(self):
        steps = build_steps("cumulative", TABLES)
        for prev, cur in zip(steps, steps[1:]):
            added = set(cur.included_tables) - set(prev.included_tables)
            assert added == {cur.added_table}
            assert len(cur.included_tables) == len(prev.included_tables) + 1

    def test_labels_and_step_numbers(self):
        steps = build_steps("cumulative", TABLES)
        assert [s.step for s in steps] == [1, 2, 3, 4, 5, 6]
        assert [s.label for s in steps] == [
            APPLICATION_ONLY, "+bureau", "+previous_application",
            "+installments", "+credit_card", "+pos_cash",
        ]

    def test_last_step_holds_all_tables(self):
        assert list(build_steps("cumulative", TABLES)[-1].included_tables) == TABLES

    def test_no_removed_table_in_cumulative_mode(self):
        assert all(s.removed_table is None for s in build_steps("cumulative", TABLES))

    def test_single_table_degenerates_cleanly(self):
        steps = build_steps("cumulative", ["bureau"])
        assert len(steps) == 2
        assert steps[1].added_table == "bureau"

    def test_empty_table_list_gives_baseline_only(self):
        steps = build_steps("cumulative", [])
        assert len(steps) == 1 and steps[0].label == APPLICATION_ONLY

    def test_steps_are_immutable(self):
        with pytest.raises(Exception):
            build_steps("cumulative", TABLES)[0].step = 99


@pytest.mark.unit
class TestBuildStepsLoo:
    def test_shape(self):
        steps = build_steps("loo", TABLES)
        assert len(steps) == len(TABLES) + 2 == 7
        assert steps[0].label == APPLICATION_ONLY
        assert steps[1].label == FULL
        assert list(steps[1].included_tables) == TABLES

    def test_each_drop_step_omits_exactly_its_table(self):
        for s in build_steps("loo", TABLES)[2:]:
            assert s.removed_table is not None
            assert s.removed_table not in s.included_tables
            assert len(s.included_tables) == len(TABLES) - 1
            assert set(s.included_tables) == set(TABLES) - {s.removed_table}

    def test_step_numbers_are_contiguous(self):
        assert [s.step for s in build_steps("loo", TABLES)] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.unit
def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        build_steps("sideways", TABLES)


# ──────────────────────────────────────────────────────────────
# 2. Fingerprint
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFingerprint:
    def test_is_stable_across_calls(self):
        s = build_setup("cumulative", TABLES, _cfg())
        assert config_fingerprint(s) == config_fingerprint(s)

    def test_insensitive_to_key_order(self):
        s = build_setup("cumulative", TABLES, _cfg())
        shuffled = dict(reversed(list(s.items())))
        assert config_fingerprint(shuffled) == config_fingerprint(s)

    def test_ignores_cosmetic_fields(self):
        """A changed prose note must not invalidate finished steps."""
        s = build_setup("cumulative", TABLES, _cfg())
        base = config_fingerprint(s)
        s["not_comparable_to"] = "reworded explanation"
        assert config_fingerprint(s) == base

    @pytest.mark.parametrize("mutate", [
        pytest.param(lambda c: c["models"]["gbt"].__setitem__("maxDepth", 6), id="maxDepth"),
        pytest.param(lambda c: c["models"]["gbt"].__setitem__("maxIter", 50), id="maxIter"),
        pytest.param(lambda c: c.__setitem__("random_seed", 7), id="seed"),
        pytest.param(lambda c: c["data"]["split"].__setitem__("test_ratio", 0.2), id="split"),
        pytest.param(lambda c: c["spark"].__setitem__("master", "local[*]"), id="master"),
        pytest.param(lambda c: c["spark"].__setitem__("shuffle_partitions", 8),
                     id="shuffle_partitions"),
    ])
    def test_changes_when_comparability_critical_setting_changes(self, mutate):
        base = config_fingerprint(build_setup("cumulative", TABLES, _cfg()))
        cfg = _cfg()
        mutate(cfg)
        assert config_fingerprint(build_setup("cumulative", TABLES, cfg)) != base

    def test_changes_with_mode(self):
        cfg = _cfg()
        assert (config_fingerprint(build_setup("cumulative", TABLES, cfg))
                != config_fingerprint(build_setup("loo", TABLES, cfg)))

    def test_changes_with_table_order(self):
        cfg = _cfg()
        base = config_fingerprint(build_setup("cumulative", TABLES, cfg))
        reordered = [TABLES[1], TABLES[0], *TABLES[2:]]
        assert config_fingerprint(build_setup("cumulative", reordered, cfg)) != base

    def test_train_ratio_folds_in_the_val_split(self):
        s = build_setup("cumulative", TABLES, _cfg())
        assert s["split"]["train_ratio"] == pytest.approx(0.85)
        assert s["split"]["test_ratio"] == pytest.approx(0.15)

    def test_looks_like_a_sha256(self):
        fp = config_fingerprint(build_setup("cumulative", TABLES, _cfg()))
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 64


@pytest.mark.unit
class TestCheckFingerprint:
    def test_passes_on_match(self):
        check_fingerprint({"config_fingerprint": "sha256:abc"}, "sha256:abc")

    def test_passes_when_no_report_yet(self):
        check_fingerprint(None, "sha256:abc")

    def test_passes_when_report_has_no_fingerprint(self):
        """An older report without the field should not block work."""
        check_fingerprint({"steps": []}, "sha256:abc")

    def test_refuses_on_mismatch_and_names_both(self):
        with pytest.raises(RuntimeError, match="Refusing to resume") as e:
            check_fingerprint({"config_fingerprint": "sha256:old"}, "sha256:new")
        assert "sha256:old" in str(e.value) and "sha256:new" in str(e.value)


# ──────────────────────────────────────────────────────────────
# 3. Resume / skip
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestResume:
    def test_completed_steps_on_empty_inputs(self):
        assert completed_steps(None) == {}
        assert completed_steps({}) == {}
        assert completed_steps({"steps": []}) == {}

    def test_completed_steps_indexes_by_number(self):
        report = {"steps": [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, "+bureau", 0.72)]}
        done = completed_steps(report)
        assert set(done) == {1, 2}
        assert done[2]["label"] == "+bureau"

    def test_completed_steps_tolerates_malformed_entries(self):
        report = {"steps": [_rec(1, APPLICATION_ONLY, 0.70), {"no_step": True}, None]}
        assert set(completed_steps(report)) == {1}

    def test_completed_steps_coerces_string_numbers(self):
        """JSON round-trips can hand back strings; indexing must still work."""
        assert set(completed_steps({"steps": [{"step": "3"}]})) == {3}

    def test_all_pending_when_nothing_done(self):
        steps = build_steps("cumulative", TABLES)
        assert [s.step for s in pending_steps(steps, None)] == [1, 2, 3, 4, 5, 6]

    def test_finished_steps_are_skipped(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, "+bureau", 0.72)]}
        assert [s.step for s in pending_steps(steps, report)] == [3, 4, 5, 6]

    def test_resume_after_a_mid_study_crash(self):
        """Steps 1-3 done, 4 died: only 4,5,6 should be queued."""
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(i, f"s{i}", 0.7) for i in (1, 2, 3)]}
        assert [s.step for s in pending_steps(steps, report)] == [4, 5, 6]

    def test_nothing_pending_when_study_complete(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(i, f"s{i}", 0.7) for i in range(1, 7)]}
        assert pending_steps(steps, report) == []

    def test_only_step_narrows_to_one(self):
        steps = build_steps("cumulative", TABLES)
        got = pending_steps(steps, None, only_step=4)
        assert len(got) == 1 and got[0].step == 4

    def test_only_step_yields_nothing_if_already_done(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(4, "+installments", 0.75)]}
        assert pending_steps(steps, report, only_step=4) == []

    def test_force_reruns_everything(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(i, f"s{i}", 0.7) for i in range(1, 7)]}
        assert len(pending_steps(steps, report, force=True)) == 6

    def test_force_combines_with_only_step(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(2, "+bureau", 0.72)]}
        got = pending_steps(steps, report, only_step=2, force=True)
        assert len(got) == 1 and got[0].step == 2

    def test_out_of_order_completions_are_handled(self):
        steps = build_steps("cumulative", TABLES)
        report = {"steps": [_rec(6, "+pos_cash", 0.77), _rec(1, APPLICATION_ONLY, 0.70)]}
        assert [s.step for s in pending_steps(steps, report)] == [2, 3, 4, 5]


# ──────────────────────────────────────────────────────────────
# 4. Deltas between steps
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCumulativeDeltas:
    def test_baseline_has_no_delta(self):
        out = attach_deltas([_rec(1, APPLICATION_ONLY, 0.7000)], "cumulative")
        assert out[0]["delta_roc_auc_vs_prev"] is None

    def test_delta_is_against_the_previous_step(self):
        recs = [
            _rec(1, APPLICATION_ONLY, 0.7000, pr=0.2000, ks=0.3000),
            _rec(2, "+bureau", 0.7250, pr=0.2200, ks=0.3300),
            _rec(3, "+previous_application", 0.7300, pr=0.2250, ks=0.3350),
        ]
        out = attach_deltas(recs, "cumulative")
        assert out[1]["delta_roc_auc_vs_prev"] == pytest.approx(0.0250)
        assert out[1]["delta_pr_auc_vs_prev"] == pytest.approx(0.0200)
        assert out[1]["delta_ks_vs_prev"] == pytest.approx(0.0300)
        assert out[2]["delta_roc_auc_vs_prev"] == pytest.approx(0.0050)

    def test_negative_delta_is_reported_not_clamped(self):
        """A table that makes things worse must show a negative number."""
        out = attach_deltas(
            [_rec(1, APPLICATION_ONLY, 0.7400), _rec(2, "+bureau", 0.7350)],
            "cumulative",
        )
        assert out[1]["delta_roc_auc_vs_prev"] == pytest.approx(-0.0050)

    def test_zero_delta_is_zero_not_none(self):
        """'This table contributed nothing' is a finding, not missing data."""
        out = attach_deltas(
            [_rec(1, APPLICATION_ONLY, 0.7400), _rec(2, "+bureau", 0.7400)],
            "cumulative",
        )
        assert out[1]["delta_roc_auc_vs_prev"] == 0.0

    def test_missing_neighbour_gives_none(self):
        """Partial resume: step 4 without step 3 cannot have a marginal delta."""
        out = attach_deltas(
            [_rec(1, APPLICATION_ONLY, 0.70), _rec(4, "+installments", 0.76)],
            "cumulative",
        )
        assert {s["step"]: s["delta_roc_auc_vs_prev"] for s in out} == {1: None, 4: None}

    def test_none_metric_propagates_as_none(self):
        recs = [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, "+bureau", 0.72)]
        recs[1]["ks_statistic"] = None
        out = attach_deltas(recs, "cumulative")
        assert out[1]["delta_ks_vs_prev"] is None
        assert out[1]["delta_roc_auc_vs_prev"] == pytest.approx(0.02)

    def test_input_records_are_not_mutated(self):
        recs = [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, "+bureau", 0.72)]
        attach_deltas(recs, "cumulative")
        assert "delta_roc_auc_vs_prev" not in recs[1]

    def test_output_is_sorted_by_step(self):
        recs = [_rec(3, "c", 0.74), _rec(1, "a", 0.70), _rec(2, "b", 0.72)]
        assert [s["step"] for s in attach_deltas(recs, "cumulative")] == [1, 2, 3]


@pytest.mark.unit
class TestLooDeltas:
    def test_drop_steps_measure_the_fall_from_full(self):
        recs = [
            _rec(1, APPLICATION_ONLY, 0.7000),
            _rec(2, FULL, 0.7700),
            _rec(3, "drop_bureau", 0.7650, removed_table="bureau"),
            _rec(4, "drop_installments", 0.7500, removed_table="installments"),
        ]
        out = {s["label"]: s for s in attach_deltas(recs, "loo")}
        assert out["drop_bureau"]["delta_roc_auc_vs_full"] == pytest.approx(0.0050)
        assert out["drop_installments"]["delta_roc_auc_vs_full"] == pytest.approx(0.0200)

    def test_baselines_get_no_delta(self):
        recs = [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, FULL, 0.77)]
        out = {s["label"]: s for s in attach_deltas(recs, "loo")}
        assert out[APPLICATION_ONLY]["delta_roc_auc_vs_full"] is None
        assert out[FULL]["delta_roc_auc_vs_full"] is None

    def test_no_full_baseline_yet_means_no_deltas(self):
        recs = [_rec(3, "drop_bureau", 0.765, removed_table="bureau")]
        assert attach_deltas(recs, "loo")[0]["delta_roc_auc_vs_full"] is None


# ──────────────────────────────────────────────────────────────
# Summary + report assembly
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSummarize:
    def _full_run(self):
        rocs = [0.7000, 0.7250, 0.7300, 0.7600, 0.7610, 0.7700]
        labels = build_steps("cumulative", TABLES)
        recs = [
            _rec(s.step, s.label, roc, added_table=s.added_table)
            for s, roc in zip(labels, rocs)
        ]
        return attach_deltas(recs, "cumulative")

    def test_total_lift_spans_first_to_last(self):
        out = summarize(self._full_run(), "cumulative")
        assert out["total_lift_vs_application_only"]["roc_auc"] == pytest.approx(0.07)
        assert out["total_lift_vs_application_only"]["from"] == APPLICATION_ONLY
        assert out["total_lift_vs_application_only"]["to"] == "+pos_cash"

    def test_ranking_is_by_marginal_gain_descending(self):
        out = summarize(self._full_run(), "cumulative")
        tables = [r["table"] for r in out["marginal_ranking_by_delta_roc_auc"]]
        assert tables[0] == "installments"      # +0.0300, the biggest jump
        assert tables[-1] == "credit_card"      # +0.0010, the smallest
        deltas = [r["delta_roc_auc"] for r in out["marginal_ranking_by_delta_roc_auc"]]
        assert deltas == sorted(deltas, reverse=True)

    def test_empty_study_summarizes_to_nothing(self):
        assert summarize([], "cumulative") == {}

    def test_single_step_has_no_lift(self):
        out = summarize(attach_deltas([_rec(1, APPLICATION_ONLY, 0.7)], "cumulative"),
                        "cumulative")
        assert out["total_lift_vs_application_only"] == {}

    def test_loo_ranking_is_by_drop_descending(self):
        recs = attach_deltas([
            _rec(2, FULL, 0.7700),
            _rec(3, "drop_bureau", 0.7650, removed_table="bureau"),
            _rec(4, "drop_installments", 0.7500, removed_table="installments"),
        ], "loo")
        ranking = summarize(recs, "loo")["importance_ranking_by_roc_auc_drop"]
        assert [r["table"] for r in ranking] == ["installments", "bureau"]


@pytest.mark.unit
class TestBuildReport:
    def test_carries_schema_fingerprint_and_progress(self):
        setup = build_setup("cumulative", TABLES, _cfg())
        report = build_report("cumulative", setup, [_rec(1, APPLICATION_ONLY, 0.70)])
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["mode"] == "cumulative"
        assert report["config_fingerprint"] == config_fingerprint(setup)
        assert report["n_steps_total"] == 6
        assert report["n_steps_completed"] == 1
        assert report["table_order"] == TABLES

    def test_note_flags_bureau_balance_and_split_difference(self):
        setup = build_setup("cumulative", TABLES, _cfg())
        note = build_report("cumulative", setup, [])["note"]
        assert "bureau_balance" in note
        assert "order" in note.lower()
        assert "README" in note

    def test_partial_report_is_still_well_formed(self):
        setup = build_setup("cumulative", TABLES, _cfg())
        report = build_report("cumulative", setup,
                              [_rec(1, APPLICATION_ONLY, 0.70), _rec(2, "+bureau", 0.72)])
        assert report["n_steps_completed"] == 2
        assert report["steps"][1]["delta_roc_auc_vs_prev"] == pytest.approx(0.02)


# ──────────────────────────────────────────────────────────────
# Atomic write / read-back
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAtomicWrite:
    def test_round_trips(self, tmp_path):
        p = tmp_path / "ablation_report.json"
        setup = build_setup("cumulative", TABLES, _cfg())
        report = build_report("cumulative", setup, [_rec(1, APPLICATION_ONLY, 0.70)])
        atomic_write_json(p, report)
        assert load_report(p)["config_fingerprint"] == report["config_fingerprint"]

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "r.json"
        atomic_write_json(p, {"ok": True})
        assert json.loads(p.read_text(encoding="utf-8")) == {"ok": True}

    def test_leaves_no_temp_file_behind(self, tmp_path):
        p = tmp_path / "r.json"
        atomic_write_json(p, {"ok": True})
        assert list(tmp_path.iterdir()) == [p]

    def test_overwrite_replaces_content(self, tmp_path):
        p = tmp_path / "r.json"
        atomic_write_json(p, {"v": 1})
        atomic_write_json(p, {"v": 2})
        assert load_report(p) == {"v": 2}

    def test_missing_file_reads_as_none(self, tmp_path):
        assert load_report(tmp_path / "nope.json") is None

    def test_corrupt_file_reads_as_none_rather_than_raising(self, tmp_path):
        """A truncated report must not crash a resume."""
        p = tmp_path / "r.json"
        p.write_text('{"steps": [', encoding="utf-8")
        assert load_report(p) is None

    def test_checkpoint_then_resume_cycle(self, tmp_path):
        """Write step 1, resume, append step 2 — the shape ablate() relies on."""
        p = tmp_path / "r.json"
        setup = build_setup("cumulative", TABLES, _cfg())
        steps = build_steps("cumulative", TABLES)

        atomic_write_json(p, build_report("cumulative", setup,
                                          [_rec(1, APPLICATION_ONLY, 0.70)]))
        reloaded = load_report(p)
        check_fingerprint(reloaded, config_fingerprint(setup))
        assert [s.step for s in pending_steps(steps, reloaded)] == [2, 3, 4, 5, 6]

        records = list(completed_steps(reloaded).values()) + [_rec(2, "+bureau", 0.72)]
        atomic_write_json(p, build_report("cumulative", setup, records))
        final = load_report(p)
        assert final["n_steps_completed"] == 2
        assert final["steps"][1]["delta_roc_auc_vs_prev"] == pytest.approx(0.02)


# ──────────────────────────────────────────────────────────────
# Guard: the real table order must match what PLAN T1.2 specifies
# ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_declared_table_order_matches_plan():
    """PLAN T1.2 fixes this order; the pipeline asserts its aggregator map agrees."""
    assert DEFAULT_TABLE_ORDER == TABLES


@pytest.mark.unit
def test_modes_write_to_separate_files():
    """LOO must never overwrite the cumulative report."""
    assert REPORT_FILENAMES["cumulative"] == "ablation_report.json"
    assert REPORT_FILENAMES["loo"] != REPORT_FILENAMES["cumulative"]
    assert len(set(REPORT_FILENAMES.values())) == len(REPORT_FILENAMES)


@pytest.mark.unit
def test_pipeline_aggregator_order_agrees_when_spark_is_available():
    """The pipeline's assert is the real guard; verify it here when importable."""
    pytest.importorskip("pyspark.ml", reason="pyspark.ml needs distutils on py3.12")
    from pipelines.ablation_pipeline import CHILD_TABLES
    assert list(CHILD_TABLES.keys()) == DEFAULT_TABLE_ORDER
