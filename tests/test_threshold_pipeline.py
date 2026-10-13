"""End-to-end tests for the P1.1 pipeline — pandas on temp Parquet, no Spark.

`pipelines/threshold_pipeline.py` imports no Spark, so the join and the EAD
checks are executed here rather than first exercised on the VM. EAD multiplies
every cost in the study, so the guard that refuses to run on a null or
non-positive exposure is the one that matters most: silently dropping those
rows would change the population every threshold is computed on.

Runs against the real `conf/base/config.yaml`, so the configured margin rates
and EAD column must agree with what the pipeline reads.
"""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pipelines.threshold_pipeline as tp

from credit_risk.config import get_config

ID = "SK_ID_CURR"
TARGET = "TARGET"
N_PER_FOLD = 400


def _oof_frame(fold: int, ids: range) -> "pd.DataFrame":
    rows = []
    for i in ids:
        pd_score = ((i * 7919) % 1000) / 1000.0 * 0.6
        y = 1 if ((i * 31) % 1000) / 1000.0 < pd_score else 0
        rows.append((i, y, pd_score, fold))
    return pd.DataFrame(rows, columns=[ID, TARGET, "pd_score", "fold"])


def _attr_frame(ids: range, ead=lambda i: 100000.0 + (i % 11) * 10000.0):
    return pd.DataFrame([(i, ead(i)) for i in ids], columns=[ID, "AMT_CREDIT"])


@pytest.fixture
def world(tmp_path, monkeypatch):
    oof_dir = tmp_path / "cv_oof"
    model_input = tmp_path / "05_model_input"
    reporting = tmp_path / "08_reporting"
    for d in (oof_dir, model_input, reporting):
        d.mkdir(parents=True)

    folds = {f: range(1 + (f - 1) * N_PER_FOLD, 1 + f * N_PER_FOLD) for f in (1, 2, 3)}
    for f, ids in folds.items():
        _oof_frame(f, ids).to_parquet(oof_dir / f"fold_{f}.parquet", index=False)

    all_ids = [i for ids in folds.values() for i in ids]
    half = len(all_ids) // 2
    _attr_frame(range(1, half + 1)).to_parquet(model_input / "train.parquet", index=False)
    _attr_frame(range(half + 1, len(all_ids) + 1)).to_parquet(
        model_input / "val.parquet", index=False)

    cv_report = {
        "config_fingerprint": "sha256:deadbeef",
        "n_folds_total": 3,
        "n_folds_completed": 3,
        "folds": [{"fold": f, "n_valid": N_PER_FOLD} for f in folds],
        "summary": {"roc_auc": {"mean": 0.80}},
    }
    (reporting / "cv_report.json").write_text(json.dumps(cv_report), encoding="utf-8")

    monkeypatch.setattr(tp, "_OOF_DIR", oof_dir)
    monkeypatch.setattr(tp, "_MODEL_INPUT", model_input)
    monkeypatch.setattr(tp, "_REPORT_DIR", reporting)
    monkeypatch.setattr(tp, "_CV_REPORT", reporting / "cv_report.json")
    monkeypatch.setattr(tp, "_THRESHOLD_REPORT", reporting / "threshold_report.json")

    return {"oof_dir": oof_dir, "model_input": model_input, "reporting": reporting,
            "cv_report": cv_report, "n_rows": len(all_ids)}


# ──────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────

def test_runs_end_to_end_and_writes_the_report(world):
    report = tp.run_threshold()
    written = json.loads(
        (world["reporting"] / "threshold_report.json").read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(report))

    assert report["population"]["n"] == world["n_rows"]
    assert report["cv_config_fingerprint"] == "sha256:deadbeef"
    assert report["setup"]["source"]["cv_folds_used"] == [1, 2, 3]
    assert report["setup"]["source"]["oof_rows"] == world["n_rows"]


def test_one_scenario_per_configured_margin_rate(world):
    configured = get_config()["threshold"]["margin_rates"]
    report = tp.run_threshold()
    assert [s["margin_rate"] for s in report["scenarios"]] == sorted(set(configured))
    assert report["n_scenarios"] == len(set(configured))


def test_every_scenario_has_an_optimum_and_a_plateau(world):
    for sc in tp.run_threshold()["scenarios"]:
        assert sc["optimum"]["threshold"] is not None
        assert sc["plateau"]["lower"] <= sc["optimum"]["threshold"] <= sc["plateau"]["upper"]
        assert 0.0 <= sc["optimum"]["approval_rate"] <= 1.0


def test_rotating_holdout_uses_the_fold_column(world):
    sc = tp.run_threshold()["scenarios"][0]
    rh = sc["rotating_holdout"]
    assert [r["held_out_fold"] for r in rh["folds"]] == [1, 2, 3]
    assert rh["threshold_mean"] is not None


def test_optimum_rises_with_the_assumed_margin(world):
    thresholds = [s["optimum"]["threshold"] for s in tp.run_threshold()["scenarios"]]
    assert thresholds == sorted(thresholds)


def test_current_policy_is_scored_from_segmentation_thresholds(world):
    cfg = get_config()["risk"]["segmentation"]["thresholds"]
    sc = tp.run_threshold()["scenarios"][0]
    fixed = sc["references"]["current_fixed"]
    assert [r["threshold"] for r in fixed] == [cfg["medium"], cfg["high"]]
    assert all(r["implied_margin_rate"] is not None for r in fixed)


def test_summary_renders(world):
    out = tp.format_summary(tp.run_threshold())
    assert "COST-BASED CUT-OFF" in out and "break-even PD" in out


# ──────────────────────────────────────────────────────────────
# Guards
# ──────────────────────────────────────────────────────────────

def test_missing_cv_report_says_why_oof_is_required(world):
    (world["reporting"] / "cv_report.json").unlink()
    with pytest.raises(FileNotFoundError, match="not test.parquet"):
        tp.run_threshold()


def test_null_exposure_stops_the_run(world):
    """Dropping these rows would change the population silently."""
    path = world["model_input"] / "train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[:3], "AMT_CREDIT"] = None
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="3 null and 0 non-positive"):
        tp.run_threshold()


def test_non_positive_exposure_stops_the_run(world):
    path = world["model_input"] / "val.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[:2], "AMT_CREDIT"] = 0.0
    frame.loc[frame.index[2:4], "AMT_CREDIT"] = -5.0
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="0 null and 4 non-positive"):
        tp.run_threshold()


def test_missing_ead_column_names_the_config_key(world):
    for name in ("train", "val"):
        path = world["model_input"] / f"{name}.parquet"
        pd.read_parquet(path).drop(columns=["AMT_CREDIT"]).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="risk.expected_loss.ead_column"):
        tp.run_threshold()


def test_predictions_without_attributes_are_rejected(world):
    val = pd.read_parquet(world["model_input"] / "val.parquet")
    val.head(len(val) - 10).to_parquet(
        world["model_input"] / "val.parquet", index=False)
    with pytest.raises(RuntimeError, match="different ETL runs"):
        tp.run_threshold()


def test_stale_oof_files_are_rejected(world):
    extra = _oof_frame(3, range(90_001, 90_031))
    pd.concat([_oof_frame(3, range(1 + 2 * N_PER_FOLD, 1 + 3 * N_PER_FOLD)), extra]) \
      .to_parquet(world["oof_dir"] / "fold_3.parquet", index=False)
    with pytest.raises(RuntimeError, match="different run"):
        tp.run_threshold()


def test_build_rows_groups_by_fold():
    frame = pd.DataFrame({
        "pd_score": [0.1, 0.2, 0.3],
        TARGET: [0, 1, 0],
        "AMT_CREDIT": [100.0, 200.0, 300.0],
        "fold": [1, 1, 2],
    })
    rows, by_fold = tp.build_rows(frame, TARGET, "AMT_CREDIT")
    assert rows == [(0.1, 0, 100.0), (0.2, 1, 200.0), (0.3, 0, 300.0)]
    assert sorted(by_fold) == [1, 2]
    assert len(by_fold[1]) == 2 and len(by_fold[2]) == 1


def test_build_rows_without_a_fold_column():
    frame = pd.DataFrame({"pd_score": [0.1], TARGET: [0], "AMT_CREDIT": [100.0]})
    rows, by_fold = tp.build_rows(frame, TARGET, "AMT_CREDIT")
    assert rows and by_fold == {}
