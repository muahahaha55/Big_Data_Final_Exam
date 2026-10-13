"""End-to-end tests for the P0.4 pipeline — pandas on temp Parquet, no Spark.

`pipelines/segment_pipeline.py` imports no Spark at all (the predictions were
written by T1.4), so the whole thing is testable here: the join between the
out-of-fold predictions and the segment attributes, and every guard around it.

That join is where this step can go wrong silently — a stale Parquet file, a
fanned-out aggregate, or predictions from a different ETL run would all still
produce a plausible-looking report. T1.4 shipped a runtime check that had never
been executed and it crashed on the VM; these tests execute each of these.

They also run against the **real** `conf/base/config.yaml`, so the four
configured dimensions must exist in the frame this builds — if someone renames
a column in config, a test fails here rather than on the VM.
"""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pipelines.segment_pipeline as sp

from credit_risk.config import get_config

ID = "SK_ID_CURR"
TARGET = "TARGET"
N_PER_FOLD = 1600          # above segments.min_rows for both contract types


def _oof_frame(fold: int, ids: range) -> "pd.DataFrame":
    """Scored rows with real signal, so the AUC is well away from 0.5."""
    rows = []
    for i in ids:
        target = 1 if i % 5 == 0 else 0
        score = 0.05 + 0.3 * target + ((i * 7919) % 1000) / 2000.0
        rows.append((i, target, min(score, 0.99), fold))
    return pd.DataFrame(rows, columns=[ID, TARGET, "pd_score", "fold"])


def _attr_frame(ids: range) -> "pd.DataFrame":
    rows = []
    for i in ids:
        rows.append((
            i,
            "Cash loans" if i % 2 else "Revolving loans",
            "F" if i % 3 else "M",
            -365.25 * (21 + (i % 45)),
            50000.0 + (i % 17) * 25000.0,
        ))
    return pd.DataFrame(rows, columns=[
        ID, "NAME_CONTRACT_TYPE", "CODE_GENDER", "DAYS_BIRTH", "AMT_INCOME_TOTAL"])


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A complete, consistent set of inputs; tests then break one thing."""
    oof_dir = tmp_path / "cv_oof"
    model_input = tmp_path / "05_model_input"
    reporting = tmp_path / "08_reporting"
    for d in (oof_dir, model_input, reporting):
        d.mkdir(parents=True)

    fold_1 = range(1, N_PER_FOLD + 1)
    fold_2 = range(N_PER_FOLD + 1, 2 * N_PER_FOLD + 1)
    _oof_frame(1, fold_1).to_parquet(oof_dir / "fold_1.parquet", index=False)
    _oof_frame(2, fold_2).to_parquet(oof_dir / "fold_2.parquet", index=False)

    all_ids = list(fold_1) + list(fold_2)
    half = len(all_ids) // 2
    _attr_frame(range(1, half + 1)).to_parquet(
        model_input / "train.parquet", index=False)
    _attr_frame(range(half + 1, len(all_ids) + 1)).to_parquet(
        model_input / "val.parquet", index=False)

    cv_report = {
        "config_fingerprint": "sha256:deadbeef",
        "n_folds_total": 2,
        "n_folds_completed": 2,
        # The fixture's scores are deliberately easy, so its AUC sits near
        # 0.92. The reference has to be near the data, not near the real
        # project's 0.7663, or the structural cross-check fires correctly.
        "folds": [
            {"fold": 1, "n_valid": N_PER_FOLD, "roc_auc": 0.9150},
            {"fold": 2, "n_valid": N_PER_FOLD, "roc_auc": 0.9250},
        ],
        "summary": {"roc_auc": {"mean": 0.92}},
    }
    (reporting / "cv_report.json").write_text(json.dumps(cv_report), encoding="utf-8")

    monkeypatch.setattr(sp, "_OOF_DIR", oof_dir)
    monkeypatch.setattr(sp, "_MODEL_INPUT", model_input)
    monkeypatch.setattr(sp, "_REPORT_DIR", reporting)
    monkeypatch.setattr(sp, "_CV_REPORT", reporting / "cv_report.json")
    monkeypatch.setattr(sp, "_SEGMENT_REPORT", reporting / "segment_report.json")

    return {"oof_dir": oof_dir, "model_input": model_input,
            "reporting": reporting, "cv_report": cv_report,
            "n_rows": len(all_ids)}


# ──────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────

def test_runs_end_to_end_and_writes_the_report(world):
    report = sp.run_segments()

    written = json.loads(
        (world["reporting"] / "segment_report.json").read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(report))     # same object on disk

    assert report["overall"]["n"] == world["n_rows"]
    assert report["cv_config_fingerprint"] == "sha256:deadbeef"
    assert report["setup"]["source"]["cv_folds_used"] == [1, 2]
    assert report["setup"]["source"]["oof_rows"] == world["n_rows"]


def test_report_covers_every_configured_dimension(world):
    """Guards config drift: a renamed column fails here, not on the VM."""
    configured = [d["name"] for d in get_config()["segments"]["dimensions"]]
    report = sp.run_segments()
    assert [d["name"] for d in report["dimensions"]] == configured
    assert len(configured) == 4


def test_quantile_dimensions_bin_from_the_data(world):
    report = sp.run_segments()
    by_name = {d["name"]: d for d in report["dimensions"]}

    age = by_name["age_group"]
    assert age["bin_edges"] and all(20 < e < 70 for e in age["bin_edges"])
    assert age["derivation"] == "days_to_years(DAYS_BIRTH)"

    income = by_name["income_group"]
    assert income["bin_edges"] and all(e > 0 for e in income["bin_edges"])


def test_group_counts_add_up_to_the_population(world):
    report = sp.run_segments()
    for dim in report["dimensions"]:
        assert sum(g["n"] for g in dim["groups"]) == report["overall"]["n"]


def test_reference_compares_against_the_cv_fold_mean(world):
    report = sp.run_segments()
    assert report["reference"]["cv_roc_auc_mean"] == 0.92
    assert report["reference"]["delta"] is not None


def test_summary_renders(world):
    out = sp.format_summary(sp.run_segments())
    assert "SEGMENT VALIDATION" in out
    assert "contract_type" in out and "income_group" in out


# ──────────────────────────────────────────────────────────────
# Guards
# ──────────────────────────────────────────────────────────────

def test_missing_cv_report_is_a_clear_error(world, monkeypatch):
    (world["reporting"] / "cv_report.json").unlink()
    with pytest.raises(FileNotFoundError, match="cv-fold"):
        sp.run_segments()


def test_missing_fold_parquet_names_the_fold(world):
    (world["oof_dir"] / "fold_2.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="fold 2 is recorded as finished"):
        sp.run_segments()


def test_stale_oof_files_are_rejected(world):
    """Row count must match what the folds recorded, or it is another run."""
    extra = _oof_frame(2, range(10_001, 10_051))
    pd.concat([_oof_frame(2, range(N_PER_FOLD + 1, 2 * N_PER_FOLD + 1)), extra]) \
      .to_parquet(world["oof_dir"] / "fold_2.parquet", index=False)
    with pytest.raises(RuntimeError, match="different run"):
        sp.run_segments()


def test_overlapping_folds_are_rejected(world):
    """Folds must be disjoint; the same customer must not be scored twice."""
    world["cv_report"]["folds"][1]["n_valid"] = N_PER_FOLD
    _oof_frame(2, range(1, N_PER_FOLD + 1)).to_parquet(
        world["oof_dir"] / "fold_2.parquet", index=False)
    with pytest.raises(RuntimeError, match="disjoint"):
        sp.run_segments()


def test_duplicate_ids_in_the_pool_are_rejected(world):
    """A fanned-out aggregate join would double-count rows in every segment."""
    train = pd.read_parquet(world["model_input"] / "train.parquet")
    pd.concat([train, train.head(3)]).to_parquet(
        world["model_input"] / "train.parquet", index=False)
    with pytest.raises(RuntimeError, match="fanned out"):
        sp.run_segments()


def test_predictions_without_attributes_are_rejected(world):
    """Predictions and pool from different ETL runs must not quietly subset."""
    val = pd.read_parquet(world["model_input"] / "val.parquet")
    val.head(len(val) - 25).to_parquet(
        world["model_input"] / "val.parquet", index=False)
    with pytest.raises(RuntimeError, match="different ETL runs"):
        sp.run_segments()


def test_missing_attribute_file_points_at_etl(world):
    (world["model_input"] / "val.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="make etl"):
        sp.run_segments()


def test_missing_segment_column_names_the_config_key(world):
    """Renaming a column in config without the data must fail loudly."""
    for name in ("train", "val"):
        path = world["model_input"] / f"{name}.parquet"
        pd.read_parquet(path).drop(columns=["CODE_GENDER"]).to_parquet(
            path, index=False)
    with pytest.raises(RuntimeError, match="segments.dimensions column"):
        sp.run_segments()


def test_no_finished_folds_is_a_clear_error(world):
    world["cv_report"]["folds"] = []
    (world["reporting"] / "cv_report.json").write_text(
        json.dumps(world["cv_report"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no finished folds"):
        sp.run_segments()
