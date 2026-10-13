"""Cross-validation pipeline (T1.4) — confidence intervals + calibration.

Answers "is ROC-AUC 0.7663 a stable number or one lucky draw?" by training the
same model k times on stratified folds, then reporting mean ± std, a
t-interval, and an out-of-fold calibration table.

Two phases
----------
`--phase folds` (Spark, one training per fold, resumable)
    Pool = `05_model_input/train.parquet ∪ val.parquet` (~85% of master).
    `test.parquet` is **never read**, so it stays a clean hold-out.
    Folds come from `xxhash64(SK_ID_CURR, seed)` ordering inside each TARGET
    stratum — deterministic regardless of Spark partitioning, and exactly
    balanced on TARGET. The assignment is persisted so it can be reused.
    Each fold: train on the other k−1 folds, score the held-out fold, record
    ROC-AUC / PR-AUC / KS, write out-of-fold predictions, checkpoint the report.

`--phase report` (pandas only, no SparkSession)
    Reads the per-fold out-of-fold Parquet files and builds the decile
    calibration table + ECE. Fast, and unit-testable without Spark.

Reproducing Production exactly
------------------------------
Same feature pipeline, same `models.gbt` hyperparams, same seed, `tune=False`,
same `features.missing_threshold`, and **no class weighting** —
`add_class_weights()` exists in trainer.py but is never called and no
classifier receives `weightCol`, so adding it here would measure a different
model (and would break the calibration that §7.4 relies on).

Resume
------
`cv_report.json` is rewritten atomically after every fold and doubles as the
checkpoint. A `config_fingerprint` guards against resuming into a report built
with different settings. Prefer `--fold N`, one process per fold: the JVM heap
is handed back between trainings, which matters on a 3.8 GB VM.

Run:
    make cv-fold N=1      # … through N=5
    make cv-report
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F

from credit_risk.config import get_config, project_root
from credit_risk.cv import (
    build_calibration_report,
    build_cv_report,
    build_setup,
    check_fold_balance,
    column_audit,
    completed_folds,
    config_fingerprint,
    dropped_by_prefix,
    fold_ids,
    format_calibration_summary,
    format_cv_summary,
    pending_folds,
)
from credit_risk.cv.study import REPORT_FILENAMES
from credit_risk.features.pipeline import build_pipeline, identify_feature_columns
from credit_risk.models.trainer import train_model
from credit_risk.reporting import (
    atomic_write_json,
    check_fingerprint,
    load_report,
    require_mlflow,
)
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.cv")

_MODEL_INPUT = project_root() / "data" / "05_model_input"
_FOLD_DIR = project_root() / "data" / "02_intermediate" / "cv_folds"
_FOLD_PARQUET = _FOLD_DIR / "fold_assignment.parquet"
_OOF_DIR = project_root() / "data" / "07_model_output" / "cv_oof"
_REPORT_DIR = project_root() / "data" / "08_reporting"

_CV_REPORT = _REPORT_DIR / REPORT_FILENAMES["cv"]
_CALIBRATION_REPORT = _REPORT_DIR / REPORT_FILENAMES["calibration"]

# Aggregate column prefixes per source table, so `dropped_columns` can be
# attributed back to the table it came from (T1.4 item 4).
_TABLE_PREFIXES = {
    "bureau": "BUREAU_",
    "previous_application": "PREV_",
    "installments_payments": "INSTAL_",
    "credit_card_balance": "CC_",
    "pos_cash_balance": "POS_",
}


# ══════════════════════════════════════════════════════════════
# Pool
# ══════════════════════════════════════════════════════════════

def load_pool(spark) -> tuple[DataFrame, dict]:
    """Read train ∪ val as the CV pool. `test.parquet` is deliberately not read."""
    train_path = _MODEL_INPUT / "train.parquet"
    val_path = _MODEL_INPUT / "val.parquet"
    test_path = _MODEL_INPUT / "test.parquet"

    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"Run `make etl` first — {p} not found")

    train_df = spark.read.parquet(str(train_path))
    val_df = spark.read.parquet(str(val_path))
    if set(train_df.columns) != set(val_df.columns):
        only_t = sorted(set(train_df.columns) - set(val_df.columns))
        only_v = sorted(set(val_df.columns) - set(train_df.columns))
        raise RuntimeError(
            f"train/val schemas differ — train-only {only_t[:10]}, "
            f"val-only {only_v[:10]}"
        )

    pool = train_df.unionByName(val_df)
    n_train, n_val = train_df.count(), val_df.count()

    # One row per customer is load-bearing: a duplicated SK_ID_CURR would be
    # split across folds by the window, so the same customer could end up in
    # both the training and the validation side of one fold — leakage that
    # inflates every metric. Checked here, in seconds, rather than in the
    # report phase after an hour of training.
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    n_distinct = pool.select(id_col).distinct().count()
    if n_distinct != n_train + n_val:
        raise RuntimeError(
            f"{id_col} is not unique in the CV pool: {n_train + n_val} rows but "
            f"{n_distinct} distinct ids ({n_train + n_val - n_distinct} duplicates). "
            "Folds would leak the same customer into train and validation. "
            "Check the aggregate joins in build_master_dataset for fan-out."
        )

    meta = {
        "n_distinct_ids": n_distinct,
        "n_rows": n_train + n_val,
        "n_from_train_parquet": n_train,
        "n_from_val_parquet": n_val,
        # D10: master column count, taken from the pool rather than re-running ETL.
        "n_master_columns": len(pool.columns),
        "holdout_rows": None,
        "holdout_note": "test.parquet is not read by this pipeline",
    }
    if test_path.exists():
        # Row count only, as evidence the hold-out is intact and excluded.
        meta["holdout_rows"] = spark.read.parquet(str(test_path)).count()

    log.info("cv_pool_loaded", **{k: v for k, v in meta.items() if v is not None})
    return pool, meta


# ══════════════════════════════════════════════════════════════
# Fold assignment
# ══════════════════════════════════════════════════════════════

def build_fold_assignment(spark, pool: DataFrame, k: int, force: bool = False) -> DataFrame:
    """Stratified, hash-ordered fold assignment; persisted and reused.

    `xxhash64(SK_ID_CURR, seed)` gives a deterministic pseudo-random order that
    does not depend on how Spark partitioned the input — unlike `randomSplit`.
    Ranking inside each TARGET stratum and taking `rank mod k` makes the folds
    exactly balanced on the label (±1 row).
    """
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]
    seed = cfg.get("random_seed", 42)

    if _FOLD_PARQUET.exists() and not force:
        folds = spark.read.parquet(str(_FOLD_PARQUET))
        log.info("cv_folds_cached", path=str(_FOLD_PARQUET), rows=folds.count())
        return folds

    _FOLD_DIR.mkdir(parents=True, exist_ok=True)
    h = F.xxhash64(F.col(id_col).cast("string"), F.lit(seed))
    # Tie-break on the id so the ordering is total and fully reproducible.
    w = Window.partitionBy(target_col).orderBy(h, F.col(id_col))

    folds = (
        pool.select(id_col, target_col)
            .withColumn("fold", F.pmod(F.row_number().over(w) - F.lit(1), F.lit(k)) + F.lit(1))
    )
    folds.write.mode("overwrite").parquet(str(_FOLD_PARQUET))
    folds = spark.read.parquet(str(_FOLD_PARQUET))
    log.info("cv_folds_saved", path=str(_FOLD_PARQUET), k=k, seed=seed)
    return folds


def measure_fold_balance(folds: DataFrame, k: int) -> list[dict]:
    """Per-fold size / positives / default rate, then assert stratification."""
    cfg = get_config()
    target_col = cfg["data"]["target_column"]

    rows = (
        folds.groupBy("fold")
             .agg(F.count("*").alias("n"), F.sum(F.col(target_col)).alias("n_positive"))
             .orderBy("fold")
             .collect()
    )
    balance = []
    for r in rows:
        n = int(r["n"])
        pos = int(r["n_positive"] or 0)
        balance.append({
            "fold": int(r["fold"]),
            "n": n,
            "n_positive": pos,
            # Recorded so the stratification invariant is auditable from the
            # report: each class must be balanced to within one row per fold.
            "n_negative": n - pos,
            "default_rate": round(pos / n, 6),
        })
    if len(balance) != k:
        raise RuntimeError(f"expected {k} folds, assignment produced {len(balance)}")
    # Fails loudly if the Spark rule ever drifts from the tested pure-Python rule.
    check_fold_balance(balance)
    log.info("cv_fold_balance_ok", balance=balance)
    return balance


# ══════════════════════════════════════════════════════════════
# One fold = one training run
# ══════════════════════════════════════════════════════════════

def run_fold(spark, pool: DataFrame, folds: DataFrame, fold: int, k: int) -> dict:
    """Train on the other k−1 folds, score this one, write out-of-fold preds."""
    from pyspark.ml.functions import vector_to_array

    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]
    started = time.monotonic()

    assign = folds.select(id_col, "fold")
    joined = pool.join(assign, on=id_col, how="inner")
    train_df = joined.filter(F.col("fold") != fold).drop("fold").cache()
    valid_df = joined.filter(F.col("fold") == fold).drop("fold").cache()
    n_train, n_valid = train_df.count(), valid_df.count()
    log.info("cv_fold_data", fold=fold, train_rows=n_train, valid_rows=n_valid)

    try:
        feature_cols = identify_feature_columns(
            train_df,
            target_col=target_col,
            id_cols=[id_col],
            drop_threshold=cfg["features"]["missing_threshold"],
        )
        audit = column_audit(
            all_columns=train_df.columns,
            numeric=feature_cols.numeric,
            categorical=feature_cols.categorical,
            excluded=[target_col, id_col],
        )
        feature_pipeline = build_pipeline(feature_cols, scaler=True)

        result = train_model(
            train_df=train_df,
            test_df=valid_df,
            feature_pipeline=feature_pipeline,
            model_type="gbt",
            target_col=target_col,
            tune=False,
            register=False,   # never pollute the model registry during CV
            run_name=f"cv_k{k}_fold{fold:02d}",
        )

        # Out-of-fold predictions: this fold was not in training.
        _OOF_DIR.mkdir(parents=True, exist_ok=True)
        oof = (
            result.pipeline.transform(valid_df)
            .withColumn("pd_score", vector_to_array("probability")[1])
            .select(id_col, target_col, "pd_score")
            .withColumn("fold", F.lit(fold))
        )
        oof.coalesce(1).write.mode("overwrite").parquet(str(_OOF_DIR / f"fold_{fold}.parquet"))
        log.info("cv_oof_saved", fold=fold, path=str(_OOF_DIR / f"fold_{fold}.parquet"))
    finally:
        train_df.unpersist()
        valid_df.unpersist()

    m = result.metrics
    record = {
        "fold": fold,
        "n_train": n_train,
        "n_valid": n_valid,
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "ks_statistic": m.get("ks_statistic"),
        **audit,
        "dropped_by_table": dropped_by_prefix(audit["dropped_columns"], _TABLE_PREFIXES),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "mlflow_run_id": result.mlflow_run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info("cv_fold_done", **{
        k2: record[k2] for k2 in
        ("fold", "roc_auc", "pr_auc", "ks_statistic", "n_source_columns", "elapsed_sec")
    })
    return record


def run_folds_phase(
    spark,
    only_fold: int | None = None,
    force: bool = False,
) -> dict:
    """Run the pending folds, checkpointing cv_report.json after each."""
    cfg = get_config()
    k = int(cfg.get("cv", {}).get("k_folds", 5))

    if only_fold is not None and only_fold not in fold_ids(k):
        raise ValueError(f"--fold {only_fold} is out of range (valid: 1..{k})")

    pool, pool_meta = load_pool(spark)
    setup = build_setup(k, cfg, pool=pool_meta)
    fingerprint = config_fingerprint(setup)

    existing = None if force else load_report(_CV_REPORT)
    check_fingerprint(existing, fingerprint)
    records = dict(completed_folds(existing))

    folds = build_fold_assignment(spark, pool, k, force=force)
    balance = measure_fold_balance(folds, k)

    todo = pending_folds(k, existing, only_fold=only_fold, force=force)
    if not todo:
        log.info("cv_nothing_to_do", completed=sorted(records))
        report = build_cv_report(setup, list(records.values()), balance)
        atomic_write_json(_CV_REPORT, report)
        return report

    log.info("cv_plan", k=k, already_done=sorted(records), to_run=todo)

    for fold in todo:
        records[fold] = run_fold(spark, pool, folds, fold, k)
        report = build_cv_report(setup, list(records.values()), balance)
        atomic_write_json(_CV_REPORT, report)   # checkpoint after every fold
        log.info(
            "cv_checkpoint",
            fold=fold,
            completed=report["n_folds_completed"],
            total=report["n_folds_total"],
            path=str(_CV_REPORT),
        )

    return build_cv_report(setup, list(records.values()), balance)


# ══════════════════════════════════════════════════════════════
# Report phase — pandas only, no SparkSession
# ══════════════════════════════════════════════════════════════

def run_report_phase() -> dict:
    """Build calibration_report.json from the out-of-fold Parquet files."""
    import pandas as pd

    cv_report = load_report(_CV_REPORT)
    if not cv_report:
        raise FileNotFoundError(
            f"{_CV_REPORT} not found — run `make cv-fold N=1` … `N=5` first"
        )

    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]
    n_deciles = int(cfg.get("cv", {}).get("n_deciles", 10))

    done = sorted(completed_folds(cv_report))
    expected = cv_report.get("n_folds_total")
    if len(done) != expected:
        log.warning("cv_report_partial", completed=done, expected=expected)

    frames = []
    for fold in done:
        path = _OOF_DIR / f"fold_{fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"fold {fold} is recorded as finished but {path} is missing"
            )
        frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError("no out-of-fold predictions found")

    oof = pd.concat(frames, ignore_index=True)
    dup = int(oof[id_col].duplicated().sum())
    if dup:
        raise RuntimeError(
            f"{dup} ids appear in more than one fold — folds must be disjoint"
        )
    log.info("cv_oof_loaded", rows=len(oof), folds=len(frames), duplicate_ids=dup)

    rows = list(zip(oof[id_col].tolist(),
                    oof[target_col].astype(int).tolist(),
                    oof["pd_score"].astype(float).tolist()))

    report = build_calibration_report(
        rows,
        config_fingerprint_value=cv_report.get("config_fingerprint", ""),
        n_folds=len(frames),
        n_deciles=n_deciles,
    )
    atomic_write_json(_CALIBRATION_REPORT, report)
    log.info("cv_calibration_saved", path=str(_CALIBRATION_REPORT),
             ece=report["ece"], n_total=report["n_total"])
    return report


# ══════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation: confidence intervals + calibration.",
    )
    parser.add_argument(
        "--phase", choices=["folds", "report", "all"], default="all",
        help="folds = the training runs (Spark); report = calibration (pandas only)",
    )
    parser.add_argument(
        "--fold", type=int, default=None, metavar="N",
        help="run exactly one fold then exit (one process per fold keeps peak heap low)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="redo work already on disk (fold assignment / finished folds)",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("cv_started", phase=args.phase, fold=args.fold)

    if args.phase in ("folds", "all"):
        require_mlflow("cross-validation study")
        spark = get_spark("cv-pipeline")
        try:
            report = run_folds_phase(spark, only_fold=args.fold, force=args.force)
            print("\n" + format_cv_summary(report) + "\n")
        except Exception:
            log.exception("cv_folds_failed")
            raise
        finally:
            stop_spark()

    if args.phase in ("report", "all"):
        try:
            calib = run_report_phase()
            print("\n" + format_calibration_summary(calib) + "\n")
        except Exception:
            log.exception("cv_report_failed")
            raise

    log.info("cv_completed", phase=args.phase)


if __name__ == "__main__":
    main()
