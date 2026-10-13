"""Baseline comparison pipeline (P0.3) — Spark orchestration only.

Study design, resume bookkeeping and the comparison arithmetic live in
`credit_risk/baseline/study.py`, which imports no Spark.

What it answers: how much does the 7-table, 58.4M-row pipeline buy over a
model you could fit in a spreadsheet? Four cheap variants decompose it into
"the features" versus "the model class" — see the study module's docstring.

Comparability is the whole point, so this reuses the **persisted ablation
split** (`data/02_intermediate/ablation_split/`) and the ablation's cached
`_application.parquet`. Same 45,820 test customers, same seed, same
`evaluate_predictions`. The reference numbers are read out of
`ablation_report.json` rather than hardcoded, and the run refuses to compare
if the split has since been regenerated.

Run:
    make baseline                 # all pending variants
    make baseline V=gbt_ext_all3  # one variant
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.baseline import (
    MODEL_RULE,
    REPORT_FILENAME,
    VariantSpec,
    build_report,
    build_setup,
    build_variants,
    check_split_match,
    completed_variants,
    config_fingerprint,
    format_summary,
    load_references,
    pending_variants,
)
from credit_risk.config import get_config, project_root
from credit_risk.features.pipeline import FeatureColumns, build_pipeline
from credit_risk.models.trainer import evaluate_predictions, train_model
from credit_risk.reporting import (
    atomic_write_json,
    check_fingerprint,
    load_report,
    require_mlflow,
)
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.baseline")

# Reused from T1.2 so the numbers line up with README §12.1.
_AGG_DIR = project_root() / "data" / "02_intermediate" / "ablation_agg"
_APP_PARQUET = _AGG_DIR / "_application.parquet"
_ID_DIR = project_root() / "data" / "02_intermediate" / "ablation_split"

_REPORT_DIR = project_root() / "data" / "08_reporting"
_BASELINE_REPORT = _REPORT_DIR / REPORT_FILENAME
_ABLATION_REPORT = _REPORT_DIR / "ablation_report.json"


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def load_split(spark) -> tuple[DataFrame, DataFrame, dict]:
    """Application rows split by the persisted ablation ids."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]

    for p in (_APP_PARQUET, _ID_DIR / "train_ids.parquet", _ID_DIR / "test_ids.parquet"):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p} — run `make ablation-precompute` first. The "
                "baseline deliberately reuses the ablation split so its numbers "
                "are comparable to README §12.1."
            )

    application = spark.read.parquet(str(_APP_PARQUET))
    train_ids = spark.read.parquet(str(_ID_DIR / "train_ids.parquet"))
    test_ids = spark.read.parquet(str(_ID_DIR / "test_ids.parquet"))

    train_df = application.join(train_ids, on=id_col, how="inner")
    test_df = application.join(test_ids, on=id_col, how="inner")
    n_train, n_test = train_df.count(), test_df.count()

    meta = {"train_rows": n_train, "test_rows": n_test,
            "source": "data/02_intermediate/ablation_split (T1.2)"}
    log.info("baseline_split_loaded", **meta)
    return train_df, test_df, meta


def _null_rates(df: DataFrame, columns: list[str], n_rows: int) -> dict:
    """Null rate per feature — explains why EXT_SOURCE_1 is contentious."""
    if not n_rows:
        return {c: None for c in columns}
    agg = df.agg(*[
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns
    ]).collect()[0].asDict()
    return {c: round((agg[c] or 0) / n_rows, 4) for c in columns}


# ══════════════════════════════════════════════════════════════
# One variant
# ══════════════════════════════════════════════════════════════

def _score_rule(test_df: DataFrame, features: list[str], target_col: str) -> DataFrame:
    """Rank by the mean of the available EXT_SOURCE columns — no training.

    Risk score = 1 − mean(non-null EXT_SOURCE), because a higher EXT_SOURCE
    means better credit. Rows with every column null fall back to 0.5, i.e.
    the middle of the ranking rather than either extreme.

    A `probability` vector and a dummy `prediction` are synthesised so the
    same `evaluate_predictions` the ablation used can score it. Only the
    ranking metrics (ROC-AUC, PR-AUC, KS) are read back: this is a score, not
    a calibrated PD, and the report says so.
    """
    from pyspark.ml.functions import array_to_vector

    present = [F.col(c).isNotNull().cast("double") for c in features]
    filled = [F.coalesce(F.col(c), F.lit(0.0)) for c in features]
    n_present = sum(present[1:], present[0])
    total = sum(filled[1:], filled[0])

    mean_ext = F.when(n_present > 0, total / n_present).otherwise(F.lit(0.5))
    score = F.lit(1.0) - mean_ext

    return (
        test_df
        .withColumn("pd_score", score)
        .withColumn("probability",
                    array_to_vector(F.array(F.lit(1.0) - F.col("pd_score"),
                                             F.col("pd_score"))))
        .withColumn("prediction", F.lit(0.0))
        .select(target_col, "pd_score", "probability", "prediction")
    )


def run_variant(spark, spec: VariantSpec, train_df: DataFrame, test_df: DataFrame) -> dict:
    """Fit (or not) one baseline variant and score it on the shared test set."""
    cfg = get_config()
    target_col = cfg["data"]["target_column"]
    started = time.monotonic()

    missing = [c for c in spec.features if c not in train_df.columns]
    if missing:
        raise RuntimeError(
            f"variant {spec.name!r} needs {missing}, absent from the "
            "application parquet"
        )

    features = list(spec.features)
    nulls = _null_rates(train_df, features, train_df.count())
    run_id = None

    if spec.model_type == MODEL_RULE:
        scored = _score_rule(test_df, features, target_col)
        metrics = evaluate_predictions(scored, target_col)
    else:
        feature_cols = FeatureColumns(numeric=features, categorical=[], all_assembled=[])
        result = train_model(
            train_df=train_df,
            test_df=test_df,
            feature_pipeline=build_pipeline(feature_cols, scaler=True),
            model_type=spec.model_type,
            target_col=target_col,
            tune=False,
            register=False,   # never pollute the model registry
            run_name=f"baseline_{spec.name}",
        )
        metrics, run_id = result.metrics, result.mlflow_run_id

    record = {
        "name": spec.name,
        "model": spec.model_type,
        "description": spec.description,
        "features": features,
        "n_features": len(features),
        "feature_null_rate_train": nulls,
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "ks_statistic": metrics.get("ks_statistic"),
        "calibrated": spec.model_type != MODEL_RULE,
        "elapsed_sec": round(time.monotonic() - started, 1),
        "mlflow_run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info("baseline_variant_done", **{
        k: record[k] for k in
        ("name", "model", "n_features", "roc_auc", "pr_auc", "ks_statistic", "elapsed_sec")
    })
    return record


# ══════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════

def run_baseline(spark, only_variant: str | None = None, force: bool = False) -> dict:
    """Run the pending variants, checkpointing the report after each."""
    cfg = get_config()
    variants = build_variants(cfg)
    names = [v.name for v in variants]

    if only_variant is not None and only_variant not in names:
        raise ValueError(f"--variant {only_variant!r} not in {names}")

    train_df, test_df, split_meta = load_split(spark)
    setup = build_setup(cfg, variants, split=split_meta)
    fingerprint = config_fingerprint(setup)

    references = load_references(load_report(_ABLATION_REPORT))
    if not references:
        log.warning("baseline_no_ablation_reference", path=str(_ABLATION_REPORT))
    else:
        # A regenerated split would silently invalidate every delta.
        check_split_match(split_meta["test_rows"], references)

    existing = None if force else load_report(_BASELINE_REPORT)
    check_fingerprint(existing, fingerprint)
    records = dict(completed_variants(existing))

    todo = pending_variants(variants, existing, only_variant=only_variant, force=force)
    if not todo:
        log.info("baseline_nothing_to_do", completed=sorted(records))
        report = build_report(setup, list(records.values()), references)
        atomic_write_json(_BASELINE_REPORT, report)
        return report

    log.info("baseline_plan", already_done=sorted(records), to_run=[v.name for v in todo])

    train_df.cache()
    test_df.cache()
    try:
        for spec in todo:
            records[spec.name] = run_variant(spark, spec, train_df, test_df)
            report = build_report(setup, list(records.values()), references)
            atomic_write_json(_BASELINE_REPORT, report)   # checkpoint per variant
            log.info("baseline_checkpoint", variant=spec.name,
                     completed=report["n_variants_completed"],
                     total=report["n_variants_total"], path=str(_BASELINE_REPORT))
    finally:
        train_df.unpersist()
        test_df.unpersist()

    return build_report(setup, list(records.values()), references)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline comparison: is the multi-table pipeline worth it?",
    )
    parser.add_argument(
        "--variant", default=None, metavar="NAME",
        help="run exactly one variant then exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="redo variants already recorded in the report",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("baseline_started", variant=args.variant)
    require_mlflow("baseline study")

    spark = get_spark("baseline-pipeline")
    try:
        report = run_baseline(spark, only_variant=args.variant, force=args.force)
        print("\n" + format_summary(report) + "\n")
        log.info("baseline_completed")
    except Exception:
        log.exception("baseline_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    main()
