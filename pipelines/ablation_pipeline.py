"""Ablation pipeline — how much does each child table actually contribute?

Spark orchestration only. The study design, checkpointing and delta logic live
in `credit_risk/ablation/study.py`, which imports no Spark so it can be unit
tested without a session (tests/test_ablation.py).

Two designs, selected with --mode:

  cumulative (default, T1.2)
      application_train → +bureau → +previous_application
      → +installments → +credit_card → +pos_cash
      Records the MARGINAL gain of each added table. Order-dependent.

  loo
      Leave-one-table-out plus both baselines — the unique contribution of
      each table given the others. Writes to a SEPARATE file; the two modes
      never overwrite each other.

IMPORTANT — what actually gets joined
-------------------------------------
Exactly FIVE child aggregates feed the model:
    bureau, previous_application, installments, credit_card, pos_cash
`bureau_balance` is declared in config.yaml but has NO aggregator, so it is
never joined. This study therefore covers 5 tables, not 6 (see T1.3). That
caveat is carried in the output JSON too.

Phases
------
1. PRECOMPUTE (expensive, once, skips work already on disk):
   - Aggregate each child table to one row / SK_ID_CURR → Parquet.
   - Convert application_train.csv → Parquet. The CSV is read with
     inferSchema, which costs extra full passes over 166 MB *every* time it is
     opened; the ablate phase would otherwise pay that on every step.
   - Verify the Parquet copy against the CSV (schema + row count) and against
     `data.tables.application.expected_rows`.
   - Split application IDs train/test ONCE with a fixed seed and persist, so
     every step is scored on the SAME test customers.
2. ABLATE (one training run per step, resumable):
   - Read application + the allowed aggregates from Parquet, left-join, add
     domain features, split by the shared ID sets, cache, train GBT, record
     roc_auc / pr_auc / ks_statistic / elapsed_sec.

Resumability
------------
The report JSON is rewritten atomically after EVERY step and doubles as the
checkpoint: a finished step is skipped unless --force. Prefer --step N, one
process per step: it hands the JVM heap back between trainings, which matters
on a 3.8 GB VM, and a crash costs one step instead of the whole study.

A `config_fingerprint` covers everything that makes steps comparable
(hyperparams, seed, split, table order, mode, and the Spark knobs that change
task partitioning and hence GBT subsampling). Resuming into a report with a
different fingerprint is refused rather than silently mixing numbers that
cannot be compared.

Run:
    make ablation-precompute
    make ablation-step N=1      # … through N=6
    # or, all pending steps in one process:
    python pipelines/ablation_pipeline.py --phase ablate
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Sequence

from credit_risk.ablation import (
    DEFAULT_TABLE_ORDER,
    REPORT_FILENAMES,
    StepSpec,
    atomic_write_json,
    build_report,
    build_setup,
    build_steps,
    check_fingerprint,
    completed_steps,
    config_fingerprint,
    load_report,
    pending_steps,
)
from credit_risk.ablation.study import format_summary
from credit_risk.config import get_config, project_root
from credit_risk.data.ingestion.multi_table import (
    aggregate_bureau,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_application,
    load_table,
)
from credit_risk.features.pipeline import (
    add_domain_features,
    build_pipeline,
    identify_feature_columns,
)
from credit_risk.models.trainer import train_model
from credit_risk.reporting import require_mlflow
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.ablation")

# The 5 child tables that actually contribute features, in the order the
# cumulative study adds them (PLAN T1.2). Order matters — see the docstring.
CHILD_TABLES = {
    "bureau": aggregate_bureau,
    "previous_application": aggregate_previous_application,
    "installments": aggregate_installments,
    "credit_card": aggregate_credit_card,
    "pos_cash": aggregate_pos_cash,
}

_AGG_DIR = project_root() / "data" / "02_intermediate" / "ablation_agg"
_ID_DIR = project_root() / "data" / "02_intermediate" / "ablation_split"
_APP_PARQUET = _AGG_DIR / "_application.parquet"

_OUT_PATHS = {
    mode: project_root() / "data" / "08_reporting" / fname
    for mode, fname in REPORT_FILENAMES.items()
}

# The aggregator mapping and the study's declared order must not drift apart.
assert list(CHILD_TABLES) == DEFAULT_TABLE_ORDER, (
    f"CHILD_TABLES order {list(CHILD_TABLES)} != "
    f"DEFAULT_TABLE_ORDER {DEFAULT_TABLE_ORDER}"
)


# ══════════════════════════════════════════════════════════════
# Preflight
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# Phase 1 — precompute
# ══════════════════════════════════════════════════════════════

def _verify_application_parquet(spark, csv_df, expected_rows: int | None) -> dict:
    """Check the Parquet copy matches the CSV in schema and row count."""
    pq = spark.read.parquet(str(_APP_PARQUET))

    csv_fields = [(f.name, f.dataType.simpleString()) for f in csv_df.schema.fields]
    pq_fields = [(f.name, f.dataType.simpleString()) for f in pq.schema.fields]
    if csv_fields != pq_fields:
        only_csv = [f for f in csv_fields if f not in pq_fields]
        only_pq = [f for f in pq_fields if f not in csv_fields]
        raise RuntimeError(
            "application Parquet schema does not match the CSV.\n"
            f"  csv-only:     {only_csv[:10]}\n"
            f"  parquet-only: {only_pq[:10]}"
        )

    pq_rows = pq.count()
    csv_rows = csv_df.count()
    if pq_rows != csv_rows:
        raise RuntimeError(
            f"application Parquet row count {pq_rows} != CSV row count {csv_rows}"
        )
    if expected_rows is not None and pq_rows != expected_rows:
        raise RuntimeError(
            f"application row count {pq_rows} != expected {expected_rows} "
            "(config: data.tables.application.expected_rows)"
        )

    log.info("application_parquet_verified", rows=pq_rows, cols=len(pq_fields))
    return {"rows": pq_rows, "columns": len(pq_fields)}


def precompute(spark, force: bool = False) -> None:
    """Aggregate child tables, cache application as Parquet, fix the split."""
    _AGG_DIR.mkdir(parents=True, exist_ok=True)
    _ID_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_config()
    id_col = cfg["data"]["id_column"]

    # ── Child aggregates → Parquet ────────────────────────────
    for name, fn in CHILD_TABLES.items():
        path = _AGG_DIR / f"{name}.parquet"
        if path.exists() and not force:
            log.info("ablation_agg_cached", table=name, path=str(path))
            continue
        agg = fn(spark)
        if agg is None:
            log.warning("ablation_agg_missing", table=name)
            continue
        agg.write.mode("overwrite").parquet(str(path))
        log.info("ablation_agg_saved", table=name, path=str(path))

    # ── application CSV → Parquet (kills repeated inferSchema cost) ──
    application = load_table(spark, "application")
    if application is None:
        raise RuntimeError("application table is required but not found")

    if _APP_PARQUET.exists() and not force:
        log.info("application_parquet_cached", path=str(_APP_PARQUET))
    else:
        application.write.mode("overwrite").parquet(str(_APP_PARQUET))
        log.info("application_parquet_saved", path=str(_APP_PARQUET))

    expected = cfg["data"]["tables"]["application"].get("expected_rows")
    _verify_application_parquet(spark, application, expected)

    # ── Fixed train/test split on IDs — shared by every step ──
    train_path = _ID_DIR / "train_ids.parquet"
    test_path = _ID_DIR / "test_ids.parquet"
    if train_path.exists() and test_path.exists() and not force:
        log.info("ablation_split_cached", path=str(_ID_DIR))
        return

    split = cfg["data"]["split"]
    seed = cfg.get("random_seed", 42)
    ids = spark.read.parquet(str(_APP_PARQUET)).select(id_col)
    train_ids, test_ids = ids.randomSplit(
        [split["train_ratio"] + split["val_ratio"], split["test_ratio"]],
        seed=seed,
    )
    train_ids.write.mode("overwrite").parquet(str(train_path))
    test_ids.write.mode("overwrite").parquet(str(test_path))
    log.info(
        "ablation_split_saved",
        train_ids=train_ids.count(),
        test_ids=test_ids.count(),
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════
# Phase 2 — one step = one training run
# ══════════════════════════════════════════════════════════════

def _build_master(spark, included_tables: Sequence[str]):
    """Application (Parquet) + only the listed child aggregates."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]

    if not _APP_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {_APP_PARQUET} — run `make ablation-precompute` first"
        )
    master = spark.read.parquet(str(_APP_PARQUET))

    for name in included_tables:
        path = _AGG_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} — run `make ablation-precompute` first"
            )
        master = master.join(spark.read.parquet(str(path)), on=id_col, how="left")

    return add_domain_features(master)


def run_step(spark, spec: StepSpec, mode: str) -> dict:
    """Train one step; return its record."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]
    started = time.monotonic()

    master = _build_master(spark, spec.included_tables)

    train_ids = spark.read.parquet(str(_ID_DIR / "train_ids.parquet"))
    test_ids = spark.read.parquet(str(_ID_DIR / "test_ids.parquet"))
    # Cached: train_model and identify_feature_columns take several actions
    # each, and without this the whole join chain is recomputed every time.
    train_df = master.join(train_ids, on=id_col, how="inner").cache()
    test_df = master.join(test_ids, on=id_col, how="inner").cache()
    n_train, n_test = train_df.count(), test_df.count()
    log.info("ablation_step_data", step=spec.step, train_rows=n_train, test_rows=n_test)

    try:
        feature_cols = identify_feature_columns(
            train_df,
            target_col=target_col,
            id_cols=[id_col],
            drop_threshold=cfg["features"]["missing_threshold"],
        )
        feature_pipeline = build_pipeline(feature_cols, scaler=True)

        result = train_model(
            train_df=train_df,
            test_df=test_df,
            feature_pipeline=feature_pipeline,
            model_type="gbt",
            target_col=target_col,
            tune=False,
            register=False,   # never pollute the model registry during ablation
            run_name=f"ablation_{mode}_{spec.step:02d}_{spec.label}",
        )
    finally:
        train_df.unpersist()
        test_df.unpersist()

    m = result.metrics
    record = {
        "step": spec.step,
        "label": spec.label,
        "added_table": spec.added_table,
        "removed_table": spec.removed_table,
        "included_tables": list(spec.included_tables),
        # Source columns fed to the feature pipeline — NOT the width of the
        # assembled vector, which one-hot encoding expands further.
        "n_source_columns": len(feature_cols.numeric) + len(feature_cols.categorical),
        "n_numeric": len(feature_cols.numeric),
        "n_categorical": len(feature_cols.categorical),
        "train_rows": n_train,
        "test_rows": n_test,
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "ks_statistic": m.get("ks_statistic"),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "mlflow_run_id": result.mlflow_run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info("ablation_step_done", **{
        k: record[k] for k in
        ("step", "label", "roc_auc", "pr_auc", "ks_statistic", "elapsed_sec")
    })
    return record


def ablate(
    spark,
    mode: str = "cumulative",
    only_step: int | None = None,
    force: bool = False,
) -> dict:
    """Run the pending steps, checkpointing the report after each one."""
    cfg = get_config()
    tables = list(CHILD_TABLES.keys())
    steps = build_steps(mode, tables)
    setup = build_setup(mode, tables, cfg)
    fingerprint = config_fingerprint(setup)
    out_path = _OUT_PATHS[mode]

    if only_step is not None and not any(s.step == only_step for s in steps):
        raise ValueError(
            f"--step {only_step} is out of range for mode {mode!r} "
            f"(valid: 1..{len(steps)})"
        )

    existing = None if force else load_report(out_path)
    check_fingerprint(existing, fingerprint)
    records = dict(completed_steps(existing))

    todo = pending_steps(steps, existing, only_step=only_step, force=force)
    if not todo:
        log.info("ablation_nothing_to_do", mode=mode, completed=sorted(records))
        report = build_report(mode, setup, list(records.values()))
        atomic_write_json(out_path, report)
        return report

    log.info(
        "ablation_plan",
        mode=mode,
        total=len(steps),
        already_done=sorted(records),
        to_run=[s.step for s in todo],
    )

    report = build_report(mode, setup, list(records.values()))
    for spec in todo:
        records[spec.step] = run_step(spark, spec, mode)
        report = build_report(mode, setup, list(records.values()))
        atomic_write_json(out_path, report)   # checkpoint after every step
        log.info(
            "ablation_checkpoint",
            step=spec.step,
            completed=report["n_steps_completed"],
            total=report["n_steps_total"],
            path=str(out_path),
        )

    return report


# ══════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study over the Home Credit child tables.",
    )
    parser.add_argument(
        "--phase", choices=["precompute", "ablate", "all"], default="all",
        help="precompute = aggregates + application parquet + split; "
             "ablate = the training runs",
    )
    parser.add_argument(
        "--mode", choices=["cumulative", "loo"], default="cumulative",
        help="cumulative = marginal gain per added table (default); "
             "loo = leave-one-table-out",
    )
    parser.add_argument(
        "--step", type=int, default=None, metavar="N",
        help="run exactly one step then exit (one process per step keeps peak heap low)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="redo work already on disk (precompute artifacts / finished steps)",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("ablation_started", phase=args.phase, mode=args.mode, step=args.step)

    if args.phase in ("ablate", "all"):
        require_mlflow("ablation study")

    spark = get_spark("ablation-pipeline")
    try:
        if args.phase in ("precompute", "all"):
            precompute(spark, force=args.force)
        if args.phase in ("ablate", "all"):
            report = ablate(spark, mode=args.mode, only_step=args.step, force=args.force)
            print("\n" + format_summary(report) + "\n")
        log.info("ablation_completed", phase=args.phase, mode=args.mode)
    except Exception:
        log.exception("ablation_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    main()
