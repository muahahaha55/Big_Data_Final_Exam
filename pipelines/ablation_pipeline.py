"""Ablation pipeline — leave-one-table-out study (Phương án A).

Measures each child table's contribution to model performance by training
the GBT model with one source removed at a time, plus an application-only
baseline and a full-table baseline.

IMPORTANT — what actually gets joined
-------------------------------------
`build_master_dataset` joins exactly FIVE child aggregates:
    bureau, previous_application, installments, credit_card, pos_cash
`bureau_balance` is declared in config.yaml but has NO aggregator, so it is
never joined. This study therefore ablates the 5 tables that truly feed the
model — not 6. This is reported honestly in the output JSON.

Design
------
1. PRECOMPUTE phase (expensive, done once):
   - Aggregate each child table to one row / SK_ID_CURR, write to Parquet.
   - Split application IDs into train/test ONCE (fixed seed) and persist,
     so every ablation config is evaluated on the SAME test customers
     (fair comparison).
2. ABLATE phase (cheap, 7 runs):
   - For each config, read application + the *allowed* aggregate Parquets,
     left-join, add domain features, split by the shared ID sets, train GBT,
     record roc_auc / pr_auc / ks_statistic.

The two phases can be run separately (--phase precompute, then --phase ablate)
to keep peak RAM low on a constrained VM — the heavy 13.6M-row aggregation
happens only in the precompute phase.

Run:
    python pipelines/ablation_pipeline.py --phase all
    # or, RAM-tight VM:
    python pipelines/ablation_pipeline.py --phase precompute
    python pipelines/ablation_pipeline.py --phase ablate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.ablation")

# The 5 child tables that actually contribute features, with their aggregators.
CHILD_TABLES = {
    "bureau": aggregate_bureau,
    "previous_application": aggregate_previous_application,
    "installments": aggregate_installments,
    "credit_card": aggregate_credit_card,
    "pos_cash": aggregate_pos_cash,
}

# Intermediate locations for precomputed artifacts.
_AGG_DIR = project_root() / "data" / "02_intermediate" / "ablation_agg"
_ID_DIR = project_root() / "data" / "02_intermediate" / "ablation_split"
_OUT_PATH = project_root() / "data" / "08_reporting" / "ablation_results.json"


# ──────────────────────────────────────────────────────────────
# Phase 1 — precompute aggregates + shared train/test ID split
# ──────────────────────────────────────────────────────────────

def precompute(spark) -> None:
    """Aggregate each child table once and persist a fixed ID split."""
    _AGG_DIR.mkdir(parents=True, exist_ok=True)
    _ID_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_config()
    id_col = cfg["data"]["id_column"]

    # Aggregate each child table once → Parquet (avoids re-reading 13.6M rows).
    for name, fn in CHILD_TABLES.items():
        agg = fn(spark)
        if agg is None:
            log.warning("ablation_agg_missing", table=name)
            continue
        path = _AGG_DIR / f"{name}.parquet"
        agg.write.mode("overwrite").parquet(str(path))
        log.info("ablation_agg_saved", table=name, path=str(path))

    # Fixed train/test split on application IDs — shared by all configs.
    application = load_table(spark, "application")
    if application is None:
        raise RuntimeError("application table is required but not found")

    split = cfg["data"]["split"]
    seed = cfg.get("random_seed", 42)
    ids = application.select(id_col)
    train_ids, test_ids = ids.randomSplit(
        [split["train_ratio"] + split["val_ratio"], split["test_ratio"]],
        seed=seed,
    )
    train_ids.write.mode("overwrite").parquet(str(_ID_DIR / "train_ids.parquet"))
    test_ids.write.mode("overwrite").parquet(str(_ID_DIR / "test_ids.parquet"))
    log.info(
        "ablation_split_saved",
        train_ids=train_ids.count(),
        test_ids=test_ids.count(),
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────
# Phase 2 — build a config's master, train, return metrics
# ──────────────────────────────────────────────────────────────

def _build_master(spark, included_tables: list[str]):
    """Application + only the listed child aggregates (read from Parquet)."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]

    master = load_table(spark, "application")
    if master is None:
        raise RuntimeError("application table is required but not found")

    for name in included_tables:
        path = _AGG_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} — run `--phase precompute` first"
            )
        agg = spark.read.parquet(str(path))
        master = master.join(agg, on=id_col, how="left")

    return add_domain_features(master)


def _run_config(spark, label: str, included_tables: list[str]) -> dict:
    """Train one ablation config; return its metrics + feature count."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]

    master = _build_master(spark, included_tables)

    train_ids = spark.read.parquet(str(_ID_DIR / "train_ids.parquet"))
    test_ids = spark.read.parquet(str(_ID_DIR / "test_ids.parquet"))
    train_df = master.join(train_ids, on=id_col, how="inner")
    test_df = master.join(test_ids, on=id_col, how="inner")

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
    )

    m = result.metrics
    record = {
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "ks_statistic": m.get("ks_statistic"),
        "n_features": len(feature_cols.numeric) + len(feature_cols.categorical),
        "included_tables": included_tables,
    }
    log.info("ablation_config_done", config=label, **{
        k: record[k] for k in ("roc_auc", "pr_auc", "ks_statistic", "n_features")
    })
    return record


def ablate(spark) -> dict:
    """Run all 7 configs and assemble the comparison report."""
    all_tables = list(CHILD_TABLES.keys())

    configs: dict[str, list[str]] = {
        "application_only": [],
        "full": all_tables,
    }
    for t in all_tables:
        configs[f"drop_{t}"] = [x for x in all_tables if x != t]

    results = {label: _run_config(spark, label, tbls)
               for label, tbls in configs.items()}

    # ── Build the comparison view ─────────────────────────────
    full = results["full"]
    leave_one_out = {}
    for t in all_tables:
        cfg_res = results[f"drop_{t}"]
        leave_one_out[f"drop_{t}"] = {
            **cfg_res,
            "delta_roc_auc": _delta(full["roc_auc"], cfg_res["roc_auc"]),
            "delta_ks": _delta(full["ks_statistic"], cfg_res["ks_statistic"]),
        }
    # Rank tables by how much ROC-AUC drops when removed (bigger = more important).
    ranking = sorted(
        leave_one_out.items(),
        key=lambda kv: (kv[1]["delta_roc_auc"] or 0),
        reverse=True,
    )

    report = {
        "note": (
            "Leave-one-table-out over the 5 child tables that actually feed "
            "the model. bureau_balance is declared in config but has no "
            "aggregator, so it is not part of the join or this study."
        ),
        "child_tables_ablated": all_tables,
        "baseline_full": full,
        "application_only": results["application_only"],
        "multi_table_lift_roc_auc": _delta(
            full["roc_auc"], results["application_only"]["roc_auc"]
        ),
        "multi_table_lift_ks": _delta(
            full["ks_statistic"], results["application_only"]["ks_statistic"]
        ),
        "leave_one_out": leave_one_out,
        "importance_ranking_by_roc_auc_drop": [k for k, _ in ranking],
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(report, indent=2))
    log.info("ablation_report_saved", path=str(_OUT_PATH))
    return report


def _delta(a, b):
    """a - b, rounded; None-safe."""
    if a is None or b is None:
        return None
    return round(a - b, 4)


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["precompute", "ablate", "all"],
        default="all",
        help="precompute = aggregates+split only; ablate = 7 training runs; all = both",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("ablation_started", phase=args.phase)
    spark = get_spark("ablation-pipeline")
    try:
        if args.phase in ("precompute", "all"):
            precompute(spark)
        if args.phase in ("ablate", "all"):
            ablate(spark)
        log.info("ablation_completed", phase=args.phase)
    except Exception:
        log.exception("ablation_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    main()