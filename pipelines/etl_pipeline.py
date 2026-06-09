"""ETL pipeline — ingest, join, clean, save processed data.

Run via: make etl  (or python pipelines/etl_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

from credit_risk.config import get_config, project_root
from credit_risk.data.ingestion.multi_table import build_master_dataset
from credit_risk.features.pipeline import add_domain_features
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_etl(use_multi_table: bool = True) -> None:
    """Full ETL: load → join → enrich → save to Parquet."""
    configure_logging()
    log = get_logger("pipeline.etl")
    log.info("etl_started", multi_table=use_multi_table)

    spark = get_spark("etl-pipeline")

    try:
        # ── Load + join all tables ────────────────────────────
        master = build_master_dataset(spark, use_multi_table=use_multi_table)
        log.info("master_built", rows=master.count(), cols=len(master.columns))

        # ── Add domain features ───────────────────────────────
        master = add_domain_features(master)

        # ── Save to intermediate layer ────────────────────────
        out_path = project_root() / "data" / "02_intermediate" / "master.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        master.write.mode("overwrite").parquet(str(out_path))
        log.info("etl_saved", path=str(out_path))

        # ── Save train/test split ─────────────────────────────
        cfg = get_config()["data"]["split"]
        train_df, val_df, test_df = master.randomSplit(
            [cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"]],
            seed=get_config().get("random_seed", 42),
        )

        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = project_root() / "data" / "05_model_input" / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write.mode("overwrite").parquet(str(path))
            log.info("split_saved", name=name, rows=df.count(), path=str(path))

        log.info("etl_completed_successfully")
    except Exception:
        log.exception("etl_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-multi-table", action="store_true", help="Use only application table")
    args = parser.parse_args()
    run_etl(use_multi_table=not args.no_multi_table)
