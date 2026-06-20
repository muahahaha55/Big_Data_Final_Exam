"""Categorical drift pipeline — Jensen-Shannon divergence detection.

Complements the numeric PSI+KS monitoring (monitoring_pipeline.py) by
checking categorical features for distributional drift between reference
and current data, using JS divergence.

Run via: python pipelines/categorical_drift_pipeline.py

Reference:
    - Fair and Explainable Credit-Scoring under Concept Drift (2025).
      arXiv:2511.03807.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from credit_risk.config import project_root
from credit_risk.monitoring.categorical_drift import detect_categorical_drift
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


# Auto-detect candidate categorical columns: string type, low cardinality
MAX_CARDINALITY = 20
EXCLUDE_COLUMNS = {"SK_ID_CURR", "TARGET"}


def find_categorical_columns(df, max_categories: int = MAX_CARDINALITY) -> list[str]:
    """Find string columns with manageable cardinality for drift checks."""
    string_cols = [
        f.name for f in df.schema.fields
        if str(f.dataType) == "StringType()" and f.name not in EXCLUDE_COLUMNS
    ]
    candidates = []
    for c in string_cols:
        n_distinct = df.select(c).distinct().count()
        if 1 < n_distinct <= max_categories:
            candidates.append(c)
    return candidates


def run_categorical_drift_check() -> None:
    """Detect categorical drift between reference (train) and current (test) data."""
    configure_logging()
    log = get_logger("pipeline.categorical_drift")
    log.info("categorical_drift_check_started")

    spark = get_spark("categorical-drift-pipeline")

    try:
        ref_path = str(project_root() / "data" / "05_model_input" / "train.parquet")
        cur_path = str(project_root() / "data" / "05_model_input" / "test.parquet")

        reference_df = spark.read.parquet(ref_path)
        current_df = spark.read.parquet(cur_path)
        log.info("data_loaded", reference=reference_df.count(), current=current_df.count())

        categorical_features = find_categorical_columns(reference_df)
        log.info("categorical_columns_found", n=len(categorical_features), columns=categorical_features[:10])

        if not categorical_features:
            print("\n  No categorical columns with cardinality <= "
                  f"{MAX_CARDINALITY} found. Skipping.\n")
            return

        results = detect_categorical_drift(reference_df, current_df, categorical_features)

        print("\n" + "=" * 60)
        print(f"  CATEGORICAL DRIFT REPORT (JS Divergence)  "
              f"({datetime.now(timezone.utc).isoformat()})")
        print("=" * 60)
        for r in results:
            new_cats_str = f", new={r.new_categories}" if r.new_categories else ""
            print(f"  {r.feature:<30} JS={r.js_divergence:.4f}  [{r.severity}]{new_cats_str}")
        print("=" * 60)
        print("\n  Reference: Fair and Explainable Credit-Scoring under Concept Drift")
        print("             (2025), arXiv:2511.03807\n")

        out_path = project_root() / "data" / "08_reporting" / "categorical_drift_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        n_drifting = sum(1 for r in results if r.severity != "none")
        log.info(
            "categorical_drift_check_completed",
            n_checked=len(results),
            n_drifting=n_drifting,
        )

    except Exception:
        log.exception("categorical_drift_check_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    run_categorical_drift_check()