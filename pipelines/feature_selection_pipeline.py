"""Feature selection pipeline — Kruskal-Wallis non-parametric ranking.

Runs on the engineered train dataset, ranking monitored features by their
Kruskal-Wallis H-statistic against TARGET. Saves a JSON report consumable
by the report/slide generation step.

Run via: python pipelines/feature_selection_pipeline.py

Reference:
    - Ashofteh, A. & Bravo, J.M. (2021). Expert Systems with Applications, 176.
    - Predicting mortgage credit defaults in Morocco (2025). Discover AI, Springer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from credit_risk.config import get_config, project_root
from credit_risk.features.kruskal_selection import kruskal_wallis_ranking
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_feature_selection() -> None:
    """Rank monitored features by Kruskal-Wallis H-statistic."""
    configure_logging()
    log = get_logger("pipeline.feature_selection")
    cfg = get_config()
    log.info("feature_selection_started")

    spark = get_spark("feature-selection-pipeline")

    try:
        train_path = str(project_root() / "data" / "05_model_input" / "train.parquet")
        df = spark.read.parquet(train_path)
        log.info("data_loaded", rows=df.count())

        # Reuse the same feature list already verified in drift monitoring
        features = cfg["monitoring"]["drift"]["monitored_features"]

        ranking = kruskal_wallis_ranking(df, features, target="TARGET", top_k=len(features))

        print("\n" + "=" * 60)
        print(f"  KRUSKAL-WALLIS FEATURE RANKING  ({datetime.now(timezone.utc).isoformat()})")
        print("=" * 60)
        print(ranking.to_string(index=False))
        print("=" * 60)
        print("\n  Reference: Ashofteh & Bravo (2021), Expert Systems with Applications")
        print("  Reference: Morocco mortgage credit defaults (2025), Discover AI\n")

        out_path = project_root() / "data" / "08_reporting" / "feature_ranking_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_json(out_path, orient="records", indent=2)

        n_significant = int(ranking["significant"].sum())
        log.info(
            "feature_selection_completed",
            n_features=len(features),
            n_significant=n_significant,
        )

    except Exception:
        log.exception("feature_selection_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    run_feature_selection()