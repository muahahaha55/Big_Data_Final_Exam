"""Training pipeline — load processed data, fit model, register to MLflow.

Run via: make train
"""

from __future__ import annotations

from credit_risk.config import get_config, project_root
from credit_risk.features.pipeline import (
    build_pipeline,
    identify_feature_columns,
)
from credit_risk.models.registry import MLflowRegistry
from credit_risk.models.trainer import train_model
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_training(
    model_type: str | None = None,
    tune: bool = False,
    promote_to_production: bool = True,
) -> None:
    """Train + evaluate + register a model."""
    configure_logging()
    log = get_logger("pipeline.training")
    cfg = get_config()
    model_type = model_type or cfg["models"]["default_model"]

    log.info("training_started", model=model_type, tune=tune, promote=promote_to_production)
    spark = get_spark("training-pipeline")

    try:
        # ── Load splits ───────────────────────────────────────
        train_path = project_root() / "data" / "05_model_input" / "train.parquet"
        test_path = project_root() / "data" / "05_model_input" / "test.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Run `make etl` first — {train_path} not found")

        train_df = spark.read.parquet(str(train_path))
        test_df = spark.read.parquet(str(test_path))
        log.info("data_loaded", train_rows=train_df.count(), test_rows=test_df.count())

        # ── Build feature pipeline ────────────────────────────
        feature_cols = identify_feature_columns(
            train_df,
            target_col=cfg["data"]["target_column"],
            id_cols=[cfg["data"]["id_column"]],
            drop_threshold=cfg["features"]["missing_threshold"],
        )
        feature_pipeline = build_pipeline(feature_cols, scaler=True)

        # ── Train ────────────────────────────────────────────
        result = train_model(
            train_df=train_df,
            test_df=test_df,
            feature_pipeline=feature_pipeline,
            model_type=model_type,
            target_col=cfg["data"]["target_column"],
            tune=tune,
            register=True,
        )

        log.info(
            "training_metrics",
            **{k: v for k, v in result.metrics.items() if isinstance(v, (int, float))},
        )

        # ── Promote to Production (if AUC is acceptable) ─────
        if promote_to_production and result.metrics.get("roc_auc", 0) >= 0.70:
            registry = MLflowRegistry()
            model_name = cfg["mlflow"]["registered_model_name"]
            latest = registry.list_versions(model_name)
            if latest:
                newest = max(latest, key=lambda v: int(v.version))
                registry.promote_model(model_name, newest.version, stage="Production")
                log.info("model_promoted", name=model_name, version=newest.version)

        log.info("training_pipeline_completed", roc_auc=result.metrics.get("roc_auc"))

    except Exception:
        log.exception("training_pipeline_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="logistic | random_forest | gbt")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--no-promote", action="store_true", help="Skip auto-promotion to Production")
    args = parser.parse_args()
    run_training(model_type=args.model, tune=args.tune, promote_to_production=not args.no_promote)
