"""Evaluation pipeline — compare multiple registered models.

Run via: make evaluate
"""

from __future__ import annotations

import json
from datetime import datetime

from credit_risk.config import get_config, project_root
from credit_risk.models.registry import MLflowRegistry
from credit_risk.models.trainer import evaluate_predictions
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_evaluation() -> None:
    """Evaluate all registered model versions on the held-out test set."""
    configure_logging()
    log = get_logger("pipeline.evaluation")
    cfg = get_config()
    log.info("evaluation_started")

    spark = get_spark("evaluation-pipeline")

    try:
        test_path = project_root() / "data" / "05_model_input" / "test.parquet"
        test_df = spark.read.parquet(str(test_path))

        registry = MLflowRegistry()
        model_name = cfg["mlflow"]["registered_model_name"]
        versions = registry.list_versions(model_name)

        if not versions:
            log.warning("no_registered_models", name=model_name)
            return

        results: list[dict] = []
        target_col = cfg["data"]["target_column"]

        for v in versions:
            try:
                import mlflow.spark
                model = mlflow.spark.load_model(v.uri)
                predictions = model.transform(test_df)
                metrics = evaluate_predictions(predictions, target_col)
                row = {
                    "version": v.version, "stage": v.stage, "run_id": v.run_id, **metrics,
                }
                results.append(row)
                log.info("version_evaluated", version=v.version, roc_auc=metrics["roc_auc"])
            except Exception as e:
                log.warning("version_evaluation_failed", version=v.version, error=str(e))

        # Save report
        report = {
            "model_name": model_name,
            "evaluated_at": datetime.utcnow().isoformat(),
            "test_set_size": test_df.count(),
            "n_versions": len(results),
            "results": sorted(results, key=lambda r: r.get("roc_auc", 0), reverse=True),
        }

        out_path = project_root() / "data" / "08_reporting" / "evaluation_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print headline
        print("\n" + "=" * 70)
        print(f"  MODEL EVALUATION  ({model_name})")
        print("=" * 70)
        print(f"  {'Version':<10}{'Stage':<14}{'ROC-AUC':<12}{'PR-AUC':<12}{'KS':<10}")
        print("-" * 70)
        for r in report["results"]:
            print(f"  {r['version']:<10}{r['stage']:<14}"
                  f"{r['roc_auc']:<12.4f}{r['pr_auc']:<12.4f}{r.get('ks_statistic', 0):<10.4f}")
        print("=" * 70 + "\n")

        log.info("evaluation_completed")

    except Exception:
        log.exception("evaluation_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    run_evaluation()
