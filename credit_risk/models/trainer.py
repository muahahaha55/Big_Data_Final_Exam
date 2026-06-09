"""Model trainer with full MLflow tracking.

Trains Logistic Regression / Random Forest / GBT on a Spark DataFrame,
logs everything (params, metrics, model artifacts) to MLflow, and
registers the best model in the model registry.

Supports CrossValidator hyperparameter tuning when enabled in config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.config import get_config
from credit_risk.models.registry import MLflowRegistry
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Result of one training run."""

    model_name: str
    pipeline: PipelineModel | Pipeline
    metrics: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    feature_count: int = 0
    train_rows: int = 0
    test_rows: int = 0
    mlflow_run_id: str | None = None
    mlflow_model_uri: str | None = None


# ──────────────────────────────────────────────────────────────
# Classifier factory
# ──────────────────────────────────────────────────────────────

def build_classifier(model_type: str, target_col: str = "TARGET", features_col: str = "features"):
    """Build a Spark MLlib classifier based on model_type and config."""
    cfg = get_config()["models"]

    if model_type == "logistic":
        p = cfg.get("logistic", {})
        return LogisticRegression(
            labelCol=target_col, featuresCol=features_col,
            maxIter=p.get("maxIter", 100),
            regParam=p.get("regParam", 0.01),
            elasticNetParam=p.get("elasticNetParam", 0.0),
            standardization=p.get("standardization", True),
        )

    if model_type == "random_forest":
        p = cfg.get("random_forest", {})
        return RandomForestClassifier(
            labelCol=target_col, featuresCol=features_col, seed=get_config().get("random_seed", 42),
            numTrees=p.get("numTrees", 100),
            maxDepth=p.get("maxDepth", 10),
            maxBins=p.get("maxBins", 64),
            subsamplingRate=p.get("subsamplingRate", 0.8),
            featureSubsetStrategy=p.get("featureSubsetStrategy", "sqrt"),
        )

    if model_type == "gbt":
        p = cfg.get("gbt", {})
        return GBTClassifier(
            labelCol=target_col, featuresCol=features_col, seed=get_config().get("random_seed", 42),
            maxIter=p.get("maxIter", 100),
            maxDepth=p.get("maxDepth", 5),
            stepSize=p.get("stepSize", 0.1),
            subsamplingRate=p.get("subsamplingRate", 0.8),
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# ──────────────────────────────────────────────────────────────
# Class-imbalance handling (Home Credit is ~92/8 imbalanced)
# ──────────────────────────────────────────────────────────────

def add_class_weights(df: DataFrame, target_col: str = "TARGET") -> DataFrame:
    """Add a weightCol that balances positive vs negative class."""
    pos_count = df.filter(F.col(target_col) == 1).count()
    neg_count = df.filter(F.col(target_col) == 0).count()
    total = pos_count + neg_count
    if pos_count == 0 or neg_count == 0:
        return df.withColumn("classWeight", F.lit(1.0))

    weight_pos = total / (2.0 * pos_count)
    weight_neg = total / (2.0 * neg_count)

    log.info("class_weights_computed", pos_weight=weight_pos, neg_weight=weight_neg)
    return df.withColumn(
        "classWeight",
        F.when(F.col(target_col) == 1, F.lit(weight_pos)).otherwise(F.lit(weight_neg))
    )


# ──────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ──────────────────────────────────────────────────────────────

def build_param_grid(model_type: str, classifier) -> list:
    """Build a ParamGrid from config for the given model."""
    cfg = get_config()["models"]["tuning"]
    grid_cfg = cfg.get("grids", {}).get(model_type, {})
    if not grid_cfg:
        return []

    builder = ParamGridBuilder()
    for param_name, values in grid_cfg.items():
        param = getattr(classifier, param_name, None)
        if param is None:
            log.warning("unknown_param", param=param_name, model=model_type)
            continue
        builder = builder.addGrid(param, values)
    return builder.build()


# ──────────────────────────────────────────────────────────────
# Main training entrypoint
# ──────────────────────────────────────────────────────────────

def train_model(
    train_df: DataFrame,
    test_df: DataFrame,
    feature_pipeline: Pipeline,
    model_type: str = "gbt",
    target_col: str = "TARGET",
    tune: bool = False,
    register: bool = True,
) -> TrainingResult:
    """Full training: fit pipeline, optionally tune, evaluate, log to MLflow.

    Args:
        train_df: training set (with label column)
        test_df: held-out test set
        feature_pipeline: preprocessing Pipeline (Imputer/Indexer/OHE/Assembler/Scaler)
        model_type: 'logistic' | 'random_forest' | 'gbt'
        target_col: label column name
        tune: run CrossValidator hyperparameter search
        register: register model in MLflow registry

    Returns:
        TrainingResult with metrics, model, mlflow run id
    """
    registry = MLflowRegistry()
    cfg = get_config()
    run_name = f"{model_type}_{'tuned' if tune else 'baseline'}"

    with registry.start_run(run_name=run_name, tags={"model_type": model_type, "tuned": str(tune)}) as run:
        # ── Build full pipeline (preprocessing + classifier) ──
        classifier = build_classifier(model_type, target_col=target_col)
        full_pipeline = Pipeline(stages=list(feature_pipeline.getStages()) + [classifier])

        # ── Fit (with or without tuning) ──
        if tune and cfg["models"]["tuning"].get("enabled", False):
            param_grid = build_param_grid(model_type, classifier)
            if param_grid:
                log.info("hyperparameter_tuning", n_combos=len(param_grid), folds=cfg["models"]["tuning"]["num_folds"])
                evaluator = BinaryClassificationEvaluator(
                    labelCol=target_col, metricName="areaUnderROC",
                )
                cv = CrossValidator(
                    estimator=full_pipeline,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=cfg["models"]["tuning"]["num_folds"],
                    parallelism=2,
                    seed=cfg.get("random_seed", 42),
                )
                cv_model = cv.fit(train_df)
                fitted = cv_model.bestModel
                best_params = _extract_best_params(cv_model)
            else:
                fitted = full_pipeline.fit(train_df)
                best_params = {}
        else:
            fitted = full_pipeline.fit(train_df)
            best_params = {}

        # ── Evaluate on test ──
        predictions = fitted.transform(test_df)
        metrics = evaluate_predictions(predictions, target_col)

        # ── Log to MLflow ──
        registry.log_params({
            "model_type": model_type,
            "tuned": tune,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            **best_params,
        })
        registry.log_metrics(metrics)

        model_uri = None
        if register:
            model_uri = registry.log_spark_model(
                fitted,
                artifact_path="model",
                registered_model_name=cfg["mlflow"]["registered_model_name"],
            )

        result = TrainingResult(
            model_name=run_name,
            pipeline=fitted,
            metrics=metrics,
            best_params=best_params,
            feature_count=_count_features(fitted),
            train_rows=train_df.count(),
            test_rows=test_df.count(),
            mlflow_run_id=run.info.run_id,
            mlflow_model_uri=model_uri,
        )

        log.info(
            "training_completed",
            model=model_type, roc_auc=metrics.get("roc_auc"),
            pr_auc=metrics.get("pr_auc"), run_id=run.info.run_id,
        )
        return result


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_predictions(predictions: DataFrame, target_col: str = "TARGET") -> dict[str, float]:
    """Compute classification metrics from a predictions DataFrame."""
    bin_eval = BinaryClassificationEvaluator(labelCol=target_col)
    multi_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction")

    metrics = {
        "roc_auc": bin_eval.setMetricName("areaUnderROC").evaluate(predictions),
        "pr_auc": bin_eval.setMetricName("areaUnderPR").evaluate(predictions),
        "accuracy": multi_eval.setMetricName("accuracy").evaluate(predictions),
        "f1": multi_eval.setMetricName("f1").evaluate(predictions),
        "weighted_precision": multi_eval.setMetricName("weightedPrecision").evaluate(predictions),
        "weighted_recall": multi_eval.setMetricName("weightedRecall").evaluate(predictions),
    }

    # KS statistic from probability column
    try:
        prob_pdf = predictions.select(target_col, "probability").toPandas()
        prob_pdf["prob_default"] = prob_pdf["probability"].apply(lambda v: float(v[1]))
        from scipy.stats import ks_2samp
        ks = ks_2samp(
            prob_pdf.loc[prob_pdf[target_col] == 1, "prob_default"],
            prob_pdf.loc[prob_pdf[target_col] == 0, "prob_default"],
        )
        metrics["ks_statistic"] = float(ks.statistic)
    except Exception as e:
        log.warning("ks_compute_failed", error=str(e))

    return {k: round(float(v), 4) for k, v in metrics.items()}


def _extract_best_params(cv_model) -> dict:
    """Extract chosen hyperparameters from a CrossValidatorModel."""
    try:
        best_stages = cv_model.bestModel.stages
        classifier = best_stages[-1]
        params = classifier.extractParamMap()
        return {p.name: v for p, v in params.items() if p.name in {
            "maxDepth", "maxIter", "regParam", "elasticNetParam",
            "numTrees", "stepSize", "subsamplingRate"
        }}
    except Exception as e:
        log.warning("best_params_extraction_failed", error=str(e))
        return {}


def _count_features(pipeline_model: PipelineModel) -> int:
    """Count number of input features going into the classifier."""
    try:
        # VectorAssembler stage knows the input vector size
        for stage in pipeline_model.stages:
            if hasattr(stage, "getInputCols"):
                cols = stage.getInputCols()
                if cols:
                    return len(cols)
    except Exception:
        pass
    return 0
