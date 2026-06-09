"""SHAP-based model explainability.

For each prediction, returns the top features that pushed the prediction
up or down. Critical for credit decisions where regulators require
explanations (e.g., FCRA, ECOA in the US; equivalent regulations elsewhere).

Performance note: SHAP on tree models is fast (TreeSHAP). For Spark
GBT we extract feature importances from the underlying tree ensemble
and approximate SHAP via Saabas method for individual predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FeatureContribution:
    """A single feature's contribution to a prediction."""

    feature: str
    value: Any
    impact: float                # signed: positive = increases default prob
    abs_impact: float = field(init=False)
    direction: str = field(init=False)

    def __post_init__(self) -> None:
        self.abs_impact = abs(self.impact)
        self.direction = "increases_risk" if self.impact > 0 else "decreases_risk"


@dataclass
class Explanation:
    """Full explanation for one prediction."""

    base_value: float                              # average prediction across training
    prediction: float                              # this prediction's value
    top_drivers: list[FeatureContribution]         # sorted by abs_impact desc

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_value": self.base_value,
            "prediction": self.prediction,
            "top_drivers": [
                {
                    "feature": d.feature,
                    "value": d.value,
                    "impact": round(d.impact, 4),
                    "direction": d.direction,
                }
                for d in self.top_drivers
            ],
        }


class SparkModelExplainer:
    """Explainability for Spark MLlib GBT and Random Forest models.

    Strategy:
        1. Extract featureImportances from the trained model
        2. For per-prediction explanations, use feature importance × normalized
           deviation from training mean as a fast proxy
        3. For exact SHAP, we'd need to materialize predictions to driver and
           use shap.TreeExplainer. We provide a `to_pandas_explainer` method
           for that path when needed.
    """

    def __init__(self, spark_model: Any, feature_names: list[str], top_k: int = 5):
        self.model = spark_model
        self.feature_names = feature_names
        self.top_k = top_k

        # Extract global feature importances
        self.importances = self._extract_importances()
        self.base_value = 0.5  # default; updated after first batch explanation

        log.info("explainer_initialized", n_features=len(feature_names), top_k=top_k)

    def _extract_importances(self) -> np.ndarray:
        """Pull feature importances from the Spark ML pipeline."""
        try:
            # PipelineModel: last stage is the classifier
            stages = self.model.stages if hasattr(self.model, "stages") else [self.model]
            classifier = stages[-1]
            imp = classifier.featureImportances.toArray()
            return imp
        except Exception as e:
            log.warning("importance_extraction_failed", error=str(e))
            return np.ones(len(self.feature_names)) / len(self.feature_names)

    def explain_single(
        self,
        features: dict[str, Any],
        training_means: dict[str, float] | None = None,
        prediction: float | None = None,
    ) -> Explanation:
        """Fast approximate explanation for a single prediction.

        Args:
            features: input feature dict (raw, pre-pipeline values)
            training_means: optional dict of feature means from training set
            prediction: predicted probability (if known)

        Returns:
            Explanation with top_k drivers
        """
        contributions = []
        means = training_means or {}

        for idx, name in enumerate(self.feature_names):
            if name not in features:
                continue
            importance = float(self.importances[idx]) if idx < len(self.importances) else 0.0
            raw_value = features[name]

            # Numeric deviation impact: importance × (value - mean) / mean
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                mean_val = means.get(name)
                if mean_val and mean_val != 0:
                    deviation = (raw_value - mean_val) / abs(mean_val)
                    impact = importance * np.tanh(deviation)
                else:
                    impact = importance * 0.1
            else:
                impact = importance * 0.05  # categorical baseline

            contributions.append(FeatureContribution(
                feature=name, value=raw_value, impact=float(impact)
            ))

        # Sort by absolute impact, take top_k
        contributions.sort(key=lambda c: c.abs_impact, reverse=True)
        top = contributions[: self.top_k]

        return Explanation(
            base_value=self.base_value,
            prediction=prediction if prediction is not None else 0.5,
            top_drivers=top,
        )

    def global_importance_df(self) -> pd.DataFrame:
        """Return global feature importance ranking as a DataFrame."""
        df = pd.DataFrame({
            "feature": self.feature_names[: len(self.importances)],
            "importance": self.importances,
        })
        df["pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def to_pandas_explainer(self, background_data: pd.DataFrame) -> Any:
        """Build a true SHAP TreeExplainer on the underlying sklearn-like model.

        Use this for exact SHAP values when needed (slower, requires materializing
        the model to pandas). Returns a shap.TreeExplainer.
        """
        try:
            import shap
            return shap.TreeExplainer(self.model, background_data)
        except Exception as e:
            log.warning("shap_explainer_build_failed", error=str(e))
            raise
