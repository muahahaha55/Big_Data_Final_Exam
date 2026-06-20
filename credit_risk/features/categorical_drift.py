"""Jensen-Shannon divergence for categorical feature drift detection.

PSI and KS test work well for numeric features but cannot handle categorical
ones meaningfully. JS divergence is bounded [0, 1], symmetric (unlike KL),
and interpretable as a distance metric — making it the canonical choice for
detecting categorical drift in production credit scoring systems.

References:
    - Fair and Explainable Credit-Scoring under Concept Drift (2025).
      arXiv:2511.03807.
    - Lin, J. (1991). Divergence measures based on the Shannon entropy.
      IEEE Transactions on Information Theory, 37(1), 145-151.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from pyspark.sql import DataFrame
from scipy.spatial.distance import jensenshannon

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CategoricalDriftResult:
    """Drift detection result for one categorical feature."""

    feature: str
    js_divergence: float
    severity: str  # 'none', 'low', 'medium', 'high'
    n_categories_reference: int
    n_categories_current: int
    new_categories: list[str]

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "js_divergence": round(self.js_divergence, 4),
            "severity": self.severity,
            "n_categories_reference": self.n_categories_reference,
            "n_categories_current": self.n_categories_current,
            "new_categories": self.new_categories,
        }


def classify_js_severity(js: float) -> str:
    """Classify JS divergence severity using thresholds from drift literature."""
    if js < 0.05:
        return "none"
    elif js < 0.15:
        return "low"
    elif js < 0.30:
        return "medium"
    else:
        return "high"


def compute_js_divergence(reference: list, current: list) -> float:
    """Compute Jensen-Shannon divergence between two categorical samples.

    Args:
        reference: list of categorical values from reference period.
        current: list of categorical values from current period.

    Returns:
        JS divergence in [0, 1]. 0 = identical distributions, 1 = no overlap.
    """
    all_categories = sorted(set(reference) | set(current))

    ref_counts = Counter(reference)
    cur_counts = Counter(current)
    ref_total = len(reference)
    cur_total = len(current)

    ref_dist = np.array([ref_counts.get(c, 0) / ref_total for c in all_categories])
    cur_dist = np.array([cur_counts.get(c, 0) / cur_total for c in all_categories])

    js = float(jensenshannon(ref_dist, cur_dist))
    return js if not np.isnan(js) else 0.0


def detect_categorical_drift(
    reference_df: DataFrame,
    current_df: DataFrame,
    categorical_features: list[str],
) -> list[CategoricalDriftResult]:
    """Detect drift on categorical features using JS divergence.

    Args:
        reference_df: Spark DataFrame from training period (baseline).
        current_df: Spark DataFrame from current period.
        categorical_features: List of categorical column names to check.

    Returns:
        List of CategoricalDriftResult, one per feature.
    """
    log.info(
        "categorical_drift_started",
        n_features=len(categorical_features),
    )

    results = []
    for feature in categorical_features:
        ref_values = [
            row[feature]
            for row in reference_df.select(feature).collect()
            if row[feature] is not None
        ]
        cur_values = [
            row[feature]
            for row in current_df.select(feature).collect()
            if row[feature] is not None
        ]

        if not ref_values or not cur_values:
            continue

        js = compute_js_divergence(ref_values, cur_values)
        ref_set = set(ref_values)
        cur_set = set(cur_values)
        new_cats = list(cur_set - ref_set)

        result = CategoricalDriftResult(
            feature=feature,
            js_divergence=js,
            severity=classify_js_severity(js),
            n_categories_reference=len(ref_set),
            n_categories_current=len(cur_set),
            new_categories=new_cats,
        )
        results.append(result)

        log.info(
            "categorical_drift_computed",
            feature=feature,
            js=round(js, 4),
            severity=result.severity,
        )

    return results