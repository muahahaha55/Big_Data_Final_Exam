"""Data drift detection using PSI (Population Stability Index) and KS test.

Two complementary methods:
    - PSI: classical credit-risk industry metric (Basel-aligned thresholds)
        PSI < 0.10: no significant change
        0.10 ≤ PSI < 0.25: moderate shift, monitor
        PSI ≥ 0.25: significant shift, retrain
    - KS test: non-parametric, statistical p-value approach for any distribution

Why this matters: production models silently degrade when input distributions
drift (customer demographics change, economic conditions shift, data pipelines
break). Catching drift before performance metrics degrade is critical.

Reference: SR 11-7 (Federal Reserve), Basel ML risk management guidance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from scipy import stats

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


class DriftSeverity(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


@dataclass
class FeatureDriftResult:
    """Drift assessment for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: DriftSeverity
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    sample_size_reference: int
    sample_size_current: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DriftReport:
    """Overall drift report across many features."""

    timestamp: str
    n_features_checked: int
    n_features_drifting: int
    overall_severity: DriftSeverity
    feature_results: list[FeatureDriftResult]
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_features_checked": self.n_features_checked,
            "n_features_drifting": self.n_features_drifting,
            "overall_severity": self.overall_severity.value,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "recommendations": self.recommendations,
        }


# ──────────────────────────────────────────────────────────────
# PSI computation
# ──────────────────────────────────────────────────────────────

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index.

    PSI = Σ (current% - reference%) × ln(current% / reference%)

    Binning strategy: use quantile bins from reference distribution so we
    compare like-for-like even if current values are extreme.

    Args:
        reference: baseline values (e.g., training set)
        current: new values (e.g., last week of production)
        n_bins: number of bins (10 is industry standard)
        epsilon: avoid log(0) and division by zero

    Returns:
        PSI score (≥ 0).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Bin edges from reference quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(reference, quantiles)
    bin_edges = np.unique(bin_edges)  # drop dupes (when reference has ties)
    if len(bin_edges) < 3:
        return 0.0  # can't compute meaningfully

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts / ref_counts.sum()) + epsilon
    cur_pct = (cur_counts / cur_counts.sum()) + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def classify_psi(psi: float, warn_threshold: float = 0.10, alert_threshold: float = 0.25) -> DriftSeverity:
    """Classify PSI into severity bucket. Industry-standard thresholds."""
    if psi < warn_threshold:
        return DriftSeverity.NONE
    if psi < alert_threshold:
        return DriftSeverity.MODERATE
    return DriftSeverity.SIGNIFICANT


# ──────────────────────────────────────────────────────────────
# Per-feature drift
# ──────────────────────────────────────────────────────────────

def detect_feature_drift(
    feature_name: str,
    reference: np.ndarray,
    current: np.ndarray,
    warn_threshold: float = 0.10,
    alert_threshold: float = 0.25,
) -> FeatureDriftResult:
    """Run full drift assessment on a single feature."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    psi = calculate_psi(ref_clean, cur_clean)
    severity = classify_psi(psi, warn_threshold, alert_threshold)

    # KS test as a second opinion
    if len(ref_clean) > 1 and len(cur_clean) > 1:
        ks_stat, ks_pvalue = stats.ks_2samp(ref_clean, cur_clean)
    else:
        ks_stat, ks_pvalue = 0.0, 1.0

    return FeatureDriftResult(
        feature=feature_name,
        psi=round(psi, 4),
        ks_statistic=round(float(ks_stat), 4),
        ks_pvalue=round(float(ks_pvalue), 6),
        severity=severity,
        reference_mean=float(np.mean(ref_clean)) if len(ref_clean) else 0.0,
        current_mean=float(np.mean(cur_clean)) if len(cur_clean) else 0.0,
        reference_std=float(np.std(ref_clean)) if len(ref_clean) else 0.0,
        current_std=float(np.std(cur_clean)) if len(cur_clean) else 0.0,
        sample_size_reference=len(ref_clean),
        sample_size_current=len(cur_clean),
    )


# ──────────────────────────────────────────────────────────────
# Full report
# ──────────────────────────────────────────────────────────────

def generate_drift_report(
    reference_df,
    current_df,
    features: list[str],
    warn_threshold: float = 0.10,
    alert_threshold: float = 0.25,
) -> DriftReport:
    """Generate a full drift report.

    Works with either pandas DataFrames or Spark DataFrames (auto-detected).
    """
    # Convert to pandas if Spark
    if hasattr(reference_df, "toPandas"):
        reference_df = reference_df.select(*features).toPandas()
    if hasattr(current_df, "toPandas"):
        current_df = current_df.select(*features).toPandas()

    results: list[FeatureDriftResult] = []
    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            log.warning("drift_feature_missing", feature=feat)
            continue

        result = detect_feature_drift(
            feature_name=feat,
            reference=reference_df[feat].values,
            current=current_df[feat].values,
            warn_threshold=warn_threshold,
            alert_threshold=alert_threshold,
        )
        results.append(result)

        log.info(
            "drift_computed",
            feature=feat,
            psi=result.psi,
            severity=result.severity.value,
        )

    drifting = [r for r in results if r.severity != DriftSeverity.NONE]
    significant = [r for r in results if r.severity == DriftSeverity.SIGNIFICANT]

    overall = DriftSeverity.NONE
    if significant:
        overall = DriftSeverity.SIGNIFICANT
    elif drifting:
        overall = DriftSeverity.MODERATE

    recommendations = _build_recommendations(results, overall)

    report = DriftReport(
        timestamp=datetime.utcnow().isoformat(),
        n_features_checked=len(results),
        n_features_drifting=len(drifting),
        overall_severity=overall,
        feature_results=results,
        recommendations=recommendations,
    )

    log.info(
        "drift_report_generated",
        checked=report.n_features_checked,
        drifting=report.n_features_drifting,
        severity=overall.value,
    )
    return report


def _build_recommendations(results: list[FeatureDriftResult], overall: DriftSeverity) -> list[str]:
    recs: list[str] = []
    if overall == DriftSeverity.NONE:
        recs.append("No significant drift detected. Continue routine monitoring.")
        return recs

    significant = [r for r in results if r.severity == DriftSeverity.SIGNIFICANT]
    if significant:
        names = ", ".join(r.feature for r in significant[:5])
        recs.append(
            f"SIGNIFICANT drift on {len(significant)} feature(s): {names}. "
            "Recommend retraining model and investigating upstream data sources."
        )

    moderate = [r for r in results if r.severity == DriftSeverity.MODERATE]
    if moderate:
        names = ", ".join(r.feature for r in moderate[:5])
        recs.append(
            f"MODERATE drift on {len(moderate)} feature(s): {names}. "
            "Monitor closely; consider retraining if trend persists."
        )

    return recs
