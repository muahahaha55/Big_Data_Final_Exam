"""Weight of Evidence (WoE) and Information Value (IV) scorecard analysis.

Industry-standard credit risk methodology:
    WoE measures the predictive strength of a feature bin
        WoE = ln(% good / % bad)

    IV aggregates WoE across all bins, giving a single feature-strength score
        IV = Σ (% good - % bad) × WoE

Interpretation (Siddiqi 2006):
    IV < 0.02    : useless
    0.02 - 0.10  : weak
    0.10 - 0.30  : medium
    0.30 - 0.50  : strong
    > 0.50       : suspicious (data leakage?)

Reference: Siddiqi, N. (2006). Credit Risk Scorecards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IVResult:
    """Information Value for a single feature."""

    feature: str
    iv: float
    strength: str
    n_bins: int
    bin_details: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "iv": round(self.iv, 4),
            "strength": self.strength,
            "n_bins": self.n_bins,
            "bins": self.bin_details.to_dict(orient="records"),
        }


def classify_iv(iv: float) -> str:
    """Bucket IV into Siddiqi's predictive-power categories."""
    if iv < 0.02:
        return "useless"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspicious"


# ──────────────────────────────────────────────────────────────
# Numeric binning
# ──────────────────────────────────────────────────────────────

def compute_woe_iv_numeric(
    df: DataFrame,
    feature: str,
    target: str = "TARGET",
    n_bins: int = 10,
    epsilon: float = 0.5,
) -> IVResult:
    """Compute WoE and IV for a numeric feature using quantile binning."""
    pdf = df.select(feature, target).toPandas()
    pdf = pdf.dropna()

    if len(pdf) == 0 or pdf[feature].nunique() < 2:
        return IVResult(feature, 0.0, "useless", 0, pd.DataFrame())

    try:
        pdf["bin"] = pd.qcut(pdf[feature], q=n_bins, duplicates="drop", labels=False)
    except ValueError:
        return IVResult(feature, 0.0, "useless", 0, pd.DataFrame())

    return _aggregate_bins(pdf, feature, target, epsilon)


def compute_woe_iv_categorical(
    df: DataFrame,
    feature: str,
    target: str = "TARGET",
    epsilon: float = 0.5,
) -> IVResult:
    """Compute WoE and IV for a categorical feature (one bin per category)."""
    pdf = df.select(feature, target).toPandas()
    pdf = pdf.dropna()
    pdf["bin"] = pdf[feature].astype(str)

    if len(pdf) == 0 or pdf["bin"].nunique() < 2:
        return IVResult(feature, 0.0, "useless", 0, pd.DataFrame())

    return _aggregate_bins(pdf, feature, target, epsilon)


def _aggregate_bins(pdf: pd.DataFrame, feature: str, target: str, epsilon: float) -> IVResult:
    """Aggregate per-bin counts → WoE → IV."""
    total_good = (pdf[target] == 0).sum()
    total_bad = (pdf[target] == 1).sum()

    grouped = pdf.groupby("bin", observed=True).agg(
        n=("bin", "size"),
        bad=(target, "sum"),
    ).reset_index()
    grouped["good"] = grouped["n"] - grouped["bad"]
    grouped["bad"] = grouped["bad"].clip(lower=epsilon)
    grouped["good"] = grouped["good"].clip(lower=epsilon)

    grouped["pct_bad"] = grouped["bad"] / (total_bad + epsilon)
    grouped["pct_good"] = grouped["good"] / (total_good + epsilon)
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv_contribution"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    iv = float(grouped["iv_contribution"].sum())
    strength = classify_iv(iv)

    grouped = grouped.round(4)

    log.info("woe_iv_computed", feature=feature, iv=round(iv, 4), strength=strength)

    return IVResult(
        feature=feature,
        iv=iv,
        strength=strength,
        n_bins=len(grouped),
        bin_details=grouped,
    )


# ──────────────────────────────────────────────────────────────
# Bulk analysis
# ──────────────────────────────────────────────────────────────

def analyze_features(
    df: DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str = "TARGET",
    n_bins: int = 10,
    min_iv: float = 0.02,
) -> pd.DataFrame:
    """Compute IV for many features, return ranked summary table."""
    results: list[IVResult] = []

    for feat in numeric_features:
        try:
            r = compute_woe_iv_numeric(df, feat, target, n_bins)
            results.append(r)
        except Exception as e:
            log.warning("woe_iv_failed", feature=feat, error=str(e))

    for feat in categorical_features:
        try:
            r = compute_woe_iv_categorical(df, feat, target)
            results.append(r)
        except Exception as e:
            log.warning("woe_iv_failed", feature=feat, error=str(e))

    summary = pd.DataFrame([
        {"feature": r.feature, "iv": r.iv, "strength": r.strength, "n_bins": r.n_bins}
        for r in results
    ])
    summary = summary[summary["iv"] >= min_iv].sort_values("iv", ascending=False).reset_index(drop=True)

    log.info("feature_analysis_completed", n_features=len(results), n_above_threshold=len(summary))
    return summary
