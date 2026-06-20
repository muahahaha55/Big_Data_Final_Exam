"""Non-parametric feature selection using Kruskal-Wallis H-test.

Kruskal-Wallis is a rank-based alternative to ANOVA, suitable for credit
risk data where features are typically skewed and non-normal. Unlike
parametric methods, it makes no distributional assumptions and is robust
to outliers — both common in financial data.

References:
    - Ashofteh, A. & Bravo, J.M. (2021). A conservative approach for online
      credit scoring. Expert Systems with Applications, 176.
    - Predicting mortgage credit defaults in Morocco using machine learning
      approaches (2025). Discover Artificial Intelligence, Springer.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from scipy.stats import kruskal

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


def kruskal_wallis_ranking(
    df: DataFrame,
    features: list[str],
    target: str = "TARGET",
    top_k: int = 20,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Rank features by Kruskal-Wallis H-statistic against the binary target.

    Args:
        df: Spark DataFrame containing features and target column.
        features: List of numeric feature column names to rank.
        target: Binary target column name (default 'TARGET').
        top_k: Number of top features to return (default 20).
        significance_level: P-value threshold for significance (default 0.05).

    Returns:
        Pandas DataFrame with columns:
            - feature: feature name
            - h_statistic: Kruskal-Wallis H statistic (higher = stronger signal)
            - p_value: statistical significance
            - significant: bool flag for p < significance_level
            - rank: ranking position (1 = strongest)

    Notes:
        Higher H-statistic indicates the feature distribution differs more
        between default and non-default groups. Significant features
        (p < 0.05) are recommended for inclusion in the model.
    """
    log.info("kruskal_wallis_started", n_features=len(features), top_k=top_k)

    # Pull data to driver — small enough for this analysis
    pdf = df.select(features + [target]).toPandas()

    results = []
    for feat in features:
        valid = pdf[[feat, target]].dropna()
        if len(valid) < 30:
            continue

        group_0 = valid.loc[valid[target] == 0, feat]
        group_1 = valid.loc[valid[target] == 1, feat]

        if len(group_0) == 0 or len(group_1) == 0:
            continue

        try:
            h_stat, p_value = kruskal(group_0, group_1)
            results.append({
                "feature": feat,
                "h_statistic": round(float(h_stat), 2),
                "p_value": float(p_value),
                "significant": p_value < significance_level,
            })
        except Exception as e:
            log.warning("kruskal_failed", feature=feat, error=str(e))

    result_df = pd.DataFrame(results).sort_values(
        "h_statistic", ascending=False
    ).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1

    n_significant = int(result_df["significant"].sum())
    log.info(
        "kruskal_wallis_completed",
        n_tested=len(result_df),
        n_significant=n_significant,
    )

    return result_df.head(top_k)


def select_significant_features(
    df: DataFrame,
    features: list[str],
    target: str = "TARGET",
    significance_level: float = 0.05,
) -> list[str]:
    """Return only features with statistically significant Kruskal-Wallis test."""
    ranking = kruskal_wallis_ranking(
        df, features, target,
        top_k=len(features),
        significance_level=significance_level,
    )
    return ranking[ranking["significant"]]["feature"].tolist()