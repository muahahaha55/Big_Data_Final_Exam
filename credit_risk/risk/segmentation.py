"""Risk segmentation — assign each customer to a discrete risk tier.

4-tier segmentation aligned with Basel II:
    Low Risk      : PD < 10%       — auto-approve
    Medium Risk   : 10% ≤ PD < 20% — auto-approve with monitoring
    High Risk     : 20% ≤ PD < 35% — manual review
    Very High     : PD ≥ 35%       — auto-reject or high-margin only

Thresholds are configurable via conf/base/config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


TIER_LOW = "Low Risk"
TIER_MEDIUM = "Medium Risk"
TIER_HIGH = "High Risk"
TIER_VERY_HIGH = "Very High Risk"
TIER_ORDER = [TIER_LOW, TIER_MEDIUM, TIER_HIGH, TIER_VERY_HIGH]


@dataclass
class SegmentationSummary:
    """Summary stats per tier."""

    tier_counts: dict[str, int] = field(default_factory=dict)
    tier_pct: dict[str, float] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "tier_counts": self.tier_counts,
            "tier_pct": self.tier_pct,
        }


def assign_risk_tier(
    df: DataFrame,
    pd_column: str = "pd_score",
    output_column: str = "risk_tier",
    thresholds: dict[str, float] | None = None,
) -> DataFrame:
    """Add a risk_tier column based on PD thresholds.

    Args:
        df: input DataFrame with a PD column
        pd_column: name of the PD column
        output_column: name of the tier column to create
        thresholds: dict with keys 'low', 'medium', 'high' (defaults from config)
    """
    if thresholds is None:
        thresholds = get_config()["risk"]["segmentation"]["thresholds"]

    low = float(thresholds["low"])
    medium = float(thresholds["medium"])
    high = float(thresholds["high"])

    log.info("assigning_risk_tier", low=low, medium=medium, high=high)

    return df.withColumn(
        output_column,
        F.when(F.col(pd_column) < low, TIER_LOW)
         .when(F.col(pd_column) < medium, TIER_MEDIUM)
         .when(F.col(pd_column) < high, TIER_HIGH)
         .otherwise(TIER_VERY_HIGH)
    )


def summarize_segmentation(
    df: DataFrame,
    tier_column: str = "risk_tier",
) -> SegmentationSummary:
    """Compute per-tier customer counts and percentages."""
    total = df.count()
    if total == 0:
        return SegmentationSummary()

    counts_df = df.groupBy(tier_column).count().collect()
    tier_counts = {row[tier_column]: row["count"] for row in counts_df}

    # Ensure all tiers present (with 0 if missing)
    tier_counts = {tier: tier_counts.get(tier, 0) for tier in TIER_ORDER}
    tier_pct = {tier: round(count / total * 100, 2) for tier, count in tier_counts.items()}

    summary = SegmentationSummary(
        tier_counts=tier_counts,
        tier_pct=tier_pct,
        total=total,
    )
    log.info("segmentation_summary", summary=summary.to_dict())
    return summary
