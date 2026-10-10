"""Expected Loss computation — Basel II foundation approach.

EL = PD × LGD × EAD

Where:
    PD  = Probability of Default (from model, 1-year horizon)
    LGD = Loss Given Default (Basel II foundation: 0.45 for unsecured)
    EAD = Exposure at Default (loan amount outstanding)

Portfolio-level metrics:
    Total EL              — expected dollar loss across portfolio
    Portfolio EL rate     — EL / total EAD (a single comparable number)
    Per-tier EL breakdown — where the loss concentration sits

Capital is NOT computed here. It is non-linear in per-obligor PD, so it cannot
be derived from portfolio aggregates — see `credit_risk/risk/capital.py`, which
adds a per-row `irb_capital` column that this module only sums up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.config import get_config
from credit_risk.risk.capital import get_capital_config, rule_of_thumb_cross_check
from credit_risk.risk.segmentation import TIER_ORDER
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk aggregates."""

    total_customers: int = 0
    total_ead: float = 0.0
    total_expected_loss: float = 0.0
    portfolio_el_rate: float = 0.0
    avg_pd: float = 0.0
    avg_lgd: float = 0.45
    actual_default_rate: float = 0.0
    tier_breakdown: list[dict] = field(default_factory=list)
    total_irb_capital: float = 0.0
    has_irb_capital: bool = False

    def to_dict(self) -> dict:
        return {
            "portfolio_overview": {
                "total_customers": self.total_customers,
                "total_ead_million": round(self.total_ead / 1e6, 2),
                "total_expected_loss_million": round(self.total_expected_loss / 1e6, 2),
                "portfolio_el_rate": round(self.portfolio_el_rate, 4),
                "portfolio_el_rate_pct": round(self.portfolio_el_rate * 100, 3),
                "interpretation": self._interpret(),
                "avg_pd": round(self.avg_pd, 4),
                "avg_lgd": round(self.avg_lgd, 4),
                "actual_default_rate_pct": round(self.actual_default_rate * 100, 2),
            },
            "risk_tier_breakdown": self.tier_breakdown,
            "capital_requirement": self._capital_requirement(),
        }

    def _interpret(self) -> str:
        rate = self.portfolio_el_rate * 100
        if rate < 1:
            return "Very healthy (<1%)"
        if rate < 3:
            return "Healthy (1-3%)"
        if rate < 5:
            return "Moderate risk (3-5%)"
        if rate < 10:
            return "Elevated risk (5-10%)"
        return "High risk (>10%) — requires attention"

    def _capital_requirement(self) -> dict:
        """IRB capital requirement (estimate), summed from the per-obligor column.

        The headline number is the Basel II retail IRB figure. The old EL-multiple
        heuristic is reported alongside it, labelled as an assumption, purely to
        show how far it lands from the formula.
        """
        el = self.total_expected_loss
        cap_cfg = get_capital_config()

        if not self.has_irb_capital:
            return {
                "method": "unavailable",
                "note": "no irb_capital column present — run add_irb_capital() before "
                        "compute_portfolio_metrics() to get a capital figure",
                "expected_loss_million": round(el / 1e6, 2),
            }

        cap = self.total_irb_capital
        rwa = cap * cap_cfg.rwa_multiplier
        return {
            "method": "basel_ii_irb_retail_asrf",
            "label": "IRB capital requirement (estimate)",
            "confidence_level": cap_cfg.confidence_level,
            "pd_floor": cap_cfg.pd_floor,
            "expected_loss_million": round(el / 1e6, 2),
            "irb_capital_requirement_million": round(cap / 1e6, 2),
            "irb_rwa_million": round(rwa / 1e6, 2),
            "capital_to_ead_pct": round((cap / self.total_ead * 100) if self.total_ead > 0 else 0.0, 3),
            "rwa_density_pct": round((rwa / self.total_ead * 100) if self.total_ead > 0 else 0.0, 2),
            "rule_of_thumb_cross_check": rule_of_thumb_cross_check(el, cap, cap_cfg),
        }


def add_expected_loss(
    df: DataFrame,
    pd_column: str = "pd_score",
    ead_column: str = "AMT_CREDIT",
    lgd: float | None = None,
    output_columns: tuple[str, str, str] = ("lgd", "ead", "expected_loss"),
) -> DataFrame:
    """Add LGD, EAD, and Expected Loss columns to a DataFrame."""
    if lgd is None:
        lgd = float(get_config()["risk"]["expected_loss"]["lgd_fixed"])

    lgd_col, ead_col, el_col = output_columns

    log.info("computing_expected_loss", lgd=lgd, pd_col=pd_column, ead_col=ead_column)

    return (
        df.withColumn(lgd_col, F.lit(lgd))
          .withColumn(ead_col, F.col(ead_column))
          .withColumn(el_col, F.col(pd_column) * F.col(lgd_col) * F.col(ead_col))
    )


def compute_portfolio_metrics(
    df: DataFrame,
    pd_column: str = "pd_score",
    el_column: str = "expected_loss",
    ead_column: str = "ead",
    lgd_column: str = "lgd",
    target_column: str | None = "TARGET",
    tier_column: str = "risk_tier",
    capital_column: str = "irb_capital",
) -> PortfolioMetrics:
    """Aggregate portfolio-level risk metrics.

    If `capital_column` is present (added by `capital.add_irb_capital`), its sum
    becomes the portfolio IRB capital requirement.
    """
    df.cache()

    has_capital = capital_column in df.columns

    aggs = [
        F.count("*").alias("n"),
        F.sum(ead_column).alias("total_ead"),
        F.sum(el_column).alias("total_el"),
        F.avg(pd_column).alias("avg_pd"),
        F.avg(lgd_column).alias("avg_lgd"),
        F.avg(target_column).alias("actual_dr") if target_column else F.lit(0.0).alias("actual_dr"),
        F.sum(capital_column).alias("total_capital") if has_capital
        else F.lit(0.0).alias("total_capital"),
    ]
    overall = df.agg(*aggs).collect()[0]

    total_ead = float(overall["total_ead"] or 0.0)
    total_el = float(overall["total_el"] or 0.0)

    # Per-tier breakdown
    tier_agg = (
        df.groupBy(tier_column)
          .agg(
              F.count("*").alias("count"),
              F.sum(ead_column).alias("ead"),
              F.sum(el_column).alias("el"),
              F.avg(pd_column).alias("avg_pd"),
          )
          .collect()
    )

    total_customers = int(overall["n"])
    tier_breakdown = []
    tier_dict = {row[tier_column]: row for row in tier_agg}

    for tier in TIER_ORDER:
        row = tier_dict.get(tier)
        if row is None:
            tier_breakdown.append({
                "risk_tier": tier, "count": 0, "pct_customers": 0.0,
                "ead_m": 0.0, "el_m": 0.0, "el_rate_pct": 0.0, "avg_pd": 0.0,
            })
            continue
        ead = float(row["ead"] or 0.0)
        el = float(row["el"] or 0.0)
        tier_breakdown.append({
            "risk_tier": tier,
            "count": int(row["count"]),
            "pct_customers": round(row["count"] / total_customers * 100, 2) if total_customers else 0.0,
            "ead_m": round(ead / 1e6, 2),
            "el_m": round(el / 1e6, 2),
            "el_rate_pct": round((el / ead * 100) if ead > 0 else 0.0, 2),
            "avg_pd": round(float(row["avg_pd"] or 0.0), 4),
        })

    df.unpersist()

    metrics = PortfolioMetrics(
        total_customers=total_customers,
        total_ead=total_ead,
        total_expected_loss=total_el,
        portfolio_el_rate=(total_el / total_ead) if total_ead > 0 else 0.0,
        avg_pd=float(overall["avg_pd"] or 0.0),
        avg_lgd=float(overall["avg_lgd"] or 0.45),
        actual_default_rate=float(overall["actual_dr"] or 0.0),
        tier_breakdown=tier_breakdown,
        total_irb_capital=float(overall["total_capital"] or 0.0),
        has_irb_capital=has_capital,
    )

    log.info(
        "portfolio_metrics_computed",
        customers=total_customers,
        el_million=round(total_el / 1e6, 2),
        el_rate_pct=round(metrics.portfolio_el_rate * 100, 3),
        irb_capital_million=round(metrics.total_irb_capital / 1e6, 2) if has_capital else None,
    )
    return metrics
