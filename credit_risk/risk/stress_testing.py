"""Portfolio stress testing under economic scenarios.

Industry-standard practice (CCAR for US banks, EBA stress tests in EU).
Given current PD/LGD/EAD per customer, simulate how the portfolio's
expected loss would change under different macroeconomic scenarios.

Scenarios are multipliers applied to base PD and LGD:
    baseline:       1.0× / 1.0×  — current state
    mild_recession: 1.5× / 1.1×  — GDP -2%, unemployment +3pp
    severe:         2.5× / 1.3×  — 2008-style crisis
    black_swan:     4.0× / 1.5×  — 99.9th percentile tail event

The output drives capital adequacy decisions: how much reserve does
the bank need to hold to survive scenario X?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScenarioConfig:
    """Definition of a single stress scenario."""

    name: str
    pd_multiplier: float
    lgd_multiplier: float
    description: str = ""


@dataclass
class ScenarioResult:
    """Aggregate results for one scenario across the portfolio."""

    scenario: str
    description: str
    pd_multiplier: float
    lgd_multiplier: float
    total_ead: float
    total_el: float
    el_rate_pct: float
    avg_pd: float
    avg_lgd: float
    customers_high_risk: int      # PD ≥ 0.20 after stress
    customers_default_zone: int   # PD ≥ 0.50 after stress
    capital_required_estimate: float  # EL × 3 (simplified UL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "pd_multiplier": self.pd_multiplier,
            "lgd_multiplier": self.lgd_multiplier,
            "total_ead_million": round(self.total_ead / 1e6, 2),
            "total_el_million": round(self.total_el / 1e6, 2),
            "el_rate_pct": round(self.el_rate_pct, 3),
            "avg_pd": round(self.avg_pd, 4),
            "avg_lgd": round(self.avg_lgd, 4),
            "customers_high_risk": self.customers_high_risk,
            "customers_default_zone": self.customers_default_zone,
            "capital_required_estimate_million": round(self.capital_required_estimate / 1e6, 2),
        }


@dataclass
class StressTestReport:
    """Full stress test report across all scenarios."""

    n_customers: int
    base_el: float
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_customers": self.n_customers,
            "base_el_million": round(self.base_el / 1e6, 2),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "comparison": self._build_comparison(),
        }

    def _build_comparison(self) -> dict[str, Any]:
        """Compare scenarios as multiplier vs baseline."""
        baseline = next((s for s in self.scenarios if s.scenario == "baseline"), None)
        if baseline is None or baseline.total_el == 0:
            return {}
        return {
            s.scenario: {
                "el_multiplier_vs_baseline": round(s.total_el / baseline.total_el, 2),
                "additional_loss_million": round((s.total_el - baseline.total_el) / 1e6, 2),
            }
            for s in self.scenarios
        }


# ──────────────────────────────────────────────────────────────
# Computation
# ──────────────────────────────────────────────────────────────

def apply_scenario(
    df: DataFrame,
    scenario: ScenarioConfig,
    pd_col: str = "pd_score",
    lgd_col: str = "lgd",
    ead_col: str = "ead",
) -> DataFrame:
    """Apply scenario multipliers and recompute EL.

    PD is clipped at [0, 1] since it's a probability.
    LGD is clipped at [0, 1] (can't lose more than the loan).
    """
    return df.withColumn(
        f"pd_{scenario.name}",
        F.least(F.lit(1.0), F.col(pd_col) * F.lit(scenario.pd_multiplier))
    ).withColumn(
        f"lgd_{scenario.name}",
        F.least(F.lit(1.0), F.col(lgd_col) * F.lit(scenario.lgd_multiplier))
    ).withColumn(
        f"el_{scenario.name}",
        F.col(f"pd_{scenario.name}") * F.col(f"lgd_{scenario.name}") * F.col(ead_col)
    )


def aggregate_scenario(df: DataFrame, scenario: ScenarioConfig, ead_col: str = "ead") -> ScenarioResult:
    """Compute portfolio-level aggregates for one scenario."""
    pd_col = f"pd_{scenario.name}"
    lgd_col = f"lgd_{scenario.name}"
    el_col = f"el_{scenario.name}"

    agg = df.agg(
        F.count("*").alias("n"),
        F.sum(ead_col).alias("total_ead"),
        F.sum(el_col).alias("total_el"),
        F.avg(pd_col).alias("avg_pd"),
        F.avg(lgd_col).alias("avg_lgd"),
        F.sum(F.when(F.col(pd_col) >= 0.20, 1).otherwise(0)).alias("high_risk"),
        F.sum(F.when(F.col(pd_col) >= 0.50, 1).otherwise(0)).alias("default_zone"),
    ).collect()[0]

    total_ead = float(agg["total_ead"] or 0.0)
    total_el = float(agg["total_el"] or 0.0)
    el_rate = (total_el / total_ead * 100) if total_ead > 0 else 0.0

    return ScenarioResult(
        scenario=scenario.name,
        description=scenario.description,
        pd_multiplier=scenario.pd_multiplier,
        lgd_multiplier=scenario.lgd_multiplier,
        total_ead=total_ead,
        total_el=total_el,
        el_rate_pct=el_rate,
        avg_pd=float(agg["avg_pd"] or 0.0),
        avg_lgd=float(agg["avg_lgd"] or 0.0),
        customers_high_risk=int(agg["high_risk"] or 0),
        customers_default_zone=int(agg["default_zone"] or 0),
        capital_required_estimate=total_el * 3.0,  # simplified UL proxy
    )


def run_stress_test(
    df: DataFrame,
    scenarios: list[ScenarioConfig] | None = None,
    pd_col: str = "pd_score",
    lgd_col: str = "lgd",
    ead_col: str = "ead",
) -> StressTestReport:
    """Run stress test across all configured scenarios.

    Args:
        df: DataFrame with PD, LGD, EAD per customer
        scenarios: list of scenarios to test (default: from config)

    Returns:
        StressTestReport with results per scenario + comparisons
    """
    if scenarios is None:
        scenarios = load_scenarios_from_config()

    df.cache()
    n_customers = df.count()
    log.info("stress_test_started", customers=n_customers, scenarios=len(scenarios))

    # Apply all scenarios at once (more efficient than sequential)
    df_stressed = df
    for sc in scenarios:
        df_stressed = apply_scenario(df_stressed, sc, pd_col, lgd_col, ead_col)
    df_stressed.cache()

    # Aggregate each
    results: list[ScenarioResult] = []
    for sc in scenarios:
        result = aggregate_scenario(df_stressed, sc, ead_col)
        results.append(result)
        log.info(
            "scenario_completed",
            scenario=sc.name,
            el_million=round(result.total_el / 1e6, 2),
            high_risk=result.customers_high_risk,
        )

    base = next((r for r in results if r.scenario == "baseline"), results[0] if results else None)
    base_el = base.total_el if base else 0.0

    df.unpersist()
    df_stressed.unpersist()

    return StressTestReport(
        n_customers=n_customers,
        base_el=base_el,
        scenarios=results,
    )


def load_scenarios_from_config() -> list[ScenarioConfig]:
    """Build ScenarioConfig list from conf/base/config.yaml."""
    cfg = get_config().get("risk", {}).get("stress_testing", {}).get("scenarios", {})
    scenarios = []
    for name, params in cfg.items():
        scenarios.append(ScenarioConfig(
            name=name,
            pd_multiplier=float(params.get("pd_multiplier", 1.0)),
            lgd_multiplier=float(params.get("lgd_multiplier", 1.0)),
            description=params.get("description", ""),
        ))
    return scenarios
