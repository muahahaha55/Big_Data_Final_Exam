"""Financial Risk Modeling (Basel II) — scorecard, segmentation, EL, stress testing."""

from credit_risk.risk.expected_loss import (
    PortfolioMetrics,
    add_expected_loss,
    compute_portfolio_metrics,
)
from credit_risk.risk.scorecard import (
    IVResult,
    analyze_features,
    classify_iv,
    compute_woe_iv_categorical,
    compute_woe_iv_numeric,
)
from credit_risk.risk.segmentation import (
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    TIER_ORDER,
    TIER_VERY_HIGH,
    SegmentationSummary,
    assign_risk_tier,
    summarize_segmentation,
)
from credit_risk.risk.stress_testing import (
    ScenarioConfig,
    ScenarioResult,
    StressTestReport,
    apply_scenario,
    load_scenarios_from_config,
    run_stress_test,
)

__all__ = [
    # scorecard
    "IVResult", "analyze_features", "classify_iv",
    "compute_woe_iv_numeric", "compute_woe_iv_categorical",
    # segmentation
    "SegmentationSummary", "assign_risk_tier", "summarize_segmentation",
    "TIER_LOW", "TIER_MEDIUM", "TIER_HIGH", "TIER_VERY_HIGH", "TIER_ORDER",
    # expected loss
    "PortfolioMetrics", "add_expected_loss", "compute_portfolio_metrics",
    # stress testing
    "ScenarioConfig", "ScenarioResult", "StressTestReport",
    "apply_scenario", "load_scenarios_from_config", "run_stress_test",
]
