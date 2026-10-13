"""Cost-based cut-off study (P1.1) — turning a PD score into a decision.

Spark-free by design so the cost model, the plateau arithmetic and the
rotating-holdout bias measurement are unit-testable in milliseconds
(tests/test_threshold.py). The pandas I/O lives in
pipelines/threshold_pipeline.py; no SparkSession is involved, because the
predictions were written by T1.4.
"""

from credit_risk.threshold.study import (
    DEFAULT_CURVE_POINTS,
    DEFAULT_GRID_POINTS,
    DEFAULT_MARGIN_RATES,
    DEFAULT_PLATEAU_N_SE,
    NOTE,
    REPORT_FILENAME,
    REVIEW_AS_APPROVE,
    REVIEW_AS_REJECT,
    SCHEMA_VERSION,
    CostParams,
    Prepared,
    approve_all,
    breakeven_pd,
    build_report,
    build_scenario,
    build_setup,
    config_fingerprint,
    current_policy,
    format_summary,
    implied_margin_rate,
    margin_rates,
    optimum,
    paired_se,
    pearson,
    plateau,
    policy_value,
    population_summary,
    prepare,
    quantile_grid,
    random_at_rate,
    reject_all,
    rotating_holdout,
    row_value,
    sweep,
)

__all__ = [
    "DEFAULT_CURVE_POINTS", "DEFAULT_GRID_POINTS", "DEFAULT_MARGIN_RATES",
    "DEFAULT_PLATEAU_N_SE", "NOTE", "REPORT_FILENAME", "REVIEW_AS_APPROVE",
    "REVIEW_AS_REJECT", "SCHEMA_VERSION", "CostParams", "Prepared",
    "approve_all", "breakeven_pd", "build_report", "build_scenario",
    "build_setup", "config_fingerprint", "current_policy", "format_summary",
    "implied_margin_rate", "margin_rates", "optimum", "paired_se", "pearson", "plateau",
    "policy_value", "population_summary", "prepare", "quantile_grid",
    "random_at_rate", "reject_all", "rotating_holdout", "row_value", "sweep",
]
