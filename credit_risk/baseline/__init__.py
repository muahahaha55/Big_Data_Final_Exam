"""Baseline comparison study (P0.3) — is the Big Data apparatus worth it?

Spark-free by design so the whole study design is unit-testable without a
SparkSession (tests/test_baseline.py). Spark execution lives in
pipelines/baseline_pipeline.py.
"""

from credit_risk.baseline.study import (
    DEFAULT_VARIANTS,
    EXT_SOURCES,
    METRICS,
    MODEL_GBT,
    MODEL_LOGISTIC,
    MODEL_RULE,
    NOTE,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    VALID_MODELS,
    VariantSpec,
    attach_deltas,
    build_report,
    build_setup,
    build_variants,
    check_split_match,
    completed_variants,
    config_fingerprint,
    format_summary,
    load_references,
    pending_variants,
    summarize,
)

__all__ = [
    "DEFAULT_VARIANTS", "EXT_SOURCES", "METRICS", "MODEL_GBT", "MODEL_LOGISTIC",
    "MODEL_RULE", "NOTE", "REPORT_FILENAME", "SCHEMA_VERSION", "VALID_MODELS",
    "VariantSpec", "attach_deltas", "build_report", "build_setup",
    "build_variants", "check_split_match", "completed_variants",
    "config_fingerprint", "format_summary", "load_references",
    "pending_variants", "summarize",
]
