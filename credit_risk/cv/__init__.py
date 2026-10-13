"""K-fold cross-validation study (T1.4) — confidence intervals + calibration.

Spark-free by design so the whole study design is unit-testable without a
SparkSession (tests/test_cv.py). Spark execution lives in
pipelines/cv_pipeline.py.
"""

from credit_risk.cv.study import (
    CALIBRATION_NOTE,
    CV_NOTE,
    DEFAULT_CONFIDENCE,
    DEFAULT_K,
    DEFAULT_N_DECILES,
    METRICS,
    REPORT_FILENAMES,
    SCHEMA_VERSION,
    assign_folds,
    build_calibration_report,
    build_cv_report,
    build_setup,
    check_fold_balance,
    column_audit,
    completed_folds,
    config_fingerprint,
    decile_table,
    dropped_by_prefix,
    expected_calibration_error,
    fold_balance,
    fold_ids,
    format_calibration_summary,
    format_cv_summary,
    pending_folds,
    summarize_folds,
    t_critical,
    t_interval,
)

__all__ = [
    "CALIBRATION_NOTE", "CV_NOTE", "DEFAULT_CONFIDENCE", "DEFAULT_K",
    "DEFAULT_N_DECILES", "METRICS", "REPORT_FILENAMES", "SCHEMA_VERSION",
    "assign_folds", "build_calibration_report", "build_cv_report",
    "build_setup", "check_fold_balance", "column_audit", "completed_folds",
    "config_fingerprint", "decile_table", "dropped_by_prefix",
    "expected_calibration_error", "fold_balance", "fold_ids",
    "format_calibration_summary", "format_cv_summary", "pending_folds",
    "summarize_folds", "t_critical", "t_interval",
]
