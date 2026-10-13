"""Segment validation study (P0.4) — does the model work for every group?

Spark-free by design so the binning, the AUC estimator and the noise bounds
are unit-testable in milliseconds (tests/test_segments.py). The pandas I/O
lives in pipelines/segment_pipeline.py; no SparkSession is involved at all,
because the predictions were already written by T1.4.
"""

from credit_risk.segments.study import (
    AUC_HARD_TOLERANCE,
    AUC_SOFT_TOLERANCE,
    DEFAULT_MIN_POSITIVES,
    DEFAULT_MIN_ROWS,
    DEFAULT_N_BINS,
    KIND_CATEGORICAL,
    KIND_QUANTILE,
    MISSING_LABEL,
    NOTE,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    TRANSFORMS,
    VALID_KINDS,
    SegmentSpec,
    auc_standard_error,
    bin_bounds,
    bin_label,
    build_dimension,
    build_segment_report,
    build_setup,
    build_specs,
    compare_overall_auc,
    config_fingerprint,
    format_summary,
    group_metrics,
    ks_statistic,
    overall_metrics,
    quantile_edges,
    roc_auc,
    source_columns,
)

__all__ = [
    "AUC_HARD_TOLERANCE", "AUC_SOFT_TOLERANCE", "DEFAULT_MIN_POSITIVES",
    "DEFAULT_MIN_ROWS", "DEFAULT_N_BINS", "KIND_CATEGORICAL", "KIND_QUANTILE",
    "MISSING_LABEL", "NOTE", "REPORT_FILENAME", "SCHEMA_VERSION",
    "TRANSFORMS", "VALID_KINDS", "SegmentSpec", "auc_standard_error",
    "bin_bounds", "bin_label", "build_dimension", "build_segment_report",
    "build_setup", "build_specs", "compare_overall_auc", "config_fingerprint",
    "format_summary", "group_metrics", "ks_statistic", "overall_metrics",
    "quantile_edges", "roc_auc", "source_columns",
]
