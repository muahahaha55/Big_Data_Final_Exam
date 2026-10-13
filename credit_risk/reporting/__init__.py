"""Shared reporting infrastructure for long, resumable studies.

Extracted from the T1.2 ablation pipeline so the ablation study and the T1.4
cross-validation study use one implementation of:

    - a comparability fingerprint that refuses to mix incomparable runs
    - atomic JSON writes that a crash cannot truncate
    - loading a partial report as a resume checkpoint

Deliberately free of any Spark import, so everything here is unit-testable
without a SparkSession.
"""

from credit_risk.reporting.checkpoint import (
    atomic_write_json,
    canonical_fingerprint,
    check_fingerprint,
    completed_records,
    load_report,
)
from credit_risk.reporting.mlflow_preflight import require_mlflow
from credit_risk.reporting.oof import (
    cv_fold_mean,
    join_attributes,
    load_attributes,
    load_oof,
)

__all__ = [
    "cv_fold_mean",
    "join_attributes",
    "load_attributes",
    "load_oof",
    "atomic_write_json",
    "canonical_fingerprint",
    "check_fingerprint",
    "completed_records",
    "load_report",
    "require_mlflow",
]
