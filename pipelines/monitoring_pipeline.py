"""Monitoring pipeline — daily drift detection.

Compares the reference distribution (training set) against the current
production data, computes PSI + KS per feature, saves a drift report.

Run via: make drift-check
"""

from __future__ import annotations

import json
from datetime import datetime

from credit_risk.config import get_config, project_root
from credit_risk.monitoring import generate_drift_report
from credit_risk.utils import get_spark, stop_spark
from credit_risk.utils.logging import configure_logging, get_logger


def run_drift_check(
    reference_path: str | None = None,
    current_path: str | None = None,
) -> None:
    """Run drift detection comparing current vs reference data."""
    configure_logging()
    log = get_logger("pipeline.monitoring")
    cfg = get_config()
    log.info("drift_check_started")

    spark = get_spark("monitoring-pipeline")

    try:
        # ── Default paths: train (reference) vs test (current) ──
        # In production, "current" would be the latest week of scored requests.
        ref_path = reference_path or str(project_root() / "data" / "05_model_input" / "train.parquet")
        cur_path = current_path or str(project_root() / "data" / "05_model_input" / "test.parquet")

        reference_df = spark.read.parquet(ref_path)
        current_df = spark.read.parquet(cur_path)
        log.info("data_loaded", reference=reference_df.count(), current=current_df.count())

        # ── Run drift detection ─────────────────────────────
        monitored = cfg["monitoring"]["drift"]["monitored_features"]
        warn_thr = cfg["monitoring"]["drift"]["psi_threshold_warning"]
        alert_thr = cfg["monitoring"]["drift"]["psi_threshold_alert"]

        report = generate_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            features=monitored,
            warn_threshold=warn_thr,
            alert_threshold=alert_thr,
        )

        # ── Save report ─────────────────────────────────────
        out_path = project_root() / "data" / "08_reporting" / "drift_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # ── Headline output ──────────────────────────────────
        print("\n" + "=" * 60)
        print(f"  DRIFT DETECTION REPORT  ({datetime.utcnow().isoformat()})")
        print("=" * 60)
        print(f"  Features checked:   {report.n_features_checked}")
        print(f"  Features drifting:  {report.n_features_drifting}")
        print(f"  Overall severity:   {report.overall_severity.value.upper()}")
        print("  Per-feature PSI:")
        for r in report.feature_results:
            print(f"    {r.feature:<25}PSI={r.psi:.4f}  [{r.severity.value}]")
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")
        print("=" * 60 + "\n")

        log.info("drift_check_completed", severity=report.overall_severity.value)

    except Exception:
        log.exception("drift_check_failed")
        raise
    finally:
        stop_spark()


if __name__ == "__main__":
    run_drift_check()
