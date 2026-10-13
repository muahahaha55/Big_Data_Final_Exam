"""Segment validation pipeline (P0.4) — pandas only, no Spark, no training.

ROC-AUC 0.7663 is an average over every kind of customer. This splits the
**out-of-fold** predictions from T1.4 by contract type, gender, age band and
income band, and measures discrimination and calibration inside each slice.
Nothing is retrained: the predictions already exist, so the whole run is a
few seconds of pandas on a single process.

All the study logic — binning, the Mann-Whitney AUC, the Hanley-McNeil noise
bound, the suppression rules — lives in `credit_risk/segments/study.py`, which
imports neither Spark nor pandas. This file only does I/O and the join.

Inputs (all produced by earlier steps, none regenerated here):
    data/08_reporting/cv_report.json          which folds finished, fold AUCs
    data/07_model_output/cv_oof/fold_*.parquet  id, TARGET, pd_score, fold
    data/05_model_input/{train,val}.parquet   the segment attributes

Output:
    data/08_reporting/segment_report.json

Run:
    make segments
"""

from __future__ import annotations

import argparse

from credit_risk.config import get_config, project_root
from credit_risk.cv import completed_folds
from credit_risk.cv.study import REPORT_FILENAMES
from credit_risk.reporting import atomic_write_json, load_report
from credit_risk.reporting.oof import join_attributes, load_attributes, load_oof
from credit_risk.segments import (
    REPORT_FILENAME,
    build_dimension,
    build_segment_report,
    build_setup,
    build_specs,
    compare_overall_auc,
    format_summary,
    overall_metrics,
    source_columns,
)
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.segments")

_MODEL_INPUT = project_root() / "data" / "05_model_input"
_OOF_DIR = project_root() / "data" / "07_model_output" / "cv_oof"
_REPORT_DIR = project_root() / "data" / "08_reporting"

_SEGMENT_REPORT = _REPORT_DIR / REPORT_FILENAME
_CV_REPORT = _REPORT_DIR / REPORT_FILENAMES["cv"]


# ══════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════

def run_segments() -> dict:
    """Build segment_report.json from the T1.4 out-of-fold predictions."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]

    specs = build_specs(cfg)
    cv_report = load_report(_CV_REPORT)
    if not cv_report:
        raise FileNotFoundError(
            f"{_CV_REPORT} not found — P0.4 reads the out-of-fold predictions "
            "T1.4 produced. Run `make cv-fold N=1` … `N=5` first."
        )

    oof, folds = load_oof(cv_report, _OOF_DIR, _CV_REPORT)
    attrs = load_attributes(
        source_columns(specs), id_col, _MODEL_INPUT,
        hint="every segments.dimensions column",
    )
    merged = join_attributes(oof, attrs, id_col)
    log.info("segments_inputs_ready", oof_rows=len(merged), folds=folds)

    targets = merged[target_col].astype(int).tolist()
    scores = merged["pd_score"].astype(float).tolist()
    overall = overall_metrics(list(zip(targets, scores)))

    cv_mean = ((cv_report.get("summary") or {}).get("roc_auc") or {}).get("mean")
    reference = compare_overall_auc(overall.get("roc_auc"), cv_mean)
    log.info("segments_overall", n=overall["n"], roc_auc=overall["roc_auc"],
             cv_fold_mean=cv_mean, delta=reference.get("delta"))

    setup = build_setup(cfg, specs, source={
        "cv_folds_used": folds,
        "oof_rows": int(len(merged)),
        "cv_report": f"data/08_reporting/{REPORT_FILENAMES['cv']}",
    })

    dimensions = []
    for spec in specs:
        rows = list(zip(merged[spec.column].tolist(), targets, scores))
        dim = build_dimension(
            spec, rows,
            min_rows=setup["min_rows"],
            min_positives=setup["min_positives"],
        )
        dimensions.append(dim)
        log.info("segments_dimension_done", name=dim["name"],
                 groups=dim["n_groups"], evaluated=dim["n_groups_evaluated"],
                 auc_spread=dim["auc_spread"],
                 spread_vs_noise=dim["spread_vs_noise"])

    report = build_segment_report(
        setup=setup,
        dimensions=dimensions,
        overall=overall,
        reference=reference,
        cv_config_fingerprint=cv_report.get("config_fingerprint", ""),
    )
    atomic_write_json(_SEGMENT_REPORT, report)
    log.info("segments_saved", path=str(_SEGMENT_REPORT),
             n_dimensions=report["n_dimensions"],
             beyond_noise=[d["name"] for d in
                           report["dimensions_with_spread_beyond_noise"]])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment validation on out-of-fold predictions (P0.4). "
                    "No Spark, no training, no arguments: every input is "
                    "already on disk and the slices come from config.",
    )
    parser.parse_args()

    configure_logging()
    log.info("segments_started")
    try:
        report = run_segments()
        print("\n" + format_summary(report) + "\n")
        log.info("segments_completed")
    except Exception:
        log.exception("segments_failed")
        raise


if __name__ == "__main__":
    main()
