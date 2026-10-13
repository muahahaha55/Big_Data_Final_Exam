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
# Inputs
# ══════════════════════════════════════════════════════════════

def load_oof(cv_report: dict):
    """Concatenate the per-fold out-of-fold predictions.

    Cross-checks the row count against the `n_valid` the folds recorded, so a
    stale Parquet file left over from an earlier run cannot silently become
    the population every segment number is computed on.
    """
    import pandas as pd

    folds = sorted(completed_folds(cv_report))
    if not folds:
        raise RuntimeError(
            f"{_CV_REPORT} records no finished folds — run "
            "`make cv-fold N=1` … `N=5` first"
        )

    frames = []
    for fold in folds:
        path = _OOF_DIR / f"fold_{fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"fold {fold} is recorded as finished but {path} is missing"
            )
        frames.append(pd.read_parquet(path))

    oof = pd.concat(frames, ignore_index=True)

    expected = sum(int(f.get("n_valid") or 0)
                   for f in cv_report.get("folds", [])
                   if int(f.get("fold", -1)) in folds)
    if expected and len(oof) != expected:
        raise RuntimeError(
            f"out-of-fold files hold {len(oof)} rows but cv_report.json "
            f"records {expected} validation rows across folds {folds}. The "
            "Parquet files are from a different run — delete "
            f"{_OOF_DIR} and re-run the folds."
        )

    log.info("segments_oof_loaded", rows=len(oof), folds=folds)
    return oof, folds


def load_attributes(columns: list[str], id_col: str):
    """Read only the segment columns from the CV pool, via column pruning.

    The pool is `train.parquet ∪ val.parquet`, exactly what T1.4 drew folds
    from, so every out-of-fold id must be found here. Reading five columns
    instead of 112 keeps this inside a few hundred MB on the 3.8 GB VM.
    """
    import pandas as pd

    wanted = [id_col] + [c for c in columns if c != id_col]
    frames = []
    for name in ("train", "val"):
        path = _MODEL_INPUT / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} — the segment attributes come from the same "
                "CV pool T1.4 used. Run `make etl` first."
            )
        try:
            frames.append(pd.read_parquet(path, columns=wanted))
        except Exception as exc:                      # narrow message, wide catch
            raise RuntimeError(
                f"could not read {wanted} from {path}: {exc}. Check that every "
                "segments.dimensions column exists in the master dataset."
            ) from exc

    attrs = pd.concat(frames, ignore_index=True)
    dup = int(attrs[id_col].duplicated().sum())
    if dup:
        raise RuntimeError(
            f"{dup} duplicate {id_col} values in the CV pool — an aggregate "
            "join fanned out and the segment join would double-count rows."
        )
    log.info("segments_attributes_loaded", rows=len(attrs), columns=wanted)
    return attrs


def join_attributes(oof, attrs, id_col: str):
    """Attach the attributes to every out-of-fold prediction, or fail loudly."""
    dup = int(oof[id_col].duplicated().sum())
    if dup:
        raise RuntimeError(
            f"{dup} ids appear in more than one out-of-fold file — folds must "
            "be disjoint"
        )

    merged = oof.merge(
        attrs, on=id_col, how="left", validate="one_to_one", indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(
            f"{unmatched} of {len(merged)} out-of-fold ids have no row in "
            "train.parquet ∪ val.parquet. The predictions and the pool are "
            "from different ETL runs; segment numbers would be computed on a "
            "subset without saying so."
        )
    return merged.drop(columns=["_merge"])


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

    oof, folds = load_oof(cv_report)
    attrs = load_attributes(source_columns(specs), id_col)
    merged = join_attributes(oof, attrs, id_col)

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
