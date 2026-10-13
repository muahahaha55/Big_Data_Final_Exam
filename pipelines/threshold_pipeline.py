"""Cost-based cut-off pipeline (P1.1) — pandas only, no Spark, no training.

The repo's approve/reject boundary is three round numbers in
`risk.segmentation.thresholds` that nothing derives. This sweeps every
candidate threshold over the **out-of-fold** predictions from T1.4 under an
explicit cost model, and reports where the optimum is, how wide the
statistically indistinguishable plateau around it is, and how the current fixed
policy compares.

The threshold MUST come from out-of-fold predictions rather than
`test.parquet`: choosing a cut-off on the same rows used to report its value is
optimistically biased. The report measures that bias too, by rotating which
fold the threshold is chosen on.

All study logic lives in `credit_risk/threshold/study.py`, which imports
neither Spark nor pandas. This file does I/O, the join, and the EAD checks.

Inputs (all produced by earlier steps, none regenerated here):
    data/08_reporting/cv_report.json            which folds finished
    data/07_model_output/cv_oof/fold_*.parquet  id, TARGET, pd_score, fold
    data/05_model_input/{train,val}.parquet     the EAD column

Output:
    data/08_reporting/threshold_report.json

Run:
    make threshold
"""

from __future__ import annotations

import argparse

from credit_risk.config import get_config, project_root
from credit_risk.cv.study import REPORT_FILENAMES
from credit_risk.reporting import atomic_write_json, load_report
from credit_risk.reporting.oof import join_attributes, load_attributes, load_oof
from credit_risk.threshold import (
    REPORT_FILENAME,
    CostParams,
    build_report,
    build_scenario,
    build_setup,
    format_summary,
    margin_rates,
    population_summary,
)
from credit_risk.utils.logging import configure_logging, get_logger

log = get_logger("pipeline.threshold")

_MODEL_INPUT = project_root() / "data" / "05_model_input"
_OOF_DIR = project_root() / "data" / "07_model_output" / "cv_oof"
_REPORT_DIR = project_root() / "data" / "08_reporting"

_THRESHOLD_REPORT = _REPORT_DIR / REPORT_FILENAME
_CV_REPORT = _REPORT_DIR / REPORT_FILENAMES["cv"]


# ══════════════════════════════════════════════════════════════
# Rows
# ══════════════════════════════════════════════════════════════

def build_rows(merged, target_col: str, ead_column: str):
    """(pd_score, target, ead) per row, plus the same grouped by fold.

    EAD is validated rather than cleaned: dropping rows with a bad exposure
    would quietly change the population every number below is computed on, so
    a bad exposure stops the run and says how many there are.
    """
    ead = merged[ead_column]
    n_null = int(ead.isna().sum())
    n_nonpos = int((ead.fillna(0) <= 0).sum()) - n_null
    if n_null or n_nonpos:
        raise RuntimeError(
            f"{ead_column} has {n_null} null and {n_nonpos} non-positive "
            f"values out of {len(merged)}. EAD multiplies every cost in this "
            "study, so these are not dropped silently — decide explicitly how "
            "to treat them (a floor, an exclusion recorded in the report, or a "
            "different EAD proxy) before quoting a threshold."
        )

    rows = list(zip(
        merged["pd_score"].astype(float).tolist(),
        merged[target_col].astype(int).tolist(),
        ead.astype(float).tolist(),
    ))

    rows_by_fold: dict[int, list] = {}
    if "fold" in merged.columns:
        for fold, row in zip(merged["fold"].astype(int).tolist(), rows):
            rows_by_fold.setdefault(fold, []).append(row)

    return rows, rows_by_fold


# ══════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════

def run_threshold() -> dict:
    """Build threshold_report.json from the T1.4 out-of-fold predictions."""
    cfg = get_config()
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]

    rates = margin_rates(cfg)
    cv_report = load_report(_CV_REPORT)
    if not cv_report:
        raise FileNotFoundError(
            f"{_CV_REPORT} not found — P1.1 must be built on out-of-fold "
            "predictions, not test.parquet. Run `make cv-fold N=1` … `N=5` "
            "first."
        )

    oof, folds = load_oof(cv_report, _OOF_DIR, _CV_REPORT)

    setup = build_setup(cfg, rates, source={
        "cv_folds_used": folds,
        "cv_report": f"data/08_reporting/{REPORT_FILENAMES['cv']}",
    })
    ead_column = setup["ead_column"]

    attrs = load_attributes(
        [ead_column], id_col, _MODEL_INPUT,
        hint=f"the EAD column {ead_column} (risk.expected_loss.ead_column)",
    )
    merged = join_attributes(oof, attrs, id_col)
    rows, rows_by_fold = build_rows(merged, target_col, ead_column)
    setup["source"]["oof_rows"] = len(rows)

    population = population_summary(rows, CostParams(rates[0], setup["lgd"]))
    log.info("threshold_population", **{
        k: population[k] for k in
        ("n", "n_default", "base_default_rate", "ead_pd_correlation")
    })

    scenarios = []
    for rate in rates:
        params = CostParams(margin_rate=rate, lgd=setup["lgd"])
        scenario = build_scenario(
            rows, params,
            approve_below=setup["current_policy"]["approve_below"],
            review_below=setup["current_policy"]["review_below"],
            rows_by_fold=rows_by_fold or None,
            n_points=setup["grid_points"],
            curve_points=setup["curve_points"],
            plateau_n_se=setup["plateau_n_se"],
        )
        scenarios.append(scenario)
        best = scenario["optimum"] or {}
        log.info("threshold_scenario_done", margin_rate=rate,
                 breakeven_pd=scenario["breakeven_pd"],
                 optimum=best.get("threshold"),
                 plateau=[scenario["plateau"]["lower"], scenario["plateau"]["upper"]],
                 approval_rate=best.get("approval_rate"),
                 bad_rate_approved=best.get("bad_rate_approved"))

    report = build_report(
        setup=setup,
        population=population,
        scenarios=scenarios,
        cv_config_fingerprint=cv_report.get("config_fingerprint", ""),
    )
    atomic_write_json(_THRESHOLD_REPORT, report)
    log.info("threshold_saved", path=str(_THRESHOLD_REPORT),
             n_scenarios=report["n_scenarios"],
             optimum_range=[report["sensitivity"]["optimum_threshold_min"],
                            report["sensitivity"]["optimum_threshold_max"]]
             if report.get("sensitivity") else None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-based cut-off on out-of-fold predictions (P1.1). "
                    "No Spark, no training, no arguments: the margin rates and "
                    "the grid come from config.",
    )
    parser.parse_args()

    configure_logging()
    log.info("threshold_started")
    try:
        report = run_threshold()
        print("\n" + format_summary(report) + "\n")
        log.info("threshold_completed")
    except Exception:
        log.exception("threshold_failed")
        raise


if __name__ == "__main__":
    main()
