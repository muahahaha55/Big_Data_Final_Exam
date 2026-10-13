"""Cross-validation study logic — pure Python, no Spark.

Holds everything that decides whether the T1.4 confidence intervals are
trustworthy: fold assignment, the comparability fingerprint, resume
bookkeeping, the t-interval formula, and the out-of-fold calibration table.
All of it is unit-testable in milliseconds (tests/test_cv.py).

Design notes that matter when reading the output
------------------------------------------------
**Pool.** Folds are drawn from `train.parquet ∪ val.parquet` (~85% of master).
`test.parquet` (15%) is never read, so it stays a clean hold-out for later
work. With k=5 each fold trains on ~68% of master, against ~70% for the
Production run in README §7.4 — close enough that the interval says something
about that number, but the evaluation sets differ, so it is *not* a confidence
interval for 0.7663 itself.

**Fold assignment.** `xxhash64(SK_ID_CURR, seed)` orders the rows inside each
TARGET stratum; fold = rank mod k. Deterministic regardless of Spark
partitioning (unlike `randomSplit`), reusable across any master variant, and
exactly balanced on TARGET by construction.

**Why stratify.** PR-AUC is sensitive to prevalence. Unstratified folds would
vary ~±0.12pp in default rate, injecting variance that is a sampling artifact
rather than model instability — the opposite of what this study measures. The
cost: stratification slightly *narrows* the observed spread, so the interval
is a little optimistic. Stated in the report rather than hidden.

**Why the interval is anti-conservative.** Any two folds share (k−2)/(k−1) of
their training rows — 75% at k=5. The t-interval assumes independent draws, so
it is narrower than the truth. Bengio & Grandvalet (2004) show no unbiased
estimator of k-fold CV variance exists. Hence mean ± std, min and max are
reported alongside, and the caveat ships inside the JSON.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

from credit_risk.reporting.checkpoint import canonical_fingerprint, completed_records

SCHEMA_VERSION = 1

REPORT_FILENAMES = {
    "cv": "cv_report.json",
    "calibration": "calibration_report.json",
}

METRICS = ("roc_auc", "pr_auc", "ks_statistic")

DEFAULT_K = 5
DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_DECILES = 10

CV_NOTE = (
    "The interval is for the MEAN across folds, not a prediction interval for "
    "a single run. Folds share training data ((k-2)/(k-1) of the rows), so the "
    "t-interval assumes more independence than exists and is "
    "anti-conservative; mean/std/min/max are reported alongside. Folds are "
    "stratified on TARGET, which slightly narrows the observed spread. Folds "
    "train on ~68% of master vs ~70% for the Production run in README 7.4, and "
    "evaluate on different sets, so this is not a confidence interval for "
    "0.7663 itself. test.parquet is never read."
)

CALIBRATION_NOTE = (
    "Built from out-of-fold predictions: every row is scored by a model that "
    "did not see it in training. Deciles are equal-count groups by predicted "
    "PD (ties broken by SK_ID_CURR for determinism)."
)


# ──────────────────────────────────────────────────────────────
# Fold assignment
# ──────────────────────────────────────────────────────────────

def fold_ids(k: int) -> list[int]:
    """Fold numbers, 1-based so they match `--fold N`."""
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    return list(range(1, k + 1))


def assign_folds(
    rows: Iterable[tuple[Any, int]],
    k: int,
    hash_fn: Callable[[Any], int],
) -> dict[Any, int]:
    """Assign each id to a fold, stratified on the label.

    `rows` is (id, label) pairs. Inside each label stratum the ids are ordered
    by `hash_fn(id)` (ties broken by `repr(id)` so the order is total), and the
    fold is the rank modulo k, shifted to 1-based.

    This is the rule the Spark implementation applies with `F.xxhash64`; the
    hash is a parameter so the *rule* can be tested without Spark.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")

    strata: dict[int, list[Any]] = {}
    for ident, label in rows:
        strata.setdefault(int(label), []).append(ident)

    out: dict[Any, int] = {}
    for label in sorted(strata):
        ordered = sorted(strata[label], key=lambda i: (hash_fn(i), repr(i)))
        for rank, ident in enumerate(ordered):
            out[ident] = (rank % k) + 1
    return out


def fold_balance(
    rows: Iterable[tuple[Any, int]],
    assignment: dict[Any, int],
    k: int,
) -> list[dict]:
    """Per-fold size, positive count and default rate — auditable balance."""
    counts = {f: {"n": 0, "n_positive": 0} for f in fold_ids(k)}
    for ident, label in rows:
        f = assignment[ident]
        counts[f]["n"] += 1
        counts[f]["n_positive"] += int(label)
    out = []
    for f in fold_ids(k):
        n, pos = counts[f]["n"], counts[f]["n_positive"]
        out.append({
            "fold": f,
            "n": n,
            "n_positive": pos,
            "default_rate": round(pos / n, 6) if n else 0.0,
        })
    return out


def check_fold_balance(balance: Sequence[dict], tolerance_n: int = 1) -> None:
    """Assert stratification actually worked; raise loudly if it did not.

    Runs against the Spark-produced assignment so a drift between the two
    implementations fails the pipeline instead of quietly skewing the metrics.
    """
    if not balance:
        raise ValueError("empty fold balance")
    sizes = [b["n"] for b in balance]
    positives = [b["n_positive"] for b in balance]
    if max(sizes) - min(sizes) > tolerance_n:
        raise RuntimeError(
            f"fold sizes differ by more than {tolerance_n}: {sizes}")
    if max(positives) - min(positives) > tolerance_n:
        raise RuntimeError(
            f"fold positive counts differ by more than {tolerance_n} — "
            f"stratification failed: {positives}")


# ──────────────────────────────────────────────────────────────
# Setup block + fingerprint
# ──────────────────────────────────────────────────────────────

def build_setup(k: int, cfg: dict, pool: dict | None = None) -> dict:
    """Everything that defines comparability between folds."""
    sp = cfg.get("spark", {})
    cv = cfg.get("cv", {})
    return {
        "k_folds": k,
        "fold_assignment": "xxhash64(SK_ID_CURR, seed) ordering within TARGET "
                           "stratum; fold = rank mod k (1-based)",
        "stratified_by": cfg["data"]["target_column"],
        "cv_pool": "data/05_model_input/train.parquet + val.parquet",
        "holdout_untouched": "data/05_model_input/test.parquet",
        "model_type": "gbt",
        "hyperparams": dict(cfg["models"].get("gbt", {})),
        "random_seed": cfg.get("random_seed", 42),
        "tune": False,
        # Reproducing Production exactly: add_class_weights() exists in
        # trainer.py but is never called and no classifier receives weightCol.
        "class_weighting": False,
        "missing_threshold": cfg["features"]["missing_threshold"],
        "confidence_level": float(cv.get("confidence_level", DEFAULT_CONFIDENCE)),
        "spark": {
            "master": sp.get("master", "local[1]"),
            "driver_memory": sp.get("driver_memory", "2g"),
            "shuffle_partitions": sp.get("shuffle_partitions", 50),
        },
        "pool": pool or {},
        "not_comparable_to": (
            "README 7.4 ROC-AUC 0.7663 — folds train on ~68% of master vs ~70% "
            "there, and evaluate on fold hold-outs rather than test.parquet"
        ),
    }


def config_fingerprint(setup: dict) -> str:
    """Hash of the comparability-critical settings only.

    Spark master and shuffle_partitions are included because they change task
    partitioning, and GBT's per-partition subsampling can shift results even
    with a fixed seed. Prose and measured pool counts are excluded.
    """
    return canonical_fingerprint({
        "k_folds": setup.get("k_folds"),
        "fold_assignment": setup.get("fold_assignment"),
        "stratified_by": setup.get("stratified_by"),
        "cv_pool": setup.get("cv_pool"),
        "model_type": setup.get("model_type"),
        "hyperparams": setup.get("hyperparams"),
        "random_seed": setup.get("random_seed"),
        "tune": setup.get("tune"),
        "class_weighting": setup.get("class_weighting"),
        "missing_threshold": setup.get("missing_threshold"),
        "spark_master": (setup.get("spark") or {}).get("master"),
        "spark_shuffle_partitions": (setup.get("spark") or {}).get("shuffle_partitions"),
    })


# ──────────────────────────────────────────────────────────────
# Resume bookkeeping
# ──────────────────────────────────────────────────────────────

def completed_folds(report: dict | None) -> dict[int, dict]:
    """Fold number → record, for folds already finished."""
    return completed_records(report, records_key="folds", id_key="fold")


def pending_folds(
    k: int,
    report: dict | None,
    only_fold: int | None = None,
    force: bool = False,
) -> list[int]:
    """Which folds still need running, honouring --fold and --force."""
    done = set() if force else set(completed_folds(report))
    out = [f for f in fold_ids(k) if f not in done]
    if only_fold is not None:
        out = [f for f in out if f == only_fold]
    return out


# ──────────────────────────────────────────────────────────────
# Column audit — does the model actually see credit_card features?
# ──────────────────────────────────────────────────────────────

def column_audit(
    all_columns: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
    excluded: Sequence[str],
) -> dict:
    """Which candidate columns made it into the model, and which were dropped.

    `identify_feature_columns` only logs a dropped *count*, so the names are
    reconstructed here: candidates are all columns minus the label and ids;
    whatever is not kept as numeric or categorical was dropped (high-null, or
    an unsupported dtype).
    """
    excluded_set = set(excluded)
    kept = set(numeric) | set(categorical)
    candidates = [c for c in all_columns if c not in excluded_set]
    dropped = [c for c in candidates if c not in kept]
    return {
        "n_master_columns": len(all_columns),
        "n_candidates": len(candidates),
        "n_source_columns": len(numeric) + len(categorical),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "dropped_columns": sorted(dropped),
        "n_dropped": len(dropped),
    }


def dropped_by_prefix(dropped: Sequence[str], prefixes: dict[str, str]) -> dict:
    """Group dropped column names by source-table prefix.

    Turns "credit_card_balance contributes nothing" from an inference into a
    list of names, for each table whose aggregates carry a known prefix.
    """
    out = {}
    for table, prefix in prefixes.items():
        hit = sorted(c for c in dropped if c.startswith(prefix))
        out[table] = {"prefix": prefix, "n_dropped": len(hit), "columns": hit}
    return out


# ──────────────────────────────────────────────────────────────
# Confidence interval
# ──────────────────────────────────────────────────────────────

def t_critical(confidence: float, df: int) -> float:
    """Two-sided t critical value. scipy if available, else a small table."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    try:
        from scipy import stats
        return float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        table_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        if abs(confidence - 0.95) > 1e-9 or df not in table_95:
            raise
        return table_95[df]


def t_interval(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """mean ± t(1-α/2, n-1) · s/√n, plus the raw spread.

    `s` is the sample standard deviation (ddof=1). With a single value there is
    no spread to estimate, so std and the interval are None rather than 0 —
    reporting 0 would read as "perfectly stable".
    """
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "ci95_lower": None, "ci95_upper": None,
                "t_critical": None, "df": None, "confidence_level": confidence}

    mean = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": round(mean, 6), "std": None,
                "min": round(mean, 6), "max": round(mean, 6),
                "ci95_lower": None, "ci95_upper": None,
                "t_critical": None, "df": 0, "confidence_level": confidence}

    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    df = n - 1
    tcrit = t_critical(confidence, df)
    half = tcrit * std / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "ci95_lower": round(mean - half, 6),
        "ci95_upper": round(mean + half, 6),
        "half_width": round(half, 6),
        "t_critical": round(tcrit, 4),
        "df": df,
        "confidence_level": confidence,
    }


def summarize_folds(folds: Sequence[dict], confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """t-interval per metric across whatever folds have finished."""
    return {
        m: t_interval([f.get(m) for f in folds], confidence) for m in METRICS
    }


def build_cv_report(
    setup: dict,
    fold_records: Sequence[dict],
    fold_balance_rows: Sequence[dict] | None = None,
    generated_at: str | None = None,
) -> dict:
    """Assemble cv_report.json from whatever folds exist."""
    folds = sorted((dict(f) for f in fold_records), key=lambda f: f["fold"])
    confidence = float(setup.get("confidence_level", DEFAULT_CONFIDENCE))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now(),
        "config_fingerprint": config_fingerprint(setup),
        "setup": setup,
        "note": CV_NOTE,
        "n_folds_total": setup.get("k_folds"),
        "n_folds_completed": len(folds),
        "fold_balance": list(fold_balance_rows or []),
        "folds": folds,
        "summary": summarize_folds(folds, confidence),
    }


# ──────────────────────────────────────────────────────────────
# Calibration on out-of-fold predictions
# ──────────────────────────────────────────────────────────────

def decile_table(
    rows: Sequence[tuple[Any, int, float]],
    n_deciles: int = DEFAULT_N_DECILES,
) -> list[dict]:
    """Equal-count groups by predicted PD, ascending.

    `rows` is (id, target, pd_score). Ranking breaks ties by id so the table is
    deterministic. Groups are contiguous rank ranges, so sizes differ by at
    most one — the NTILE convention. Decile 1 is the lowest predicted risk.
    """
    if n_deciles < 2:
        raise ValueError(f"n_deciles must be at least 2, got {n_deciles}")
    data = [(float(pd_), int(t), ident) for ident, t, pd_ in rows]
    n = len(data)
    if n == 0:
        return []

    data.sort(key=lambda r: (r[0], repr(r[2])))
    buckets: dict[int, list[tuple[float, int]]] = {}
    for rank, (pd_, target, _ident) in enumerate(data):
        d = (rank * n_deciles) // n + 1
        buckets.setdefault(d, []).append((pd_, target))

    out = []
    for d in sorted(buckets):
        vals = buckets[d]
        cnt = len(vals)
        pds = [v[0] for v in vals]
        defaults = sum(v[1] for v in vals)
        mean_pd = sum(pds) / cnt
        actual = defaults / cnt
        out.append({
            "decile": d,
            "n": cnt,
            "n_default": defaults,
            "pd_min": round(min(pds), 6),
            "pd_max": round(max(pds), 6),
            "pd_mean_predicted": round(mean_pd, 6),
            "actual_default_rate": round(actual, 6),
            "gap": round(mean_pd - actual, 6),
        })
    return out


def expected_calibration_error(deciles: Sequence[dict]) -> float | None:
    """ECE = Σ (nᵢ/N) · |predictedᵢ − actualᵢ|, weighted by bucket size."""
    total = sum(d["n"] for d in deciles)
    if not total:
        return None
    return round(
        sum(d["n"] * abs(d["pd_mean_predicted"] - d["actual_default_rate"])
            for d in deciles) / total,
        6,
    )


def build_calibration_report(
    rows: Sequence[tuple[Any, int, float]],
    config_fingerprint_value: str,
    n_folds: int,
    n_deciles: int = DEFAULT_N_DECILES,
    generated_at: str | None = None,
) -> dict:
    """Assemble calibration_report.json from out-of-fold predictions.

    `verdict` is left empty on purpose: PLAN T1.4 requires the "well
    calibrated" claim to be rewritten against the evidence, so the wording is
    decided once the numbers exist, not pre-filled here.
    """
    deciles = decile_table(rows, n_deciles)
    n_total = sum(d["n"] for d in deciles)
    n_default = sum(d["n_default"] for d in deciles)
    mean_pd = (
        sum(d["pd_mean_predicted"] * d["n"] for d in deciles) / n_total
        if n_total else None
    )
    actual = n_default / n_total if n_total else None
    worst = max(deciles, key=lambda d: abs(d["gap"])) if deciles else None

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now(),
        "config_fingerprint": config_fingerprint_value,
        "source": f"out-of-fold predictions, {n_folds} folds, "
                  "data/07_model_output/cv_oof/",
        "note": CALIBRATION_NOTE,
        "n_total": n_total,
        "n_default": n_default,
        "overall": {
            "mean_predicted_pd": round(mean_pd, 6) if mean_pd is not None else None,
            "actual_default_rate": round(actual, 6) if actual is not None else None,
            "gap": round(mean_pd - actual, 6)
                   if mean_pd is not None and actual is not None else None,
        },
        "n_deciles": n_deciles,
        "deciles": deciles,
        "ece": expected_calibration_error(deciles),
        "max_abs_gap": round(abs(worst["gap"]), 6) if worst else None,
        "max_abs_gap_decile": worst["decile"] if worst else None,
        "verdict": "",
    }


# ──────────────────────────────────────────────────────────────
# Console rendering
# ──────────────────────────────────────────────────────────────

def format_cv_summary(report: dict) -> str:
    """Per-fold table plus the interval per metric."""
    lines = [
        "=" * 78,
        f"  CV REPORT — k={report.get('n_folds_total')}  "
        f"({report.get('n_folds_completed')}/{report.get('n_folds_total')} folds)",
        "=" * 78,
        f"  {'fold':>5}{'train':>10}{'valid':>9}{'cols':>7}"
        f"{'ROC-AUC':>10}{'PR-AUC':>9}{'KS':>9}{'sec':>9}",
    ]
    for f in report.get("folds", []):
        lines.append(
            f"  {f.get('fold', '?'):>5}{f.get('n_train', '-'):>10}"
            f"{f.get('n_valid', '-'):>9}{f.get('n_source_columns', '-'):>7}"
            f"{_fmt(f.get('roc_auc')):>10}{_fmt(f.get('pr_auc')):>9}"
            f"{_fmt(f.get('ks_statistic')):>9}{f.get('elapsed_sec', '-'):>9}"
        )
    lines.append("-" * 78)
    for m, s in (report.get("summary") or {}).items():
        if s.get("mean") is None:
            continue
        std = f"± {s['std']:.4f}" if s.get("std") is not None else "± —"
        ci = (f"[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]"
              if s.get("ci95_lower") is not None else "[—]")
        lines.append(f"  {m:<14} {s['mean']:.4f} {std}   CI95 {ci}"
                     f"   min {s['min']:.4f}  max {s['max']:.4f}")
    lines.append("=" * 78)
    return "\n".join(lines)


def format_calibration_summary(report: dict) -> str:
    """Decile table plus ECE."""
    lines = [
        "=" * 78,
        f"  OUT-OF-FOLD CALIBRATION — {report.get('n_total')} rows, "
        f"{report.get('n_default')} defaults",
        "=" * 78,
        f"  {'decile':>7}{'n':>9}{'defaults':>10}"
        f"{'mean PD':>11}{'actual':>10}{'gap':>10}",
    ]
    for d in report.get("deciles", []):
        lines.append(
            f"  {d['decile']:>7}{d['n']:>9}{d['n_default']:>10}"
            f"{d['pd_mean_predicted']:>11.4f}{d['actual_default_rate']:>10.4f}"
            f"{d['gap']:>+10.4f}"
        )
    ov = report.get("overall") or {}
    lines.append("-" * 78)
    if ov.get("mean_predicted_pd") is not None:
        lines.append(
            f"  overall: mean PD {ov['mean_predicted_pd']:.4f} vs actual "
            f"{ov['actual_default_rate']:.4f}  (gap {ov['gap']:+.4f})"
        )
    if report.get("ece") is not None:
        lines.append(f"  ECE {report['ece']:.4f}   "
                     f"max |gap| {report['max_abs_gap']:.4f} "
                     f"at decile {report['max_abs_gap_decile']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    return "—" if v is None else f"{v:.4f}"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
