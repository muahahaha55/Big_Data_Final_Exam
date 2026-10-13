"""Segment validation study (P0.4) — pure Python, no Spark, no pandas.

A single ROC-AUC is an average over every kind of customer. It can hide a
segment where the model is close to useless, and in credit risk that is both a
model-risk problem and a fairness problem. This module slices the
**out-of-fold** predictions from T1.4 and measures discrimination and
calibration inside each slice.

Why out-of-fold and not `test.parquet`
--------------------------------------
Every row in `data/07_model_output/cv_oof/` was scored by a model that did not
train on it, so segment numbers are honest hold-out numbers — and there are
~261k of them instead of 45k, which matters because a segment of a hold-out
set is small enough that the AUC is mostly noise.

Reading the output
------------------
**Noise is reported, not hidden.** Each segment AUC carries a Hanley-McNeil
standard error and a 95% interval. A dimension's `auc_spread` is compared
against the noise of its best and worst group (`spread_vs_noise`), so "group X
is worse" is only claimed when the gap is larger than the uncertainty. The
approximation assumes exponential score distributions and treats the two
groups as independent samples — true of the rows, not of the model that scored
them, so the ratio is indicative rather than a formal test.

**Small groups are suppressed, not reported badly.** A group below
`min_rows` or `min_positives` gets `evaluated: false` and a reason. With a
~8% default rate, a group needs ~600 rows before it holds 50 defaults, and an
AUC built on a handful of defaults moves by whole percentage points when one
of them changes.

**No PR-AUC per segment.** The repo's PR-AUC comes from Spark's
`areaUnderPR` (trapezoid over the recall axis), which is not the same
estimator as average precision. Rather than publish a number computed by a
different definition than every other PR-AUC in the repo, this module reports
ROC-AUC, KS and the calibration gap. ROC-AUC here is the Mann-Whitney
statistic with mid-ranks for ties, and KS matches `scipy.stats.ks_2samp` —
the same definition `trainer.evaluate_predictions` uses.

**Gender is measured because the model uses it.** `CODE_GENDER` is in
`features.categorical_columns`, so it is an input to the model, not just a
reporting key. Measuring the disparity is the minimum; this is a measurement,
not a legal fairness assessment, and the report says so.

No resume machinery (unlike the ablation and CV studies): this runs in seconds
and is fully derived from files those studies already checkpointed.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from credit_risk.reporting.checkpoint import canonical_fingerprint

SCHEMA_VERSION = 1
REPORT_FILENAME = "segment_report.json"

KIND_CATEGORICAL = "categorical"
KIND_QUANTILE = "quantile"
VALID_KINDS = (KIND_CATEGORICAL, KIND_QUANTILE)

MISSING_LABEL = "__missing__"

DEFAULT_MIN_ROWS = 1000
DEFAULT_MIN_POSITIVES = 50
DEFAULT_N_BINS = 5

# Pooled out-of-fold AUC and the mean of the per-fold AUCs are different
# estimands, so they are not expected to be equal. A gap this large means
# something structural is wrong (score column, label flip, misaligned join),
# not sampling.
AUC_HARD_TOLERANCE = 0.05
AUC_SOFT_TOLERANCE = 0.01

# Named, auditable transforms. No eval: a config file must not be able to run
# arbitrary code, and a reader must be able to see exactly what was applied.
TRANSFORMS: dict[str, Callable[[float], float]] = {
    "none": lambda v: v,
    "days_to_years": lambda v: -v / 365.25,
}

NOTE = (
    "Segment metrics are computed on out-of-fold predictions (T1.4): every row "
    "was scored by a model that did not train on it. ROC-AUC is the "
    "Mann-Whitney statistic with mid-ranks for ties; KS matches "
    "scipy.stats.ks_2samp, the definition used elsewhere in the repo. PR-AUC "
    "is deliberately absent — Spark's areaUnderPR is a different estimator "
    "from average precision and the two must not be mixed. Each AUC carries a "
    "Hanley-McNeil standard error; a dimension's spread is only meaningful "
    "when spread_vs_noise is above ~2. Groups below min_rows or min_positives "
    "are suppressed rather than reported with an unstable AUC. Quantile bin "
    "edges are computed from the data, not imposed. CODE_GENDER is an input "
    "feature of the model, so the gender rows measure a disparity in a model "
    "that sees gender; this is a measurement, not a fairness assessment, and "
    "no legal conclusion follows from it."
)


# ──────────────────────────────────────────────────────────────
# Dimension specs
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentSpec:
    """One way of slicing the population."""

    name: str
    column: str
    kind: str = KIND_CATEGORICAL
    n_bins: int = DEFAULT_N_BINS
    transform: str = "none"
    sensitive: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("segment spec needs a name")
        if not self.column:
            raise ValueError(f"segment {self.name!r} needs a source column")
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"segment {self.name!r}: kind {self.kind!r} not in {VALID_KINDS}"
            )
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"segment {self.name!r}: transform {self.transform!r} not in "
                f"{sorted(TRANSFORMS)} — transforms are a fixed registry on "
                "purpose, config must not be able to run arbitrary code"
            )
        if self.kind == KIND_QUANTILE and self.n_bins < 2:
            raise ValueError(
                f"segment {self.name!r}: n_bins must be at least 2, got {self.n_bins}"
            )

    @property
    def derivation(self) -> str:
        """Human-readable statement of what was actually measured."""
        if self.transform == "none":
            return self.column
        return f"{self.transform}({self.column})"


def build_specs(cfg: dict) -> list[SegmentSpec]:
    """Read the dimensions out of `segments.dimensions` in config."""
    section = cfg.get("segments") or {}
    raw = section.get("dimensions") or []
    if not raw:
        raise ValueError(
            "config has no segments.dimensions — P0.4 is config-driven so the "
            "slices that back README claims are visible in one place"
        )
    default_bins = int(section.get("n_bins", DEFAULT_N_BINS))
    specs = [
        SegmentSpec(
            name=str(d["name"]),
            column=str(d["column"]),
            kind=str(d.get("kind", KIND_CATEGORICAL)),
            n_bins=int(d.get("n_bins", default_bins)),
            transform=str(d.get("transform", "none")),
            sensitive=bool(d.get("sensitive", False)),
            description=str(d.get("description", "")),
        )
        for d in raw
    ]
    names = [s.name for s in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate segment names in config: {sorted(dupes)}")
    return specs


def source_columns(specs: Sequence[SegmentSpec]) -> list[str]:
    """Columns the pipeline must read — used for Parquet column pruning."""
    seen: list[str] = []
    for s in specs:
        if s.column not in seen:
            seen.append(s.column)
    return seen


# ──────────────────────────────────────────────────────────────
# Binning
# ──────────────────────────────────────────────────────────────

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:   # NaN
        return True
    return False


def quantile_edges(values: Sequence[float], n_bins: int) -> list[float]:
    """Interior cut points at equal-count ranks, duplicates removed.

    Edges come from the data, so the bands are not imposed round numbers. Ties
    can collapse two cut points into one (a heavily repeated value such as a
    modal income); the duplicate is dropped and the dimension reports fewer
    groups than requested rather than emitting an empty band.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    clean = sorted(float(v) for v in values if not _is_missing(v))
    n = len(clean)
    if n == 0:
        return []
    edges: list[float] = []
    for i in range(1, n_bins):
        edge = clean[(i * n) // n_bins]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def bin_label(value: Any, edges: Sequence[float]) -> str:
    """`q1`..`qk` by cut point; `__missing__` when the value is absent.

    A value equal to an edge falls in the upper band, matching the rank-based
    construction of the edges themselves.
    """
    if _is_missing(value):
        return MISSING_LABEL
    return f"q{bisect.bisect_right(list(edges), float(value)) + 1}"


def bin_bounds(label: str, edges: Sequence[float]) -> tuple[float | None, float | None]:
    """Half-open bounds [lower, upper) of a quantile band; None at the ends."""
    if not label.startswith("q"):
        return None, None
    try:
        idx = int(label[1:]) - 1
    except ValueError:
        return None, None
    lower = edges[idx - 1] if idx > 0 else None
    upper = edges[idx] if idx < len(edges) else None
    return lower, upper


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def roc_auc(pairs: Sequence[tuple[int, float]]) -> float | None:
    """Mann-Whitney U / (n_pos·n_neg), with mid-ranks for tied scores.

    `pairs` is (target, score). None when either class is absent — an AUC is
    undefined there, and returning 0.5 would look like a measurement.
    """
    data = sorted(((float(s), int(t)) for t, s in pairs), key=lambda r: r[0])
    n = len(data)
    n_pos = sum(t for _, t in data)
    n_neg = n - n_pos
    if not n_pos or not n_neg:
        return None

    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and data[j + 1][0] == data[i][0]:
            j += 1
        mid_rank = ((i + 1) + (j + 1)) / 2.0          # ranks are 1-based
        rank_sum_pos += mid_rank * sum(t for _, t in data[i:j + 1])
        i = j + 1

    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auc_standard_error(auc: float | None, n_pos: int, n_neg: int) -> float | None:
    """Hanley-McNeil (1982) standard error of an AUC.

    Q1 = A/(2−A) and Q2 = 2A²/(1+A) assume exponential score distributions.
    It is the standard closed form and needs no resampling, which keeps this
    module free of numpy; the report says it is an approximation.
    """
    if auc is None or n_pos <= 0 or n_neg <= 0:
        return None
    a = float(auc)
    q1 = a / (2.0 - a) if a < 2.0 else 0.0
    q2 = 2.0 * a * a / (1.0 + a)
    var = (
        a * (1.0 - a)
        + (n_pos - 1) * (q1 - a * a)
        + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    return math.sqrt(var) if var > 0 else 0.0


def ks_statistic(pairs: Sequence[tuple[int, float]]) -> float | None:
    """max |F_default(s) − F_good(s)|, evaluated after each block of ties.

    Evaluating only at distinct score values makes the result independent of
    the order rows arrive in, and matches `scipy.stats.ks_2samp`.
    """
    data = sorted(((float(s), int(t)) for t, s in pairs), key=lambda r: r[0])
    n = len(data)
    n_pos = sum(t for _, t in data)
    n_neg = n - n_pos
    if not n_pos or not n_neg:
        return None

    cum_pos = cum_neg = 0
    ks = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and data[j + 1][0] == data[i][0]:
            j += 1
        for _, t in data[i:j + 1]:
            cum_pos += t
            cum_neg += 1 - t
        ks = max(ks, abs(cum_pos / n_pos - cum_neg / n_neg))
        i = j + 1
    return ks


def group_metrics(
    rows: Sequence[tuple[int, float]],
    min_rows: int = DEFAULT_MIN_ROWS,
    min_positives: int = DEFAULT_MIN_POSITIVES,
) -> dict:
    """Discrimination + calibration for one group, or a suppression reason."""
    n = len(rows)
    n_pos = sum(int(t) for t, _ in rows)
    mean_pd = sum(float(p) for _, p in rows) / n if n else None
    actual = n_pos / n if n else None

    record: dict[str, Any] = {
        "n": n,
        "n_positive": n_pos,
        "default_rate": round(actual, 6) if actual is not None else None,
        "mean_predicted_pd": round(mean_pd, 6) if mean_pd is not None else None,
        "gap": round(mean_pd - actual, 6)
               if mean_pd is not None and actual is not None else None,
        "roc_auc": None,
        "auc_se": None,
        "auc_ci95_lower": None,
        "auc_ci95_upper": None,
        "ks_statistic": None,
        "evaluated": False,
        "suppressed_reason": None,
    }

    if n < min_rows:
        record["suppressed_reason"] = f"n={n} below min_rows={min_rows}"
        return record
    if n_pos < min_positives:
        record["suppressed_reason"] = (
            f"n_positive={n_pos} below min_positives={min_positives}"
        )
        return record
    if n_pos == n:
        record["suppressed_reason"] = "no non-default rows — AUC undefined"
        return record

    auc = roc_auc(rows)
    se = auc_standard_error(auc, n_pos, n - n_pos)
    record.update({
        "roc_auc": round(auc, 6) if auc is not None else None,
        "auc_se": round(se, 6) if se is not None else None,
        "ks_statistic": round(ks_statistic(rows) or 0.0, 6),
        "evaluated": auc is not None,
    })
    if auc is not None and se is not None:
        record["auc_ci95_lower"] = round(auc - 1.96 * se, 6)
        record["auc_ci95_upper"] = round(auc + 1.96 * se, 6)
    return record


# ──────────────────────────────────────────────────────────────
# One dimension
# ──────────────────────────────────────────────────────────────

def build_dimension(
    spec: SegmentSpec,
    rows: Sequence[tuple[Any, int, float]],
    min_rows: int = DEFAULT_MIN_ROWS,
    min_positives: int = DEFAULT_MIN_POSITIVES,
) -> dict:
    """Group `rows` = (raw value, target, pd_score) by `spec` and measure each.

    Binning lives here rather than in the pipeline so the cut points — the
    part a reader will question — are unit-testable without Spark or pandas.
    """
    fn = TRANSFORMS[spec.transform]
    edges: list[float] = []

    if spec.kind == KIND_QUANTILE:
        values = [fn(float(v)) for v, _, _ in rows if not _is_missing(v)]
        edges = quantile_edges(values, spec.n_bins)

    grouped: dict[str, list[tuple[int, float]]] = {}
    for value, target, pd_score in rows:
        if spec.kind == KIND_QUANTILE:
            label = (MISSING_LABEL if _is_missing(value)
                     else bin_label(fn(float(value)), edges))
        else:
            label = MISSING_LABEL if _is_missing(value) else str(value)
        grouped.setdefault(label, []).append((int(target), float(pd_score)))

    groups = []
    for label in _sorted_labels(grouped, spec.kind):
        record = {"group": label}
        if spec.kind == KIND_QUANTILE:
            lower, upper = bin_bounds(label, edges)
            record["lower"] = round(lower, 4) if lower is not None else None
            record["upper"] = round(upper, 4) if upper is not None else None
        record.update(group_metrics(grouped[label], min_rows, min_positives))
        groups.append(record)

    return {
        "name": spec.name,
        "column": spec.column,
        "kind": spec.kind,
        "transform": spec.transform,
        "derivation": spec.derivation,
        "sensitive": spec.sensitive,
        "description": spec.description,
        "n_bins_requested": spec.n_bins if spec.kind == KIND_QUANTILE else None,
        "bin_edges": [round(e, 4) for e in edges] if edges else None,
        "n_groups": len(groups),
        "n_groups_evaluated": sum(1 for g in groups if g["evaluated"]),
        "n_rows": sum(g["n"] for g in groups),
        "groups": groups,
        **_dimension_spread(groups),
    }


def _sorted_labels(grouped: dict[str, list], kind: str) -> list[str]:
    """Quantile bands in band order; categories by size, missing last."""
    labels = [l for l in grouped if l != MISSING_LABEL]
    if kind == KIND_QUANTILE:
        labels.sort(key=lambda l: int(l[1:]) if l[1:].isdigit() else 0)
    else:
        labels.sort(key=lambda l: (-len(grouped[l]), l))
    if MISSING_LABEL in grouped:
        labels.append(MISSING_LABEL)
    return labels


def _dimension_spread(groups: Sequence[dict]) -> dict:
    """Best/worst evaluated group and whether the gap clears the noise."""
    scored = [g for g in groups if g["evaluated"] and g["roc_auc"] is not None]
    if len(scored) < 2:
        return {
            "auc_min": scored[0]["roc_auc"] if scored else None,
            "auc_max": scored[0]["roc_auc"] if scored else None,
            "auc_spread": None,
            "worst_group": scored[0]["group"] if scored else None,
            "best_group": scored[0]["group"] if scored else None,
            "spread_vs_noise": None,
            "spread_within_noise": None,
            "max_abs_gap": abs(scored[0]["gap"]) if scored and scored[0]["gap"] is not None else None,
            "max_abs_gap_group": scored[0]["group"] if scored else None,
        }

    worst = min(scored, key=lambda g: g["roc_auc"])
    best = max(scored, key=lambda g: g["roc_auc"])
    spread = best["roc_auc"] - worst["roc_auc"]
    se_a, se_b = worst.get("auc_se"), best.get("auc_se")
    ratio = None
    if se_a and se_b:
        ratio = spread / math.sqrt(se_a ** 2 + se_b ** 2)

    gapped = [g for g in scored if g["gap"] is not None]
    worst_gap = max(gapped, key=lambda g: abs(g["gap"])) if gapped else None

    return {
        "auc_min": worst["roc_auc"],
        "auc_max": best["roc_auc"],
        "auc_spread": round(spread, 6),
        "worst_group": worst["group"],
        "best_group": best["group"],
        "spread_vs_noise": round(ratio, 2) if ratio is not None else None,
        "spread_within_noise": (ratio < 2.0) if ratio is not None else None,
        "max_abs_gap": round(abs(worst_gap["gap"]), 6) if worst_gap else None,
        "max_abs_gap_group": worst_gap["group"] if worst_gap else None,
    }


# ──────────────────────────────────────────────────────────────
# Setup + comparability
# ──────────────────────────────────────────────────────────────

def build_setup(cfg: dict, specs: Sequence[SegmentSpec], source: dict | None = None) -> dict:
    """What was sliced, how, and with what support thresholds."""
    section = cfg.get("segments") or {}
    return {
        "min_rows": int(section.get("min_rows", DEFAULT_MIN_ROWS)),
        "min_positives": int(section.get("min_positives", DEFAULT_MIN_POSITIVES)),
        "n_bins_default": int(section.get("n_bins", DEFAULT_N_BINS)),
        "predictions": "data/07_model_output/cv_oof (out-of-fold, T1.4)",
        "attributes": "data/05_model_input/train.parquet + val.parquet",
        "dimensions": [
            {
                "name": s.name,
                "column": s.column,
                "kind": s.kind,
                "transform": s.transform,
                "n_bins": s.n_bins if s.kind == KIND_QUANTILE else None,
                "sensitive": s.sensitive,
            }
            for s in specs
        ],
        "metrics": ["roc_auc", "auc_se", "ks_statistic", "gap"],
        "pr_auc_excluded": (
            "Spark areaUnderPR is a different estimator from average "
            "precision; mixing the two would not be comparable"
        ),
        "source": source or {},
    }


def config_fingerprint(setup: dict) -> str:
    """Hash of what decides whether two segment reports are comparable."""
    return canonical_fingerprint({
        "min_rows": setup.get("min_rows"),
        "min_positives": setup.get("min_positives"),
        "dimensions": setup.get("dimensions"),
        "predictions": setup.get("predictions"),
        "attributes": setup.get("attributes"),
    })


def compare_overall_auc(
    oof_auc: float | None,
    cv_mean_auc: float | None,
    hard_tolerance: float = AUC_HARD_TOLERANCE,
    soft_tolerance: float = AUC_SOFT_TOLERANCE,
) -> dict:
    """Cross-check the pure-Python pooled AUC against T1.4's fold mean.

    They are different estimands (one pooled hold-out vs the mean of five), so
    a small gap is expected and only recorded. A large gap means the join or
    the score column is wrong, which would silently corrupt every segment
    number, so that raises.
    """
    if oof_auc is None or cv_mean_auc is None:
        return {
            "cv_roc_auc_mean": cv_mean_auc,
            "oof_pooled_roc_auc": oof_auc,
            "delta": None,
            "within_soft_tolerance": None,
            "note": "no reference available in cv_report.json",
        }

    delta = float(oof_auc) - float(cv_mean_auc)
    if abs(delta) > hard_tolerance:
        raise RuntimeError(
            f"pooled out-of-fold ROC-AUC {oof_auc:.4f} differs from the T1.4 "
            f"fold mean {cv_mean_auc:.4f} by {delta:+.4f}, beyond "
            f"{hard_tolerance}. These are different estimands so a small gap "
            "is normal, but a gap this large means something structural is "
            "wrong — wrong score column, inverted label, or attributes joined "
            "to the wrong ids. Every segment number below would inherit it."
        )
    return {
        "cv_roc_auc_mean": round(float(cv_mean_auc), 6),
        "oof_pooled_roc_auc": round(float(oof_auc), 6),
        "delta": round(delta, 6),
        "within_soft_tolerance": abs(delta) <= soft_tolerance,
        "note": (
            "pooled out-of-fold AUC vs the mean of per-fold AUCs — different "
            "estimands, equality is not expected"
        ),
    }


# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────

def build_segment_report(
    setup: dict,
    dimensions: Sequence[dict],
    overall: dict,
    reference: dict,
    cv_config_fingerprint: str = "",
    generated_at: str | None = None,
) -> dict:
    """Assemble segment_report.json.

    `verdict` stays empty: the wording is written once the numbers exist, from
    the numbers, not pre-filled here.
    """
    flagged = [
        {
            "name": d["name"],
            "worst_group": d["worst_group"],
            "auc_min": d["auc_min"],
            "auc_max": d["auc_max"],
            "auc_spread": d["auc_spread"],
            "spread_vs_noise": d["spread_vs_noise"],
            "sensitive": d["sensitive"],
        }
        for d in dimensions
        if d.get("spread_within_noise") is False
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now(),
        "config_fingerprint": config_fingerprint(setup),
        "cv_config_fingerprint": cv_config_fingerprint,
        "note": NOTE,
        "setup": setup,
        "overall": overall,
        "reference": reference,
        "n_dimensions": len(dimensions),
        "dimensions_with_spread_beyond_noise": flagged,
        "dimensions": list(dimensions),
        "verdict": "",
    }


def overall_metrics(rows: Sequence[tuple[int, float]]) -> dict:
    """Pooled out-of-fold metrics — the number every segment is compared to."""
    record = group_metrics(rows, min_rows=0, min_positives=0)
    record.pop("suppressed_reason", None)
    return record


# ──────────────────────────────────────────────────────────────
# Console rendering
# ──────────────────────────────────────────────────────────────

def format_summary(report: dict) -> str:
    """One block per dimension, groups as rows."""
    ov = report.get("overall") or {}
    lines = [
        "=" * 86,
        f"  SEGMENT VALIDATION — {ov.get('n', '?')} out-of-fold rows, "
        f"{ov.get('n_positive', '?')} defaults",
        "=" * 86,
    ]
    if ov.get("roc_auc") is not None:
        lines.append(
            f"  pooled OOF: ROC-AUC {ov['roc_auc']:.4f} "
            f"(SE {_fmt(ov.get('auc_se'))})  KS {_fmt(ov.get('ks_statistic'))}  "
            f"mean PD {_fmt(ov.get('mean_predicted_pd'))} vs actual "
            f"{_fmt(ov.get('default_rate'))}"
        )
    ref = report.get("reference") or {}
    if ref.get("delta") is not None:
        lines.append(
            f"  vs T1.4 fold mean {ref['cv_roc_auc_mean']:.4f}: "
            f"{ref['delta']:+.4f} (different estimands)"
        )

    for d in report.get("dimensions", []):
        lines.append("-" * 86)
        flag = "  [sensitive]" if d.get("sensitive") else ""
        lines.append(f"  {d['name']}  ({d['derivation']}, {d['kind']}){flag}")
        if d.get("bin_edges"):
            lines.append(f"    edges: {d['bin_edges']}")
        lines.append(
            f"    {'group':<22}{'n':>8}{'bad':>7}{'rate':>8}"
            f"{'ROC-AUC':>10}{'SE':>8}{'KS':>8}{'gap':>9}"
        )
        for g in d.get("groups", []):
            if not g.get("evaluated"):
                lines.append(
                    f"    {str(g['group'])[:22]:<22}{g['n']:>8}{g['n_positive']:>7}"
                    f"{_fmt(g.get('default_rate')):>8}"
                    f"{'  suppressed: ' + str(g.get('suppressed_reason')):>35}"
                )
                continue
            lines.append(
                f"    {str(g['group'])[:22]:<22}{g['n']:>8}{g['n_positive']:>7}"
                f"{_fmt(g.get('default_rate')):>8}{_fmt(g.get('roc_auc')):>10}"
                f"{_fmt(g.get('auc_se')):>8}{_fmt(g.get('ks_statistic')):>8}"
                f"{_fmt(g.get('gap'), signed=True):>9}"
            )
        if d.get("auc_spread") is not None:
            verdict = ("within noise" if d.get("spread_within_noise")
                       else "beyond noise")
            ratio = d.get("spread_vs_noise")
            lines.append(
                f"    spread {d['auc_spread']:.4f} "
                f"({d['worst_group']} -> {d['best_group']}), "
                f"spread/noise {'—' if ratio is None else format(ratio, '.2f')}"
                f" — {verdict}"
            )
    lines.append("=" * 86)
    return "\n".join(lines)


def _fmt(v: Any, signed: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
