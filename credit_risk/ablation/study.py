"""Study design and bookkeeping for the table ablation — pure Python.

No Spark, no MLflow, no I/O beyond reading and writing the report JSON. That
keeps the parts that decide whether a multi-hour run is *trustworthy* — step
enumeration, the comparability fingerprint, resume logic, and the deltas that
get quoted in the README — testable in milliseconds.

Two study designs:

  cumulative
      Add one table at a time, in a fixed order, recording the MARGINAL gain:
          application_train → +bureau → +previous_application
          → +installments → +credit_card → +pos_cash
      Answers "what did joining this table buy us?". Marginal contributions
      are ORDER-DEPENDENT: a table that adds nothing here may still carry
      signal an earlier table already supplied.

  loo
      Leave-one-table-out, plus application-only and full baselines. Answers
      the different question "what is unique to this table, given the others?"
      The two designs disagree whenever tables overlap — and here they do
      (installments, pos_cash and credit_card all describe repayment history
      reached through SK_ID_PREV), so neither can be derived from the other.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1

APPLICATION_ONLY = "application_only"
FULL = "full"

# The 5 child tables that feed the model, in the order PLAN T1.2 adds them.
# pipelines/ablation_pipeline.py maps these to aggregator functions and
# asserts its mapping matches this order.
DEFAULT_TABLE_ORDER = [
    "bureau", "previous_application", "installments", "credit_card", "pos_cash",
]

# Each mode gets its own file so neither study overwrites the other.
REPORT_FILENAMES = {
    "cumulative": "ablation_report.json",
    "loo": "ablation_loo_report.json",
}

_METRICS = ("roc_auc", "pr_auc", "ks_statistic")

NOTE = (
    "bureau_balance has no aggregator, so it is not part of the join or this "
    "study (see T1.3). Marginal contributions depend on the order tables are "
    "added. Metrics here are NOT comparable to README §7 (different split)."
)


# ──────────────────────────────────────────────────────────────
# Step enumeration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepSpec:
    """One training configuration in the study."""

    step: int
    label: str
    included_tables: tuple[str, ...]
    added_table: str | None = None     # cumulative mode
    removed_table: str | None = None   # loo mode


def build_steps(mode: str, tables: Sequence[str]) -> list[StepSpec]:
    """Enumerate the study's steps.

    cumulative: len(tables) + 1 steps, each a prefix of `tables`.
    loo:        application_only, full, then one step per removed table.
    """
    tables = list(tables)

    if mode == "cumulative":
        steps = [StepSpec(1, APPLICATION_ONLY, ())]
        for i, t in enumerate(tables, start=1):
            steps.append(StepSpec(
                step=i + 1,
                label=f"+{t}",
                included_tables=tuple(tables[:i]),
                added_table=t,
            ))
        return steps

    if mode == "loo":
        steps = [
            StepSpec(1, APPLICATION_ONLY, ()),
            StepSpec(2, FULL, tuple(tables)),
        ]
        for i, t in enumerate(tables, start=3):
            steps.append(StepSpec(
                step=i,
                label=f"drop_{t}",
                included_tables=tuple(x for x in tables if x != t),
                removed_table=t,
            ))
        return steps

    raise ValueError(f"Unknown mode: {mode!r} (expected 'cumulative' or 'loo')")


# ──────────────────────────────────────────────────────────────
# Setup block + comparability fingerprint
# ──────────────────────────────────────────────────────────────

def build_setup(mode: str, tables: Sequence[str], cfg: dict) -> dict:
    """Assemble the `setup` block: everything that defines comparability."""
    split = cfg["data"]["split"]
    sp = cfg.get("spark", {})
    return {
        "model_type": "gbt",
        "hyperparams": dict(cfg["models"].get("gbt", {})),
        "random_seed": cfg.get("random_seed", 42),
        "tune": False,
        "split": {
            "method": "randomSplit on SK_ID_CURR, persisted and shared by all steps",
            "train_ratio": round(split["train_ratio"] + split["val_ratio"], 4),
            "test_ratio": split["test_ratio"],
            "seed": cfg.get("random_seed", 42),
        },
        "spark": {
            "master": sp.get("master", "local[1]"),
            "driver_memory": sp.get("driver_memory", "2g"),
            "shuffle_partitions": sp.get("shuffle_partitions", 50),
        },
        "table_order": list(tables),
        "mode": mode,
        "not_comparable_to": (
            "README §7 ROC-AUC 0.7663 — that model used etl_pipeline's 3-way "
            "70/15/15 split taken on the master, not this 2-way split on IDs"
        ),
    }


def config_fingerprint(setup: dict) -> str:
    """Stable hash of the comparability-critical settings.

    Spark master and shuffle_partitions are included on purpose: they change
    task partitioning, and GBT's per-partition subsampling means results can
    shift even with a fixed seed. Prose fields are excluded so rewording a
    note never invalidates finished steps.
    """
    material = {
        "model_type": setup.get("model_type"),
        "hyperparams": setup.get("hyperparams"),
        "random_seed": setup.get("random_seed"),
        "tune": setup.get("tune"),
        "split": setup.get("split"),
        "table_order": setup.get("table_order"),
        "mode": setup.get("mode"),
        "spark_master": (setup.get("spark") or {}).get("master"),
        "spark_shuffle_partitions": (setup.get("spark") or {}).get("shuffle_partitions"),
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_fingerprint(report: dict | None, fingerprint: str) -> None:
    """Refuse to extend a report produced under different settings."""
    if not report:
        return
    existing = report.get("config_fingerprint")
    if existing and existing != fingerprint:
        raise RuntimeError(
            "Refusing to resume: the existing report was produced with "
            "different settings, so its steps are not comparable to new ones.\n"
            f"  on disk: {existing}\n"
            f"  now:     {fingerprint}\n"
            "Re-run from scratch with --force, or restore the previous "
            "hyperparams / seed / split / table order / spark settings."
        )


# ──────────────────────────────────────────────────────────────
# Resume bookkeeping
# ──────────────────────────────────────────────────────────────

def completed_steps(report: dict | None) -> dict[int, dict]:
    """Step number → record, for steps already finished."""
    if not report:
        return {}
    return {
        int(s["step"]): s
        for s in report.get("steps", [])
        if isinstance(s, dict) and s.get("step") is not None
    }


def pending_steps(
    steps: Sequence[StepSpec],
    report: dict | None,
    only_step: int | None = None,
    force: bool = False,
) -> list[StepSpec]:
    """Which steps still need running, honouring --step and --force."""
    done = set() if force else set(completed_steps(report))
    out = [s for s in steps if s.step not in done]
    if only_step is not None:
        out = [s for s in out if s.step == only_step]
    return out


# ──────────────────────────────────────────────────────────────
# Deltas and summary
# ──────────────────────────────────────────────────────────────

def _delta(a: float | None, b: float | None) -> float | None:
    """a - b, rounded; None-safe."""
    if a is None or b is None:
        return None
    return round(a - b, 4)


def attach_deltas(steps: Sequence[dict], mode: str) -> list[dict]:
    """Add per-step deltas, returning new dicts (inputs are not mutated).

    cumulative: versus the immediately preceding step (the marginal gain).
                None when that neighbour has not been run yet.
    loo:        versus the `full` baseline, for the drop_* steps only.
    """
    ordered = sorted((dict(s) for s in steps), key=lambda s: s["step"])

    if mode == "cumulative":
        by_step = {s["step"]: s for s in ordered}
        for s in ordered:
            prev = by_step.get(s["step"] - 1)
            for m in _METRICS:
                key = f"delta_{'ks' if m == 'ks_statistic' else m}_vs_prev"
                s[key] = _delta(s.get(m), prev.get(m)) if prev else None
        return ordered

    full = next((s for s in ordered if s.get("label") == FULL), None)
    for s in ordered:
        for m in _METRICS:
            key = f"delta_{'ks' if m == 'ks_statistic' else m}_vs_full"
            if full is not None and s.get("removed_table"):
                s[key] = _delta(full.get(m), s.get(m))
            else:
                s[key] = None
    return ordered


def summarize(steps: Sequence[dict], mode: str) -> dict:
    """Totals and ranking, tolerant of a partially finished study."""
    ordered = sorted(steps, key=lambda s: s["step"])
    if not ordered:
        return {}

    if mode == "cumulative":
        first = next((s for s in ordered if s.get("label") == APPLICATION_ONLY), None)
        last = ordered[-1]
        lift = {}
        if first is not None and first is not last:
            lift = {
                "from": first["label"],
                "to": last["label"],
                "roc_auc": _delta(last.get("roc_auc"), first.get("roc_auc")),
                "pr_auc": _delta(last.get("pr_auc"), first.get("pr_auc")),
                "ks": _delta(last.get("ks_statistic"), first.get("ks_statistic")),
            }
        ranked = sorted(
            (s for s in ordered
             if s.get("added_table") and s.get("delta_roc_auc_vs_prev") is not None),
            key=lambda s: s["delta_roc_auc_vs_prev"],
            reverse=True,
        )
        return {
            "total_lift_vs_application_only": lift,
            "marginal_ranking_by_delta_roc_auc": [
                {"table": s["added_table"], "delta_roc_auc": s["delta_roc_auc_vs_prev"]}
                for s in ranked
            ],
        }

    ranked = sorted(
        (s for s in ordered
         if s.get("removed_table") and s.get("delta_roc_auc_vs_full") is not None),
        key=lambda s: s["delta_roc_auc_vs_full"],
        reverse=True,
    )
    return {
        "importance_ranking_by_roc_auc_drop": [
            {"table": s["removed_table"], "roc_auc_drop": s["delta_roc_auc_vs_full"]}
            for s in ranked
        ],
    }


def build_report(mode: str, setup: dict, step_records: Sequence[dict]) -> dict:
    """Assemble the full report document from whatever steps exist."""
    steps = attach_deltas(step_records, mode)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(setup),
        "setup": setup,
        "note": NOTE,
        "table_order": setup.get("table_order", []),
        "n_steps_total": len(build_steps(mode, setup.get("table_order", []))),
        "n_steps_completed": len(steps),
        "steps": steps,
        **summarize(steps, mode),
    }


# ──────────────────────────────────────────────────────────────
# Report persistence
# ──────────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via a temp file + os.replace, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_report(path: Path) -> dict | None:
    """Read an existing report; None if absent or unparseable.

    A truncated report must never crash a resume, so a decode error reads as
    "no checkpoint" rather than propagating.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def format_summary(report: dict) -> str:
    """Render the console table for a finished (or partial) study."""
    mode = report.get("mode")
    dkey = "delta_roc_auc_vs_prev" if mode == "cumulative" else "delta_roc_auc_vs_full"
    head = "Δ ROC-AUC vs prev" if mode == "cumulative" else "ROC-AUC drop"

    lines = [
        "=" * 78,
        f"  ABLATION REPORT — mode={mode}  "
        f"({report.get('n_steps_completed')}/{report.get('n_steps_total')} steps)",
        "=" * 78,
        f"  {'step':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'KS':>8}"
        f"{head:>19}{'cols':>7}{'sec':>8}",
    ]
    for s in report.get("steps", []):
        lines.append(
            f"  {s.get('label', '?'):<26}{_fmt(s.get('roc_auc')):>9}"
            f"{_fmt(s.get('pr_auc')):>9}{_fmt(s.get('ks_statistic')):>8}"
            f"{_fmt(s.get(dkey), signed=True):>19}"
            f"{s.get('n_source_columns', '-'):>7}{s.get('elapsed_sec', '-'):>8}"
        )
    lift = report.get("total_lift_vs_application_only")
    if lift:
        lines += [
            "-" * 78,
            f"  Total lift {lift.get('from')} → {lift.get('to')}: "
            f"ROC-AUC {_fmt(lift.get('roc_auc'), signed=True)}  "
            f"PR-AUC {_fmt(lift.get('pr_auc'), signed=True)}  "
            f"KS {_fmt(lift.get('ks'), signed=True)}",
        ]
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt(v: Any, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}" if signed else f"{v:.4f}"
