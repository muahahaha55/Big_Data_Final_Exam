"""Baseline comparison study (P0.3) — pure Python, no Spark.

The question this answers
-------------------------
The repo has measured the marginal value of the five child tables (+0.0106
ROC-AUC, T1.2) but never the marginal value of the 109 features beyond
`EXT_SOURCE`. The Kruskal-Wallis ranking in README §2 says those two columns
dominate everything else by 3-60x:

    EXT_SOURCE_3  H = 4,672.97      DAYS_EMPLOYED     H =   144.83
    EXT_SOURCE_2  H = 4,649.61      AMT_INCOME_TOTAL  H =    72.98
    DAYS_BIRTH    H = 1,371.48      AMT_CREDIT        H =    70.51

So: how much does the whole 7-table, 58.4M-row, Spark/HDFS/YARN apparatus buy
over a model you could fit in a spreadsheet? Whatever the answer, knowing it
is a stronger position than being asked and not knowing.

Why four variants rather than one number
----------------------------------------
A single "logistic on EXT_SOURCE" number changes two things at once — the
feature set *and* the model class — so it cannot tell you which mattered. The
variants decompose it:

    rule_ext_mean      no training at all; rank by mean EXT_SOURCE
    logit_ext2_ext3    only the two columns that survive missing_threshold
    logit_ext_all3     all three, median-imputed
    gbt_ext_all3       same three features, same model as the real pipeline

    logit_ext_all3 vs gbt_ext_all3   -> effect of the model class
    gbt_ext_all3   vs ablation full  -> effect of the other 109 features
    rule_ext_mean  vs everything     -> what zero training already buys

Comparability
-------------
Every variant trains and scores on the **persisted ablation split**
(`data/02_intermediate/ablation_split/`), so the numbers sit directly beside
the six steps already published in README §12.1 — same 45,820 test customers,
same seed. Metrics come from the same `evaluate_predictions` the ablation used.

One deliberate difference, recorded in the report: `EXT_SOURCE_1` is roughly
56% null in this dataset and is therefore dropped by the pipeline's global
`features.missing_threshold: 0.5`. The `*_all3` variants keep it anyway, with
median imputation, which is why `logit_ext2_ext3` exists as the "what the real
pipeline would actually see" version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from credit_risk.reporting.checkpoint import canonical_fingerprint, completed_records

SCHEMA_VERSION = 1
REPORT_FILENAME = "baseline_report.json"

METRICS = ("roc_auc", "pr_auc", "ks_statistic")

MODEL_RULE = "rule"
MODEL_LOGISTIC = "logistic"
MODEL_GBT = "gbt"
VALID_MODELS = (MODEL_RULE, MODEL_LOGISTIC, MODEL_GBT)

EXT_SOURCES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")

NOTE = (
    "Baselines train and score on the persisted ablation split, so they are "
    "directly comparable to README 12.1 (same 45,820 test customers, same "
    "seed, same evaluate_predictions). rule_ext_mean does no training and its "
    "score is a ranking only, not a calibrated PD — read its ROC-AUC/KS, not "
    "its calibration. EXT_SOURCE_1 is ~56% null and the real pipeline drops it "
    "via features.missing_threshold; the *_all3 variants keep it with median "
    "imputation, which logit_ext2_ext3 exists to contrast against."
)


# ──────────────────────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VariantSpec:
    """One baseline configuration."""

    name: str
    model_type: str
    features: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if self.model_type not in VALID_MODELS:
            raise ValueError(
                f"unknown model_type {self.model_type!r} for variant "
                f"{self.name!r} (expected one of {VALID_MODELS})"
            )
        if not self.features:
            raise ValueError(f"variant {self.name!r} has no features")


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "rule_ext_mean", MODEL_RULE, EXT_SOURCES,
        "No training: rank by the mean of the available EXT_SOURCE columns",
    ),
    VariantSpec(
        "logit_ext2_ext3", MODEL_LOGISTIC, ("EXT_SOURCE_2", "EXT_SOURCE_3"),
        "Logistic on only the EXT_SOURCE columns that pass missing_threshold",
    ),
    VariantSpec(
        "logit_ext_all3", MODEL_LOGISTIC, EXT_SOURCES,
        "Logistic on all three EXT_SOURCE columns, median-imputed",
    ),
    VariantSpec(
        "gbt_ext_all3", MODEL_GBT, EXT_SOURCES,
        "GBT on the same three features — isolates model class from feature set",
    ),
)


def build_variants(cfg: dict) -> list[VariantSpec]:
    """Variants from `baseline.variants` in config, else the defaults."""
    declared = (cfg.get("baseline") or {}).get("variants")
    if not declared:
        return list(DEFAULT_VARIANTS)

    out = []
    for v in declared:
        out.append(VariantSpec(
            name=str(v["name"]),
            model_type=str(v["model"]),
            features=tuple(v["features"]),
            description=str(v.get("description", "")),
        ))
    names = [v.name for v in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate variant names: {names}")
    return out


# ──────────────────────────────────────────────────────────────
# Setup + fingerprint
# ──────────────────────────────────────────────────────────────

def build_setup(cfg: dict, variants: Sequence[VariantSpec], split: dict | None = None) -> dict:
    """Everything that defines comparability with the ablation numbers."""
    sp = cfg.get("spark", {})
    return {
        "variants": [
            {"name": v.name, "model": v.model_type, "features": list(v.features),
             "description": v.description}
            for v in variants
        ],
        "split": "data/02_intermediate/ablation_split (reused from T1.2)",
        "random_seed": cfg.get("random_seed", 42),
        "tune": False,
        "class_weighting": False,
        "hyperparams": {
            "gbt": dict(cfg["models"].get("gbt", {})),
            "logistic": dict(cfg["models"].get("logistic", {})),
        },
        "spark": {
            "master": sp.get("master", "local[1]"),
            "driver_memory": sp.get("driver_memory", "2g"),
            "shuffle_partitions": sp.get("shuffle_partitions", 50),
        },
        "split_observed": split or {},
        "comparable_to": (
            "README 12.1 / ablation_report.json — same persisted split, same "
            "seed, same metric code"
        ),
    }


def config_fingerprint(setup: dict) -> str:
    """Hash of the settings that make variants comparable to each other."""
    return canonical_fingerprint({
        "variants": setup.get("variants"),
        "split": setup.get("split"),
        "random_seed": setup.get("random_seed"),
        "tune": setup.get("tune"),
        "class_weighting": setup.get("class_weighting"),
        "hyperparams": setup.get("hyperparams"),
        "spark_master": (setup.get("spark") or {}).get("master"),
        "spark_shuffle_partitions": (setup.get("spark") or {}).get("shuffle_partitions"),
    })


# ──────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────

def completed_variants(report: dict | None) -> dict[str, dict]:
    """Variant name → record, for variants already finished."""
    return completed_records(report, records_key="variants", id_key="name", coerce=str)


def pending_variants(
    variants: Sequence[VariantSpec],
    report: dict | None,
    only_variant: str | None = None,
    force: bool = False,
) -> list[VariantSpec]:
    """Which variants still need running, honouring --variant and --force."""
    done = set() if force else set(completed_variants(report))
    out = [v for v in variants if v.name not in done]
    if only_variant is not None:
        out = [v for v in out if v.name == only_variant]
    return out


# ──────────────────────────────────────────────────────────────
# References from the ablation report
# ──────────────────────────────────────────────────────────────

def load_references(ablation_report: dict | None) -> dict:
    """Pull the two comparison points out of ablation_report.json.

    Read from the file rather than hardcoded so every delta traces back to a
    report (PLAN rule 7). Returns {} when the ablation report is absent, and
    the baseline report then says so instead of inventing a comparison.
    """
    if not ablation_report:
        return {}
    steps = [s for s in (ablation_report.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return {}

    by_label = {s.get("label"): s for s in steps}
    last = max(steps, key=lambda s: s.get("step", 0))

    out = {}
    app_only = by_label.get("application_only")
    if app_only:
        out["ablation_application_only"] = _reference(app_only)
    if last is not app_only:
        out["ablation_full"] = _reference(last)
    return out


def _reference(step: dict) -> dict:
    return {
        "label": step.get("label"),
        "n_source_columns": step.get("n_source_columns"),
        "test_rows": step.get("test_rows"),
        **{m: step.get(m) for m in METRICS},
    }


def check_split_match(baseline_test_rows: int, references: dict) -> None:
    """Refuse to compare against an ablation run on a different split.

    If the persisted split was regenerated after the ablation, the baseline
    would be scored on different customers and every delta would be wrong.
    """
    for key, ref in references.items():
        ref_rows = ref.get("test_rows")
        if ref_rows is not None and int(ref_rows) != int(baseline_test_rows):
            raise RuntimeError(
                "Refusing to compare: the baseline scored "
                f"{baseline_test_rows} test rows but {key} was scored on "
                f"{ref_rows}. The persisted ablation split has changed, so the "
                "deltas would be meaningless. Re-run the ablation, or delete "
                "data/02_intermediate/ablation_split and re-run both."
            )


# ──────────────────────────────────────────────────────────────
# Comparison + report
# ──────────────────────────────────────────────────────────────

def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def attach_deltas(records: Sequence[dict], references: dict) -> list[dict]:
    """Add, per variant, the gap to each reference. Inputs are not mutated.

    Sign convention: `reference − variant`, so a **positive** number is how
    much the full pipeline is *ahead* of the baseline.
    """
    out = []
    for rec in sorted((dict(r) for r in records), key=lambda r: r.get("name", "")):
        gaps = {}
        for key, ref in references.items():
            gaps[key] = {
                f"delta_{m if m != 'ks_statistic' else 'ks'}": _delta(ref.get(m), rec.get(m))
                for m in METRICS
            }
        rec["gap_to_reference"] = gaps
        out.append(rec)
    return out


def summarize(records: Sequence[dict], references: dict) -> dict:
    """Headline: the best baseline and how far the full pipeline is ahead."""
    scored = [r for r in records if r.get("roc_auc") is not None]
    if not scored:
        return {}

    best = max(scored, key=lambda r: r["roc_auc"])
    full = references.get("ablation_full") or {}
    out = {
        "best_baseline": {
            "name": best.get("name"),
            "model": best.get("model"),
            "n_features": best.get("n_features"),
            **{m: best.get(m) for m in METRICS},
        },
        "ranking_by_roc_auc": [
            {"name": r["name"], "roc_auc": r["roc_auc"]}
            for r in sorted(scored, key=lambda r: r["roc_auc"], reverse=True)
        ],
    }
    if full.get("roc_auc") is not None:
        out["full_pipeline_advantage_over_best_baseline"] = {
            "reference": full.get("label"),
            "reference_n_source_columns": full.get("n_source_columns"),
            "baseline_n_features": best.get("n_features"),
            **{
                f"delta_{m if m != 'ks_statistic' else 'ks'}":
                    _delta(full.get(m), best.get(m))
                for m in METRICS
            },
        }
    return out


def build_report(
    setup: dict,
    records: Sequence[dict],
    references: dict,
    generated_at: str | None = None,
) -> dict:
    """Assemble baseline_report.json from whatever variants have finished."""
    variants = attach_deltas(records, references)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(setup),
        "setup": setup,
        "note": NOTE,
        "references": references,
        "n_variants_total": len(setup.get("variants") or []),
        "n_variants_completed": len(variants),
        "variants": variants,
        "summary": summarize(variants, references),
    }


# ──────────────────────────────────────────────────────────────
# Console rendering
# ──────────────────────────────────────────────────────────────

def format_summary(report: dict) -> str:
    refs = report.get("references") or {}
    lines = [
        "=" * 82,
        f"  BASELINE COMPARISON  "
        f"({report.get('n_variants_completed')}/{report.get('n_variants_total')} variants)",
        "=" * 82,
        f"  {'variant':<20}{'model':<10}{'feat':>5}"
        f"{'ROC-AUC':>10}{'PR-AUC':>9}{'KS':>9}{'sec':>8}",
    ]
    for v in report.get("variants", []):
        lines.append(
            f"  {v.get('name', '?'):<20}{v.get('model', '?'):<10}"
            f"{v.get('n_features', '-'):>5}{_fmt(v.get('roc_auc')):>10}"
            f"{_fmt(v.get('pr_auc')):>9}{_fmt(v.get('ks_statistic')):>9}"
            f"{v.get('elapsed_sec', '-'):>8}"
        )
    if refs:
        lines.append("-" * 82)
        lines.append("  reference (ablation, same split):")
        for key, r in refs.items():
            lines.append(
                f"  {r.get('label', key):<20}{'gbt':<10}"
                f"{r.get('n_source_columns', '-'):>5}{_fmt(r.get('roc_auc')):>10}"
                f"{_fmt(r.get('pr_auc')):>9}{_fmt(r.get('ks_statistic')):>9}"
            )
    adv = (report.get("summary") or {}).get("full_pipeline_advantage_over_best_baseline")
    if adv:
        lines += [
            "-" * 82,
            f"  Full pipeline ({adv.get('reference_n_source_columns')} cols) minus best "
            f"baseline ({adv.get('baseline_n_features')} cols):",
            f"    ROC-AUC {_fmt(adv.get('delta_roc_auc'), signed=True)}   "
            f"PR-AUC {_fmt(adv.get('delta_pr_auc'), signed=True)}   "
            f"KS {_fmt(adv.get('delta_ks'), signed=True)}",
        ]
    lines.append("=" * 82)
    return "\n".join(lines)


def _fmt(v: Any, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}" if signed else f"{v:.4f}"
