"""Loading out-of-fold predictions and joining them to customer attributes.

Shared by the two studies that read T1.4's output without Spark: segment
validation (P0.4) and the cost-based cut-off (P1.1). Both need the same three
steps and, more importantly, the same guards — a stale Parquet file, a
fanned-out pool id, or predictions from a different ETL run would all still
produce a plausible-looking report.

Paths are arguments rather than module constants so each pipeline keeps
ownership of its own layout, and so tests can point the whole thing at a temp
directory. pandas is imported inside the functions: everything else in
`credit_risk.reporting` is import-light and stays that way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def load_oof(cv_report: dict, oof_dir: Path, cv_report_path: Path | None = None):
    """Concatenate the per-fold out-of-fold predictions.

    Cross-checks the row count against the `n_valid` the folds recorded, so a
    Parquet file left over from an earlier run cannot silently become the
    population every downstream number is computed on.
    """
    import pandas as pd

    # Imported here, not at module scope: credit_risk.cv imports
    # credit_risk.reporting.checkpoint, so a top-level import would make
    # `credit_risk.reporting` and `credit_risk.cv` circular.
    from credit_risk.cv import completed_folds

    folds = sorted(completed_folds(cv_report))
    if not folds:
        raise RuntimeError(
            f"{cv_report_path or 'cv_report.json'} records no finished folds — "
            "run `make cv-fold N=1` … `N=5` first"
        )

    frames = []
    for fold in folds:
        path = oof_dir / f"fold_{fold}.parquet"
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
            f"Parquet files are from a different run — delete {oof_dir} and "
            "re-run the folds."
        )
    return oof, folds


def load_attributes(
    columns: Sequence[str],
    id_col: str,
    model_input: Path,
    hint: str = "every configured column",
):
    """Read only the needed columns from the CV pool, via column pruning.

    The pool is `train.parquet ∪ val.parquet`, exactly what T1.4 drew folds
    from, so every out-of-fold id must be found here. Reading a handful of
    columns instead of 112 keeps this inside a few hundred MB on the 3.8 GB VM.
    """
    import pandas as pd

    wanted = [id_col] + [c for c in columns if c != id_col]
    frames = []
    for name in ("train", "val"):
        path = model_input / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} — the attributes come from the same CV pool "
                "T1.4 used. Run `make etl` first."
            )
        try:
            frames.append(pd.read_parquet(path, columns=wanted))
        except Exception as exc:                      # narrow message, wide catch
            raise RuntimeError(
                f"could not read {wanted} from {path}: {exc}. Check that "
                f"{hint} exists in the master dataset."
            ) from exc

    attrs = pd.concat(frames, ignore_index=True)
    dup = int(attrs[id_col].duplicated().sum())
    if dup:
        raise RuntimeError(
            f"{dup} duplicate {id_col} values in the CV pool — an aggregate "
            "join fanned out and the join below would double-count rows."
        )
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
            "from different ETL runs; the numbers would be computed on a "
            "subset without saying so."
        )
    return merged.drop(columns=["_merge"])


def cv_fold_mean(cv_report: dict, metric: str = "roc_auc") -> Any:
    """The per-fold mean of a metric, for cross-checking a pooled recompute."""
    return ((cv_report.get("summary") or {}).get(metric) or {}).get("mean")
