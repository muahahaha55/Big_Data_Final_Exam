"""Checkpointing primitives for multi-hour, resumable studies.

A study that takes hours on a 3.8 GB VM needs three things to be trustworthy:

1. **A fingerprint** over everything that makes its parts comparable, so
   resuming after someone changed a hyperparameter is refused instead of
   silently mixing numbers that cannot be compared.
2. **Atomic writes**, so a crash mid-write cannot leave a truncated report
   that destroys the work already done.
3. **A report that doubles as the checkpoint**, so "which parts are done" is
   read from the same file a human reads.

No Spark import — all of this is plain Python and unit-tested as such.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


def canonical_fingerprint(material: dict) -> str:
    """Stable `sha256:...` hash of a dict, independent of key order.

    Callers pass only the fields that decide comparability; prose fields are
    left out on purpose so rewording a note never invalidates finished work.
    """
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_fingerprint(report: dict | None, fingerprint: str) -> None:
    """Refuse to extend a report produced under different settings.

    A report with no fingerprint at all (older format) is allowed through
    rather than blocking work.
    """
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


def completed_records(
    report: dict | None,
    records_key: str = "steps",
    id_key: str = "step",
    coerce: Callable[[Any], Any] = int,
) -> dict[Any, dict]:
    """Map id → record for the parts of a study that already finished.

    Tolerates malformed entries. `coerce` normalises the id: `int` for studies
    keyed by step/fold number (JSON can hand numbers back as strings), `str`
    for studies keyed by name.
    """
    if not report:
        return {}
    out: dict[Any, dict] = {}
    for rec in report.get(records_key, []) or []:
        if not isinstance(rec, dict) or rec.get(id_key) is None:
            continue
        try:
            out[coerce(rec[id_key])] = rec
        except (TypeError, ValueError):
            continue
    return out
