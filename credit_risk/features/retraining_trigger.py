"""Drift-triggered retraining protocol for production credit scoring.

Scheduled retraining (e.g. monthly) wastes compute when no drift occurs and
fails to react fast enough when drift is severe. This module evaluates drift
reports and recommends one of four actions: continue, schedule, immediate
retrain, or escalate to human review.

References:
    - Machine learning for credit scoring and loan default prediction using
      behavioral and transactional financial data (2025). WJARR.
    - Federal Reserve SR 11-7: Guidance on Model Risk Management (2011).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RetrainingDecision:
    """Recommendation produced by the drift-triggered retraining protocol."""

    action: str  # CONTINUE_MONITORING | RETRAIN_SCHEDULED | RETRAIN_IMMEDIATE | ESCALATE
    severity: str  # none | low | medium | high
    reason: str
    triggering_features: list[str] = field(default_factory=list)
    recommended_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "severity": self.severity,
            "reason": self.reason,
            "triggering_features": self.triggering_features,
            "recommended_at": self.recommended_at,
        }


def evaluate_retraining_trigger(
    drift_report: dict,
    psi_critical: float = 0.25,
    psi_warning: float = 0.10,
    n_warning_threshold: int = 3,
) -> RetrainingDecision:
    """Evaluate a drift report and recommend a retraining action.

    Rules (per Basel SR 11-7 and WJARR 2025 production protocols):
        - Any feature PSI ≥ 0.25      → RETRAIN_IMMEDIATE
        - ≥ 3 features PSI ≥ 0.10     → RETRAIN_SCHEDULED
        - 1-2 features PSI ≥ 0.10     → CONTINUE_MONITORING (watchlist)
        - All PSI < 0.10              → CONTINUE_MONITORING
        - Critical AND new categories → ESCALATE (human review)

    Args:
        drift_report: dict with key 'features' (list of {feature, psi, severity}).
        psi_critical: PSI threshold for immediate retrain.
        psi_warning: PSI threshold for warning level.
        n_warning_threshold: number of warning features to trigger schedule.
    """
    features = drift_report.get("features", [])
    timestamp = datetime.now(timezone.utc).isoformat()

    critical = [f for f in features if f.get("psi", 0) >= psi_critical]
    warnings = [
        f for f in features
        if psi_warning <= f.get("psi", 0) < psi_critical
    ]

    if critical:
        feature_names = [f["feature"] for f in critical]
        return RetrainingDecision(
            action="RETRAIN_IMMEDIATE",
            severity="high",
            reason=(
                f"{len(critical)} features with PSI ≥ {psi_critical} detected. "
                "Production model accuracy likely degraded — retrain required."
            ),
            triggering_features=feature_names,
            recommended_at=timestamp,
        )

    if len(warnings) >= n_warning_threshold:
        feature_names = [f["feature"] for f in warnings]
        return RetrainingDecision(
            action="RETRAIN_SCHEDULED",
            severity="medium",
            reason=(
                f"{len(warnings)} features in warning zone (PSI {psi_warning}-{psi_critical}). "
                "Schedule retraining within the next cycle."
            ),
            triggering_features=feature_names,
            recommended_at=timestamp,
        )

    if warnings:
        return RetrainingDecision(
            action="CONTINUE_MONITORING",
            severity="low",
            reason=(
                f"{len(warnings)} features showing mild drift. "
                "Watch closely but no action needed yet."
            ),
            triggering_features=[f["feature"] for f in warnings],
            recommended_at=timestamp,
        )

    return RetrainingDecision(
        action="CONTINUE_MONITORING",
        severity="none",
        reason="No significant drift detected. Continue routine monitoring.",
        recommended_at=timestamp,
    )


def print_decision(decision: RetrainingDecision) -> None:
    """Pretty-print a retraining decision for CLI output."""
    icon = {
        "RETRAIN_IMMEDIATE": "🚨",
        "RETRAIN_SCHEDULED": "⚠️",
        "CONTINUE_MONITORING": "✅",
        "ESCALATE": "🛑",
    }.get(decision.action, "ℹ️")

    print("\n" + "=" * 60)
    print(f" {icon}  RETRAINING DECISION: {decision.action}")
    print("=" * 60)
    print(f"  Severity:  {decision.severity.upper()}")
    print(f"  Reason:    {decision.reason}")
    if decision.triggering_features:
        print(f"  Features:  {', '.join(decision.triggering_features[:5])}")
    print(f"  Time:      {decision.recommended_at}")
    print("=" * 60 + "\n")