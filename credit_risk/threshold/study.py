"""Cost-based cut-off study (P1.1) — pure Python, no Spark, no pandas.

The repo decides APPROVE / REVIEW / REJECT from three round numbers
(`risk.segmentation.thresholds`: 0.10 / 0.20 / 0.35) that nothing derives. This
module turns the cut-off into a decision: for every candidate threshold it
computes what approving that population is worth under an explicit cost model,
and reports where the optimum sits, how wide its plateau is, and how the
current fixed policy compares.

The cost model
--------------
Approve a loan and it either performs or defaults:

    value(row) = (1 - y)·m·EAD  -  y·LGD·EAD          if approved
                 0                                     if rejected

`m` (margin rate) is the expected margin on a performing loan as a fraction of
EAD, over the life of the loan, net of funding and operating cost. **It is a
pure assumption** — nothing in Home Credit measures it — so the study sweeps
several values and reports the sensitivity rather than picking one. LGD and EAD
carry the assumptions already documented in README §8.2 (LGD = 0.45 constant,
EAD = `AMT_CREDIT` with no CCF).

Why a single threshold is the right shape here
----------------------------------------------
Per obligor, approving is worth more than rejecting when
(1−p)·m·EAD > p·LGD·EAD, i.e. when **p < m/(m+LGD)** — EAD cancels. So under a
constant margin and LGD the optimal per-obligor rule *is* one shared
threshold, and `breakeven_pd` is its analytic value. That gives a free
cross-check: if the model's PDs were perfectly calibrated, the empirical
optimum of the sweep would sit at the breakeven. The distance between them
measures **miscalibration** (and any EAD-PD correlation, which is reported
alongside so the gap can be attributed rather than guessed at).

Reading the output
------------------
**The optimum of a swept curve is noisy.** Every threshold is scored on the
same rows, so thresholds are compared with a **paired** standard error — only
the rows between two thresholds differ — and the report gives the `plateau`:
the whole range of thresholds that is statistically indistinguishable from the
best. A single argmax quoted alone would be false precision.

**Choosing a threshold on the rows you then report is optimistic.** That bias
is measured, not argued away: `rotating_holdout` picks the threshold on k−1
folds and scores it on the held-out fold, which the `fold` column in the
out-of-fold predictions makes free.

**No money claims.** Amounts are in the currency units of the Kaggle 2018
dataset and correspond to no real portfolio, so the headline numbers are the
threshold, the approval rate, the bad rate among approved, and value
**relative** to approving everyone — never an absolute saving.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Sequence

from credit_risk.reporting.checkpoint import canonical_fingerprint

SCHEMA_VERSION = 1
REPORT_FILENAME = "threshold_report.json"

DEFAULT_MARGIN_RATES = (0.02, 0.05, 0.08, 0.12)
DEFAULT_GRID_POINTS = 200
DEFAULT_CURVE_POINTS = 40
DEFAULT_PLATEAU_N_SE = 1.0

# The repo's live decision rule, from streaming/spark_streaming.py:156-158:
# tier Low/Medium -> APPROVE, High -> REVIEW, Very High -> REJECT. A two-action
# cost model cannot price a manual review, so the current policy is bracketed
# by treating REVIEW as both.
REVIEW_AS_REJECT = "review_as_reject"
REVIEW_AS_APPROVE = "review_as_approve"

NOTE = (
    "Thresholds are swept over out-of-fold predictions (T1.4), so every row "
    "was scored by a model that did not train on it. The margin rate is an "
    "ASSUMPTION with no source in the data — several values are reported and "
    "the conclusion must hold across them, or it is a conclusion about the "
    "assumption. LGD = 0.45 constant and EAD = AMT_CREDIT are the assumptions "
    "of README 8.2. Amounts are in Kaggle 2018 currency units and match no "
    "real portfolio: use the threshold, the approval rate, the bad rate among "
    "approved and the ratio to approve-all, never an absolute saving. "
    "Thresholds are compared with a paired standard error and the plateau of "
    "statistically indistinguishable thresholds is reported, because the "
    "argmax of a noisy curve is not a decision. The optimism of choosing a "
    "threshold on the rows it is then scored on is measured by rotating "
    "holdout across folds."
)


# ──────────────────────────────────────────────────────────────
# Cost parameters
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostParams:
    """One scenario of the cost model. `margin_rate` is an assumption."""

    margin_rate: float
    lgd: float

    def __post_init__(self) -> None:
        if not (0.0 < self.margin_rate < 10.0):
            raise ValueError(
                f"margin_rate must be a positive fraction of EAD, got {self.margin_rate}"
            )
        if not (0.0 < self.lgd <= 1.0):
            raise ValueError(f"lgd must be in (0, 1], got {self.lgd}")

    @property
    def breakeven_pd(self) -> float:
        """m/(m+LGD) — the PD where approving stops paying, per obligor."""
        return self.margin_rate / (self.margin_rate + self.lgd)


def breakeven_pd(margin_rate: float, lgd: float) -> float:
    """Analytic break-even PD; EAD cancels out, so it is portfolio-free."""
    return CostParams(margin_rate=margin_rate, lgd=lgd).breakeven_pd


def implied_margin_rate(threshold: float, lgd: float) -> float | None:
    """Inverse of `breakeven_pd`: the margin that makes a threshold optimal.

    This is what turns the repo's three unsourced round numbers into something
    arguable. A cut-off at PD = t is the optimal one *only if* the margin is
    t·LGD/(1−t). With LGD = 0.45, approving below 0.20 implies a margin of
    11.25% of EAD; if the real margin were 5%, the optimal cut-off would be
    0.10. The number stops being arbitrary and becomes an assumption someone
    in the business can confirm or reject.
    """
    t = float(threshold)
    if not (0.0 <= t < 1.0):
        return None
    return t * float(lgd) / (1.0 - t)


# ──────────────────────────────────────────────────────────────
# Prefix sums: every threshold is a prefix of the sorted scores
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Prepared:
    """Rows sorted by PD with cumulative sums, so a sweep is O(grid + n log n).

    Without this, scoring 200 thresholds over 261k rows is 52M row visits in
    pure Python. With it, each threshold is two array lookups.
    """

    params: CostParams
    scores: list[float]
    cum_value: list[float]      # Σ value over the first k rows
    cum_value_sq: list[float]   # Σ value² — for the standard errors
    cum_bad: list[int]
    cum_ead: list[float]

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def n_bad(self) -> int:
        return self.cum_bad[-1]

    @property
    def total_value_if_all_approved(self) -> float:
        return self.cum_value[-1]

    def approved_count(self, threshold: float) -> int:
        """Rows with pd_score < threshold — strict, matching assign_risk_tier."""
        return bisect.bisect_left(self.scores, float(threshold))


def row_value(target: int, ead: float, params: CostParams) -> float:
    """What approving this one row is worth under the cost model."""
    if int(target):
        return -params.lgd * float(ead)
    return params.margin_rate * float(ead)


def prepare(rows: Sequence[tuple[float, int, float]], params: CostParams) -> Prepared:
    """Sort by PD and accumulate. `rows` is (pd_score, target, ead)."""
    data = sorted(
        ((float(pd_), int(y), float(ead)) for pd_, y, ead in rows),
        key=lambda r: r[0],
    )
    scores: list[float] = []
    cum_value = [0.0]
    cum_value_sq = [0.0]
    cum_bad = [0]
    cum_ead = [0.0]
    for pd_, y, ead in data:
        v = row_value(y, ead, params)
        scores.append(pd_)
        cum_value.append(cum_value[-1] + v)
        cum_value_sq.append(cum_value_sq[-1] + v * v)
        cum_bad.append(cum_bad[-1] + y)
        cum_ead.append(cum_ead[-1] + ead)
    return Prepared(params, scores, cum_value, cum_value_sq, cum_bad, cum_ead)


# ──────────────────────────────────────────────────────────────
# One policy
# ──────────────────────────────────────────────────────────────

def _total_se(sum_v: float, sum_v2: float, n: int) -> float | None:
    """SE of a sum of n row values, treating rejected rows as exact zeros."""
    if n < 2:
        return None
    mean = sum_v / n
    var = (sum_v2 - n * mean * mean) / (n - 1)
    if var <= 0:
        return 0.0
    return math.sqrt(n * var)


def policy_value(prep: Prepared, threshold: float) -> dict:
    """Score the policy "approve iff pd_score < threshold"."""
    k = prep.approved_count(threshold)
    n = prep.n
    sum_v = prep.cum_value[k]
    bad = prep.cum_bad[k]
    ead = prep.cum_ead[k]
    total_all = prep.total_value_if_all_approved

    return {
        "threshold": round(float(threshold), 6),
        "n_approved": k,
        "approval_rate": round(k / n, 6) if n else None,
        "n_bad_approved": bad,
        "bad_rate_approved": round(bad / k, 6) if k else None,
        "bad_captured_share": round(bad / prep.n_bad, 6) if prep.n_bad else None,
        "exposure_approved": round(ead, 2),
        "expected_margin": round(
            prep.params.margin_rate * (ead - _bad_exposure(prep, k)), 2),
        "expected_loss": round(prep.params.lgd * _bad_exposure(prep, k), 2),
        "net_value": round(sum_v, 2),
        "net_value_se": _round(_total_se(sum_v, prep.cum_value_sq[k], n)),
        "net_value_per_row": round(sum_v / n, 6) if n else None,
        "net_minus_approve_all": round(sum_v - total_all, 2),
        # A ratio against a negative baseline reads backwards (a *better*
        # policy would show a negative multiple), so it is only reported when
        # approving everyone is itself worth something. The difference above
        # is always well defined.
        "net_vs_approve_all_ratio": (
            round(sum_v / total_all, 6) if total_all > 0 else None),
        "approve_all_net_value": round(total_all, 2),
    }


def _bad_exposure(prep: Prepared, k: int) -> float:
    """Exposure of the defaulting rows inside the first k — from the loss term.

    value = m·EAD for good rows and −LGD·EAD for bad, so the bad exposure is
    recoverable from the accumulated value without a second pass.
    """
    m, lgd = prep.params.margin_rate, prep.params.lgd
    ead = prep.cum_ead[k]
    sum_v = prep.cum_value[k]
    # sum_v = m·(ead - bad_ead) - lgd·bad_ead  ->  bad_ead = (m·ead - sum_v)/(m + lgd)
    return max(0.0, (m * ead - sum_v) / (m + lgd))


def paired_se(prep: Prepared, threshold_a: float, threshold_b: float) -> float | None:
    """SE of the *difference* between two policies on the same rows.

    Only the rows between the two thresholds change decision, so the paired SE
    is far tighter than comparing two independent totals — and it is the right
    quantity for asking whether two thresholds really differ.
    """
    ka = prep.approved_count(threshold_a)
    kb = prep.approved_count(threshold_b)
    lo, hi = (ka, kb) if ka <= kb else (kb, ka)
    n = prep.n
    if n < 2 or lo == hi:
        return 0.0 if n >= 2 else None
    sum_d = prep.cum_value[hi] - prep.cum_value[lo]
    sum_d2 = prep.cum_value_sq[hi] - prep.cum_value_sq[lo]
    mean = sum_d / n
    var = (sum_d2 - n * mean * mean) / (n - 1)
    if var <= 0:
        return 0.0
    return math.sqrt(n * var)


# ──────────────────────────────────────────────────────────────
# Grid and sweep
# ──────────────────────────────────────────────────────────────

def quantile_grid(
    scores: Sequence[float],
    n_points: int = DEFAULT_GRID_POINTS,
    extra: Sequence[float] = (),
) -> list[float]:
    """Candidate thresholds at score quantiles, plus any required extras.

    A grid of round numbers would spend most of its points where no customer
    sits. Quantiles put the candidates where the decisions actually change.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    clean = sorted(float(s) for s in scores)
    if not clean:
        return sorted({round(float(e), 6) for e in extra})
    grid = {round(clean[(i * len(clean)) // n_points], 6) for i in range(n_points)}
    grid.update(round(float(e), 6) for e in extra)
    grid.add(round(clean[-1], 6))
    return sorted(grid)


def sweep(prep: Prepared, grid: Sequence[float]) -> list[dict]:
    """Score every candidate threshold."""
    return [policy_value(prep, t) for t in grid]


def optimum(rows: Sequence[dict]) -> dict | None:
    """The best-scoring policy; ties go to the *lower* threshold.

    A lower threshold approves fewer customers for the same expected value,
    which is the conservative side to err on in credit.
    """
    scored = [r for r in rows if r.get("net_value") is not None]
    if not scored:
        return None
    best = max(r["net_value"] for r in scored)
    return min((r for r in scored if r["net_value"] == best),
               key=lambda r: r["threshold"])


def plateau(
    prep: Prepared,
    rows: Sequence[dict],
    n_se: float = DEFAULT_PLATEAU_N_SE,
) -> dict:
    """Thresholds statistically indistinguishable from the optimum.

    A threshold stays in the plateau when the paired difference against the
    optimum is within `n_se` paired standard errors of zero. The plateau's
    width, not the argmax, is what the README should quote.
    """
    best = optimum(rows)
    if best is None:
        return {"optimum_threshold": None, "lower": None, "upper": None,
                "n_within": 0, "n_se": n_se}

    inside = []
    for r in rows:
        if r.get("net_value") is None:
            continue
        gap = best["net_value"] - r["net_value"]
        se = paired_se(prep, best["threshold"], r["threshold"]) or 0.0
        if gap <= n_se * se:
            inside.append(r["threshold"])

    return {
        "optimum_threshold": best["threshold"],
        "lower": min(inside) if inside else best["threshold"],
        "upper": max(inside) if inside else best["threshold"],
        "n_within": len(inside),
        "n_se": n_se,
        "note": (
            "any threshold in [lower, upper] is within "
            f"{n_se} paired SE of the best — the argmax alone is false precision"
        ),
    }


# ──────────────────────────────────────────────────────────────
# Reference policies
# ──────────────────────────────────────────────────────────────

def approve_all(prep: Prepared) -> dict:
    out = policy_value(prep, math.inf)
    out["threshold"] = None
    out["policy"] = "approve_all"
    return out


def reject_all(prep: Prepared) -> dict:
    out = policy_value(prep, 0.0)
    out["threshold"] = None
    out["policy"] = "reject_all"
    return out


def random_at_rate(prep: Prepared, approval_rate: float) -> dict:
    """Expected value of approving a random `approval_rate` share.

    Exact in expectation: a random policy approves each row with probability a
    regardless of its label, so E[net] = a · Σ value over all rows. This is the
    reference that isolates what the **ranking** buys, as opposed to what
    lending less buys.
    """
    a = float(approval_rate)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"approval_rate must be in [0, 1], got {a}")
    net = a * prep.total_value_if_all_approved
    total_all = prep.total_value_if_all_approved
    return {
        "policy": "random_ranking_same_approval_rate",
        "threshold": None,
        "approval_rate": round(a, 6),
        "n_approved": round(a * prep.n),
        "bad_rate_approved": round(prep.n_bad / prep.n, 6) if prep.n else None,
        "net_value": round(net, 2),
        "net_value_per_row": round(net / prep.n, 6) if prep.n else None,
        "net_minus_approve_all": round(net - total_all, 2),
        "net_vs_approve_all_ratio": (
            round(net / total_all, 6) if total_all > 0 else None),
        "note": "expected value, computed analytically — not a simulation",
    }


def current_policy(prep: Prepared, approve_below: float, review_below: float) -> list[dict]:
    """The repo's live rule, bracketed because REVIEW has no priced cost."""
    out = []
    for name, t in ((REVIEW_AS_REJECT, approve_below), (REVIEW_AS_APPROVE, review_below)):
        row = policy_value(prep, t)
        row["policy"] = f"current_fixed_{name}"
        row["source"] = (
            "risk.segmentation.thresholds via the decision rule in "
            "streaming/spark_streaming.py:156-158"
        )
        row["implied_margin_rate"] = _round(
            implied_margin_rate(t, prep.params.lgd), 6)
        row["implied_margin_note"] = (
            "the margin that would make THIS threshold the optimal one, "
            "t*LGD/(1-t) — compare it with the margin the business actually "
            "earns"
        )
        out.append(row)
    return out


# ──────────────────────────────────────────────────────────────
# Rotating holdout — how optimistic is the chosen threshold?
# ──────────────────────────────────────────────────────────────

def rotating_holdout(
    rows_by_fold: dict[Any, Sequence[tuple[float, int, float]]],
    params: CostParams,
    n_points: int = DEFAULT_GRID_POINTS,
) -> dict:
    """Choose the threshold on k−1 folds, score it on the held-out fold.

    Picking a threshold on the same rows that then report its value is
    optimistically biased. This measures the bias instead of asserting it is
    small: `optimism_per_row` is the in-fold optimum minus the value the
    threshold chosen elsewhere actually delivers on that fold.
    """
    folds = sorted(rows_by_fold)
    if len(folds) < 2:
        return {"folds": [], "note": "needs at least 2 folds"}

    records = []
    for held in folds:
        train_rows = [r for f in folds if f != held for r in rows_by_fold[f]]
        test_rows = list(rows_by_fold[held])
        if not train_rows or not test_rows:
            continue

        train_prep = prepare(train_rows, params)
        chosen = optimum(sweep(train_prep, quantile_grid(train_prep.scores, n_points)))
        test_prep = prepare(test_rows, params)
        applied = policy_value(test_prep, chosen["threshold"])
        in_fold_best = optimum(
            sweep(test_prep, quantile_grid(test_prep.scores, n_points)))

        records.append({
            "held_out_fold": held,
            "threshold_chosen_elsewhere": chosen["threshold"],
            "threshold_best_in_fold": in_fold_best["threshold"],
            "net_value_per_row_applied": applied["net_value_per_row"],
            "net_value_per_row_in_fold_best": in_fold_best["net_value_per_row"],
            "optimism_per_row": _round(
                (in_fold_best["net_value_per_row"] or 0.0)
                - (applied["net_value_per_row"] or 0.0), 6),
            "approval_rate_applied": applied["approval_rate"],
        })

    chosen_thresholds = [r["threshold_chosen_elsewhere"] for r in records]
    optimisms = [r["optimism_per_row"] for r in records if r["optimism_per_row"] is not None]
    return {
        "folds": records,
        "threshold_mean": _round(_mean(chosen_thresholds), 6),
        "threshold_std": _round(_std(chosen_thresholds), 6),
        "threshold_min": min(chosen_thresholds) if chosen_thresholds else None,
        "threshold_max": max(chosen_thresholds) if chosen_thresholds else None,
        "optimism_per_row_mean": _round(_mean(optimisms), 6),
        "note": (
            "threshold chosen on the other folds, scored on the held-out fold. "
            "threshold_std is the stability of the choice; "
            "optimism_per_row_mean is how much of the in-sample optimum does "
            "not survive being chosen elsewhere"
        ),
    }


# ──────────────────────────────────────────────────────────────
# Scenario + report
# ──────────────────────────────────────────────────────────────

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Correlation, so an optimum away from break-even can be attributed."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def population_summary(
    rows: Sequence[tuple[float, int, float]],
    params: CostParams,
) -> dict:
    """What is being decided on, before any threshold is applied."""
    prep = prepare(rows, params)
    n = prep.n
    return {
        "n": n,
        "n_default": prep.n_bad,
        "base_default_rate": round(prep.n_bad / n, 6) if n else None,
        "total_exposure": round(prep.cum_ead[-1], 2),
        "mean_exposure": round(prep.cum_ead[-1] / n, 2) if n else None,
        "mean_predicted_pd": round(sum(prep.scores) / n, 6) if n else None,
        "ead_pd_correlation": _round(
            pearson([r[0] for r in rows], [r[2] for r in rows]), 4),
    }


def build_scenario(
    rows: Sequence[tuple[float, int, float]],
    params: CostParams,
    approve_below: float,
    review_below: float,
    rows_by_fold: dict[Any, Sequence[tuple[float, int, float]]] | None = None,
    n_points: int = DEFAULT_GRID_POINTS,
    curve_points: int = DEFAULT_CURVE_POINTS,
    plateau_n_se: float = DEFAULT_PLATEAU_N_SE,
) -> dict:
    """One margin-rate scenario, end to end."""
    prep = prepare(rows, params)
    grid = quantile_grid(prep.scores, n_points,
                         extra=(approve_below, review_below, params.breakeven_pd))
    rows_swept = sweep(prep, grid)
    best = optimum(rows_swept)
    band = plateau(prep, rows_swept, plateau_n_se)

    at_breakeven = policy_value(prep, params.breakeven_pd)
    references = {
        "approve_all": approve_all(prep),
        "reject_all": reject_all(prep),
        "random_ranking_at_optimum_rate": random_at_rate(
            prep, best["approval_rate"] if best else 0.0),
        "at_analytic_breakeven": at_breakeven,
        "current_fixed": current_policy(prep, approve_below, review_below),
    }

    gain_vs_random = None
    if best and references["random_ranking_at_optimum_rate"]["net_value"] is not None:
        gain_vs_random = _round(
            best["net_value"] - references["random_ranking_at_optimum_rate"]["net_value"], 2)

    return {
        "margin_rate": params.margin_rate,
        "lgd": params.lgd,
        "breakeven_pd": round(params.breakeven_pd, 6),
        "optimum": best,
        "plateau": band,
        "optimum_minus_breakeven": (
            _round(best["threshold"] - params.breakeven_pd, 6) if best else None),
        "breakeven_inside_plateau": (
            band["lower"] <= params.breakeven_pd <= band["upper"]
            if band["lower"] is not None else None),
        "gain_vs_random_at_same_rate": gain_vs_random,
        "references": references,
        "rotating_holdout": (
            rotating_holdout(rows_by_fold, params, n_points) if rows_by_fold else None),
        "curve": _thin(rows_swept, curve_points),
        "n_grid_points": len(grid),
    }


def _thin(rows: Sequence[dict], keep: int) -> list[dict]:
    """Keep ~`keep` evenly spaced points so the JSON stays readable."""
    if keep < 2 or len(rows) <= keep:
        chosen = list(rows)
    else:
        step = len(rows) / keep
        chosen = [rows[min(len(rows) - 1, int(i * step))] for i in range(keep)]
    fields = ("threshold", "approval_rate", "bad_rate_approved",
              "net_value", "net_value_per_row", "net_minus_approve_all",
              "net_vs_approve_all_ratio")
    return [{k: r[k] for k in fields if k in r} for r in chosen]


def build_setup(cfg: dict, margin_rates: Sequence[float], source: dict | None = None) -> dict:
    """Everything that defines what this study decided and on what."""
    el = ((cfg.get("risk") or {}).get("expected_loss") or {})
    seg = (((cfg.get("risk") or {}).get("segmentation") or {}).get("thresholds") or {})
    th = cfg.get("threshold") or {}
    return {
        "lgd": float(el.get("lgd_fixed", 0.45)),
        "lgd_source": "ASSUMPTION — risk.expected_loss.lgd_fixed, see README 8.2",
        "ead_column": el.get("ead_column", "AMT_CREDIT"),
        "ead_source": "ASSUMPTION — proxy for EAD, no CCF for undrawn limits",
        "margin_rates": [float(m) for m in margin_rates],
        "margin_source": (
            "ASSUMPTION — nothing in Home Credit measures margin; several "
            "values are swept and the conclusion must hold across them"
        ),
        "predictions": "data/07_model_output/cv_oof (out-of-fold, T1.4)",
        "current_policy": {
            "approve_below": float(seg.get("medium", 0.20)),
            "review_below": float(seg.get("high", 0.35)),
            "source": (
                "risk.segmentation.thresholds + the decision rule in "
                "streaming/spark_streaming.py:156-158 (Low/Medium -> APPROVE, "
                "High -> REVIEW, Very High -> REJECT)"
            ),
            "implied_margin_if_optimal": {
                "approve_below": _round(implied_margin_rate(
                    float(seg.get("medium", 0.20)),
                    float(el.get("lgd_fixed", 0.45))), 6),
                "review_below": _round(implied_margin_rate(
                    float(seg.get("high", 0.35)),
                    float(el.get("lgd_fixed", 0.45))), 6),
                "note": (
                    "t*LGD/(1-t): the margin rate under which each round "
                    "number would be the optimal cut-off"
                ),
            },
        },
        "grid_points": int(th.get("grid_points", DEFAULT_GRID_POINTS)),
        "curve_points": int(th.get("curve_points", DEFAULT_CURVE_POINTS)),
        "plateau_n_se": float(th.get("plateau_n_se", DEFAULT_PLATEAU_N_SE)),
        "decision_rule": "approve iff pd_score < threshold (strict, as assign_risk_tier)",
        "source": source or {},
    }


def margin_rates(cfg: dict) -> list[float]:
    """Read the assumed margin rates from config; refuse a single value."""
    th = cfg.get("threshold") or {}
    rates = th.get("margin_rates") or list(DEFAULT_MARGIN_RATES)
    rates = [float(r) for r in rates]
    if len(rates) < 2:
        raise ValueError(
            "threshold.margin_rates needs at least 2 values: the margin is an "
            "assumption, so the study reports sensitivity rather than a single "
            "number dressed up as a result"
        )
    return sorted(set(rates))


def config_fingerprint(setup: dict) -> str:
    """Hash of what decides whether two threshold reports are comparable."""
    return canonical_fingerprint({
        "lgd": setup.get("lgd"),
        "ead_column": setup.get("ead_column"),
        "margin_rates": setup.get("margin_rates"),
        "predictions": setup.get("predictions"),
        "current_policy": {
            k: v for k, v in (setup.get("current_policy") or {}).items()
            if k != "source"
        },
        "grid_points": setup.get("grid_points"),
        "plateau_n_se": setup.get("plateau_n_se"),
        "decision_rule": setup.get("decision_rule"),
    })


def build_report(
    setup: dict,
    population: dict,
    scenarios: Sequence[dict],
    cv_config_fingerprint: str = "",
    generated_at: str | None = None,
) -> dict:
    """Assemble threshold_report.json.

    `verdict` stays empty: which threshold to adopt is a business decision, and
    the wording is written from the numbers once they exist.
    """
    stable = None
    thresholds = [s["optimum"]["threshold"] for s in scenarios
                  if s.get("optimum") and s["optimum"].get("threshold") is not None]
    if thresholds:
        stable = {
            "optimum_threshold_min": min(thresholds),
            "optimum_threshold_max": max(thresholds),
            "note": (
                "the optimum moves with the assumed margin — this range is the "
                "sensitivity to an assumption, not a measurement error"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _now(),
        "config_fingerprint": config_fingerprint(setup),
        "cv_config_fingerprint": cv_config_fingerprint,
        "note": NOTE,
        "setup": setup,
        "population": population,
        "n_scenarios": len(scenarios),
        "sensitivity": stable,
        "scenarios": list(scenarios),
        "verdict": "",
    }


# ──────────────────────────────────────────────────────────────
# Console rendering
# ──────────────────────────────────────────────────────────────

def format_summary(report: dict) -> str:
    pop = report.get("population") or {}
    lines = [
        "=" * 92,
        f"  COST-BASED CUT-OFF — {pop.get('n', '?')} out-of-fold rows, "
        f"{pop.get('n_default', '?')} defaults "
        f"(base rate {_fmt(pop.get('base_default_rate'))})",
        "=" * 92,
        f"  LGD {_fmt((report.get('setup') or {}).get('lgd'))} (assumed) · "
        f"EAD = {(report.get('setup') or {}).get('ead_column')} (proxy) · "
        f"corr(PD, EAD) {_fmt(pop.get('ead_pd_correlation'))}",
    ]
    cur = ((report.get("setup") or {}).get("current_policy") or {})
    lines.append(
        f"  chính sách hiện tại: approve < {_fmt(cur.get('approve_below'))}, "
        f"review < {_fmt(cur.get('review_below'))}"
    )

    for sc in report.get("scenarios", []):
        best = sc.get("optimum") or {}
        band = sc.get("plateau") or {}
        lines += [
            "-" * 92,
            f"  margin {_fmt(sc.get('margin_rate'))} (ASSUMED)  "
            f"break-even PD {_fmt(sc.get('breakeven_pd'))}",
            f"    optimum t = {_fmt(best.get('threshold'))}  "
            f"plateau [{_fmt(band.get('lower'))}, {_fmt(band.get('upper'))}]  "
            f"({band.get('n_within', 0)} điểm trong {_fmt(band.get('n_se'), 1)} SE)",
            f"    duyệt {_pct(best.get('approval_rate'))}  "
            f"nợ xấu trong nhóm duyệt {_pct(best.get('bad_rate_approved'))}  "
            f"(bắt {_pct(best.get('bad_captured_share'))} ca vỡ nợ)",
            f"    net/row {_fmt(best.get('net_value_per_row'))}  "
            + (f"= {_fmt(best.get('net_vs_approve_all_ratio'))}x giá trị của "
               "duyệt-tất-cả"
               if best.get("net_vs_approve_all_ratio") is not None else
               f"(duyệt-tất-cả âm: {_fmt(best.get('approve_all_net_value'), 2)} "
               f"— chênh {_fmt(best.get('net_minus_approve_all'), 2)})"),
        ]
        refs = sc.get("references") or {}
        rnd = refs.get("random_ranking_at_optimum_rate") or {}
        if rnd.get("net_value_per_row") is not None:
            lines.append(
                f"    cùng tỷ lệ duyệt nhưng xếp hạng ngẫu nhiên: "
                f"net/row {_fmt(rnd.get('net_value_per_row'))}"
            )
        for row in refs.get("current_fixed") or []:
            lines.append(
                f"    {row.get('policy'):<34} t={_fmt(row.get('threshold'))}  "
                f"duyệt {_pct(row.get('approval_rate'))}  "
                f"net/row {_fmt(row.get('net_value_per_row'))}"
            )
        rh = sc.get("rotating_holdout") or {}
        if rh.get("threshold_mean") is not None:
            lines.append(
                f"    rotating holdout: t = {_fmt(rh.get('threshold_mean'))} "
                f"± {_fmt(rh.get('threshold_std'))}, "
                f"optimism/row {_fmt(rh.get('optimism_per_row_mean'))}"
            )
    lines.append("=" * 92)
    return "\n".join(lines)


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _round(v: float | None, digits: int = 2) -> float | None:
    return None if v is None else round(v, digits)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:.{digits}f}"


def _pct(v: Any) -> str:
    return "—" if v is None else f"{100.0 * v:.2f}%"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
