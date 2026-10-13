"""Unit tests for the P1.1 cost-based cut-off logic — no Spark, no pandas.

Three groups of claims are checked, because each one could be wrong in a way
that would still look plausible in a report:

1. **The arithmetic.** The prefix-sum shortcut (and the algebraic trick that
   recovers defaulting exposure from accumulated value) is checked against a
   brute-force recomputation on random data. If the shortcut were wrong, every
   number in the study would be wrong together and consistently.
2. **The theory.** On synthetic data whose PDs are calibrated *by
   construction*, the swept optimum must land on the analytic break-even
   m/(m+LGD), and the optimum must rise with the assumed margin. That ties the
   empirical machinery to the closed form.
3. **The honesty guards.** Ties resolve to the lower (more conservative)
   threshold, the plateau widens with the tolerance, a single margin rate is
   refused, and the rotating holdout reports the selection optimism.

Imports `credit_risk.threshold` rather than `pipelines.threshold_pipeline`: the
pipeline needs pandas, the study module needs nothing.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from credit_risk.threshold import (
    DEFAULT_GRID_POINTS,
    NOTE,
    REPORT_FILENAME,
    REVIEW_AS_APPROVE,
    REVIEW_AS_REJECT,
    SCHEMA_VERSION,
    CostParams,
    approve_all,
    breakeven_pd,
    build_report,
    build_scenario,
    build_setup,
    config_fingerprint,
    current_policy,
    format_summary,
    implied_margin_rate,
    margin_rates,
    optimum,
    paired_se,
    pearson,
    plateau,
    policy_value,
    population_summary,
    prepare,
    quantile_grid,
    random_at_rate,
    reject_all,
    rotating_holdout,
    row_value,
    sweep,
)

LGD = 0.45


def _cfg() -> dict:
    return {
        "data": {"target_column": "TARGET", "id_column": "SK_ID_CURR"},
        "risk": {
            "expected_loss": {"lgd_fixed": 0.45, "ead_column": "AMT_CREDIT"},
            "segmentation": {"thresholds": {"low": 0.10, "medium": 0.20, "high": 0.35}},
        },
        "threshold": {
            "margin_rates": [0.02, 0.05, 0.08, 0.12],
            "grid_points": 60,
            "curve_points": 10,
            "plateau_n_se": 1.0,
        },
    }


def _calibrated(n: int = 20000, seed: int = 17, ead: float | None = 100.0):
    """PD is the true default probability, so calibration holds by construction.

    This is the only setting where the swept optimum has a known answer, which
    is what makes it a test of the machinery rather than of itself.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = rng.random() * 0.6
        y = 1 if rng.random() < p else 0
        amount = ead if ead is not None else rng.uniform(50.0, 500.0)
        rows.append((p, y, amount))
    return rows


def _ranked(n: int = 8000, seed: int = 23):
    """A model with real but imperfect ranking power, constant exposure."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        y = 1 if rng.random() < 0.08 else 0
        score = min(0.99, max(0.001, rng.betavariate(2, 20) + 0.15 * y))
        rows.append((score, y, 100.0))
    return rows


def _brute(rows, threshold, params: CostParams) -> dict:
    """Independent recomputation, straight from the cost model."""
    appr = [(y, ead) for pd_, y, ead in rows if pd_ < threshold]
    net = sum((-params.lgd * ead) if y else (params.margin_rate * ead)
              for y, ead in appr)
    bad = sum(y for y, _ in appr)
    return {
        "n_approved": len(appr),
        "n_bad_approved": bad,
        "net_value": net,
        "exposure_approved": sum(ead for _, ead in appr),
        "bad_exposure": sum(ead for y, ead in appr if y),
    }


# ──────────────────────────────────────────────────────────────
# The cost model and its closed form
# ──────────────────────────────────────────────────────────────

def test_breakeven_is_margin_over_margin_plus_lgd():
    assert breakeven_pd(0.05, 0.45) == pytest.approx(0.10)
    assert breakeven_pd(0.45, 0.45) == pytest.approx(0.5)


def test_breakeven_does_not_depend_on_exposure():
    """EAD cancels, which is why one shared threshold can be optimal."""
    p = CostParams(0.08, LGD)
    small = row_value(0, 10.0, p) / 10.0, row_value(1, 10.0, p) / 10.0
    big = row_value(0, 1e6, p) / 1e6, row_value(1, 1e6, p) / 1e6
    assert small == pytest.approx(big)


def test_implied_margin_round_trips():
    for t in (0.05, 0.10, 0.20, 0.35, 0.5):
        m = implied_margin_rate(t, LGD)
        assert breakeven_pd(m, LGD) == pytest.approx(t)


def test_implied_margin_of_the_repo_thresholds():
    """The three round numbers become assumptions someone can confirm."""
    assert implied_margin_rate(0.10, LGD) == pytest.approx(0.05)
    assert implied_margin_rate(0.20, LGD) == pytest.approx(0.1125)
    assert implied_margin_rate(0.35, LGD) == pytest.approx(0.35 * LGD / 0.65, abs=1e-9)


def test_implied_margin_outside_range():
    assert implied_margin_rate(1.0, LGD) is None
    assert implied_margin_rate(-0.1, LGD) is None


def test_row_value_signs():
    p = CostParams(0.05, LGD)
    assert row_value(0, 1000.0, p) == pytest.approx(50.0)
    assert row_value(1, 1000.0, p) == pytest.approx(-450.0)


def test_cost_params_validation():
    with pytest.raises(ValueError, match="margin_rate"):
        CostParams(0.0, LGD)
    with pytest.raises(ValueError, match="margin_rate"):
        CostParams(-0.05, LGD)
    with pytest.raises(ValueError, match="lgd"):
        CostParams(0.05, 0.0)
    with pytest.raises(ValueError, match="lgd"):
        CostParams(0.05, 1.5)


# ──────────────────────────────────────────────────────────────
# The prefix-sum shortcut must equal the naive computation
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0])
def test_policy_value_matches_brute_force(threshold):
    params = CostParams(0.05, LGD)
    rows = _calibrated(3000, seed=29, ead=None)
    prep = prepare(rows, params)
    got = policy_value(prep, threshold)
    want = _brute(rows, threshold, params)

    assert got["n_approved"] == want["n_approved"]
    assert got["n_bad_approved"] == want["n_bad_approved"]
    # Money fields are rounded to 2 decimals in the report by design.
    assert got["net_value"] == pytest.approx(want["net_value"], abs=0.01)
    assert got["exposure_approved"] == pytest.approx(want["exposure_approved"], abs=0.01)


def test_expected_loss_and_margin_decompose_the_net():
    """The algebraic recovery of defaulting exposure has to be exact."""
    params = CostParams(0.08, LGD)
    rows = _calibrated(2000, seed=31, ead=None)
    prep = prepare(rows, params)
    for t in (0.05, 0.2, 0.45):
        got = policy_value(prep, t)
        want = _brute(rows, t, params)
        assert got["expected_loss"] == pytest.approx(params.lgd * want["bad_exposure"], abs=1e-2)
        assert got["expected_margin"] == pytest.approx(
            params.margin_rate * (want["exposure_approved"] - want["bad_exposure"]), abs=1e-2)
        assert got["expected_margin"] - got["expected_loss"] == pytest.approx(
            got["net_value"], abs=1e-2)


def test_threshold_is_strict_like_assign_risk_tier():
    """`pd < t`, matching assign_risk_tier — a row exactly at t is rejected."""
    params = CostParams(0.05, LGD)
    prep = prepare([(0.10, 0, 100.0), (0.0999, 0, 100.0)], params)
    assert policy_value(prep, 0.10)["n_approved"] == 1


def test_approve_all_and_reject_all():
    params = CostParams(0.05, LGD)
    rows = _ranked(1000, seed=37)
    prep = prepare(rows, params)
    everything = approve_all(prep)
    nothing = reject_all(prep)

    assert everything["n_approved"] == len(rows)
    assert everything["net_value"] == pytest.approx(
        _brute(rows, math.inf, params)["net_value"], abs=0.01)
    assert everything["net_vs_approve_all_ratio"] == pytest.approx(1.0)
    assert nothing["n_approved"] == 0
    assert nothing["net_value"] == 0.0


# ──────────────────────────────────────────────────────────────
# Standard errors
# ──────────────────────────────────────────────────────────────

def test_ratio_against_approve_all_is_withheld_when_that_baseline_loses_money():
    """A ratio against a negative baseline reads backwards — report None.

    At a low assumed margin, approving everyone destroys value, so "3x the
    value of approve-all" would be meaningless and "-0.1x" would look like the
    better policy is worse. The difference is reported instead.
    """
    params = CostParams(0.02, LGD)                 # break-even PD 0.0426
    rows = _calibrated(3000, seed=151)             # average PD ~0.30, so
    prep = prepare(rows, params)                   # approve-all loses money
    assert approve_all(prep)["net_value"] < 0

    best = optimum(sweep(prep, quantile_grid(prep.scores, 100)))
    assert best["net_value"] > 0
    assert best["net_vs_approve_all_ratio"] is None
    assert best["net_minus_approve_all"] > 0
    assert random_at_rate(prep, 0.5)["net_vs_approve_all_ratio"] is None


def test_ratio_is_reported_when_approve_all_is_profitable():
    params = CostParams(0.12, LGD)
    prep = prepare(_ranked(3000, seed=157), params)
    assert approve_all(prep)["net_value"] > 0
    assert approve_all(prep)["net_vs_approve_all_ratio"] == pytest.approx(1.0)
    assert policy_value(prep, 0.05)["net_vs_approve_all_ratio"] is not None


def test_paired_se_is_zero_for_identical_thresholds():
    prep = prepare(_ranked(500, seed=41), CostParams(0.05, LGD))
    assert paired_se(prep, 0.2, 0.2) == 0.0


def test_paired_se_is_tighter_than_the_total_se():
    """Pairing is the whole point: only the rows between the two differ.

    The gain depends on how many rows sit between the two thresholds, so the
    property asserted is the one that always holds — paired < total — plus a
    strong bound for a narrow gap, where almost nothing differs.
    """
    prep = prepare(_ranked(4000, seed=43), CostParams(0.05, LGD))
    total = policy_value(prep, 0.20)["net_value_se"]
    assert paired_se(prep, 0.20, 0.21) < total
    assert paired_se(prep, 0.20, 0.2005) < total / 3


def test_paired_se_grows_with_the_gap():
    prep = prepare(_ranked(4000, seed=47), CostParams(0.05, LGD))
    assert paired_se(prep, 0.05, 0.08) < paired_se(prep, 0.05, 0.40)


def test_total_se_is_none_for_a_single_row():
    prep = prepare([(0.1, 0, 100.0)], CostParams(0.05, LGD))
    assert policy_value(prep, 0.5)["net_value_se"] is None


# ──────────────────────────────────────────────────────────────
# Grid
# ──────────────────────────────────────────────────────────────

def test_grid_sits_where_the_data_is():
    scores = [r[0] for r in _ranked(2000, seed=53)]
    grid = quantile_grid(scores, 50)
    assert len(grid) <= 51
    assert grid == sorted(grid) == sorted(set(grid))
    # quantile_grid rounds candidates to 6 decimals, so allow that slack.
    assert min(grid) >= min(scores) - 1e-6
    assert max(grid) <= max(scores) + 1e-6


def test_grid_keeps_required_extras():
    grid = quantile_grid([0.01, 0.02, 0.03], 2, extra=(0.2, 0.35))
    assert 0.2 in grid and 0.35 in grid


def test_grid_rejects_a_degenerate_request():
    with pytest.raises(ValueError, match="at least 2"):
        quantile_grid([0.1, 0.2], 1)


def test_grid_on_empty_scores_returns_only_extras():
    assert quantile_grid([], 10, extra=(0.2,)) == [0.2]


# ──────────────────────────────────────────────────────────────
# The optimum matches the theory on calibrated data
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("margin", [0.02, 0.05, 0.12])
def test_optimum_lands_on_the_analytic_breakeven(margin):
    """The strongest test here: machinery vs closed form, calibrated data."""
    params = CostParams(margin, LGD)
    rows = _calibrated(20000, seed=59)
    prep = prepare(rows, params)
    best = optimum(sweep(prep, quantile_grid(prep.scores, 300)))
    assert abs(best["threshold"] - params.breakeven_pd) < 0.02, (
        f"optimum {best['threshold']:.4f} vs break-even {params.breakeven_pd:.4f}"
    )


def test_optimum_rises_with_the_assumed_margin():
    """More margin per performing loan buys tolerance for more risk."""
    rows = _calibrated(12000, seed=61)
    found = []
    for margin in (0.02, 0.05, 0.08, 0.12):
        prep = prepare(rows, CostParams(margin, LGD))
        found.append(optimum(sweep(prep, quantile_grid(prep.scores, 200)))["threshold"])
    assert found == sorted(found)
    assert found[0] < found[-1]


def test_optimum_breaks_ties_toward_the_lower_threshold():
    """Same value, fewer approvals — the conservative side in credit."""
    rows = [{"threshold": 0.2, "net_value": 10.0},
            {"threshold": 0.3, "net_value": 10.0},
            {"threshold": 0.4, "net_value": 5.0}]
    assert optimum(rows)["threshold"] == 0.2


def test_optimum_none_on_empty_sweep():
    assert optimum([]) is None


def test_bad_rate_among_approved_rises_with_the_threshold():
    prep = prepare(_ranked(6000, seed=67), CostParams(0.05, LGD))
    rates = [policy_value(prep, t)["bad_rate_approved"] for t in (0.05, 0.1, 0.2, 0.5)]
    assert all(a is not None for a in rates)
    assert rates == sorted(rates)


# ──────────────────────────────────────────────────────────────
# Plateau
# ──────────────────────────────────────────────────────────────

def test_plateau_brackets_the_optimum():
    params = CostParams(0.05, LGD)
    prep = prepare(_ranked(5000, seed=71), params)
    rows = sweep(prep, quantile_grid(prep.scores, 100))
    band = plateau(prep, rows, n_se=1.0)
    assert band["lower"] <= band["optimum_threshold"] <= band["upper"]
    assert band["n_within"] >= 1


def test_plateau_widens_with_tolerance():
    params = CostParams(0.05, LGD)
    prep = prepare(_ranked(5000, seed=73), params)
    rows = sweep(prep, quantile_grid(prep.scores, 100))
    tight = plateau(prep, rows, n_se=0.5)
    loose = plateau(prep, rows, n_se=3.0)
    assert loose["n_within"] >= tight["n_within"]
    assert loose["upper"] >= tight["upper"]
    assert loose["lower"] <= tight["lower"]


def test_plateau_on_empty_sweep():
    prep = prepare(_ranked(100, seed=79), CostParams(0.05, LGD))
    band = plateau(prep, [], n_se=1.0)
    assert band["optimum_threshold"] is None and band["n_within"] == 0


# ──────────────────────────────────────────────────────────────
# Reference policies
# ──────────────────────────────────────────────────────────────

def test_random_policy_is_exactly_a_share_of_approve_all():
    prep = prepare(_ranked(3000, seed=83), CostParams(0.05, LGD))
    total = approve_all(prep)["net_value"]
    half = random_at_rate(prep, 0.5)
    assert half["net_value"] == pytest.approx(0.5 * total, abs=0.01)
    assert random_at_rate(prep, 0.0)["net_value"] == 0.0
    assert random_at_rate(prep, 1.0)["net_value"] == pytest.approx(total, abs=0.01)


def test_random_policy_rejects_an_impossible_rate():
    prep = prepare(_ranked(100, seed=89), CostParams(0.05, LGD))
    with pytest.raises(ValueError, match="approval_rate"):
        random_at_rate(prep, 1.5)


def test_ranking_beats_random_at_the_same_approval_rate():
    """Otherwise the model adds nothing over lending less."""
    params = CostParams(0.05, LGD)
    prep = prepare(_ranked(8000, seed=97), params)
    best = optimum(sweep(prep, quantile_grid(prep.scores, 200)))
    rnd = random_at_rate(prep, best["approval_rate"])
    assert best["net_value"] > rnd["net_value"]


def test_current_policy_brackets_the_review_band():
    prep = prepare(_ranked(3000, seed=101), CostParams(0.05, LGD))
    rows = current_policy(prep, approve_below=0.20, review_below=0.35)
    assert [r["policy"] for r in rows] == [
        f"current_fixed_{REVIEW_AS_REJECT}", f"current_fixed_{REVIEW_AS_APPROVE}"]
    assert rows[0]["threshold"] == 0.2 and rows[1]["threshold"] == 0.35
    assert rows[0]["n_approved"] <= rows[1]["n_approved"]
    assert rows[0]["implied_margin_rate"] == pytest.approx(0.1125)
    assert rows[1]["implied_margin_rate"] == pytest.approx(0.35 * LGD / 0.65, abs=1e-6)
    assert "streaming/spark_streaming.py:156-158" in rows[0]["source"]


# ──────────────────────────────────────────────────────────────
# Rotating holdout — the selection bias is measured, not argued
# ──────────────────────────────────────────────────────────────

def test_rotating_holdout_reports_stability_and_optimism():
    params = CostParams(0.05, LGD)
    rows = _calibrated(10000, seed=103)
    by_fold = {f: rows[f::5] for f in range(1, 6)}
    out = rotating_holdout(by_fold, params, n_points=80)

    assert len(out["folds"]) == 5
    assert out["threshold_std"] is not None
    # Folds are slices of one homogeneous population, so the chosen threshold
    # must be stable; a large spread would mean the choice is noise-driven.
    assert out["threshold_std"] < 0.05
    assert abs(out["threshold_mean"] - params.breakeven_pd) < 0.05
    assert out["optimism_per_row_mean"] is not None
    for rec in out["folds"]:
        assert rec["net_value_per_row_applied"] is not None
        assert rec["threshold_best_in_fold"] is not None


def test_rotating_holdout_needs_two_folds():
    out = rotating_holdout({1: _ranked(100, seed=107)}, CostParams(0.05, LGD))
    assert out["folds"] == [] and "2 folds" in out["note"]


def test_rotating_holdout_skips_an_empty_fold():
    rows = _ranked(600, seed=109)
    out = rotating_holdout({1: rows[:300], 2: rows[300:], 3: []},
                           CostParams(0.05, LGD), n_points=40)
    assert [r["held_out_fold"] for r in out["folds"]] == [1, 2]


# ──────────────────────────────────────────────────────────────
# Population summary
# ──────────────────────────────────────────────────────────────

def test_population_summary_describes_what_is_decided_on():
    rows = _ranked(4000, seed=113)
    pop = population_summary(rows, CostParams(0.05, LGD))
    assert pop["n"] == 4000
    assert 0.0 < pop["base_default_rate"] < 0.2
    assert pop["total_exposure"] == pytest.approx(4000 * 100.0)
    assert pop["mean_exposure"] == pytest.approx(100.0)
    assert pop["ead_pd_correlation"] is None     # exposure is constant here


def test_pearson_matches_a_known_case():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert pearson([1], [1]) is None


def test_ead_pd_correlation_is_reported_when_exposure_varies():
    rows = [(0.01 * i, i % 2, 100.0 + 10.0 * i) for i in range(50)]
    pop = population_summary(rows, CostParams(0.05, LGD))
    assert pop["ead_pd_correlation"] == pytest.approx(1.0, abs=1e-4)


# ──────────────────────────────────────────────────────────────
# Config, setup, fingerprint
# ──────────────────────────────────────────────────────────────

def test_margin_rates_refuses_a_single_value():
    """A single assumed margin dressed up as a result is the failure mode."""
    with pytest.raises(ValueError, match="at least 2 values"):
        margin_rates({"threshold": {"margin_rates": [0.05]}})


def test_margin_rates_sorts_and_deduplicates():
    assert margin_rates({"threshold": {"margin_rates": [0.12, 0.02, 0.12]}}) == [0.02, 0.12]


def test_margin_rates_falls_back_to_defaults():
    assert len(margin_rates({})) >= 2


def test_setup_labels_every_assumption():
    setup = build_setup(_cfg(), [0.02, 0.05])
    assert setup["lgd"] == 0.45
    assert "ASSUMPTION" in setup["lgd_source"]
    assert "ASSUMPTION" in setup["ead_source"]
    assert "ASSUMPTION" in setup["margin_source"]
    assert setup["ead_column"] == "AMT_CREDIT"
    assert "out-of-fold" in setup["predictions"]


def test_setup_derives_the_current_policy_from_segmentation_thresholds():
    """One source of truth: the policy is not duplicated into threshold:."""
    cur = build_setup(_cfg(), [0.05])["current_policy"]
    assert cur["approve_below"] == 0.20 and cur["review_below"] == 0.35
    assert cur["implied_margin_if_optimal"]["approve_below"] == pytest.approx(0.1125)
    assert "spark_streaming.py:156-158" in cur["source"]


def test_fingerprint_changes_with_assumptions_but_not_with_measured_rows():
    base = build_setup(_cfg(), [0.02, 0.05], source={"oof_rows": 261652})
    same = build_setup(_cfg(), [0.02, 0.05], source={"oof_rows": 1})
    assert config_fingerprint(base) == config_fingerprint(same)

    cfg = _cfg()
    cfg["risk"]["expected_loss"]["lgd_fixed"] = 0.60
    assert config_fingerprint(build_setup(cfg, [0.02, 0.05])) != config_fingerprint(base)

    cfg = _cfg()
    cfg["risk"]["segmentation"]["thresholds"]["medium"] = 0.25
    assert config_fingerprint(build_setup(cfg, [0.02, 0.05])) != config_fingerprint(base)

    assert config_fingerprint(build_setup(_cfg(), [0.02, 0.09])) != config_fingerprint(base)


# ──────────────────────────────────────────────────────────────
# Scenario and report
# ──────────────────────────────────────────────────────────────

def _scenario(margin: float = 0.05) -> dict:
    rows = _calibrated(6000, seed=127)
    return build_scenario(
        rows, CostParams(margin, LGD),
        approve_below=0.20, review_below=0.35,
        rows_by_fold={f: rows[f::5] for f in range(1, 6)},
        n_points=80, curve_points=10,
    )


def test_scenario_has_optimum_plateau_and_references():
    sc = _scenario()
    assert sc["optimum"]["threshold"] is not None
    assert sc["plateau"]["lower"] <= sc["plateau"]["upper"]
    refs = sc["references"]
    for key in ("approve_all", "reject_all", "random_ranking_at_optimum_rate",
                "at_analytic_breakeven", "current_fixed"):
        assert key in refs
    assert len(refs["current_fixed"]) == 2
    assert sc["rotating_holdout"]["threshold_mean"] is not None


def test_scenario_compares_the_optimum_with_the_breakeven():
    sc = _scenario(0.08)
    assert sc["breakeven_pd"] == pytest.approx(0.08 / (0.08 + LGD), abs=1e-6)
    assert sc["optimum_minus_breakeven"] is not None
    assert sc["breakeven_inside_plateau"] in (True, False)
    # Calibrated data: the closed form should sit inside the plateau.
    assert sc["breakeven_inside_plateau"] is True


def test_scenario_curve_is_thinned_for_the_json():
    sc = _scenario()
    assert len(sc["curve"]) <= 10
    assert sc["n_grid_points"] > len(sc["curve"])
    assert set(sc["curve"][0]) == {
        "threshold", "approval_rate", "bad_rate_approved", "net_value",
        "net_value_per_row", "net_minus_approve_all", "net_vs_approve_all_ratio"}


def test_scenario_gain_over_random_is_positive_on_a_real_ranking():
    sc = _scenario()
    assert sc["gain_vs_random_at_same_rate"] > 0


def _report() -> dict:
    rows = _calibrated(4000, seed=131)
    setup = build_setup(_cfg(), [0.02, 0.12], source={"cv_folds_used": [1, 2, 3, 4, 5]})
    scenarios = [
        build_scenario(rows, CostParams(m, LGD), approve_below=0.20,
                       review_below=0.35, n_points=60, curve_points=8)
        for m in (0.02, 0.12)
    ]
    return build_report(setup, population_summary(rows, CostParams(0.02, LGD)),
                        scenarios, cv_config_fingerprint="sha256:17fb2128")


def test_report_structure():
    r = _report()
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["config_fingerprint"].startswith("sha256:")
    assert r["cv_config_fingerprint"] == "sha256:17fb2128"
    assert r["n_scenarios"] == 2
    assert r["note"] == NOTE
    assert r["setup"]["source"]["cv_folds_used"] == [1, 2, 3, 4, 5]


def test_report_states_sensitivity_to_the_assumed_margin():
    r = _report()
    s = r["sensitivity"]
    assert s["optimum_threshold_min"] < s["optimum_threshold_max"]
    assert "assumption" in s["note"]


def test_report_verdict_is_left_empty():
    """Which cut-off to adopt is a business decision, not a computed one."""
    assert _report()["verdict"] == ""


def test_report_note_forbids_absolute_money_claims():
    assert "never an absolute saving" in NOTE
    assert "Kaggle 2018 currency units" in NOTE


def test_report_filename():
    assert REPORT_FILENAME == "threshold_report.json"


def test_report_is_json_serialisable():
    assert "sha256:" in json.dumps(_report())


def test_format_summary_renders_the_decision():
    out = format_summary(_report())
    assert "COST-BASED CUT-OFF" in out
    assert "break-even PD" in out
    assert "plateau" in out
    assert "current_fixed_review_as_reject" in out
    assert "ASSUMED" in out               # the margin is labelled in the table


def test_format_summary_survives_an_empty_report():
    assert "COST-BASED CUT-OFF" in format_summary({})


def test_default_grid_is_not_absurdly_coarse():
    assert DEFAULT_GRID_POINTS >= 100
