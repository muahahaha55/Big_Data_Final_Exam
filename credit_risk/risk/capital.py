"""Basel II IRB capital requirement — retail risk-weight function (ASRF).

Single source of truth for capital. Both the portfolio report
(`risk/expected_loss.py`) and the stress test (`risk/stress_testing.py`)
call into this module, so the two can no longer drift apart.

Capital requirement per obligor (Basel II §328-330, retail — no maturity
adjustment applies to retail exposures):

    K = LGD · Φ[ (Φ⁻¹(PD) + √R · Φ⁻¹(q)) / √(1−R) ] − PD · LGD
    Capital = K × EAD
    RWA     = K × 12.5 × EAD

Asset correlation R depends on the retail exposure class:

    QRRE (qualifying revolving retail)   R = 0.04                   (§329)
    Other retail                         R = 0.03·w + 0.16·(1−w)    (§328)
                                         w = (1−e^(−35·PD))/(1−e^(−35))

Φ and Φ⁻¹ are evaluated with closed-form rational approximations expressed
as Spark Column arithmetic. This keeps the whole computation inside the JVM —
no Python UDF, no Arrow serialisation, no extra worker memory on a small VM.

    Φ⁻¹  Acklam's algorithm              max relative error 1.15e-9
    Φ    Abramowitz & Stegun 26.2.17     max absolute error 7.5e-8

Error on K is bounded by LGD × 7.5e-8 ≈ 3.4e-8, i.e. under USD 1k of capital
on a USD 27.4bn portfolio — far below the precision of the PD estimates.

Every scalar function here has an identical twin used by `tests/test_capital.py`,
which checks them against scipy. The Spark column functions mirror the scalar
ones formula for formula.

ASSUMPTIONS (documented in README §8.2, not Basel prescriptions):
    - contract type to exposure class mapping (Revolving to QRRE, Cash to other retail)
    - LGD is an assumed constant, not an own-estimate model (retail IRB is A-IRB only)
    - PD is point-in-time from one test split, not a through-the-cycle average
    - EAD = AMT_CREDIT is a proxy (no CCF for undrawn commitments)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CapitalConfig:
    """Basel formula constants + the documented mapping assumptions."""

    confidence_level: float = 0.999
    pd_floor: float = 0.0003
    pd_cap: float = 0.9999
    rwa_multiplier: float = 12.5
    qrre_correlation: float = 0.04
    other_retail_r_min: float = 0.03
    other_retail_r_max: float = 0.16
    other_retail_k_factor: float = 35.0
    contract_type_column: str = "NAME_CONTRACT_TYPE"
    qrre_contract_types: tuple[str, ...] = ("Revolving loans",)
    ul_multiple_of_el: float = 3.0
    ec_multiple_of_ul: float = 1.5


def get_capital_config() -> CapitalConfig:
    """Build CapitalConfig from conf/base/config.yaml (`risk.capital`)."""
    cfg = get_config().get("risk", {}).get("capital", {})
    rot = cfg.get("rule_of_thumb", {})
    return CapitalConfig(
        confidence_level=float(cfg.get("confidence_level", 0.999)),
        pd_floor=float(cfg.get("pd_floor", 0.0003)),
        pd_cap=float(cfg.get("pd_cap", 0.9999)),
        rwa_multiplier=float(cfg.get("rwa_multiplier", 12.5)),
        qrre_correlation=float(cfg.get("qrre_correlation", 0.04)),
        other_retail_r_min=float(cfg.get("other_retail_r_min", 0.03)),
        other_retail_r_max=float(cfg.get("other_retail_r_max", 0.16)),
        other_retail_k_factor=float(cfg.get("other_retail_k_factor", 35.0)),
        contract_type_column=str(cfg.get("contract_type_column", "NAME_CONTRACT_TYPE")),
        qrre_contract_types=tuple(cfg.get("qrre_contract_types", ["Revolving loans"])),
        ul_multiple_of_el=float(rot.get("ul_multiple_of_el", 3.0)),
        ec_multiple_of_ul=float(rot.get("ec_multiple_of_ul", 1.5)),
    )


# ──────────────────────────────────────────────────────────────
# Φ — normal CDF (Abramowitz & Stegun 26.2.17)
# ──────────────────────────────────────────────────────────────

_AS_B = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
_AS_P = 0.2316419
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF. Scalar twin of `norm_cdf_col`."""
    ax = abs(x)
    t = 1.0 / (1.0 + _AS_P * ax)
    b1, b2, b3, b4, b5 = _AS_B
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    tail = _INV_SQRT_2PI * math.exp(-0.5 * ax * ax) * poly   # = 1 − Φ(|x|)
    return tail if x < 0 else 1.0 - tail


def norm_cdf_col(x: Column) -> Column:
    """Standard normal CDF as Spark Column arithmetic."""
    ax = F.abs(x)
    t = F.lit(1.0) / (F.lit(1.0) + F.lit(_AS_P) * ax)
    b1, b2, b3, b4, b5 = _AS_B
    poly = t * (F.lit(b1) + t * (F.lit(b2) + t * (F.lit(b3) + t * (F.lit(b4) + t * F.lit(b5)))))
    tail = F.lit(_INV_SQRT_2PI) * F.exp(F.lit(-0.5) * ax * ax) * poly
    return F.when(x < F.lit(0.0), tail).otherwise(F.lit(1.0) - tail)


# ──────────────────────────────────────────────────────────────
# Φ⁻¹ — inverse normal CDF (Acklam)
# ──────────────────────────────────────────────────────────────

_AK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_AK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
_AK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_AK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
_AK_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF. Scalar twin of `norm_ppf_col`.

    Valid on (0, 1); callers must clip PD to [pd_floor, pd_cap] first.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")

    c0, c1, c2, c3, c4, c5 = _AK_C
    d0, d1, d2, d3 = _AK_D

    def _tail(q: float) -> float:
        num = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
        den = ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
        return num / den

    if p < _AK_P_LOW:
        return _tail(math.sqrt(-2.0 * math.log(p)))
    if p > 1.0 - _AK_P_LOW:
        return -_tail(math.sqrt(-2.0 * math.log(1.0 - p)))

    a0, a1, a2, a3, a4, a5 = _AK_A
    b0, b1, b2, b3, b4 = _AK_B
    q = p - 0.5
    r = q * q
    num = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
    den = (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
    return num / den


def norm_ppf_col(p: Column) -> Column:
    """Inverse standard normal CDF as Spark Column arithmetic.

    All three branches are evaluated, so `p` must already be clipped away
    from 0 and 1 (see `clip_pd_col`) to keep every branch finite.
    """
    c0, c1, c2, c3, c4, c5 = _AK_C
    d0, d1, d2, d3 = _AK_D

    def _tail(q: Column) -> Column:
        num = (((((F.lit(c0) * q + F.lit(c1)) * q + F.lit(c2)) * q + F.lit(c3)) * q
                 + F.lit(c4)) * q + F.lit(c5))
        den = ((((F.lit(d0) * q + F.lit(d1)) * q + F.lit(d2)) * q + F.lit(d3)) * q + F.lit(1.0))
        return num / den

    a0, a1, a2, a3, a4, a5 = _AK_A
    b0, b1, b2, b3, b4 = _AK_B
    q_mid = p - F.lit(0.5)
    r = q_mid * q_mid
    num_mid = (((((F.lit(a0) * r + F.lit(a1)) * r + F.lit(a2)) * r + F.lit(a3)) * r
                 + F.lit(a4)) * r + F.lit(a5)) * q_mid
    den_mid = (((((F.lit(b0) * r + F.lit(b1)) * r + F.lit(b2)) * r + F.lit(b3)) * r
                 + F.lit(b4)) * r + F.lit(1.0))

    return (
        F.when(p < F.lit(_AK_P_LOW), _tail(F.sqrt(F.lit(-2.0) * F.log(p))))
         .when(p > F.lit(1.0 - _AK_P_LOW), -_tail(F.sqrt(F.lit(-2.0) * F.log(F.lit(1.0) - p))))
         .otherwise(num_mid / den_mid)
    )


# ──────────────────────────────────────────────────────────────
# Asset correlation R
# ──────────────────────────────────────────────────────────────

def other_retail_correlation(pd_value: float, cfg: CapitalConfig | None = None) -> float:
    """Basel II §328 asset correlation for 'other retail'. Scalar twin."""
    c = cfg or get_capital_config()
    k = c.other_retail_k_factor
    w = (1.0 - math.exp(-k * pd_value)) / (1.0 - math.exp(-k))
    return c.other_retail_r_min * w + c.other_retail_r_max * (1.0 - w)


def correlation_col(pd_col: Column, is_qrre: Column, cfg: CapitalConfig) -> Column:
    """Asset correlation per row: QRRE constant, else PD-dependent other retail."""
    k = cfg.other_retail_k_factor
    w = (F.lit(1.0) - F.exp(F.lit(-k) * pd_col)) / F.lit(1.0 - math.exp(-k))
    other = F.lit(cfg.other_retail_r_min) * w + F.lit(cfg.other_retail_r_max) * (F.lit(1.0) - w)
    return F.when(is_qrre, F.lit(cfg.qrre_correlation)).otherwise(other)


# ──────────────────────────────────────────────────────────────
# Capital requirement K
# ──────────────────────────────────────────────────────────────

def irb_k(pd_value: float, lgd: float, r: float, cfg: CapitalConfig | None = None) -> float:
    """Basel II retail capital requirement K per unit of EAD. Scalar twin."""
    c = cfg or get_capital_config()
    p = min(max(pd_value, c.pd_floor), c.pd_cap)
    z = norm_ppf(c.confidence_level)
    conditional = norm_cdf((norm_ppf(p) + math.sqrt(r) * z) / math.sqrt(1.0 - r))
    return max(lgd * conditional - p * lgd, 0.0)


def clip_pd_col(pd_col: Column, cfg: CapitalConfig) -> Column:
    """Apply the Basel PD floor and a numerical cap away from 1.0."""
    return F.least(F.greatest(pd_col, F.lit(cfg.pd_floor)), F.lit(cfg.pd_cap))


def irb_k_col(pd_col: Column, lgd_col: Column, r_col: Column, cfg: CapitalConfig) -> Column:
    """Basel II retail capital requirement K as Spark Column arithmetic."""
    p = clip_pd_col(pd_col, cfg)
    z = norm_ppf(cfg.confidence_level)          # a constant, resolved in Python
    arg = (norm_ppf_col(p) + F.sqrt(r_col) * F.lit(z)) / F.sqrt(F.lit(1.0) - r_col)
    return F.greatest(lgd_col * norm_cdf_col(arg) - p * lgd_col, F.lit(0.0))


# ──────────────────────────────────────────────────────────────
# DataFrame-level entry points
# ──────────────────────────────────────────────────────────────

def is_qrre_col(df: DataFrame, cfg: CapitalConfig) -> Column:
    """Exposure-class flag from the contract type column.

    If the column is absent, every exposure falls back to 'other retail'
    (the more conservative class at low PD) and a warning is logged — the
    fallback is never silent.
    """
    if cfg.contract_type_column not in df.columns:
        log.warning(
            "contract_type_column_missing_defaulting_to_other_retail",
            column=cfg.contract_type_column,
        )
        return F.lit(False)
    return F.col(cfg.contract_type_column).isin(list(cfg.qrre_contract_types))


def add_irb_capital(
    df: DataFrame,
    pd_column: str = "pd_score",
    lgd_column: str = "lgd",
    ead_column: str = "ead",
    output_columns: tuple[str, str, str, str] = (
        "basel_exposure_class", "basel_correlation", "irb_k", "irb_capital",
    ),
    cfg: CapitalConfig | None = None,
) -> DataFrame:
    """Add Basel II IRB exposure class, asset correlation, K and capital columns."""
    c = cfg or get_capital_config()
    class_col, corr_col, k_col, cap_col = output_columns

    qrre = is_qrre_col(df, c)
    log.info(
        "computing_irb_capital",
        confidence=c.confidence_level,
        pd_floor=c.pd_floor,
        qrre_types=list(c.qrre_contract_types),
    )

    return (
        df.withColumn(class_col, F.when(qrre, F.lit("QRRE")).otherwise(F.lit("other_retail")))
          .withColumn(corr_col, correlation_col(F.col(pd_column), qrre, c))
          .withColumn(k_col, irb_k_col(F.col(pd_column), F.col(lgd_column), F.col(corr_col), c))
          .withColumn(cap_col, F.col(k_col) * F.col(ead_column))
    )


def scenario_capital_col(
    pd_col: Column,
    lgd_col: Column,
    ead_col: Column,
    is_qrre: Column,
    cfg: CapitalConfig,
) -> Column:
    """IRB capital for an already-stressed PD/LGD pair.

    Used by the stress test so scenario capital and portfolio capital come
    from the same formula. R is recomputed from the stressed PD, as Basel
    prescribes (R is a function of PD).
    """
    r = correlation_col(pd_col, is_qrre, cfg)
    return irb_k_col(pd_col, lgd_col, r, cfg) * ead_col


def rule_of_thumb_cross_check(
    expected_loss: float,
    irb_capital: float,
    cfg: CapitalConfig | None = None,
) -> dict:
    """The old EL-multiple heuristic, kept only to quantify its error.

    NOT Basel, NOT the headline number. `UL ≈ 3 × EL` and `EC = 1.5 × UL`
    were multiples of an aggregate; capital is non-linear in per-obligor PD,
    so no aggregate multiple can reproduce it. This block reports how far off
    those multiples land on this portfolio.
    """
    c = cfg or get_capital_config()
    ul = expected_loss * c.ul_multiple_of_el
    ec = ul * c.ec_multiple_of_ul
    return {
        "note": "ASSUMPTION, not Basel — reported only to quantify the gap vs the IRB formula",
        "ul_multiple_of_el": c.ul_multiple_of_el,
        "ec_multiple_of_ul": c.ec_multiple_of_ul,
        "ul_million": round(ul / 1e6, 2),
        "ec_million": round(ec / 1e6, 2),
        "ul_over_irb_capital": round(ul / irb_capital, 2) if irb_capital > 0 else None,
        "ec_over_irb_capital": round(ec / irb_capital, 2) if irb_capital > 0 else None,
    }
