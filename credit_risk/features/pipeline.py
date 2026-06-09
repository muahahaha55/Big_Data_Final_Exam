"""Feature engineering pipeline.

Standard Spark ML pipeline:
    Imputer (numeric)  → StringIndexer (categorical) → OneHotEncoder →
    VectorAssembler → StandardScaler → labeled output

Custom domain features added before the pipeline:
    DTI (debt-to-income), credit utilization, employment tenure ratios,
    age groups, payment-to-income ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FeatureColumns:
    numeric: list[str]
    categorical: list[str]
    all_assembled: list[str]


# ──────────────────────────────────────────────────────────────
# Domain-specific features (computed before the pipeline)
# ──────────────────────────────────────────────────────────────

def add_domain_features(df: DataFrame) -> DataFrame:
    """Add hand-crafted credit-risk features."""
    out = df

    # Debt-to-income ratio
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        out = out.withColumn(
            "FEAT_DEBT_INCOME_RATIO",
            F.when(F.col("AMT_INCOME_TOTAL") > 0,
                   F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL")).otherwise(0)
        )

    # Annuity-to-income ratio (monthly payment burden)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        out = out.withColumn(
            "FEAT_ANNUITY_INCOME_RATIO",
            F.when(F.col("AMT_INCOME_TOTAL") > 0,
                   F.col("AMT_ANNUITY") * 12 / F.col("AMT_INCOME_TOTAL")).otherwise(0)
        )

    # Loan-to-goods-price (down payment proxy)
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        out = out.withColumn(
            "FEAT_CREDIT_GOODS_RATIO",
            F.when(F.col("AMT_GOODS_PRICE") > 0,
                   F.col("AMT_CREDIT") / F.col("AMT_GOODS_PRICE")).otherwise(1)
        )

    # Credit term (months)
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        out = out.withColumn(
            "FEAT_CREDIT_TERM",
            F.when(F.col("AMT_ANNUITY") > 0,
                   F.col("AMT_CREDIT") / F.col("AMT_ANNUITY")).otherwise(0)
        )

    # Employment ratio (% of life employed)
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        out = out.withColumn(
            "FEAT_EMPLOYMENT_RATIO",
            F.when(
                (F.col("DAYS_BIRTH") < 0) & (F.col("DAYS_EMPLOYED") != 365243),
                F.col("DAYS_EMPLOYED") / F.col("DAYS_BIRTH")
            ).otherwise(0)
        )

    # External score average (powerful Home Credit signal)
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        ext_sum = sum(F.coalesce(F.col(c), F.lit(0.5)) for c in ext_cols)
        out = out.withColumn("FEAT_EXT_SOURCE_MEAN", ext_sum / len(ext_cols))

    n_new = len([c for c in out.columns if c.startswith("FEAT_")])
    log.info("domain_features_added", count=n_new)
    return out


# ──────────────────────────────────────────────────────────────
# Pipeline construction
# ──────────────────────────────────────────────────────────────

def identify_feature_columns(
    df: DataFrame,
    target_col: str = "TARGET",
    id_cols: list[str] | None = None,
    drop_threshold: float = 0.5,
) -> FeatureColumns:
    """Split columns into numeric/categorical, dropping high-missing cols.

    Args:
        df: input DataFrame
        target_col: column to exclude (the label)
        id_cols: ID columns to exclude (default ['SK_ID_CURR'])
        drop_threshold: drop cols with > this fraction missing
    """
    id_cols = id_cols or ["SK_ID_CURR"]
    excluded = set([target_col, *id_cols])

    total_rows = df.count()
    if total_rows == 0:
        return FeatureColumns([], [], [])

    # Compute null rate per column
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).collect()[0].asDict()

    keep = {c: cnt / total_rows < drop_threshold for c, cnt in null_counts.items()}

    numeric, categorical = [], []
    for field in df.schema.fields:
        if field.name in excluded:
            continue
        if not keep.get(field.name, True):
            continue
        if isinstance(field.dataType, NumericType):
            numeric.append(field.name)
        elif isinstance(field.dataType, StringType):
            categorical.append(field.name)

    log.info(
        "feature_columns_identified",
        numeric=len(numeric), categorical=len(categorical),
        dropped=sum(1 for v in keep.values() if not v),
    )
    return FeatureColumns(numeric=numeric, categorical=categorical, all_assembled=[])


def build_pipeline(
    feature_cols: FeatureColumns,
    scaler: bool = True,
) -> Pipeline:
    """Build a Spark ML preprocessing Pipeline.

    Stages:
        1. Imputer for numeric (median)
        2. StringIndexer for each categorical (handle unseen → keep)
        3. OneHotEncoder for indexed categoricals
        4. VectorAssembler combining everything
        5. StandardScaler (optional)
    """
    stages: list = []

    # ── Numeric imputation ───────────────────────────────────
    imputed_cols = [f"{c}_imp" for c in feature_cols.numeric]
    if feature_cols.numeric:
        imputer = Imputer(
            inputCols=feature_cols.numeric,
            outputCols=imputed_cols,
            strategy="median",
        )
        stages.append(imputer)

    # ── Categorical encoding ─────────────────────────────────
    indexed_cols = [f"{c}_idx" for c in feature_cols.categorical]
    onehot_cols = [f"{c}_ohe" for c in feature_cols.categorical]

    if feature_cols.categorical:
        indexer = StringIndexer(
            inputCols=feature_cols.categorical,
            outputCols=indexed_cols,
            handleInvalid="keep",
        )
        stages.append(indexer)

        encoder = OneHotEncoder(
            inputCols=indexed_cols,
            outputCols=onehot_cols,
            handleInvalid="keep",
        )
        stages.append(encoder)

    # ── Assemble ─────────────────────────────────────────────
    assemble_cols = imputed_cols + onehot_cols
    assembler = VectorAssembler(
        inputCols=assemble_cols,
        outputCol="features_raw" if scaler else "features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    # ── Scale ────────────────────────────────────────────────
    if scaler:
        scaler_stage = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=False,    # sparse-friendly
            withStd=True,
        )
        stages.append(scaler_stage)

    feature_cols.all_assembled = assemble_cols
    log.info("pipeline_built", stages=len(stages), assembled_cols=len(assemble_cols))
    return Pipeline(stages=stages)


def fit_and_transform(
    df: DataFrame,
    feature_cols: FeatureColumns,
    scaler: bool = True,
) -> tuple[PipelineModel, DataFrame]:
    """Convenience: build pipeline, fit it, transform df."""
    pipeline = build_pipeline(feature_cols, scaler=scaler)
    model = pipeline.fit(df)
    transformed = model.transform(df)
    return model, transformed
