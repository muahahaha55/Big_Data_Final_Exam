"""Shared pytest fixtures."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# ──────────────────────────────────────────────────────────────
# Spark — session-scoped (expensive to create)
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Provide a SparkSession for the entire test session."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        pytest.skip("PySpark not installed")

    spark = (
        SparkSession.builder
        .appName("credit-risk-platform-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    spark.stop()


# ──────────────────────────────────────────────────────────────
# Temp directory — function-scoped
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory; cleaned up after test."""
    d = Path(tempfile.mkdtemp(prefix="crp-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ──────────────────────────────────────────────────────────────
# Sample DataFrames
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_application_df(spark):
    """Tiny synthetic application_train DataFrame for fast tests."""
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    schema = StructType([
        StructField("SK_ID_CURR", IntegerType(), False),
        StructField("TARGET", IntegerType(), False),
        StructField("AMT_CREDIT", DoubleType(), True),
        StructField("AMT_INCOME_TOTAL", DoubleType(), True),
        StructField("AMT_ANNUITY", DoubleType(), True),
        StructField("DAYS_BIRTH", IntegerType(), True),
        StructField("DAYS_EMPLOYED", IntegerType(), True),
        StructField("CODE_GENDER", StringType(), True),
        StructField("NAME_CONTRACT_TYPE", StringType(), True),
        StructField("EXT_SOURCE_2", DoubleType(), True),
        StructField("EXT_SOURCE_3", DoubleType(), True),
    ])

    data = [
        (1001, 0, 200_000.0, 100_000.0, 10_000.0, -12_000, -2_000, "F", "Cash loans", 0.70, 0.45),
        (1002, 0, 150_000.0,  80_000.0,  8_000.0, -15_000, -3_000, "M", "Cash loans", 0.65, 0.55),
        (1003, 1, 500_000.0,  40_000.0, 25_000.0, -10_000, -200,   "M", "Cash loans", 0.25, 0.18),
        (1004, 0, 100_000.0, 120_000.0,  5_000.0, -18_000, -5_000, "F", "Revolving loans", 0.80, 0.75),
        (1005, 1, 350_000.0,  60_000.0, 18_000.0, -9_000,  -100,   "M", "Cash loans", 0.30, 0.22),
        (1006, 0, 180_000.0,  90_000.0,  9_000.0, -14_000, -4_000, "F", "Cash loans", 0.68, 0.50),
        (1007, 0, 220_000.0, 100_000.0, 11_000.0, -13_000, -2_500, "M", "Cash loans", 0.72, 0.48),
        (1008, 1, 400_000.0,  50_000.0, 22_000.0, -11_000, -300,   "F", "Cash loans", 0.28, 0.20),
        (1009, 0, 250_000.0,  85_000.0, 12_000.0, -12_500, -2_200, "M", "Cash loans", 0.66, 0.52),
        (1010, 0, 130_000.0,  95_000.0,  6_500.0, -16_000, -3_500, "F", "Revolving loans", 0.74, 0.62),
    ] * 10  # 100 rows total

    return spark.createDataFrame(data, schema)
