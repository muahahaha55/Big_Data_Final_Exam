"""Spark session manager.

Singleton SparkSession with production-grade defaults: AQE on, Kryo serializer,
appropriate partition counts. Centralizes Spark config so tuning is one-place.
"""

from __future__ import annotations

from functools import lru_cache

from pyspark.sql import SparkSession

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_spark(app_name: str = "credit-risk-platform") -> SparkSession:
    """Return a configured SparkSession (singleton).

    Master, memory and parallelism come from `spark:` in conf/base/config.yaml,
    so changing them there affects EVERY pipeline that calls this function.
    For cluster mode, overlay conf/prod/config.yaml or use $CRP_spark__master.
    """
    cfg = get_config()
    seed = cfg.get("random_seed", 42)
    sp = cfg.get("spark", {})

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(str(sp.get("master", "local[1]")))
        # ── Memory & parallelism (conf/base/config.yaml → spark:) ─
        .config("spark.driver.memory", str(sp.get("driver_memory", "2g")))
        .config("spark.executor.memory", str(sp.get("executor_memory", "1g")))
        .config("spark.sql.shuffle.partitions", str(sp.get("shuffle_partitions", 50)))
        .config("spark.default.parallelism", str(sp.get("default_parallelism", 8)))
        # ── Adaptive Query Execution ─────────────────────────────
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # ── Serialization (Kryo is ~2x faster than Java default) ─
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrationRequired", "false")
        # ── Arrow for Pandas/Spark interop ───────────────────────
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        # ── Reproducibility ──────────────────────────────────────
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Stable seeds at the SQL level too
    spark.conf.set("spark.sql.random.seed", str(seed))

    log.info(
        "spark_session_created",
        app=app_name,
        master=spark.sparkContext.master,
        version=spark.version,
        shuffle_partitions=spark.conf.get("spark.sql.shuffle.partitions"),
    )
    return spark


def stop_spark() -> None:
    """Stop the active Spark session (safe in tests)."""
    spark = SparkSession.getActiveSession()
    if spark is not None:
        spark.stop()
        get_spark.cache_clear()
        log.info("spark_session_stopped")
