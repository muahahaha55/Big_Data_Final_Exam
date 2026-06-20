"""HDFS Pipeline — đọc data từ HDFS, xử lý bằng Spark, ghi kết quả lên HDFS.

Chứng minh tích hợp Hadoop + Spark:
    HDFS (input) → PySpark ETL → Spark MLlib scoring → HDFS (output)

Usage:
    python pipelines/hdfs_pipeline.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# HDFS paths
HDFS_BASE = "hdfs://localhost:9000/user/vietphuongnguyen/credit_risk"
HDFS_RAW = f"{HDFS_BASE}/raw"
HDFS_PROCESSED = f"{HDFS_BASE}/processed"
HDFS_RESULTS = f"{HDFS_BASE}/results"


def main():
    print("=" * 60)
    print(" HDFS Pipeline — Hadoop + Spark Integration")
    print("=" * 60)

    spark = (
        SparkSession.builder
        .appName("hdfs-credit-risk-pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "1g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Đọc từ HDFS ──────────────────────────────────────
    print(f"\n→ Reading from HDFS: {HDFS_RAW}/application_train.csv")
    app = spark.read.csv(f"{HDFS_RAW}/application_train.csv", header=True, inferSchema=True)
    print(f"  Loaded: {app.count():,} rows × {len(app.columns)} columns")

    print(f"\n→ Reading bureau from HDFS...")
    bureau = spark.read.csv(f"{HDFS_RAW}/bureau.csv", header=True, inferSchema=True)
    print(f"  Loaded: {bureau.count():,} rows")

    # ── 2. Multi-table aggregation ───────────────────────────
    print("\n→ Aggregating bureau → 1 row per customer...")
    bureau_agg = bureau.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("BUREAU_COUNT"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("BUREAU_ACTIVE"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Closed", 1).otherwise(0)).alias("BUREAU_CLOSED"),
        F.mean("CREDIT_DAY_OVERDUE").alias("BUREAU_OVERDUE_MEAN"),
        F.sum("AMT_CREDIT_SUM").alias("BUREAU_CREDIT_SUM"),
    )

    # ── 3. Join ──────────────────────────────────────────────
    print("→ Joining application + bureau...")
    master = app.join(bureau_agg, on="SK_ID_CURR", how="left")
    print(f"  Master dataset: {master.count():,} rows × {len(master.columns)} columns")

    # ── 4. Feature engineering ───────────────────────────────
    print("→ Adding domain features...")
    master = master.withColumn(
        "DEBT_INCOME_RATIO",
        F.when(F.col("AMT_INCOME_TOTAL") > 0,
               F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL")).otherwise(0)
    ).withColumn(
        "EXT_SOURCE_MEAN",
        (F.coalesce(F.col("EXT_SOURCE_1"), F.lit(0.5))
         + F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0.5))
         + F.coalesce(F.col("EXT_SOURCE_3"), F.lit(0.5))) / 3.0
    )

    # ── 5. Ghi processed data lên HDFS (Parquet) ────────────
    print(f"\n→ Writing processed data to HDFS: {HDFS_PROCESSED}")
    master.write.mode("overwrite").parquet(f"{HDFS_PROCESSED}/master.parquet")
    print("  ✅ Saved as Parquet on HDFS")

    # ── 6. Risk segmentation ─────────────────────────────────
    print("\n→ Computing risk segmentation...")
    scored = master.withColumn(
        "pd_score",
        F.greatest(F.lit(0.001), F.least(F.lit(0.999),
            F.lit(0.08)
            + F.when(F.col("DEBT_INCOME_RATIO") > 5, 0.08)
               .when(F.col("DEBT_INCOME_RATIO") > 3, 0.04)
               .otherwise(-0.02)
            + (F.lit(0.5) - F.col("EXT_SOURCE_MEAN")) * 0.3
        ))
    ).withColumn(
        "risk_tier",
        F.when(F.col("pd_score") < 0.10, "Low Risk")
         .when(F.col("pd_score") < 0.20, "Medium Risk")
         .when(F.col("pd_score") < 0.35, "High Risk")
         .otherwise("Very High Risk")
    ).withColumn(
        "expected_loss",
        F.col("pd_score") * 0.45 * F.col("AMT_CREDIT")
    )

    # ── 7. Ghi kết quả lên HDFS ─────────────────────────────
    print(f"→ Writing results to HDFS: {HDFS_RESULTS}")
    scored.select(
        "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL",
        "DEBT_INCOME_RATIO", "EXT_SOURCE_MEAN", "BUREAU_COUNT",
        "pd_score", "risk_tier", "expected_loss", "TARGET"
    ).write.mode("overwrite").parquet(f"{HDFS_RESULTS}/scored.parquet")
    print("  ✅ Saved scored results on HDFS")

    # ── 8. Summary ───────────────────────────────────────────
    summary = scored.groupBy("risk_tier").agg(
        F.count("*").alias("customers"),
        F.round(F.avg("pd_score"), 4).alias("avg_pd"),
        F.round(F.sum("expected_loss"), 0).alias("total_el"),
    ).orderBy("avg_pd")

    print("\n" + "=" * 60)
    print(" HDFS Pipeline — Results")
    print("=" * 60)
    summary.show(truncate=False)

    # Verify trên HDFS
    print("→ HDFS output files:")
    import subprocess
    subprocess.run(["hdfs", "dfs", "-ls", "-R", HDFS_BASE], check=False)

    print("\n" + "=" * 60)
    print(" ✅ Pipeline complete: HDFS → Spark → HDFS")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
