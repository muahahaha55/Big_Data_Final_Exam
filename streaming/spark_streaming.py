"""Spark Structured Streaming — real-time credit risk scoring.

Consumes loan applications từ Kafka topic `loan-applications`,
áp dụng scoring logic, và ghi kết quả ra:
    1. Kafka topic `scoring-results` (production)
    2. Console (debug)
    3. Parquet (analytics)

Architecture:
    Kafka [loan-applications]
        → Spark Structured Streaming
            → Parse JSON
            → Feature engineering
            → Risk scoring (PD, tier, EL)
            → Aggregate statistics (windowed)
        → Kafka [scoring-results] + Console + Parquet

Usage:
    # Trong Docker:
    make stream-score

    # Hoặc trực tiếp:
    python streaming/spark_streaming.py --output console
    python streaming/spark_streaming.py --output kafka
    python streaming/spark_streaming.py --output parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP = "localhost:9092"
INPUT_TOPIC = "loan-applications"
OUTPUT_TOPIC = "scoring-results"

# Schema khớp với producer output
APPLICATION_SCHEMA = StructType([
    StructField("SK_ID_CURR", IntegerType(), True),
    StructField("TARGET", IntegerType(), True),
    StructField("AMT_CREDIT", DoubleType(), True),
    StructField("AMT_INCOME_TOTAL", DoubleType(), True),
    StructField("AMT_ANNUITY", DoubleType(), True),
    StructField("AMT_GOODS_PRICE", DoubleType(), True),
    StructField("NAME_CONTRACT_TYPE", StringType(), True),
    StructField("CODE_GENDER", StringType(), True),
    StructField("DAYS_BIRTH", IntegerType(), True),
    StructField("DAYS_EMPLOYED", IntegerType(), True),
    StructField("EXT_SOURCE_1", DoubleType(), True),
    StructField("EXT_SOURCE_2", DoubleType(), True),
    StructField("EXT_SOURCE_3", DoubleType(), True),
    StructField("FLAG_OWN_CAR", StringType(), True),
    StructField("FLAG_OWN_REALTY", StringType(), True),
    StructField("CNT_FAM_MEMBERS", IntegerType(), True),
    StructField("NAME_INCOME_TYPE", StringType(), True),
    StructField("NAME_EDUCATION_TYPE", StringType(), True),
    StructField("OCCUPATION_TYPE", StringType(), True),
    StructField("timestamp", StringType(), True),
])

# Risk thresholds (Basel II aligned)
THRESHOLD_LOW = 0.10
THRESHOLD_MEDIUM = 0.20
THRESHOLD_HIGH = 0.35
LGD_FIXED = 0.45


# ──────────────────────────────────────────────────────────────
# Spark Session (with Kafka packages)
# ──────────────────────────────────────────────────────────────

def get_streaming_spark(app_name: str = "credit-risk-streaming") -> SparkSession:
    """Tạo SparkSession với Kafka connector."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint")
        .getOrCreate()
    )


# ──────────────────────────────────────────────────────────────
# Scoring logic (mirrors batch pipeline)
# ──────────────────────────────────────────────────────────────

def add_scoring_features(df: DataFrame) -> DataFrame:
    """Thêm domain features và compute PD score trên streaming DataFrame.

    Trong production, đây sẽ là model.transform(). Cho demo, dùng
    heuristic scoring tương tự API để chạy được không cần trained model.
    """
    # Domain features
    scored = df.withColumn(
        "debt_income_ratio",
        F.when(F.col("AMT_INCOME_TOTAL") > 0,
               F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL")).otherwise(F.lit(5.0))
    ).withColumn(
        "ext_source_mean",
        (F.coalesce(F.col("EXT_SOURCE_1"), F.lit(0.5))
         + F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0.5))
         + F.coalesce(F.col("EXT_SOURCE_3"), F.lit(0.5))) / 3.0
    ).withColumn(
        "is_unemployed",
        F.when(F.col("DAYS_EMPLOYED") == 365243, F.lit(1)).otherwise(F.lit(0))
    )

    # PD score = heuristic (base + DTI signal + ext_source signal + employment signal)
    scored = scored.withColumn(
        "pd_score",
        F.greatest(F.lit(0.001), F.least(F.lit(0.999),
            F.lit(0.08)                                                           # base rate
            + F.when(F.col("debt_income_ratio") > 5, F.lit(0.08))
               .when(F.col("debt_income_ratio") > 3, F.lit(0.04))
               .otherwise(F.lit(-0.02))                                           # DTI
            + (F.lit(0.5) - F.col("ext_source_mean")) * 0.3                      # external scores
            + F.when(F.col("is_unemployed") == 1, F.lit(0.12))
               .when(F.col("DAYS_EMPLOYED") > -365, F.lit(0.04))
               .otherwise(F.lit(-0.03))                                           # employment
        ))
    )

    # Risk tier
    scored = scored.withColumn(
        "risk_tier",
        F.when(F.col("pd_score") < THRESHOLD_LOW, F.lit("Low Risk"))
         .when(F.col("pd_score") < THRESHOLD_MEDIUM, F.lit("Medium Risk"))
         .when(F.col("pd_score") < THRESHOLD_HIGH, F.lit("High Risk"))
         .otherwise(F.lit("Very High Risk"))
    )

    # Decision
    scored = scored.withColumn(
        "decision",
        F.when(F.col("risk_tier").isin("Low Risk", "Medium Risk"), F.lit("APPROVE"))
         .when(F.col("risk_tier") == "High Risk", F.lit("REVIEW"))
         .otherwise(F.lit("REJECT"))
    )

    # Expected Loss
    scored = scored.withColumn(
        "expected_loss",
        F.round(F.col("pd_score") * LGD_FIXED * F.col("AMT_CREDIT"), 2)
    )

    # Scoring timestamp
    scored = scored.withColumn("scored_at", F.current_timestamp())

    return scored


# ──────────────────────────────────────────────────────────────
# Read from Kafka
# ──────────────────────────────────────────────────────────────

def read_kafka_stream(spark: SparkSession, bootstrap_servers: str, topic: str) -> DataFrame:
    """Đọc streaming DataFrame từ Kafka topic."""
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")        # Chỉ đọc messages mới
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 1000)       # Batch size per trigger
        .load()
    )

    # Parse: Kafka value (bytes) → JSON → structured columns
    parsed = (
        raw.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.from_json(F.col("value").cast("string"), APPLICATION_SCHEMA).alias("data"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
        )
        .select("kafka_key", "kafka_timestamp", "kafka_partition", "kafka_offset", "data.*")
    )

    return parsed


# ──────────────────────────────────────────────────────────────
# Output sinks
# ──────────────────────────────────────────────────────────────

def write_to_console(scored_df: DataFrame, trigger_seconds: int = 5):
    """Ghi kết quả ra console — dùng cho debug/demo."""
    output_cols = [
        "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL",
        "pd_score", "risk_tier", "decision", "expected_loss", "scored_at",
    ]

    query = (
        scored_df.select(output_cols)
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )
    return query


def write_to_kafka(scored_df: DataFrame, bootstrap_servers: str, trigger_seconds: int = 5):
    """Ghi kết quả ra Kafka topic scoring-results."""
    # Serialize result thành JSON
    output = scored_df.select(
        F.col("SK_ID_CURR").cast("string").alias("key"),
        F.to_json(F.struct(
            "SK_ID_CURR", "AMT_CREDIT", "pd_score", "risk_tier",
            "decision", "expected_loss", "scored_at",
        )).alias("value"),
    )

    query = (
        output.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", OUTPUT_TOPIC)
        .option("checkpointLocation", "/tmp/spark-checkpoint/kafka-output")
        .outputMode("append")
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )
    return query


def write_to_parquet(scored_df: DataFrame, output_path: str, trigger_seconds: int = 10):
    """Ghi kết quả ra Parquet — dùng cho analytics sau."""
    query = (
        scored_df.writeStream
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", "/tmp/spark-checkpoint/parquet-output")
        .outputMode("append")
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .start()
    )
    return query


# ──────────────────────────────────────────────────────────────
# Windowed aggregation (real-time dashboard metrics)
# ──────────────────────────────────────────────────────────────

def write_windowed_stats(scored_df: DataFrame, trigger_seconds: int = 10):
    """Tính thống kê theo window 1 phút — mô phỏng real-time monitoring.

    Output mỗi phút:
        - Số đơn vay mới
        - Tỷ lệ APPROVE / REVIEW / REJECT
        - PD trung bình
        - Tổng Expected Loss
    """
    windowed = (
        scored_df
        .withWatermark("scored_at", "2 minutes")
        .groupBy(F.window("scored_at", "1 minute"))
        .agg(
            F.count("*").alias("total_applications"),
            F.sum(F.when(F.col("decision") == "APPROVE", 1).otherwise(0)).alias("approved"),
            F.sum(F.when(F.col("decision") == "REVIEW", 1).otherwise(0)).alias("review"),
            F.sum(F.when(F.col("decision") == "REJECT", 1).otherwise(0)).alias("rejected"),
            F.round(F.avg("pd_score"), 4).alias("avg_pd"),
            F.round(F.sum("expected_loss"), 2).alias("total_el"),
            F.round(F.avg("AMT_CREDIT"), 0).alias("avg_loan_amount"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "total_applications", "approved", "review", "rejected",
            "avg_pd", "total_el", "avg_loan_amount",
        )
    )

    query = (
        windowed.writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", "false")
        .trigger(processingTime=f"{trigger_seconds} seconds")
        .queryName("windowed_stats")
        .start()
    )
    return query


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(
    output: str = "console",
    bootstrap_servers: str = KAFKA_BOOTSTRAP,
    trigger_seconds: int = 5,
) -> None:
    """Main streaming pipeline."""
    print("=" * 60)
    print(" Spark Structured Streaming — Credit Risk Scoring")
    print("=" * 60)
    print(f"  Input topic:  {INPUT_TOPIC}")
    print(f"  Output:       {output}")
    print(f"  Kafka:        {bootstrap_servers}")
    print(f"  Trigger:      every {trigger_seconds}s")
    print("=" * 60)

    spark = get_streaming_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka
    print("\n→ Connecting to Kafka stream...")
    raw_stream = read_kafka_stream(spark, bootstrap_servers, INPUT_TOPIC)

    # Apply scoring
    print("→ Applying scoring pipeline...")
    scored_stream = add_scoring_features(raw_stream)

    # Start output
    queries = []

    if output in ("console", "all"):
        print("→ Starting console output...")
        queries.append(write_to_console(scored_stream, trigger_seconds))

    if output in ("kafka", "all"):
        print(f"→ Writing results to topic '{OUTPUT_TOPIC}'...")
        queries.append(write_to_kafka(scored_stream, bootstrap_servers, trigger_seconds))

    if output in ("parquet", "all"):
        parquet_path = "data/07_model_output/streaming_results"
        print(f"→ Writing results to {parquet_path}...")
        queries.append(write_to_parquet(scored_stream, parquet_path, trigger_seconds))

    # Always show windowed stats
    print("→ Starting windowed aggregation (1-min windows)...")
    queries.append(write_windowed_stats(scored_stream, max(trigger_seconds, 10)))

    print("\n✅ Streaming started! Waiting for data...\n")
    print("   Press Ctrl+C to stop.\n")

    try:
        # Wait for any query to terminate (or KeyboardInterrupt)
        for q in queries:
            q.awaitTermination()
    except KeyboardInterrupt:
        print("\n⏹ Stopping streaming...")
        for q in queries:
            q.stop()
        spark.stop()
        print("✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Streaming — real-time credit risk scoring")
    parser.add_argument("--output", choices=["console", "kafka", "parquet", "all"],
                        default="console", help="Output sink(s)")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--trigger", type=int, default=5, help="Trigger interval (seconds)")
    args = parser.parse_args()

    main(output=args.output, bootstrap_servers=args.bootstrap_servers, trigger_seconds=args.trigger)
