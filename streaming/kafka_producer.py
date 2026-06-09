"""Kafka Producer — mô phỏng luồng đơn vay phát sinh liên tục.

Đọc từ application_train.csv, gửi từng dòng dưới dạng JSON vào
Kafka topic `loan-applications`. Mô phỏng kịch bản thực tế:
ngân hàng nhận đơn vay mới mỗi giây.

Modes:
    - csv: đọc từ CSV file (default)
    - random: sinh dữ liệu ngẫu nhiên (không cần file)

Usage:
    python streaming/kafka_producer.py --mode csv --rate 10
    python streaming/kafka_producer.py --mode random --rate 50 --total 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "loan-applications"

# Subset of features to stream (mirrors API schema)
STREAM_FIELDS = [
    "SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE", "CODE_GENDER",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_FAM_MEMBERS",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE",
]


# ──────────────────────────────────────────────────────────────
# Random data generator (khi không có CSV)
# ──────────────────────────────────────────────────────────────

def generate_random_application() -> dict:
    """Sinh một đơn vay ngẫu nhiên."""
    return {
        "SK_ID_CURR": random.randint(100_000, 999_999),
        "AMT_CREDIT": round(random.uniform(50_000, 2_000_000), 2),
        "AMT_INCOME_TOTAL": round(random.uniform(20_000, 500_000), 2),
        "AMT_ANNUITY": round(random.uniform(3_000, 80_000), 2),
        "AMT_GOODS_PRICE": round(random.uniform(40_000, 1_500_000), 2),
        "NAME_CONTRACT_TYPE": random.choice(["Cash loans", "Revolving loans"]),
        "CODE_GENDER": random.choice(["M", "F"]),
        "DAYS_BIRTH": random.randint(-25000, -7300),
        "DAYS_EMPLOYED": random.choice([365243, *range(-15000, -100, 100)]),
        "EXT_SOURCE_1": round(random.uniform(0, 1), 4) if random.random() > 0.3 else None,
        "EXT_SOURCE_2": round(random.uniform(0, 1), 4) if random.random() > 0.1 else None,
        "EXT_SOURCE_3": round(random.uniform(0, 1), 4) if random.random() > 0.2 else None,
        "FLAG_OWN_CAR": random.choice(["Y", "N"]),
        "FLAG_OWN_REALTY": random.choice(["Y", "N"]),
        "CNT_FAM_MEMBERS": random.randint(1, 8),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ──────────────────────────────────────────────────────────────
# CSV reader generator
# ──────────────────────────────────────────────────────────────

def read_csv_applications(csv_path: str, limit: int | None = None):
    """Đọc từng dòng từ CSV và yield dưới dạng dict."""
    import csv
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for field in STREAM_FIELDS:
                val = row.get(field)
                if val is None or val == "" or val == "nan":
                    record[field] = None
                else:
                    # Cố gắng convert sang numeric
                    try:
                        record[field] = float(val) if "." in val else int(val)
                    except (ValueError, TypeError):
                        record[field] = val
            record["timestamp"] = datetime.utcnow().isoformat()
            yield record
            count += 1
            if limit and count >= limit:
                return


# ──────────────────────────────────────────────────────────────
# Main producer
# ──────────────────────────────────────────────────────────────

def create_producer(bootstrap_servers: str, retries: int = 5) -> KafkaProducer:
    """Tạo Kafka producer với retry logic."""
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                acks="all",                   # Đảm bảo message được replicate
                retries=3,
                batch_size=16384,             # Batch 16KB trước khi gửi
                linger_ms=10,                 # Đợi 10ms để batch thêm messages
                compression_type="gzip",      # Nén giảm bandwidth
            )
            print(f"✅ Connected to Kafka at {bootstrap_servers}")
            return producer
        except NoBrokersAvailable:
            print(f"⏳ Kafka not ready (attempt {attempt + 1}/{retries}), waiting 5s...")
            time.sleep(5)
    raise RuntimeError(f"Cannot connect to Kafka at {bootstrap_servers}")


def run_producer(
    mode: str = "random",
    csv_path: str | None = None,
    rate: int = 10,
    total: int | None = None,
    bootstrap_servers: str = BOOTSTRAP_SERVERS,
) -> None:
    """Chạy producer: gửi messages vào topic."""
    producer = create_producer(bootstrap_servers)

    print(f"📤 Producing to topic '{TOPIC}' @ {rate} msg/sec")
    print(f"   Mode: {mode}")
    if total:
        print(f"   Total: {total} messages")
    print(f"   Press Ctrl+C to stop\n")

    # Chọn data source
    if mode == "csv":
        if not csv_path:
            csv_path = str(Path(__file__).parent.parent / "data" / "01_raw" / "application_train.csv")
        if not Path(csv_path).exists():
            print(f"❌ File not found: {csv_path}")
            print("   Falling back to random mode")
            mode = "random"

    sent = 0
    start_time = time.time()
    interval = 1.0 / rate if rate > 0 else 0

    try:
        if mode == "csv":
            data_source = read_csv_applications(csv_path, limit=total)
        else:
            data_source = (generate_random_application() for _ in (range(total) if total else iter(int, 1)))

        for record in data_source:
            key = record.get("SK_ID_CURR")
            producer.send(TOPIC, key=key, value=record)
            sent += 1

            if sent % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = sent / elapsed if elapsed > 0 else 0
                print(f"  📊 Sent: {sent:,} | Rate: {actual_rate:.1f} msg/s | Elapsed: {elapsed:.1f}s")

            if interval > 0:
                time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"  Total sent:    {sent:,}")
        print(f"  Total time:    {elapsed:.1f}s")
        print(f"  Avg rate:      {sent / elapsed:.1f} msg/s" if elapsed > 0 else "")
        print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Producer — stream loan applications")
    parser.add_argument("--mode", choices=["csv", "random"], default="random",
                        help="Data source: csv (from file) or random (generated)")
    parser.add_argument("--csv-path", default=None, help="Path to application_train.csv")
    parser.add_argument("--rate", type=int, default=10, help="Messages per second (0 = max speed)")
    parser.add_argument("--total", type=int, default=None, help="Total messages to send (default: unlimited)")
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    run_producer(
        mode=args.mode,
        csv_path=args.csv_path,
        rate=args.rate,
        total=args.total,
        bootstrap_servers=args.bootstrap_servers,
    )
