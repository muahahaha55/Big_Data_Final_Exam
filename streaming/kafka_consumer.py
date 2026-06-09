"""Kafka Consumer — đọc kết quả scoring từ topic `scoring-results`.

Dùng để verify pipeline end-to-end:
    Producer → Kafka → Spark Streaming → Kafka → Consumer (this)

Usage:
    python streaming/kafka_consumer.py
    python streaming/kafka_consumer.py --topic scoring-results --limit 50
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable


BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "scoring-results"


def run_consumer(
    topic: str = TOPIC,
    bootstrap_servers: str = BOOTSTRAP_SERVERS,
    limit: int | None = None,
) -> None:
    """Consume và in kết quả scoring."""
    print(f"📥 Consuming from topic '{topic}'...")
    print(f"   Press Ctrl+C to stop\n")

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=60_000,    # timeout sau 60s không có message
        )
    except NoBrokersAvailable:
        print(f"❌ Cannot connect to Kafka at {bootstrap_servers}")
        print("   Make sure Kafka is running: make stack-up")
        return

    count = 0
    stats = {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}

    try:
        for msg in consumer:
            result = msg.value
            decision = result.get("decision", "?")
            stats[decision] = stats.get(decision, 0) + 1
            count += 1

            # In chi tiết
            print(
                f"  #{count:<5} "
                f"ID={result.get('SK_ID_CURR', '?'):<8} "
                f"PD={result.get('pd_score', 0):.3f}  "
                f"Tier={result.get('risk_tier', '?'):<16} "
                f"Decision={decision:<8} "
                f"EL=${result.get('expected_loss', 0):>10,.2f}"
            )

            # Summary mỗi 50 messages
            if count % 50 == 0:
                total = sum(stats.values())
                print(f"\n  --- Summary ({total} messages) ---")
                for d, c in sorted(stats.items()):
                    print(f"      {d}: {c} ({c/total*100:.1f}%)")
                print()

            if limit and count >= limit:
                break

    except KeyboardInterrupt:
        print(f"\n⏹ Stopped")
    finally:
        consumer.close()
        total = sum(stats.values())
        print(f"\n{'='*50}")
        print(f"  Messages consumed: {count}")
        if total > 0:
            for d, c in sorted(stats.items()):
                print(f"    {d}: {c} ({c/total*100:.1f}%)")
        print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Consumer — read scoring results")
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVERS)
    parser.add_argument("--limit", type=int, default=None, help="Max messages to consume")
    args = parser.parse_args()

    run_consumer(topic=args.topic, bootstrap_servers=args.bootstrap_servers, limit=args.limit)
