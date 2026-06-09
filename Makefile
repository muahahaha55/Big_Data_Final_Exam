.PHONY: help install install-dev clean test etl train risk drift-check pipeline notebook \
       stack-up stack-down stack-logs stream-produce stream-score stream-consume

COMPOSE := docker compose -f infrastructure/docker-compose.yml

.DEFAULT_GOAL := help

help:  ## Hiển thị danh sách commands
	@echo "Big Data Final Exam — Credit Risk Modeling with PySpark + Kafka"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

# ── Setup ──────────────────────────────────────────────────────
install:  ## Cài đặt dependencies
	pip install -e .

install-dev:  ## Cài đặt dependencies + dev tools
	pip install -e ".[dev]"

clean:  ## Xóa cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

# ── Testing ────────────────────────────────────────────────────
test:  ## Chạy test suite
	pytest -v

# ── Pipelines ──────────────────────────────────────────────────
etl:  ## ETL: Load 7 bảng → join → feature engineering → train/test split
	python pipelines/etl_pipeline.py

train:  ## Train model GBT + register MLflow
	python pipelines/training_pipeline.py

risk:  ## Risk analysis: segmentation + EL + stress testing
	python pipelines/risk_pipeline.py

drift-check:  ## Drift detection (PSI + KS test)
	python pipelines/monitoring_pipeline.py

pipeline: etl train risk  ## Chạy toàn bộ pipeline end-to-end

# ── Notebooks ──────────────────────────────────────────────────
notebook:  ## Mở Jupyter Lab
	jupyter lab notebooks/

# ── MLflow ─────────────────────────────────────────────────────
mlflow-ui:  ## Mở MLflow UI (http://localhost:5000)
	mlflow ui --host 0.0.0.0 --port 5000

# ── Docker Stack (Kafka + Spark + MLflow) ──────────────────────
stack-up:  ## Start full stack: Kafka + Zookeeper + Spark + MLflow
	$(COMPOSE) up -d
	@echo ""
	@echo "Stack is up. URLs:"
	@echo "  Jupyter:    http://localhost:8888"
	@echo "  Kafka UI:   http://localhost:8080"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Spark UI:   http://localhost:4040 (when job runs)"

stack-down:  ## Stop full stack
	$(COMPOSE) down

stack-logs:  ## Tail all stack logs
	$(COMPOSE) logs -f --tail=50

# ── Kafka Streaming ────────────────────────────────────────────
stream-produce:  ## Start Kafka producer (stream loan applications)
	python streaming/kafka_producer.py --mode random --rate 10

stream-produce-csv:  ## Stream from CSV file
	python streaming/kafka_producer.py --mode csv --rate 20

stream-score:  ## Start Spark Streaming scorer (consume → score → output)
	python streaming/spark_streaming.py --output console --trigger 5

stream-score-kafka:  ## Start Spark Streaming scorer (output to Kafka topic)
	python streaming/spark_streaming.py --output kafka --trigger 5

stream-consume:  ## Read scoring results from Kafka
	python streaming/kafka_consumer.py --limit 100

stream-demo:  ## Full demo: produce + score + consume (run in 3 terminals)
	@echo "═══════════════════════════════════════════════════════"
	@echo " Kafka Streaming Demo — run these in 3 separate terminals:"
	@echo "─────────────────────────────────────────────────────────"
	@echo "  Terminal 1:  make stream-score        # Spark consumer"
	@echo "  Terminal 2:  make stream-produce       # Kafka producer"
	@echo "  Terminal 3:  make stream-consume       # Result consumer"
	@echo "═══════════════════════════════════════════════════════"
