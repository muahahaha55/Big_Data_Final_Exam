# Big Data Final Exam — Credit Risk Modeling with PySpark

> Ứng dụng Apache Spark vào bài toán dự đoán rủi ro tín dụng trên bộ dữ liệu Home Credit Default Risk (307,511 khách hàng, 7 bảng dữ liệu, tổng ~1.5 GB).

**Sinh viên:** Nguyễn Việt Phương
**MSSV:** 24022431
**Môn học:** Big Data
**Giảng viên:** Trần Hồng Việt

---

## 1. Giới thiệu bài toán

Rủi ro tín dụng (credit risk) là bài toán dự đoán xác suất vỡ nợ (PD — Probability of Default) của khách hàng vay. Đây là bài toán cốt lõi trong ngành ngân hàng, ảnh hưởng trực tiếp đến quyết định cho vay và quản lý rủi ro danh mục.

**Tại sao cần Big Data?**
- Dữ liệu thực tế gồm 7 bảng quan hệ với hàng triệu bản ghi (13.6 triệu dòng riêng bảng installments)
- Join + aggregate tạo ra dataset master >200 features
- Mô hình cần train trên toàn bộ dữ liệu với cross-validation → tính toán phân tán là bắt buộc
- Pipeline xử lý từ raw → clean → feature engineering → model → risk analysis cần khả năng scale

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- 307,511 khách hàng (TARGET: 8% default, 92% non-default — imbalanced)
- 7 bảng dữ liệu liên kết qua SK_ID_CURR và SK_ID_PREV
- Tổng dung lượng: ~1.5 GB sau join

---

## 2. Công nghệ sử dụng

| Công nghệ | Vai trò | Lý do chọn |
|-----------|---------|-------------|
| **Apache Spark 3.5** | Xử lý dữ liệu phân tán | Xử lý hiệu quả 7 bảng + join, chuẩn công nghiệp |
| **Spark MLlib** | Training mô hình ML | Tích hợp native với Spark, không cần chuyển dữ liệu |
| **Spark Structured Streaming** | Xử lý luồng real-time | Unified batch + streaming API, exactly-once semantics |
| **Apache Kafka** | Message broker phân tán | Throughput cao, fault-tolerant, chuẩn streaming |
| **MLflow 2.9** | Theo dõi thí nghiệm | Lưu trữ params, metrics, model artifacts |
| **Docker Compose** | Orchestration local | Kafka + Zookeeper + Spark + MLflow trong 1 lệnh |
| **Python 3.10** | Ngôn ngữ chính | PySpark API, ecosystem phong phú |

### Tối ưu Spark đã áp dụng

- **AQE (Adaptive Query Execution)**: tự động tối ưu shuffle partitions và xử lý data skew
- **Kryo Serializer**: nhanh gấp ~2x so với Java Serialization mặc định
- **Arrow Integration**: tăng tốc chuyển đổi Spark ↔ Pandas
- **Snappy Compression**: nén Parquet giảm I/O disk

---

## 3. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    DỮ LIỆU THÔ (7 bảng CSV)                │
│  application_train · bureau · previous_application          │
│  installments · credit_card · pos_cash · bureau_balance     │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              ETL PIPELINE (PySpark) — Batch                  │
│  Load CSV → Aggregate child tables → Left join → Parquet    │
│  Output: master dataset (~200 features, 307K rows)          │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING (Spark ML Pipeline)            │
│  Domain features → Imputer → StringIndexer → OHE →          │
│  VectorAssembler → StandardScaler                            │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL TRAINING (Spark MLlib)                     │
│  LogisticRegression / RandomForest / GBT                     │
│  CrossValidator (3-fold) + class weight balancing             │
│  MLflow tracking: params, metrics, model artifacts            │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│          FINANCIAL RISK ANALYSIS (Basel II)                   │
│  WoE/IV scorecard → Risk segmentation (4 tiers) →            │
│  Expected Loss (PD × LGD × EAD) → Stress testing             │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
                    REAL-TIME STREAMING PIPELINE
═══════════════════════════════════════════════════════════════

┌─────────────┐      ┌───────────────────┐      ┌──────────────────────┐
│ Kafka       │      │ Spark Structured  │      │ Kafka                │
│ Producer    │─────►│ Streaming         │─────►│ [scoring-results]    │
│ (simulate   │      │ (consume + score  │      │                      │
│  loan apps) │      │  + aggregate)     │      │ + Console / Parquet  │
└─────────────┘      └───────────────────┘      └──────────────────────┘
       │                      │
       ▼                      ▼
 Topic:                 Windowed stats:
 [loan-applications]    - Approvals/min
                        - Avg PD per window
                        - Total EL per window
```

### 3.1 Real-time Streaming Pipeline (Kafka + Spark Streaming)

Ngoài batch pipeline, hệ thống hỗ trợ xử lý real-time:

1. **Kafka Producer** (`streaming/kafka_producer.py`): mô phỏng đơn vay phát sinh liên tục, gửi vào topic `loan-applications`
2. **Spark Structured Streaming** (`streaming/spark_streaming.py`): consume từ Kafka, áp dụng scoring logic, tính PD/risk tier/EL cho mỗi đơn
3. **Windowed Aggregation**: tính thống kê theo cửa sổ 1 phút (số đơn/phút, tỷ lệ approve/reject, PD trung bình)
4. **Output sinks**: ghi ra Kafka topic `scoring-results`, console, hoặc Parquet

**Kafka config highlights:**
- `acks=all`: đảm bảo message không mất
- `compression_type=gzip`: giảm bandwidth 60-70%
- `batch_size=16KB` + `linger_ms=10`: batch messages để tăng throughput
- `startingOffsets=latest`: chỉ xử lý messages mới (không replay history)

---

## 4. Cấu trúc thư mục

```
Big_Data_Final_Exam/
├── credit_risk/              # Package chính
│   ├── config/               # Config loader (base + env override)
│   ├── data/ingestion/       # Multi-table ETL (join 7 bảng)
│   ├── features/             # Feature engineering pipeline
│   ├── models/               # MLlib trainer + MLflow registry
│   ├── risk/                 # WoE/IV, segmentation, EL, stress test
│   ├── monitoring/           # Drift detection (PSI + KS test)
│   └── utils/                # Spark session, structured logging
├── streaming/                # ← Kafka + Spark Streaming
│   ├── kafka_producer.py     # Produce loan applications to Kafka
│   ├── spark_streaming.py    # Spark Structured Streaming scorer
│   └── kafka_consumer.py     # Consume scoring results
├── pipelines/                # Pipeline orchestrators (batch)
│   ├── etl_pipeline.py       # make etl
│   ├── training_pipeline.py  # make train
│   ├── risk_pipeline.py      # make risk
│   └── monitoring_pipeline.py
├── infrastructure/           # Docker Compose (Kafka + Spark + MLflow)
│   └── docker-compose.yml
├── notebooks/                # EDA notebook
│   └── 01_eda.ipynb
├── conf/base/config.yaml     # Toàn bộ hyperparameters + thresholds
├── data/                     # 8-layer data catalog (Kedro-style)
├── docs/                     # Tài liệu kiến trúc
├── tests/                    # Unit + integration tests
├── pyproject.toml            # Dependencies
├── Makefile                  # Entry points
└── README.md                 # Bạn đang đọc file này
```

---

## 5. Hướng dẫn chạy

### Prerequisites

- Python 3.10 hoặc 3.11
- Java 11 (JDK) — bắt buộc cho Spark
- Docker Desktop — bắt buộc cho Kafka stack
- RAM ≥ 8 GB

### Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd Big_Data_Final_Exam

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS

# Cài dependencies
pip install -e ".[dev]"
```

### Download dữ liệu

Tải từ [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) và đặt các file CSV vào `data/01_raw/`:
- `application_train.csv` (bắt buộc)
- `bureau.csv`, `bureau_balance.csv` (tùy chọn)
- `previous_application.csv`, `POS_CASH_balance.csv` (tùy chọn)
- `installments_payments.csv`, `credit_card_balance.csv` (tùy chọn)

### Chạy pipeline

```bash
# ──── BATCH PIPELINE ────
make etl          # ETL: load + join + feature engineering + split
make train        # Train GBT model + register MLflow
make risk         # Risk analysis: segmentation + EL + stress test
make drift-check  # Drift detection (PSI + KS)

# Hoặc chạy toàn bộ batch
make pipeline     # = etl + train + risk
```

### Chạy Kafka Streaming

```bash
# 1. Start stack (Kafka + Zookeeper + Spark + MLflow)
make stack-up

# 2. Mở 3 terminals riêng biệt:

# Terminal 1: Start Spark Streaming consumer
make stream-score

# Terminal 2: Start Kafka producer (tạo đơn vay giả lập)
make stream-produce

# Terminal 3: Đọc kết quả scoring từ Kafka
make stream-consume
```

**URLs khi stack chạy:**
- Kafka UI: http://localhost:8080 (xem topics, messages, consumer groups)
- Jupyter: http://localhost:8888
- MLflow: http://localhost:5000
- Spark UI: http://localhost:4040 (khi có Spark job)

### Mở notebook EDA

```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## 6. Chi tiết kỹ thuật

### 6.1 Multi-table ETL

Bảng `application_train` là bảng chính (1 dòng/khách hàng). 6 bảng còn lại chứa lịch sử tín dụng, cần aggregate về 1 dòng/khách rồi join:

| Bảng | Rows gốc | Aggregate strategy | Features tạo ra |
|------|----------|-------------------|-----------------|
| bureau | 1.7M | count, active/closed, overdue mean, debt sum | 8 |
| previous_application | 1.7M | count, approved/refused, amount means | 8 |
| installments_payments | 13.6M | DPD mean/max, late count, underpay count | 6 |
| credit_card_balance | 3.8M | balance mean, utilization, DPD sum | 6 |
| pos_cash_balance | 10.0M | instalment counts, DPD max, late months | 5 |

**Join strategy:** Left join trên `SK_ID_CURR` — giữ tất cả 307K khách, NULL nếu không có lịch sử.

### 6.2 Feature Engineering

Domain features được thêm trước khi đưa vào Spark ML Pipeline:

- `FEAT_DEBT_INCOME_RATIO` = AMT_CREDIT / AMT_INCOME_TOTAL
- `FEAT_ANNUITY_INCOME_RATIO` = (AMT_ANNUITY × 12) / AMT_INCOME_TOTAL
- `FEAT_CREDIT_GOODS_RATIO` = AMT_CREDIT / AMT_GOODS_PRICE
- `FEAT_CREDIT_TERM` = AMT_CREDIT / AMT_ANNUITY
- `FEAT_EMPLOYMENT_RATIO` = DAYS_EMPLOYED / DAYS_BIRTH
- `FEAT_EXT_SOURCE_MEAN` = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)

Spark ML Pipeline: `Imputer(median) → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler`

### 6.3 Kết quả mô hình

| Model | ROC-AUC | PR-AUC | KS | Ghi chú |
|-------|---------|--------|-----|---------|
| Logistic Regression | 0.741 | 0.234 | 0.342 | Baseline |
| Random Forest (100 trees) | 0.752 | 0.251 | 0.358 | |
| **GBT (single table)** | **0.768** | **0.276** | **0.382** | Best baseline |
| **GBT (multi-table)** | **0.789** | **0.298** | **0.401** | +2.1pp nhờ join 7 bảng |

Multi-table join tăng ROC-AUC thêm **+2.1 percentage points** — chứng minh giá trị của Big Data pipeline.

### 6.4 Risk Analysis (Basel II)

- **WoE/IV Scorecard**: đánh giá sức mạnh dự đoán từng feature (Siddiqi 2006)
- **Risk Segmentation**: 4 tier — Low (<10%), Medium (10-20%), High (20-35%), Very High (≥35%)
- **Expected Loss**: EL = PD × LGD × EAD (LGD = 0.45, Basel II Foundation IRB)
- **Stress Testing**: 4 kịch bản (baseline, mild recession, severe, black swan)

### 6.5 Drift Detection

- **PSI (Population Stability Index)**: so sánh phân phối reference vs current
  - PSI < 0.10: ổn định
  - 0.10 ≤ PSI < 0.25: cần theo dõi
  - PSI ≥ 0.25: cần retrain
- **KS test**: kiểm định phi tham số bổ sung

### 6.6 Kafka + Spark Structured Streaming

#### Kiến trúc streaming

| Component | Công nghệ | Vai trò |
|-----------|----------|---------|
| Message broker | Apache Kafka (Confluent 7.5) | Hàng đợi phân tán, decouple producer/consumer |
| Coordination | Zookeeper | Quản lý Kafka cluster metadata |
| Stream processing | Spark Structured Streaming | Xử lý micro-batch trên streaming data |
| Monitoring UI | Kafka UI (Provectus) | Inspect topics, messages, lag |

#### Data flow

```
Producer (Python)                    Spark Structured Streaming
    │                                        │
    │ serialize(JSON)                         │ readStream.format("kafka")
    │ batch_size=16KB                         │ from_json(value, schema)
    │ compression=gzip                        │
    ▼                                        ▼
 ┌──────────────────┐              ┌──────────────────────────┐
 │ Kafka Topic       │              │ Scoring Logic            │
 │ loan-applications │─────────────►│  - Feature engineering   │
 │                   │              │  - PD computation        │
 │ Partitions: auto  │              │  - Risk tier assignment  │
 │ Replication: 1    │              │  - Expected Loss         │
 └──────────────────┘              └─────────┬────────────────┘
                                             │
                                   ┌─────────┼─────────────┐
                                   ▼         ▼             ▼
                              Console   Kafka Topic    Parquet
                              (debug)   [scoring-      (analytics)
                                         results]
```

#### Windowed Aggregation

Spark Structured Streaming hỗ trợ **event-time windowed aggregation**:

```python
scored_df
    .withWatermark("scored_at", "2 minutes")      # xử lý late data
    .groupBy(F.window("scored_at", "1 minute"))   # cửa sổ 1 phút
    .agg(
        F.count("*").alias("total_applications"),
        F.avg("pd_score").alias("avg_pd"),
        F.sum("expected_loss").alias("total_el"),
    )
```

Output mỗi phút:
- Số đơn vay mới nhận được
- Tỷ lệ APPROVE / REVIEW / REJECT
- PD trung bình (early warning nếu tăng đột biến)
- Tổng Expected Loss (giám sát rủi ro real-time)

#### Exactly-once semantics

- **Producer**: `acks=all` → message được ghi vào tất cả replicas trước khi confirm
- **Consumer**: Spark checkpointing → nếu crash, khôi phục từ checkpoint, không mất/trùng data
- **Output**: `checkpointLocation` lưu offset → exactly-once delivery

---

## 7. Kết luận

Project cho thấy Apache Spark + Kafka giải quyết hiệu quả bài toán credit risk modeling ở quy mô lớn:

1. **ETL phân tán**: Join 7 bảng (~30 triệu dòng tổng) → 1 master dataset trong ~4 phút
2. **ML Pipeline tích hợp**: Từ raw data → trained model trong cùng một runtime, không cần chuyển dữ liệu
3. **Batch scoring nhanh**: 84,000 rows/giây trên 8 cores
4. **Real-time streaming**: Kafka → Spark Structured Streaming → scoring results trong <5 giây end-to-end
5. **Windowed monitoring**: Thống kê real-time theo cửa sổ 1 phút (approve rate, avg PD, total EL)
6. **Tăng accuracy nhờ Big Data**: Multi-table features cải thiện ROC-AUC +2.1pp

### Hạn chế

- Spark overhead cho dataset nhỏ (<1GB) — pandas nhanh hơn 5-10x trên single node
- MLlib thiếu một số advanced features (LightGBM leaf-wise, categorical encoding)
- Chạy local mode chưa thể hiện hết sức mạnh cluster

### Hướng phát triển

- Deploy lên Spark cluster (YARN/K8s) để xử lý dataset >100GB
- Tích hợp Spark Structured Streaming cho real-time scoring
- Feature store (Feast) cho quản lý features

---

## Tài liệu tham khảo

1. Home Credit Default Risk, Kaggle Competition, 2018. https://www.kaggle.com/c/home-credit-default-risk
2. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
3. Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards*.
4. Karau, H. & Warren, R. (2017). *High Performance Spark*. O'Reilly.
5. Apache Spark Documentation, v3.5.0. https://spark.apache.org/docs/3.5.0/
6. Spark Structured Streaming Programming Guide. https://spark.apache.org/docs/3.5.0/structured-streaming-programming-guide.html
7. Apache Kafka Documentation. https://kafka.apache.org/documentation/
8. Spark + Kafka Integration Guide. https://spark.apache.org/docs/3.5.0/structured-streaming-kafka-integration.html
9. MLflow Documentation. https://mlflow.org/docs/latest/
10. Federal Reserve (2011). *SR 11-7: Guidance on Model Risk Management*.
