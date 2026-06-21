# Big Data Final Exam — Credit Risk Modeling on Apache Spark, Kafka & Hadoop

> Hệ thống xử lý đầu cuối phân tán (HDFS + Spark + Kafka + MLflow) cho bài toán mô hình hóa rủi ro tín dụng trên bộ dữ liệu **Home Credit Default Risk** (307,511 khách hàng, 7 bảng quan hệ, ~58.4 triệu hàng tổng cộng).

**Sinh viên:** Nguyễn Việt Phương
**MSSV:** 24022431

---

## Mục lục

1. [Bài toán và động lực Big Data](#1-bài-toán-và-động-lực-big-data)
2. [Bộ dữ liệu Home Credit](#2-bộ-dữ-liệu-home-credit)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Stack công nghệ](#4-stack-công-nghệ)
5. [Cấu trúc dự án](#5-cấu-trúc-dự-án)
6. [Quy trình ETL đa bảng](#6-quy-trình-etl-đa-bảng)
7. [Kỹ thuật đặc trưng và huấn luyện](#7-kỹ-thuật-đặc-trưng-và-huấn-luyện)
8. [Phân tích rủi ro Basel II](#8-phân-tích-rủi-ro-basel-ii)
9. [Hệ thống streaming Kafka + Spark](#9-hệ-thống-streaming-kafka--spark)
10. [Giám sát mô hình và drift detection](#10-giám-sát-mô-hình-và-drift-detection)
11. [Hướng dẫn chạy](#11-hướng-dẫn-chạy)
12. [Kết quả thực nghiệm](#12-kết-quả-thực-nghiệm)
13. [Hạn chế trung thực](#13-hạn-chế-trung-thực)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Bài toán và động lực Big Data

### Bài toán

Dự đoán **Xác suất Vỡ nợ (PD — Probability of Default)** của khách hàng vay từ nhiều nguồn dữ liệu quan hệ, sau đó định lượng rủi ro danh mục theo khung **Basel II** (Expected Loss, Unexpected Loss, vốn kinh tế).

### Vì sao cần Big Data?

Đây không phải là một bài toán Pandas-fit-in-memory:

| Yếu tố | Số liệu thực tế |
|--------|----------------|
| **Volume** | ~2.5 GB raw, **58.4 triệu hàng** ở các bảng trung gian trước khi gộp |
| **Variety** | 7 bảng quan hệ, **3 mức hạt khác nhau** (theo khách hàng / theo khoản vay / theo tháng) |
| **Velocity** | Đơn vay phát sinh liên tục → cần đường streaming song song với batch |
| **Veracity** | Dữ liệu phi dừng — tỉ lệ vỡ nợ trên các bộ dữ liệu tín dụng công khai biến động từ ~15% (2015) lên >23% (2024) [arXiv:2511.03807] |

Cụ thể:
- **58.4M hàng** không nằm gọn trong bộ nhớ đơn nút (4–8 GB).
- **Phép join 7 bảng ở 3 mức hạt khác nhau** đòi hỏi shuffle phân tán.
- **Spark cung cấp một runtime duy nhất** cho cả batch (huấn luyện) lẫn streaming (suy luận), tránh lệch giữa môi trường training và serving.

---

## 2. Bộ dữ liệu Home Credit

### Nguồn

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — cuộc thi Kaggle do **Home Credit Group** (tập đoàn tài chính tiêu dùng hoạt động tại nhiều thị trường mới nổi, trong đó có Việt Nam) tổ chức năm 2018.

### 7 bảng dữ liệu

Bảng chính là `application_train` — mỗi dòng là một đơn vay. Sáu bảng còn lại chứa **lịch sử tín dụng** ở các mức hạt khác nhau và liên kết qua khóa `SK_ID_CURR` (khách hàng) hoặc `SK_ID_PREV` (khoản vay trước).

| Bảng | Số hàng | Số cột | Mức hạt | Ý nghĩa |
|------|--------:|-------:|---------|---------|
| `application_train.csv` | 307,511 | 122 | 1 dòng / khách hàng | Đơn vay hiện tại — thông tin nhân khẩu, tài chính, tài sản, nguồn ngoài (EXT_SOURCE_1/2/3), **biến mục tiêu TARGET** |
| `bureau.csv` | 1,716,428 | 17 | 1 dòng / khoản vay từ bureau ngoài | Lịch sử tín dụng từ các tổ chức tín dụng khác (qua Credit Bureau) |
| `bureau_balance.csv` | 27,299,925 | 3 | 1 dòng / tháng / khoản vay bureau | Trạng thái tháng-tháng của mỗi khoản vay bureau (DPD bucket) |
| `previous_application.csv` | 1,670,214 | 37 | 1 dòng / khoản vay trước với Home Credit | Lịch sử đơn vay trước đó: Approved / Refused / Canceled |
| `installments_payments.csv` | 13,605,401 | 8 | 1 dòng / lần trả góp | Lịch sử trả nợ: ngày dự kiến, ngày thực tế, số tiền |
| `credit_card_balance.csv` | 3,840,312 | 23 | 1 dòng / tháng / thẻ tín dụng | Lịch sử thẻ tín dụng: dư nợ, hạn mức, DPD |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | 1 dòng / tháng / khoản POS | Lịch sử khoản vay POS (point-of-sale loan) |
| **Tổng** | **~58.4 triệu** | — | | |

### Biến mục tiêu

`TARGET ∈ {0, 1}` — 1 nếu khách hàng gặp khó khăn trả nợ (vỡ nợ kỹ thuật) trên ít nhất một trong các đợt trả đầu tiên.

**Phân phối (mất cân bằng):**
- `TARGET = 0` (không vỡ nợ): **282,686** khách (91.9%)
- `TARGET = 1` (vỡ nợ): **24,825** khách (8.1%)

⇒ Cần **cân trọng số lớp** (`classWeightCol`) khi huấn luyện để tránh mô hình "luôn dự đoán 0".

### Đặc trưng quan trọng nhất

Qua xếp hạng **Kruskal–Wallis** (chi tiết ở §10):

| Đặc trưng | H statistic | Diễn giải |
|-----------|------------:|-----------|
| `EXT_SOURCE_3` | 4,672.97 | Điểm bureau ngoài thứ 3 (Home Credit không công khai phương pháp tính) |
| `EXT_SOURCE_2` | 4,649.61 | Điểm bureau ngoài thứ 2 |
| `DAYS_BIRTH` | 1,371.48 | Tuổi khách (âm, tính bằng ngày so với ngày vay) |
| `DAYS_EMPLOYED` | 144.83 | Số ngày làm việc |
| `AMT_INCOME_TOTAL` | 72.98 | Thu nhập |
| `AMT_CREDIT` | 70.51 | Khoản tín dụng đề nghị |

Hai điểm `EXT_SOURCE` là chỉ báo mạnh nhất — phù hợp với hiểu biết nghiệp vụ: lịch sử tín dụng từ các bureau bên thứ ba thường là tín hiệu mạnh hơn các đặc trưng nhân khẩu học.

---

## 3. Kiến trúc hệ thống

Hai đường xử lý song song, **chia sẻ chung lõi nghiệp vụ** (`credit_risk/inference/realtime.py`) để bảo đảm không lệch giữa huấn luyện và suy luận:

```
═══════════════════════════════════════════════════════════════════
                    BATCH PIPELINE (huấn luyện + phân tích)
═══════════════════════════════════════════════════════════════════

  HDFS (raw)              Spark Batch                    MLflow Tracking
  ┌─────────┐         ┌──────────────────┐              ┌────────────┐
  │ 7 CSV   │────────►│ ETL: join 7 bảng │              │ Params     │
  │ ~2.5 GB │         │ → master parquet │              │ Metrics    │
  └─────────┘         │ → feature pipe   │─────────────►│ Artifacts  │
                      │ → GBT training   │              │ Registry   │
                      └────────┬─────────┘              └────────────┘
                               │
                               ▼
                      ┌──────────────────┐              ┌────────────┐
                      │ Risk analysis    │─────────────►│ JSON       │
                      │ - Tier (4)       │              │ reports    │
                      │ - EL/UL/Capital  │              │            │
                      │ - Stress test    │              └────────────┘
                      │ Monitoring       │
                      │ - PSI + KS       │
                      │ - Jensen-Shannon │
                      │ - Kruskal-Wallis │
                      │ Retraining dec.  │
                      └──────────────────┘

═══════════════════════════════════════════════════════════════════
                    STREAMING PIPELINE (suy luận real-time)
═══════════════════════════════════════════════════════════════════

  Kafka Producer          Kafka                Spark Structured Streaming
  ┌─────────────┐    ┌──────────────┐         ┌──────────────────────┐
  │ Loan apps   │───►│ Topic:       │────────►│ readStream.kafka     │
  │ (JSON)      │    │ loan-        │         │ from_json + schema   │
  │ acks=all    │    │ applications │         │                      │
  │ gzip        │    │              │         │ ── shared core ──    │
  └─────────────┘    └──────────────┘         │ assign_tier()        │
                                              │ make_decision()      │
                                              │ expected_loss()      │
                                              │                      │
                                              │ Window 1 min:        │
                                              │ - count, PD avg      │
                                              │ - approve/reject     │
                                              │ - total EL           │
                                              └──────────┬───────────┘
                                                         │
                                              ┌──────────┼──────────┐
                                              ▼          ▼          ▼
                                         ┌─────────┐ ┌──────┐ ┌─────────┐
                                         │ Kafka   │ │Cons. │ │ Parquet │
                                         │ scoring-│ │ ole  │ │ sink    │
                                         │ results │ │      │ │         │
                                         └─────────┘ └──────┘ └─────────┘
```

### Lõi nghiệp vụ dùng chung

Module `credit_risk/inference/realtime.py` cung cấp:

- `assign_tier(pd_score)` → phân tầng rủi ro {Low / Medium / High / Very High}
- `make_decision(pd_score, threshold)` → APPROVE / REVIEW / REJECT
- `expected_loss(pd, lgd, ead)` → EL = PD × LGD × EAD

Cả pipeline batch và streaming **import cùng module này**, không sao chép code. Đây là một thiết kế có chủ đích để **tránh training–serving skew** ngay từ kiến trúc.

---

## 4. Stack công nghệ

| Tầng | Công nghệ | Lý do lựa chọn |
|------|-----------|----------------|
| **Lưu trữ phân tán** | HDFS 3.3.6 | Sao chép cấp khối (RF=3), Spark đọc song song theo định vị khối, hỗ trợ Parquet partitioning |
| **Quản lý tài nguyên** | YARN | Bộ lập lịch chuẩn của Hadoop ecosystem |
| **Tính toán theo lô** | Apache Spark 3.5 + PySpark | Vector hóa, join đa bảng native, một API duy nhất cho ETL + ML + streaming |
| **Học máy** | Spark MLlib (GBTClassifier) | Huấn luyện phân tán ngay trong Spark, không phải xuất dữ liệu ra ngoài |
| **Theo dõi thí nghiệm** | MLflow 2.9 + backend SQLite | Lưu params/metrics/artifacts, model registry, gọn nhẹ phù hợp môi trường khóa luận |
| **Hàng đợi tin nhắn** | Apache Kafka 3.5 | Hàng đợi sự kiện có thể phát lại, chuẩn cho suy luận theo sự kiện |
| **Xử lý streaming** | Spark Structured Streaming | Dùng lại code path của Spark, checkpoint đảm bảo exactly-once |
| **Điều phối** | Make + Docker Compose | Môi trường dev tái lập được; production thực sẽ dùng Airflow / Argo |
| **Ngôn ngữ** | Python 3.12 | Hỗ trợ đầy đủ PySpark, scipy, mlflow |

### Tối ưu Spark đã áp dụng

| Tối ưu | Tác dụng | Trạng thái đo lường |
|--------|----------|---------------------|
| **Adaptive Query Execution (AQE)** | Tự gộp phân vùng shuffle, xử lý skew tự động | Bật mặc định; **chưa benchmark định lượng** |
| **Kryo serializer** | Serialize nhanh hơn Java mặc định | Đã cấu hình; **chưa benchmark định lượng** |
| **PyArrow integration** | Tăng tốc Spark ↔ Pandas khi materialise | Dùng trong tính Kruskal–Wallis |
| **Snappy compression** | Giảm I/O đĩa và HDFS, vẫn splittable | Mặc định cho Parquet output |

> ⚠️ **Trung thực:** AQE và Kryo được bật ở cấu hình, nhưng **chưa có benchmark trước/sau** để chứng minh độ tăng tốc cụ thể trên dataset này. Đây là một hạn chế đã ghi nhận (§13).

### Hiệu chỉnh bộ nhớ cho VM 3.8 GB

```python
.config("spark.driver.memory", "2g")
.config("spark.executor.memory", "1g")
```

Cấp phát thêm **2 GB swap** trên VM để hệ điều hành không OOM-kill Spark khi join nặng nhất. Đây là cách thích nghi với môi trường thi cử có giới hạn tài nguyên, **không phải khuyến nghị production**.

---

## 5. Cấu trúc dự án

Áp dụng theo phong cách **8-layer data catalog của Kedro** cho thư mục `data/`, kết hợp với package Python chuẩn:

```
Big_Data_Final_Exam/
├── credit_risk/                    # Package chính (importable)
│   ├── config/                     # Config loader (base.yaml + env override)
│   ├── data/
│   │   └── ingestion/              # Multi-table ETL — join 7 bảng
│   ├── features/                   # Feature engineering (domain + Spark ML pipe)
│   ├── models/
│   │   ├── trainer.py              # GBT trainer với class weighting
│   │   └── registry.py             # MLflow registration
│   ├── risk/
│   │   ├── segmentation.py         # Phân tầng 4 tier
│   │   ├── expected_loss.py        # EL = PD × LGD × EAD
│   │   ├── basel_capital.py        # UL, vốn kinh tế
│   │   └── stress_test.py          # 4 kịch bản vĩ mô
│   ├── monitoring/
│   │   ├── psi.py                  # Population Stability Index
│   │   ├── ks_test.py              # Kolmogorov–Smirnov
│   │   ├── jensen_shannon.py       # JS divergence cho hạng mục
│   │   └── retraining_policy.py    # Quyết định tái huấn luyện theo drift
│   ├── feature_selection/
│   │   └── kruskal_wallis.py       # Xếp hạng phi tham số
│   ├── inference/
│   │   └── realtime.py             # ⭐ LÕI dùng chung batch + streaming
│   └── utils/
│       ├── spark_session.py        # Builder với HDFS + Kryo + AQE
│       └── logging.py              # Structured logging
│
├── streaming/                      # Kafka + Spark Streaming
│   ├── kafka_producer.py           # Phát đơn vay JSON (random hoặc replay)
│   ├── spark_streaming.py          # Scorer + windowed aggregation
│   └── kafka_consumer.py           # Đọc scoring-results topic
│
├── pipelines/                      # Pipeline orchestrators (batch)
│   ├── etl_pipeline.py             # Load → join → split → parquet
│   ├── training_pipeline.py        # Feature pipe → GBT → MLflow
│   ├── risk_pipeline.py            # Tier + EL + Basel capital + stress
│   ├── monitoring_pipeline.py      # PSI + KS trên đặc trưng số
│   ├── feature_selection_pipeline.py     # Kruskal–Wallis ranking
│   └── categorical_drift_pipeline.py     # JS divergence + new-category detection
│
├── infrastructure/
│   ├── docker-compose.yml          # Kafka + Zookeeper + Kafka UI + MLflow
│   └── hadoop/                     # HDFS pseudo-distributed config
│
├── notebooks/
│   └── EDA_Home_Credit_Default_Risk.ipynb   # EDA 13 cell trên Colab (8 GB driver)
│
├── conf/base/
│   └── config.yaml                 # ⭐ Toàn bộ hyperparams + ngưỡng + HDFS paths
│
├── data/                           # 8-layer data catalog
│   ├── 01_raw/                     # CSV gốc từ Kaggle
│   ├── 02_intermediate/            # (HDFS) parquet sau khi cleaned
│   ├── 03_primary/                 # (HDFS) master sau join
│   ├── 04_feature/                 # (HDFS) features sau Spark ML pipe
│   ├── 05_model_input/             # Train/val/test split
│   ├── 06_models/                  # MLflow artifacts (gắn registry)
│   ├── 07_model_output/            # PD scores, predictions
│   └── 08_reporting/               # ⭐ Báo cáo JSON cuối cùng:
│                                   #   - portfolio_risk_report.json
│                                   #   - stress_test_report.json
│                                   #   - drift_report.json
│                                   #   - retraining_decision.json
│
├── docs/                           # Kiến trúc, ADR, ghi chú demo
├── tests/                          # Unit + integration tests
│
├── pyproject.toml                  # Dependencies (PEP 621)
├── Makefile                        # Entry points (make etl / train / risk / ...)
├── BigData_Technical_Report.docx   # Báo cáo kỹ thuật ~21 trang
└── README.md                       # File bạn đang đọc
```

### Quy ước data catalog

| Tầng | Nội dung | Lưu ở |
|------|----------|-------|
| `01_raw` | CSV gốc, **không bao giờ sửa** | Local (sau đó upload HDFS) |
| `02_intermediate` | Parquet đã chuẩn hóa dtype, NA chuẩn | HDFS |
| `03_primary` | Master dataset sau join 7 bảng | HDFS |
| `04_feature` | Sau Imputer + OHE + Scaler + VectorAssembler | HDFS |
| `05_model_input` | 3 split train/val/test với hash deterministic | HDFS |
| `06_models` | MLflow run artifacts (logged via tracking URI) | SQLite + filesystem |
| `07_model_output` | PD score cho toàn danh mục | HDFS |
| `08_reporting` | JSON báo cáo cho human review | Local repo (commit) |

> ℹ️ Trên môi trường Windows hiện tại, `data/02_intermediate` và `data/04_feature` chỉ chứa `.gitkeep` vì các tầng này được vật chất hóa trên HDFS chứ không phải local FS.

---

## 6. Quy trình ETL đa bảng

### Chiến lược

Bảng chính `application_train` là **bảng grain khách hàng** (1 dòng/khách). Sáu bảng còn lại ở grain khác cần **aggregate về 1 dòng/khách** trước khi join.

### Mẫu gộp (PySpark)

```python
df_child.groupBy("SK_ID_CURR") \
    .agg(
        F.count("*").alias("CHILD_COUNT"),
        F.avg("AMT_OVERDUE").alias("CHILD_AVG_OVERDUE"),
        F.sum("AMT_DEBT").alias("CHILD_SUM_DEBT"),
        F.max("DAYS_LATE").alias("CHILD_MAX_DPD"),
    )
```

### Chiến lược aggregate cho từng bảng

| Bảng con | Hàng gốc | Aggregate strategy | Đặc trưng tạo ra |
|----------|---------:|--------------------|------------------:|
| `bureau` | 1.7M | count, count active vs closed, mean overdue, sum debt | **4** |
| `previous_application` | 1.7M | count, count approved vs refused, mean amount/term | **8** |
| `installments_payments` | 13.6M | mean/max DPD, count late, count underpay | **6** |
| `credit_card_balance` | 3.8M | mean balance, mean utilisation, sum DPD | **6** |
| `pos_cash_balance` | 10.0M | count, max DPD, late-month count | **5** |

### Join

Tất cả là **LEFT JOIN trên `SK_ID_CURR`** — giữ lại các khách hàng không có lịch sử (sẽ là NULL, được xử lý ở bước Imputer).

**Kết quả đã xác minh trên HDFS:**

| Bước | Kết quả |
|------|---------|
| Input | 7 CSV, ~58.4M hàng tổng |
| `application_train` ban đầu | 307,511 × 122 |
| Sau join `bureau` aggregation | **307,511 × 126** (+4 đặc trưng) |
| Sau toàn bộ join + domain features | **307,511 × ~200** cột |

---

## 7. Kỹ thuật đặc trưng và huấn luyện

### 7.1 Domain features (6 đặc trưng nghiệp vụ)

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `FEAT_DEBT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Tỉ lệ nợ trên thu nhập |
| `FEAT_ANNUITY_INCOME_RATIO` | (AMT_ANNUITY × 12) / AMT_INCOME_TOTAL | Gánh nặng trả nợ hàng năm |
| `FEAT_CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Đòn bẩy so với tài sản |
| `FEAT_CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Kỳ hạn vay suy ra |
| `FEAT_EMPLOYMENT_RATIO` | DAYS_EMPLOYED / DAYS_BIRTH | Tỉ lệ thời gian có việc làm trên tuổi đời |
| `FEAT_EXT_SOURCE_MEAN` | mean(EXT_SOURCE_1, 2, 3) | Điểm bureau ngoài trung bình |

### 7.2 Spark ML Pipeline

```
Imputer (median số, mode hạng mục)
  ↓
StringIndexer (cho cột hạng mục)
  ↓
OneHotEncoder
  ↓
VectorAssembler
  ↓
StandardScaler
```

Pipeline **fit trên tập huấn luyện** và áp dụng nguyên xi cho validation, test, **và dữ liệu streaming** → đảm bảo nhất quán huấn luyện – suy luận.

### 7.3 Huấn luyện GBT

**Lý do chọn GBT (Gradient Boosted Trees) trong Spark MLlib:**
- ✅ Có sẵn trong MLlib → không phải xuất dữ liệu ra ngoài Spark
- ✅ Huấn luyện phân tán trên nhiều executor
- ✅ Bắt được các tương tác phi tuyến trong đặc trưng tín dụng
- ✅ Hỗ trợ cân trọng số lớp (`weightCol`)

### 7.4 Kết quả (verified từ MLflow)

**Một lần chạy duy nhất** trên thí nghiệm `credit_risk_platform / gbt_baseline`:

| Chỉ số | Giá trị | Ý nghĩa nghiệp vụ |
|--------|--------:|-------------------|
| **ROC-AUC** | **0.7663** | Phân tách mạnh giữa nhóm vỡ nợ và không vỡ nợ |
| PR-AUC | 0.2461 | Phù hợp với base rate 8% — cao hơn random 0.08 đáng kể |
| **KS** | **0.4016** | Vượt ngưỡng **0.30** — chuẩn industry cho scorecard production |
| Accuracy | 0.9178 | Lệch do imbalance — không phải metric chính |
| F1 (weighted) | 0.8830 | — |
| Precision | 0.8839 | — |
| Recall | 0.9178 | — |
| **PD trung bình** | **8.29%** | ≈ tỉ lệ vỡ nợ thực tế **8.2%** — **hiệu chỉnh tốt** (calibration gap < 0.1pp) |

> ℹ️ **Trung thực:** Đây là **kết quả của một lần chạy**, **chưa có cross-validation** để đo phương sai. Cũng **không có ablation study** so sánh "chỉ application_train" vs "đầy đủ 7 bảng" — đây là hạn chế đã ghi nhận (§13).

---

## 8. Phân tích rủi ro Basel II

### 8.1 Tổng quan danh mục (45,859 khách hàng đã chấm điểm)

Sau khi áp dụng mô hình lên tập test và áp dụng module `risk/`:

| Chỉ số | Giá trị |
|--------|--------:|
| Tổng khách hàng chấm điểm | **45,859** |
| Tổng EAD (Exposure at Default) | 27,407.37 M |
| Tổng EL (Expected Loss) | **950.72 M** |
| Tỉ lệ EL trên danh mục | 3.47% (trung bình) |
| PD trung bình | 8.29% |
| LGD (Basel II FIRB cố định) | 45.0% |
| Tỉ lệ vỡ nợ thực tế | 8.2% |

### 8.2 Vốn kinh tế (Basel II)

| Đại lượng | Triệu |
|-----------|------:|
| Tổn thất Kỳ vọng (EL) | 950.72 |
| Tổn thất Bất thường (UL) | 2,852.17 |
| **Vốn kinh tế ước tính** | **4,278.25** |

Công thức: **EL = PD × LGD × EAD** (PD từ GBT, LGD = 0.45 cố định theo Basel II Foundation IRB, EAD = `AMT_CREDIT`).

### 8.3 Phân tầng 4 tier

Bằng chứng rằng mô hình **phân tách thực sự**, không phải chỉ có ROC-AUC đẹp trên giấy:

| Tầng | Số KH | % DM | EAD (M) | EL (M) | EL rate | PD TB |
|------|------:|-----:|--------:|-------:|--------:|------:|
| 🟢 Thấp | 34,822 | 75.93% | 21,629.76 | 455.88 | 2.11% | 4.78% |
| 🟠 Trung bình | 7,465 | 16.28% | 3,987.75 | 247.16 | 6.20% | 13.82% |
| 🔴 Cao | 2,630 | 5.73% | 1,325.93 | 153.26 | 11.56% | 25.69% |
| ⚫ **Rất cao** | **942** | **2.05%** | **463.93** | **94.42** | **20.35%** | **45.49%** |

**Nhận xét chính:**
- Đuôi mỏng (2.05% danh mục) nhưng tỉ lệ EL gấp **9.6 lần** tầng Thấp (20.35% vs 2.11%)
- Quy tắc Basel: đuôi tiêu thụ hầu hết vốn kinh tế dù chỉ chiếm <5% danh mục — đúng như quan sát thực tế

### 8.4 Stress test vĩ mô

Bốn kịch bản nhân tử áp lên PD và LGD:

| Kịch bản | PD× | LGD× | Tổng EL (M) | EL rate | Vốn (M) | KH rủi ro cao |
|----------|----:|-----:|------------:|--------:|--------:|--------------:|
| Cơ sở (baseline) | 1.0 | 1.0 | 950.72 | 3.47% | 2,852.17 | 3,572 |
| Suy thoái nhẹ | 1.5 | 1.1 | 1,567.68 | 5.72% | 4,703.05 | 7,255 |
| Khủng hoảng 2008 | 2.5 | 1.3 | 3,047.41 | 11.12% | 9,142.22 | 14,799 |
| **Thiên nga đen** | **4.0** | **1.5** | **5,376.49** | **19.62%** | **16,129.48** | **24,623** |

**Hai con số đáng nhớ:**
- Thiên nga đen: EL tăng **5.66×** (từ 950.72 M → 5,376.49 M)
- Số KH rủi ro cao tăng **~7×** (3,572 → 24,623)

> Ước lượng PD tại thời điểm gốc, dù hiệu chỉnh tốt đến đâu, vẫn không đủ nếu thiếu một tầng kiểm tra căng thẳng đánh giá lại sổ sách dưới các nhân tố bất lợi.

---

## 9. Hệ thống streaming Kafka + Spark

### 9.1 Kiến trúc

```
Producer            Kafka                Spark Structured Streaming      Sinks
┌────────┐    ┌──────────────────┐      ┌────────────────────────┐    ┌──────────┐
│ Python │───►│ loan-applications │─────►│ readStream             │───►│ Kafka    │
│ JSON   │    │                  │      │ from_json + schema     │    │ scoring- │
│ acks=  │    │ - 1 partition    │      │ ─── shared core ───    │    │ results  │
│  all   │    │ - RF=1 (dev)     │      │ assign_tier()          │    └──────────┘
│ gzip   │    │ - retention=7d   │      │ make_decision()        │    ┌──────────┐
└────────┘    └──────────────────┘      │ expected_loss()        │───►│ Console  │
                                        │                        │    │ (debug)  │
                                        │ Window 1 min:          │    └──────────┘
                                        │ - count, PD avg        │    ┌──────────┐
                                        │ - approve/reject       │───►│ Parquet  │
                                        │ - total EL             │    │ (HDFS)   │
                                        │ watermark=2min         │    └──────────┘
                                        └────────────────────────┘
```

### 9.2 Cam kết exactly-once

| Tầng | Cơ chế |
|------|--------|
| Producer | `acks=all` + replication → message ghi vào tất cả replicas trước khi confirm |
| Spark | `checkpointLocation` lưu offset → nếu crash, khôi phục từ checkpoint |
| Sink | Ghi topic idempotent → không trùng lặp khi reprocess |

### 9.3 Cấu hình Kafka

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `acks` | `all` | Đảm bảo durability |
| `compression.type` | `gzip` | Giảm **60–70%** băng thông |
| `batch.size` | 16 KB | Batch messages tăng throughput |
| `linger.ms` | 10 ms | Cho phép batch tích lũy nhẹ |
| `startingOffsets` | `latest` | Chỉ xử lý message mới |

### 9.4 Windowed aggregation

```python
scored_df \
    .withWatermark("scored_at", "2 minutes") \
    .groupBy(F.window("scored_at", "1 minute")) \
    .agg(
        F.count("*").alias("total_applications"),
        F.sum(F.when(F.col("decision") == "APPROVE", 1)).alias("approved"),
        F.sum(F.when(F.col("decision") == "REVIEW", 1)).alias("reviewed"),
        F.sum(F.when(F.col("decision") == "REJECT", 1)).alias("rejected"),
        F.avg("pd_score").alias("avg_pd"),
        F.sum("expected_loss").alias("total_el"),
    )
```

### 9.5 Kết quả đã xác minh (cửa sổ 1 phút)

Demo chạy thực tế với 30 đơn vay phát sinh trong 1 phút:

| Chỉ số | Giá trị |
|--------|--------:|
| Tổng đơn vay | 30 |
| 🟢 Duyệt (APPROVE) | 28 |
| 🟠 Xem xét (REVIEW) | 2 |
| 🔴 Từ chối (REJECT) | 0 |
| PD trung bình | 0.0927 |
| Tổng EL | 1.56 M |
| Khoản vay trung bình | 1.15 M |

---

## 10. Giám sát mô hình và drift detection

Tầng giám sát **không chỉ là PSI** — kết hợp **bốn phương pháp** bổ sung lẫn nhau, mỗi cái cho một loại drift khác nhau:

### 10.1 PSI + KS cho đặc trưng số

**Population Stability Index (PSI):** so sánh phân phối reference (train) vs current (test/production) bằng cách binning theo phân vị của phân phối tham chiếu.

**Ngưỡng Siddiqi (2006):**
- PSI < 0.10 → ổn định
- 0.10 ≤ PSI < 0.25 → cần theo dõi
- PSI ≥ 0.25 → cần tái huấn luyện

**Kolmogorov–Smirnov (KS) hai mẫu:** kiểm định phi tham số xem hai phân phối có khác nhau có ý nghĩa thống kê không. Bổ sung cho PSI để bắt drift dạng "thay đổi đuôi" mà binning có thể bỏ sót.

### 10.2 Jensen–Shannon divergence cho hạng mục

PSI hoạt động kém với cột hạng mục có **giá trị mới xuất hiện** trong production (không có trong reference). **JS divergence** giải quyết vấn đề này:

- Đối xứng, chặn trong [0, 1]
- Hiểu được như khoảng cách giữa hai phân phối xác suất
- Phát hiện thêm **hạng mục mới** — chỉ báo sớm taxonomy upstream thay đổi

**Đầu ra đã xác minh:**

```
NAME_CONTRACT_TYPE   JS=0.0019  [none]
CODE_GENDER          JS=0.0044  [none]
NAME_FAMILY_STATUS   JS=0.0047  [none]  new=['Unknown'] (!)
ORGANIZATION_TYPE    JS=0.0089  [none]
WALLSMATERIAL_MODE   JS=0.0107  [none]
```

⚠️ **Phát hiện đáng chú ý:** Tập test có hạng mục mới `'Unknown'` ở `NAME_FAMILY_STATUS` — drift mà PSI số **không phát hiện được**. Đây chính xác là lý do cần JS bổ sung.

*Nguồn lý thuyết: arXiv:2511.03807 — "Fair and Explainable Credit-Scoring under Concept Drift".*

### 10.3 Kruskal–Wallis cho tuyển chọn đặc trưng

Xếp hạng phi tham số sức mạnh dự đoán của từng đặc trưng số:

- **Không giả định tính chuẩn hay đồng phương sai** — phù hợp với đặc trưng tài chính lệch (như `AMT_CREDIT`, `DAYS_EMPLOYED`)
- Xếp hạng theo khoảng cách rank-sum giữa nhóm `TARGET=0` và `TARGET=1`
- Tất cả 6 đặc trưng top có **p-value < 1e-16** ⇒ ý nghĩa thống kê

Hai điểm `EXT_SOURCE` đứng đầu cả ở correlation (Cell 9 EDA) lẫn ở Kruskal–Wallis → **xác nhận chéo trên hai phương pháp**.

*Nguồn: Ashofteh & Bravo (2021) DOI:10.24433/CO.1963899.v1.*

### 10.4 Chính sách tái huấn luyện theo drift

Thay vì tái huấn luyện theo lịch cố định (cứ tháng/quý retrain), pipeline áp dụng **chính sách dựa trên drift signal**:

| Điều kiện | Hành động | Mức độ |
|-----------|-----------|--------|
| Bất kỳ đặc trưng nào có PSI ≥ 0.25 | `RETRAIN_IMMEDIATE` | Cao |
| Có ≥ 3 đặc trưng với PSI ∈ [0.10, 0.25) | `RETRAIN_SCHEDULED` | Trung bình |
| Có 1–2 đặc trưng với PSI ∈ [0.10, 0.25) | `CONTINUE_MONITORING` (watchlist) | Thấp |
| Tất cả PSI < 0.10 | `CONTINUE_MONITORING` | Không |

**Kết quả chạy thực tế trên VM** (lưu vào `data/08_reporting/retraining_decision.json`):

```
✓ RETRAINING DECISION: CONTINUE_MONITORING
  Severity: NONE
  Reason: No drift detected.
```

**Lợi ích:** tránh tái huấn luyện không cần thiết khi dữ liệu ổn định, kích hoạt nhanh khi thực sự có thay đổi phân phối.

*Nguồn lý thuyết: WJARR 2025 — "Machine learning for credit scoring and loan default prediction".*

---

## 11. Hướng dẫn chạy

### 11.1 Prerequisites

| Yêu cầu | Phiên bản | Ghi chú |
|---------|-----------|---------|
| Python | 3.10 hoặc 3.12 | PySpark 3.5 yêu cầu |
| Java JDK | 11 | Bắt buộc cho Spark |
| Hadoop | 3.3.6 | HDFS + YARN, pseudo-distributed |
| Docker | Desktop / Engine | Cho Kafka stack |
| RAM | ≥ 4 GB | Đã test trên VM 3.8 GB + 2 GB swap |

### 11.2 Cài đặt

```bash
# Clone
git clone https://github.com/muahahaha55/Big_Data_Final_Exam.git
cd Big_Data_Final_Exam

# Virtual environment
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# Dependencies
pip install -e ".[dev]"
```

### 11.3 Download dữ liệu

Từ [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data), tải các file CSV và đặt vào `data/01_raw/`:

```
data/01_raw/
├── application_train.csv         (bắt buộc)
├── application_test.csv          (cho inference demo)
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── credit_card_balance.csv
└── POS_CASH_balance.csv
```

### 11.4 Upload lên HDFS

```bash
# Khởi động Hadoop
start-dfs.sh && start-yarn.sh

# Upload raw data
hdfs dfs -mkdir -p /user/$USER/credit_risk/raw
hdfs dfs -put data/01_raw/*.csv /user/$USER/credit_risk/raw/

# Verify
hdfs dfs -ls /user/$USER/credit_risk/raw/
```

### 11.5 Pipeline batch

```bash
# Khởi động MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Chạy từng bước
python pipelines/etl_pipeline.py                    # Load + join + features
python pipelines/training_pipeline.py               # GBT + MLflow logging
python pipelines/risk_pipeline.py                   # Tier + EL + stress
python pipelines/monitoring_pipeline.py             # PSI + KS
python pipelines/feature_selection_pipeline.py      # Kruskal–Wallis
python pipelines/categorical_drift_pipeline.py      # JS divergence

# Hoặc qua Makefile
make pipeline
```

### 11.6 Pipeline streaming

```bash
# 1. Khởi động Kafka + Zookeeper qua Docker Compose
make stack-up

# 2. Tạo topic
kafka-topics.sh --create --topic loan-applications \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
kafka-topics.sh --create --topic scoring-results \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# 3. Terminal 1 — Spark Streaming scorer
python streaming/spark_streaming.py --output console --trigger 10

# 4. Terminal 2 — Kafka producer (phát đơn vay)
python streaming/kafka_producer.py --mode random --rate 3 --total 30

# 5. (Tùy chọn) Terminal 3 — Consumer đọc kết quả
python streaming/kafka_consumer.py --topic scoring-results
```

### 11.7 UIs khi stack chạy

| Service | URL | Vai trò |
|---------|-----|---------|
| HDFS Namenode | http://localhost:9870 | Xem block, file system |
| YARN ResourceManager | http://localhost:8088 | Xem job, container |
| Spark Application UI | http://localhost:4040 | Xem stages, executors, SQL plans |
| MLflow Tracking | http://localhost:5000 | Xem runs, metrics, registered models |
| Kafka UI (nếu enable) | http://localhost:8080 | Xem topics, messages, consumer groups |

> 💡 **Demo tip:** Trên VM Ubuntu, Firefox không ổn định. Dùng **CLI** (`hdfs dfs`, `kafka-console-consumer.sh`) thay cho web UI khi cần snapshot.

### 11.8 EDA notebook

Notebook EDA chạy trên Google Colab với driver memory 8 GB:

```bash
jupyter lab notebooks/EDA_Home_Credit_Default_Risk.ipynb
```

13 cell phân tích: tổng quan 7 bảng, phân phối TARGET, missing analysis, phân phối ĐT số theo TARGET, ma trận tương quan, tỉ lệ vỡ nợ theo hạng mục, demo join, Kruskal–Wallis ranking.

---

## 12. Kết quả thực nghiệm

### Bảng tổng hợp

| Hạng mục | Kết quả | Trạng thái |
|----------|---------|-----------|
| ETL: 7 bảng → master parquet | 307,511 × ~200 cột | ✅ Verified |
| ROC-AUC | **0.7663** | ✅ MLflow |
| KS | **0.4016** (vượt 0.30) | ✅ MLflow |
| Calibration (PD avg vs actual) | 8.29% vs 8.2% | ✅ Verified |
| Tổng EL danh mục | 950.72 M | ✅ JSON report |
| Vốn kinh tế Basel II | 4,278.25 M | ✅ JSON report |
| 4 stress scenarios | EL ×1.0 → ×5.66 | ✅ JSON report |
| Streaming throughput | 30 đơn / 1 phút window | ✅ Console verified |
| Drift decision | `CONTINUE_MONITORING` (no drift) | ✅ JSON report |
| Top feature Kruskal–Wallis | EXT_SOURCE_3 (H=4,672.97) | ✅ MLflow |

### Files output cần lưu ý

| File | Vai trò |
|------|---------|
| `data/08_reporting/portfolio_risk_report.json` | EL, vốn kinh tế, phân tầng 4 tier |
| `data/08_reporting/stress_test_report.json` | 4 kịch bản vĩ mô |
| `data/08_reporting/drift_report.json` | PSI + KS + JS theo từng đặc trưng |
| `data/08_reporting/retraining_decision.json` | Quyết định cuối cùng |
| `BigData_Technical_Report.docx` | Báo cáo kỹ thuật ~21 trang |

---

## 13. Hạn chế trung thực

Liệt kê tường minh là có chủ đích. Mỗi mục đều ánh xạ vào lộ trình mở rộng trong báo cáo kỹ thuật (§11).

### Ràng buộc kỹ thuật

- **Triển khai single-node pseudo-distributed** — chưa lên cụm thật. Đường xử lý sẽ scale theo lý thuyết nhưng chưa kiểm chứng trên cụm nhiều worker.
- **Chưa benchmark Spark optimisation định lượng** — AQE, Kryo, Snappy được bật ở cấu hình, nhưng chưa có số đo trước/sau để chứng minh độ tăng tốc cụ thể trên dataset này.
- **Spark MLlib GBT thường thua LightGBM / XGBoost 1–2 pp ROC-AUC** trên các benchmark Kaggle — đây là **đánh đổi có chủ ý** để giữ huấn luyện trong runtime Spark, tránh overhead chuyển dữ liệu.
- **Phát hiện drift offline** (so train.parquet vs test.parquet) — chưa có rolling window thực sự theo thời gian production.

### Ràng buộc nghiên cứu

- **Chưa có ablation study** để đo đóng góp cụ thể của đặc trưng đa bảng. Không thể tuyên bố "+X pp ROC-AUC nhờ join 7 bảng" mà chưa có run đối chứng.
- **Chỉ một lần chạy huấn luyện** — chưa có phương sai cross-validation. Mọi metric trong báo cáo là **point estimate**, không phải khoảng tin cậy.
- **Stress test dùng nhân tử cố định** (PD×1.5, 2.5, 4.0) — chưa có satellite model vĩ mô liên kết với GDP / unemployment thực tế.
- **Chưa kiểm định ý nghĩa thống kê** khi so sánh giữa các ngưỡng drift — quyết định tái huấn luyện hiện dựa trên ngưỡng Siddiqi rule-of-thumb chứ không phải bootstrap CI.

### Hướng phát triển

1. Deploy lên Spark cluster (YARN / K8s) để xác minh scale-out
2. Benchmark định lượng AQE / Kryo (warm/cold cache, các quy mô shuffle)
3. Ablation study: train mô hình "chỉ application_train" để đo Δ ROC-AUC từ join 7 bảng
4. K-fold cross-validation để có khoảng tin cậy cho mọi metric
5. Satellite macro model: PD × (a + b·ΔGDP + c·ΔUnemployment)
6. Rolling drift detection với cửa sổ trượt theo ngày / tuần
7. Feature store (Feast) cho quản lý đặc trưng

---

## 14. Tài liệu tham khảo

### Học thuật

1. **Ashofteh, A. & Bravo, J. M. (2021).** *A non-parametric-based computationally efficient approach for credit scoring.* Data Science for Financial Econometrics. DOI: [10.24433/CO.1963899.v1](https://doi.org/10.24433/CO.1963899.v1) — cơ sở Kruskal–Wallis feature selection và multi-state borrower transitions.

2. **"Fair and Explainable Credit-Scoring under Concept Drift."** arXiv:2511.03807 — cơ sở Jensen–Shannon cho categorical drift và quan sát PD biến động thời gian.

3. **Siddiqi, N. (2006).** *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley — ngưỡng PSI và phương pháp WoE/IV.

4. **Basel Committee on Banking Supervision (2006).** *International Convergence of Capital Measurement and Capital Standards (Basel II).* — công thức EL/UL, LGD Foundation IRB.

5. **Federal Reserve (2011).** *SR 11-7: Guidance on Model Risk Management.* — chuẩn quản lý rủi ro mô hình.

6. **WJARR (2025).** *Machine learning for credit scoring and loan default prediction.* — cơ sở chính sách tái huấn luyện theo drift.

### Công nghệ

7. **Home Credit Default Risk.** Kaggle Competition, 2018. <https://www.kaggle.com/c/home-credit-default-risk>
8. **Apache Spark Documentation v3.5.0.** <https://spark.apache.org/docs/3.5.0/>
9. **Spark Structured Streaming Programming Guide.** <https://spark.apache.org/docs/3.5.0/structured-streaming-programming-guide.html>
10. **Spark + Kafka Integration Guide.** <https://spark.apache.org/docs/3.5.0/structured-streaming-kafka-integration.html>
11. **Apache Kafka Documentation.** <https://kafka.apache.org/documentation/>
12. **MLflow Documentation.** <https://mlflow.org/docs/latest/>
13. **Hadoop HDFS Architecture.** <https://hadoop.apache.org/docs/r3.3.6/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html>
14. **Karau, H. & Warren, R. (2017).** *High Performance Spark.* O'Reilly.

---

**Tinh thần trung thực kỹ thuật:** Mọi con số trong README này đều đến từ một lần chạy thực tế của repo trên VM. Những điều chưa đo được (vd. Δ ROC-AUC ablation, benchmark AQE) đều được nêu rõ là **chưa đo**, không suy diễn.
