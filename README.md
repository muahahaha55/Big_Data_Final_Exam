# Mô hình hóa Rủi ro Tín dụng trên Apache Spark, Kafka & Hadoop

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?logo=apachespark&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-231F20?logo=apachekafka)
![Hadoop](https://img.shields.io/badge/Hadoop-3.3.6-66CCFF?logo=apachehadoop&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-2.22.5-0194E2?logo=mlflow&logoColor=white)

> Hệ thống rủi ro tín dụng phân tán đầu-cuối trên bộ dữ liệu **Home Credit Default Risk** (307.511 hồ sơ vay, 7 bảng quan hệ, ~58,4 triệu dòng): ETL đa bảng và huấn luyện GBT trên Spark/HDFS/YARN, phân tích rủi ro danh mục theo Basel II (tổn thất kỳ vọng, vốn yêu cầu theo IRB, stress testing vĩ mô), đường chấm điểm streaming Kafka + Spark Structured Streaming, và chính sách tái huấn luyện khi phát hiện drift (PSI + KS + Jensen–Shannon).

**Tác giả:** Nguyễn Việt Phương · Bài tập lớn học phần Big Data

---

## Mục lục

1. [Bài toán & Lý do dùng Big Data](#1-bài-toán--lý-do-dùng-big-data)
2. [Dữ liệu](#2-dữ-liệu)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Tech Stack](#4-tech-stack)
5. [Cấu trúc dự án](#5-cấu-trúc-dự-án)
6. [ETL đa bảng](#6-etl-đa-bảng)
7. [Xây dựng đặc trưng & Huấn luyện](#7-xây-dựng-đặc-trưng--huấn-luyện)
8. [Phân tích rủi ro Basel II](#8-phân-tích-rủi-ro-basel-ii)
9. [Kafka + Spark Streaming](#9-kafka--spark-streaming)
10. [Giám sát mô hình & Phát hiện drift](#10-giám-sát-mô-hình--phát-hiện-drift)
11. [Khởi động nhanh](#11-khởi-động-nhanh)
12. [Kết quả thực nghiệm](#12-kết-quả-thực-nghiệm)
13. [Hạn chế đã biết & Lộ trình](#13-hạn-chế-đã-biết--lộ-trình)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Bài toán & Lý do dùng Big Data

**Nhiệm vụ:** dự đoán **xác suất vỡ nợ (PD — Probability of Default)** cho từng hồ sơ vay từ nhiều nguồn dữ liệu quan hệ, sau đó lượng hóa rủi ro danh mục theo khung **Basel II** (tổn thất kỳ vọng, vốn yêu cầu theo hàm trọng số rủi ro IRB, stress testing).

Đây không phải bài toán "pandas vừa khít bộ nhớ":

| Chiều | Thực tế |
|-------|---------|
| **Volume** | ~2,5 GB raw, **58,4 triệu dòng** trên các bảng trung gian trước khi hợp nhất |
| **Variety** | 7 bảng quan hệ ở **3 cấp độ tổng hợp** khác nhau (theo khách hàng / theo khoản vay / theo tháng) |
| **Velocity** | Hồ sơ vay đến liên tục → cần đường streaming song hành với batch |
| **Veracity** | Dữ liệu không tĩnh — tỷ lệ vỡ nợ trên các bộ dữ liệu tín dụng công khai trôi từ ~15% (2015) lên >23% (2024) [arXiv:2511.03807] |

Cụ thể: 58,4 triệu dòng không vừa heap 4–8 GB của một nút đơn; join 7 bảng ở 3 cấp độ tổng hợp đòi hỏi shuffle phân tán; và Spark cung cấp **một runtime duy nhất cho cả huấn luyện batch lẫn suy luận streaming**, giảm rủi ro sai lệch huấn luyện–triển khai (training–serving divergence).

---

## 2. Dữ liệu

Nguồn: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle, 2018), do Home Credit Group công bố — công ty tài chính tiêu dùng hoạt động tại các thị trường mới nổi, trong đó có Việt Nam.

Bảng chính là `application_train` (mỗi dòng một hồ sơ vay). Sáu bảng phụ chứa lịch sử tín dụng ở các cấp độ tổng hợp khác nhau, liên kết qua `SK_ID_CURR` (khách hàng) hoặc `SK_ID_PREV` (khoản vay trước).

| Bảng | Số dòng | Số cột | Cấp độ | Nội dung |
|------|--------:|-------:|--------|----------|
| `application_train.csv` | 307.511 | 122 | khách hàng | Hồ sơ hiện tại — nhân khẩu học, tài chính, điểm bên ngoài (EXT_SOURCE_1/2/3), **nhãn TARGET** |
| `bureau.csv` | 1.716.428 | 17 | khoản vay ngoài | Lịch sử tín dụng tại tổ chức khác (qua Credit Bureau) |
| `bureau_balance.csv` | 27.299.925 | 3 | tháng × khoản vay bureau | Trạng thái hàng tháng (nhóm DPD) của từng khoản vay bureau |
| `previous_application.csv` | 1.670.214 | 37 | khoản vay HC trước | Hồ sơ trước đó: Approved / Refused / Canceled |
| `installments_payments.csv` | 13.605.401 | 8 | kỳ trả góp | Lịch sử trả nợ: ngày/số tiền đến hạn so với thực trả |
| `credit_card_balance.csv` | 3.840.312 | 23 | tháng × thẻ | Dư nợ thẻ tín dụng, hạn mức, DPD |
| `POS_CASH_balance.csv` | 10.001.358 | 8 | tháng × khoản POS | Lịch sử khoản vay tại điểm bán |
| **Tổng** | **~58,4M** | — | | |

**Nhãn:** `TARGET ∈ {0, 1}` — bằng 1 nếu khách hàng gặp khó khăn thanh toán ở các kỳ trả góp đầu.
Cân bằng lớp: 282.686 không vỡ nợ (91,9%) so với 24.825 vỡ nợ (8,1%) ⇒ accuracy không dùng được làm chỉ số chính (đoán toàn 0 đã cho ~92%), nên mọi đánh giá dựa trên **ROC-AUC / PR-AUC / KS**.

> **Lưu ý:** mô hình hiện **không** áp dụng class weighting. `conf/base/config.yaml` có khai `models.imbalance.method: class_weight`, nhưng tham số này **chưa được code nào đọc** và không classifier nào được truyền `weightCol` — xem §13.

**Các đặc trưng dự báo mạnh nhất** (xếp hạng Kruskal–Wallis, §10):

| Đặc trưng | Thống kê H | Diễn giải |
|-----------|-----------:|-----------|
| `EXT_SOURCE_3` | 4.672,97 | Điểm tín dụng bên ngoài thứ ba |
| `EXT_SOURCE_2` | 4.649,61 | Điểm tín dụng bên ngoài thứ hai |
| `DAYS_BIRTH` | 1.371,48 | Tuổi (số ngày âm) |
| `DAYS_EMPLOYED` | 144,83 | Số ngày đi làm |
| `AMT_INCOME_TOTAL` | 72,98 | Thu nhập |
| `AMT_CREDIT` | 70,51 | Số tiền vay đề nghị |

Hai điểm `EXT_SOURCE` áp đảo — nhất quán với trực giác nghiệp vụ rằng lịch sử bureau bên thứ ba dự báo tốt hơn nhân khẩu học thô.

---

## 3. Kiến trúc hệ thống

Hai đường xử lý song song, áp dụng **cùng một bộ quy tắc nghiệp vụ** (ngưỡng phân tầng rủi ro, luật quyết định, EL = PD × LGD × EAD):

```
═══════════════════════════════════════════════════════════════════
                    PIPELINE BATCH (huấn luyện + phân tích)
═══════════════════════════════════════════════════════════════════

  HDFS (raw)              Spark Batch                    MLflow Tracking
  ┌─────────┐         ┌──────────────────┐              ┌────────────┐
  │ 7 CSVs  │────────►│ ETL: join bảng   │              │ Params     │
  │ ~2.5 GB │         │ → master parquet │              │ Metrics    │
  └─────────┘         │ → feature pipe   │─────────────►│ Artifacts  │
                      │ → GBT training   │              │ Registry   │
                      └────────┬─────────┘              └────────────┘
                               │
                               ▼
                      ┌──────────────────┐              ┌────────────┐
                      │ Risk analytics   │─────────────►│ JSON       │
                      │ - 4-tier segm.   │              │ reports    │
                      │ - EL/UL/Capital  │              └────────────┘
                      │ - Stress testing │
                      │ Monitoring       │
                      │ - PSI + KS       │
                      │ - Jensen-Shannon │
                      │ - Kruskal-Wallis │
                      │ Retraining dec.  │
                      └──────────────────┘

═══════════════════════════════════════════════════════════════════
                    PIPELINE STREAMING (suy luận thời gian thực)
═══════════════════════════════════════════════════════════════════

  Kafka Producer          Kafka                Spark Structured Streaming
  ┌─────────────┐    ┌──────────────┐         ┌──────────────────────┐
  │ Hồ sơ vay   │───►│ Topic:       │────────►│ readStream.kafka     │
  │ (JSON)      │    │ loan-        │         │ from_json + schema   │
  │ acks=all    │    │ applications │         │ scoring logic:       │
  │ gzip        │    │              │         │ - PD (demo scorer)   │
  └─────────────┘    └──────────────┘         │ - risk tier          │
                                              │ - decision           │
                                              │ - expected loss      │
                                              │ Cửa sổ 1 phút:       │
                                              │ - count, avg PD      │
                                              │ - approve/reject     │
                                              │ - tổng EL            │
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

**Quy tắc nghiệp vụ dùng chung.** Cả hai đường dùng cùng ngưỡng cho: phân tầng rủi ro (Low / Medium / High / Very High), quyết định phê duyệt (APPROVE / REVIEW / REJECT), và tổn thất kỳ vọng (LGD = 0,45 — hằng số giả định, xem §8.1). Các quy tắc hiện được cài đặt trong `credit_risk/risk/` (batch) và inline trong `streaming/spark_streaming.py` (streaming); việc trích xuất thành module dùng chung `credit_risk/inference/` nằm trong lộ trình (§13).

---

## 4. Tech stack

| Tầng | Công nghệ | Lý do lựa chọn |
|------|-----------|----------------|
| Lưu trữ phân tán | HDFS 3.3.6 | Sao chép theo block, đọc song song data-local, phân vùng Parquet |
| Quản lý tài nguyên | YARN | Bộ lập lịch chuẩn của hệ sinh thái Hadoop |
| Tính toán batch | Apache Spark 3.5 + PySpark | Join đa bảng native, một API cho ETL + ML + streaming |
| Học máy | Spark MLlib (GBTClassifier) | Huấn luyện phân tán ngay trong Spark — không cần xuất dữ liệu |
| Theo dõi thí nghiệm | MLflow 2.22.5 | Params/metrics/artifacts + model registry, gọn nhẹ cho đồ án |
| Hàng đợi thông điệp | Apache Kafka 3.5 | Event log phát lại được, chuẩn cho suy luận hướng sự kiện |
| Xử lý luồng | Spark Structured Streaming | Tái dùng code path của Spark; checkpoint offset cho at-least-once (§9.1) |
| Điều phối | Make + Docker Compose | Môi trường dev tái lập được (production sẽ dùng Airflow / Argo) |
| Ngôn ngữ | Python 3.12 | Hỗ trợ đầy đủ PySpark, scipy, mlflow |

**Tối ưu Spark đã bật:** Adaptive Query Execution, tuần tự hóa Kryo, chuyển đổi Spark↔pandas tăng tốc bằng PyArrow (dùng trong Kruskal–Wallis), Parquet nén Snappy. AQE và Kryo được bật trong cấu hình nhưng **chưa đo lường định lượng** trên bộ dữ liệu này (§13).

**Điều chỉnh bộ nhớ cho VM 3,8 GB:** `spark.master=local[1]`, `spark.driver.memory=2g`, cộng 2 GB swap để hệ điều hành không OOM-kill Spark trong các join nặng nhất. Cả ba đọc từ mục `spark:` trong `conf/base/config.yaml` nên đổi một chỗ là áp cho mọi pipeline.

> **Lưu ý về `spark.executor.memory`:** trong local mode **không có JVM executor riêng** — driver và executor là cùng một tiến trình, nên `executor.memory=1g` **bị bỏ qua** và ràng buộc thật chỉ là `driver.memory=2g`. Giá trị này vẫn giữ trong config cho ngày chạy trên YARN/K8s. Cùng lý do đó, `local[1]` được chọn thay vì `local[*]`: thêm core chỉ làm nhiều task cùng tranh một heap 2 GB, tăng đỉnh bộ nhớ chứ không tăng thông lượng trên máy này.

---

## 5. Cấu trúc dự án

Thư mục `data/` theo **mô hình catalog 8 tầng kiểu Kedro**; mã nguồn là Python package cài được chuẩn.

```
Big_Data_Final_Exam/
├── credit_risk/                    # Package chính (importable)
│   ├── config/                     # Config loader (conf/base/config.yaml)
│   ├── data/
│   │   └── ingestion/
│   │       └── multi_table.py      # ETL đa bảng — aggregate + join các bảng con
│   ├── features/
│   │   ├── pipeline.py             # Spark ML pipeline (Imputer → OHE → Scaler → Assembler)
│   │   └── kruskal_selection.py    # Xếp hạng đặc trưng Kruskal–Wallis
│   ├── models/
│   │   ├── trainer.py              # Trainer GBT/RF/LR + log MLflow
│   │   ├── registry.py             # Đăng ký MLflow
│   │   └── explainer.py            # Feature attribution (Saabas) — chưa nối vào pipeline
│   ├── risk/
│   │   ├── segmentation.py         # Phân tầng rủi ro 4 tier
│   │   ├── expected_loss.py        # EL = PD × LGD × EAD
│   │   ├── stress_testing.py       # 4 kịch bản vĩ mô
│   │   └── scorecard.py            # WoE / IV (Siddiqi) — chưa nối vào pipeline
│   ├── monitoring/
│   │   ├── drift.py                # PSI + KS cho đặc trưng số
│   │   ├── categorical_drift.py    # JS divergence + phát hiện hạng mục mới
│   │   └── retraining_trigger.py   # Quyết định tái huấn luyện theo drift
│   └── utils/
│       ├── spark.py                # SparkSession builder (HDFS + Kryo + AQE)
│       └── logging.py              # Structured logging
│
├── streaming/                      # Kafka + Spark Structured Streaming
│   ├── kafka_producer.py           # Phát hồ sơ vay JSON (random hoặc replay CSV)
│   ├── spark_streaming.py          # Scorer + windowed aggregation
│   └── kafka_consumer.py           # Đọc topic scoring-results
│
├── pipelines/                      # Orchestrator các pipeline batch
│   ├── etl_pipeline.py             # Load → join → split → parquet
│   ├── training_pipeline.py        # Feature pipe → GBT → MLflow
│   ├── evaluation_pipeline.py      # So sánh các phiên bản model đã đăng ký
│   ├── ablation_pipeline.py        # Ablation: tích lũy từng bảng (+ leave-one-out)
│   ├── risk_pipeline.py            # Tier + EL + vốn Basel + stress
│   ├── monitoring_pipeline.py      # PSI + KS trên đặc trưng số
│   ├── feature_selection_pipeline.py     # Xếp hạng Kruskal–Wallis
│   ├── categorical_drift_pipeline.py     # JS divergence + hạng mục mới
│   ├── hdfs_pipeline.py            # Demo tích hợp HDFS → Spark → HDFS
│   └── yarn_submit.sh              # Submit lên YARN
│
├── infrastructure/
│   └── docker-compose.yml          # Kafka + Zookeeper + Kafka UI + Spark + MLflow
│
├── notebooks/
│   └── EDA_Home_Credit_Default_Risk.ipynb   # EDA 13 cell (Colab, driver 8 GB)
│
├── conf/base/
│   └── config.yaml                 # ⭐ Toàn bộ hyperparams + ngưỡng + đường dẫn HDFS
│
├── data/                           # Catalog dữ liệu 8 tầng
│   ├── 01_raw/                     # CSV gốc từ Kaggle
│   ├── 02_intermediate/            # (HDFS) parquet sau chuẩn hóa kiểu
│   ├── 03_primary/                 # (HDFS) master dataset sau join
│   ├── 04_feature/                 # (HDFS) đặc trưng sau pipeline
│   ├── 05_model_input/             # Split train/val/test
│   ├── 06_models/                  # Artifacts MLflow (gắn registry)
│   ├── 07_model_output/            # Điểm PD danh mục
│   └── 08_reporting/               # ⭐ Final JSON reports (committed):
│                                   #   portfolio_risk_report.json (incl. stress tests)
│                                   #   drift_report.json
│                                   #   categorical_drift_report.json
│                                   #   feature_ranking_report.json
│                                   #   retraining_decision.json
│
├── documents/
│   ├── Reports_BigDataFinalExam.pdf     # Báo cáo kỹ thuật
│   └── Slide_BigDataFinalExam.pdf       # Slide thuyết trình
│
├── tests/                          # Unit tests (module drift)
├── pyproject.toml                  # Dependencies (PEP 621)
├── Makefile                        # Entry points (make etl / train / risk / ...)
└── README.md
```

| Tầng | Nội dung | Nơi lưu |
|------|----------|---------|
| `01_raw` | CSV gốc, **không bao giờ sửa** | Local → upload HDFS |
| `02_intermediate` | Parquet chuẩn hóa kiểu | HDFS |
| `03_primary` | Master dataset sau join | HDFS |
| `04_feature` | Sau Imputer + OHE + Scaler + VectorAssembler | HDFS |
| `05_model_input` | Split train/val/test tất định | HDFS |
| `06_models` | Artifacts run MLflow | SQLite + filesystem |
| `07_model_output` | Điểm PD danh mục | HDFS |
| `08_reporting` | Báo cáo JSON đọc được | Repo (commit) |

---

## 6. ETL đa bảng

`application_train` là bảng neo cấp khách hàng (1 dòng/khách). Các bảng con phải được **gộp về cấp khách hàng** trước khi join:

```python
df_child.groupBy("SK_ID_CURR") \
    .agg(
        F.count("*").alias("CHILD_COUNT"),
        F.avg("AMT_OVERDUE").alias("CHILD_AVG_OVERDUE"),
        F.sum("AMT_DEBT").alias("CHILD_SUM_DEBT"),
        F.max("DAYS_LATE").alias("CHILD_MAX_DPD"),
    )
```

| Bảng con | Số dòng | Chiến lược tổng hợp | Đặc trưng tạo ra | **Thực vào model** |
|----------|--------:|---------------------|----------------:|-------------------:|
| `bureau` | 1,7M | count, còn hoạt động vs đã đóng, trung bình quá hạn, tổng dư nợ | 8 | **+8** |
| `previous_application` | 1,7M | count, được duyệt vs từ chối, trung bình số tiền/kỳ hạn | 8 | **+8** |
| `installments_payments` | 13,6M | trung bình/max DPD, đếm lần trễ, đếm lần trả thiếu | 6 | **+6** |
| `credit_card_balance` | 3,8M | trung bình dư nợ, trung bình hệ số sử dụng, tổng DPD | 6 | **+0** ⚠️ |
| `pos_cash_balance` | 10,0M | count, max DPD, đếm tháng trễ | 5 | **+5** |

Cột "thực vào model" lấy từ chênh lệch `n_numeric` giữa các bước trong [`ablation_report.json`](data/08_reporting/ablation_report.json) — tức số đặc trưng **thực sự** đi vào vector huấn luyện, không phải số biểu thức aggregate viết trong code.

> ⚠️ **`credit_card_balance` tạo 6 đặc trưng nhưng 0 đặc trưng vào được model.** Bảng này chỉ phủ một phần nhỏ tập khách hàng, nên sau LEFT JOIN cả 6 cột đều vượt ngưỡng `features.missing_threshold: 0.5` (bỏ cột có > 50% null) và bị loại ở `identify_feature_columns`. Bằng chứng trong JSON: bước `+credit_card` có `n_source_columns` **không đổi** (107 → 107) và cả ba chỉ số **trùng khít** với bước trước. Nghĩa là bảng này **chưa được kiểm chứng**, không phải "không có tín hiệu" — muốn dùng được thì phải hạ ngưỡng null hoặc thiết kế đặc trưng chịu được null (ví dụ cờ "có/không có thẻ" + giá trị mặc định).
>
> Điều này **không chỉ giới hạn trong nghiên cứu ablation**: `training_pipeline` cũng truyền đúng `features.missing_threshold` vào `identify_feature_columns` trên cùng master, nên model Production (§7.4) **nhiều khả năng cũng không có đặc trưng nào từ `credit_card_balance`**. Đây là suy luận từ code — chưa có artifact nào ghi danh sách cột của lần chạy Production — và sẽ được kiểm chứng bằng số liệu trong T1.4 (CV sẽ ghi số cột vào model từng fold).

Tất cả phép nối đều là **LEFT JOIN theo `SK_ID_CURR`** — giữ lại khách hàng chưa có lịch sử (NULL được xử lý ở tầng Imputer).

**Đã xác minh trên HDFS:** đầu vào 7 CSV (~58,4M dòng) → `application_train` 307.511 × 122 → sau gộp bureau 307.511 × 126 → sau toàn bộ join + đặc trưng nghiệp vụ **307.511 × ~200 cột**. Ở bước đủ bảng, **112 cột** thực sự đi vào feature pipeline (99 số + 13 hạng mục, sau khi loại cột > 50% null và bỏ `SK_ID_CURR`/`TARGET`) — xem §12.

> Lưu ý: `bureau_balance` (27,3M dòng) được khai báo trong `config.yaml` nhưng **chưa có aggregator** — trạng thái tháng của nó chưa đóng góp đặc trưng vào master dataset. Con số 58,4M phản ánh khối lượng raw lưu trữ và đọc, không phải khối lượng đã chuyển thành đặc trưng. Bổ sung aggregator theo nhóm DPD nằm trong lộ trình (§13).

---

## 7. Xây dựng đặc trưng & Huấn luyện

### 7.1 Đặc trưng nghiệp vụ

| Đặc trưng | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| `FEAT_DEBT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Tỷ lệ nợ trên thu nhập |
| `FEAT_ANNUITY_INCOME_RATIO` | (AMT_ANNUITY × 12) / AMT_INCOME_TOTAL | Gánh nặng trả nợ hàng năm |
| `FEAT_CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Đòn bẩy so với giá trị tài sản |
| `FEAT_CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Kỳ hạn vay ngầm định |
| `FEAT_EMPLOYMENT_RATIO` | DAYS_EMPLOYED / DAYS_BIRTH | Tỷ trọng thời gian đi làm trên tuổi đời |
| `FEAT_EXT_SOURCE_MEAN` | mean(EXT_SOURCE_1, 2, 3) | Điểm bên ngoài trung bình |

### 7.2 Spark ML pipeline

```
Imputer (median cho số, mode cho hạng mục)
  → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
```

Pipeline được **fit chỉ trên tập huấn luyện** và áp dụng nguyên xi cho validation và test, đảm bảo nhất quán giữa huấn luyện và đánh giá. (Đường streaming hiện chưa đi qua pipeline này — xem §9.)

### 7.3 Vì sao chọn Spark MLlib GBT

Huấn luyện phân tán ngay trong runtime Spark (không xuất dữ liệu) và bắt được tương tác phi tuyến giữa đặc trưng. Khoảng cách ~1–2 điểm phần trăm ROC-AUC so với LightGBM/XGBoost trên benchmark Kaggle là **đánh đổi có chủ đích** để giữ toàn bộ pipeline trong một runtime (§13). `GBTClassifier` có tham số `weightCol` cho mất cân bằng lớp, nhưng **pipeline hiện không dùng** (§13).

### 7.4 Kết quả (xác minh trong MLflow)

Một lần chạy, experiment `credit_risk_platform / gbt_baseline`:

| Chỉ số | Giá trị | Ý nghĩa nghiệp vụ |
|--------|--------:|-------------------|
| **ROC-AUC** | **0,7663** | Phân biệt tốt nhóm vỡ nợ và không vỡ nợ |
| PR-AUC | 0,2461 | Vượt xa baseline ngẫu nhiên 0,08 ở tỷ lệ nền 8% |
| **KS** | **0,4016** | Vượt ngưỡng **0,30** của ngành cho scorecard production |
| Accuracy | 0,9178 | Bị thổi phồng do mất cân bằng lớp — không phải chỉ số chính |
| Weighted F1 | 0,8830 | — |
| **Mean PD** | **8,29%** | ≈ tỷ lệ vỡ nợ thực tế 8,2% — **cân chỉnh tốt** (lệch < 0,1 điểm phần trăm) |

Đây là ước lượng điểm từ một lần huấn luyện; chưa báo cáo phương sai cross-validation (§13).

> **Bao nhiêu trong số này đến từ việc join đa bảng?** Nghiên cứu ablation tích lũy ở **§12.1** trả lời bằng số: **+0,0106 ROC-AUC** cho cả 5 bảng con gộp lại, phần lớn từ `previous_application` và `installments`. Lưu ý ablation dùng split 85/15 nên **các con số ở đó không so sánh trực tiếp được** với 0,7663 trong bảng trên.

---

## 8. Phân tích rủi ro Basel II

> **Nguồn số liệu:** mọi con số trong §8 trích trực tiếp từ [`data/08_reporting/portfolio_risk_report.json`](data/08_reporting/portfolio_risk_report.json), sinh bởi `make risk` trên model `credit_risk_gbt` **version 2** (stage `Production`), chạy ngày 2026-10-10. Công thức vốn nằm ở [`credit_risk/risk/capital.py`](credit_risk/risk/capital.py), được kiểm bằng 25 test trong [`tests/test_capital.py`](tests/test_capital.py).

### 8.1 Tổng quan danh mục (45.859 khách hàng được chấm điểm)

| Chỉ số | Giá trị |
|--------|--------:|
| Khách hàng được chấm | **45.859** |
| Tổng EAD (Exposure at Default) | 27.407,37 M |
| Tổng EL (tổn thất kỳ vọng) | **950,72 M** |
| Tỷ lệ EL danh mục | 3,47% |
| Mean PD | 8,29% |
| LGD (hằng số **giả định**) | 45,0% |
| Tỷ lệ vỡ nợ thực tế | 8,2% |

LGD = 0,45 là **giả định**, không phải tham số đã hiệu chỉnh: con số này thuộc Foundation IRB cho danh mục **doanh nghiệp**, mượn dùng ở đây vì chưa có mô hình LGD riêng. Basel không có Foundation IRB cho bán lẻ — bán lẻ bắt buộc dùng **A-IRB** với LGD tự ước lượng. Xem §13.

### 8.2 Vốn yêu cầu theo IRB (ước lượng)

Vốn được tính **theo từng hồ sơ** bằng hàm trọng số rủi ro bán lẻ Basel II (ASRF) rồi cộng lại, chứ không phải nhân EL với một hệ số:

```
K    = LGD · Φ[ (Φ⁻¹(PD) + √R · Φ⁻¹(0,999)) / √(1−R) ] − PD · LGD
Vốn  = K × EAD                RWA = K × 12,5 × EAD
R    = 0,04 (QRRE)  |  0,03·w + 0,16·(1−w), w = (1−e^(−35·PD))/(1−e^(−35))  (other retail)
```

| Đại lượng | Giá trị |
|-----------|--------:|
| Tổn thất kỳ vọng (EL) | 950,72 M |
| **Vốn yêu cầu theo IRB (ước lượng)** | **1.547,84 M** |
| RWA | 19.348,03 M |
| Vốn / EAD | 5,648% |
| Mật độ RWA (RWA / EAD) | 70,59% |
| Mức tin cậy | 99,9% |
| Sàn PD áp dụng | 0,03% |

**Đối chiếu với hệ số kinh nghiệm — giả định, không phải Basel:**

| Cách tính | Giá trị | So với vốn IRB |
|-----------|--------:|---------------:|
| 3 × EL | 2.852,17 M | 1,84× |
| 4,5 × EL | 4.278,25 M | 2,76× |

Hai dòng này là **giả định**, giữ lại chỉ để đo khoảng lệch. Không hệ số nhân EL nào thay được vốn, vì K phi tuyến theo PD của từng hồ sơ: tỷ lệ K/EL giảm đơn điệu khi PD tăng — tính chất được ghim bằng test `tests/test_capital.py::TestIrbK::test_capital_to_el_ratio_strictly_decreasing`. Các bản README trước ghi 4.278,25 M dưới tên "vốn kinh tế Basel II"; đó là 4,5 × EL với hệ số 1,5 hardcode không nguồn gốc, nay đã bỏ.

Công thức EL: **EL = PD × LGD × EAD** (PD từ GBT, LGD = 0,45 giả định, EAD = `AMT_CREDIT` — proxy, xem §13).

### 8.3 Phân tầng 4 tier

Bằng chứng mô hình **tách rủi ro trong thực tế**, không chỉ trên đường ROC:

| Tier | Khách hàng | % danh mục | EAD (M) | EL (M) | Tỷ lệ EL | Mean PD |
|------|-----------:|-----------:|--------:|-------:|---------:|--------:|
| 🟢 Low | 34.822 | 75,93% | 21.629,76 | 455,88 | 2,11% | 4,78% |
| 🟠 Medium | 7.465 | 16,28% | 3.987,75 | 247,16 | 6,20% | 13,82% |
| 🔴 High | 2.630 | 5,73% | 1.325,93 | 153,26 | 11,56% | 25,69% |
| ⚫ **Very High** | **942** | **2,05%** | **463,93** | **94,42** | **20,35%** | **45,49%** |

Đuôi mỏng (2,05% danh mục) mang tỷ lệ EL gấp **9,6 lần** tier Low — nhất quán với quan sát Basel rằng phần đuôi tiêu tốn phần lớn vốn.

### 8.4 Stress testing vĩ mô

Bốn kịch bản nhân hệ số áp lên PD và LGD:

| Kịch bản | PD× | LGD× | Tổng EL (M) | Tỷ lệ EL | Vốn IRB (M) | KH rủi ro cao |
|----------|----:|-----:|------------:|---------:|------------:|--------------:|
| Cơ sở | 1,0 | 1,0 | 950,72 | 3,47% | 1.547,84 | 3.572 |
| Suy thoái nhẹ | 1,5 | 1,1 | 1.567,68 | 5,72% | 1.858,43 | 7.255 |
| Khủng hoảng kiểu 2008 | 2,5 | 1,3 | 3.047,41 | 11,12% | 2.434,81 | 14.799 |
| **Thiên nga đen** | **4,0** | **1,5** | **5.376,49** | **19,62%** | **3.013,33** | **24.623** |

Cột vốn dùng cùng hàm K với §8.2, chỉ thay PD và LGD bằng giá trị đã stress (hệ số tương quan R tính lại từ PD đã stress, đúng như Basel quy định) — nên hai mục không thể lệch nhau về định nghĩa.

Ở kịch bản thiên nga đen, EL tăng **5,66 lần** và nhóm rủi ro cao tăng **~7 lần** — ước lượng PD tại một thời điểm, dù cân chỉnh tốt đến đâu, là chưa đủ nếu thiếu tầng stress testing.

**Vì sao vốn IRB tăng chậm hơn EL khi stress?** Đây không phải dấu hiệu mô hình đánh giá thấp rủi ro, mà là hệ quả trực tiếp của định nghĩa. Công thức K **trừ đi** `PD × LGD`, nên nó chỉ đo tổn thất **ngoài** kỳ vọng — phần vượt quá mức đã dự phòng. Khi PD tăng, tổn thất ở mức tin cậy 99,9% tiến dần tới trần `LGD × EAD`, trong khi EL vẫn tăng gần như tuyến tính theo PD, nên hiệu số giữa hai đại lượng co lại: rủi ro tăng thêm chuyển dần từ "bất ngờ" sang "đã biết trước". Hệ số tương quan tài sản R theo §328 cũng giảm khi PD tăng, làm K phẳng thêm. Vì vậy hai cột EL và Vốn IRB phải đọc **cùng nhau**: dưới stress, phần rủi ro tăng thêm hiện ra ở EL (tức ở dự phòng), còn vốn chỉ phủ phần vượt ngoài dự phòng đó.

---

## 9. Kafka + Spark Streaming

> **Lưu ý về bước chấm điểm:** demo streaming hiện tại tính PD bằng **heuristic scorer** (tỷ lệ nền + tín hiệu nợ/thu nhập + EXT_SOURCE + việc làm — xem `add_scoring_features()` trong `streaming/spark_streaming.py`), **không phải model GBT đã huấn luyện**. Nạp Spark ML PipelineModel song song với Kafka vượt ngân sách bộ nhớ của VM 3,8 GB. Logic tier / decision / EL phía sau dùng đúng cùng ngưỡng và công thức với batch, và kiến trúc Kafka → Structured Streaming → sinks giữ nguyên khi hoán scorer bằng `model.transform()` — mục ưu tiên trong lộ trình (§13).

### 9.1 Ngữ nghĩa phân phối (delivery semantics)

| Tầng | Cơ chế | Đảm bảo |
|------|--------|---------|
| Producer | `acks=all` — ghi được xác nhận trước khi confirm | Không mất message phía producer |
| Spark | `checkpointLocation` lưu offset; crash thì tiếp tục từ offset đã commit | Không bỏ sót message |
| Sink Kafka | Ghi ra `scoring-results` **không transactional** | Có thể **trùng lặp** khi recovery |

⇒ Đảm bảo đầu-cuối là **at-least-once**. Exactly-once đòi hỏi sink idempotent hoặc transactional (ví dụ ghi parquet theo `batchId`, hoặc Kafka transactions) — lộ trình (§13). Sink parquet vốn idempotent theo batch nên gần exactly-once hơn sink Kafka. Checkpoint hiện ở `/tmp/spark-checkpoint` (chỉ cho demo; production đặt trên HDFS).

### 9.2 Cấu hình Kafka

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `acks` | `all` | Độ bền dữ liệu |
| `compression.type` | `gzip` | Giảm ~60–70% băng thông |
| `batch.size` / `linger.ms` | 16 KB / 10 ms | Gộp nhẹ để tăng throughput |
| `startingOffsets` | `latest` | Chỉ xử lý message mới |

### 9.3 Gộp theo cửa sổ thời gian

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

### 9.4 Kết quả demo đã xác minh (cửa sổ 1 phút)

30 hồ sơ phát trong 1 phút. Các con số này kiểm chứng luồng dữ liệu đầu-cuối (Kafka → Spark → windowed aggregation), không phải chất lượng mô hình — PD ở đây từ demo scorer:

| Chỉ số | Giá trị |
|--------|--------:|
| Tổng hồ sơ | 30 |
| 🟢 APPROVE | 28 |
| 🟠 REVIEW | 2 |
| 🔴 REJECT | 0 |
| Mean PD | 0,0927 |
| Tổng EL | 1,56 M |

---

## 10. Giám sát mô hình & Phát hiện drift

Bốn phương pháp bổ trợ nhau, mỗi phương pháp bắt một kiểu drift khác nhau:

### 10.1 PSI + KS cho đặc trưng số

**Population Stability Index** so phân phối reference (train) với current (test/production) bằng binning theo quantile của reference, với ngưỡng Siddiqi (2006): < 0,10 ổn định · 0,10–0,25 theo dõi · ≥ 0,25 tái huấn luyện. **Kiểm định Kolmogorov–Smirnov hai mẫu** bổ trợ PSI, bắt được dịch chuyển ở đuôi phân phối mà binning có thể bỏ lỡ.

### 10.2 Jensen–Shannon divergence cho hạng mục

PSI suy yếu khi **hạng mục mới** xuất hiện trong production. JS divergence đối xứng, bị chặn trong [0, 1], và module còn cảnh báo hạng mục chưa từng thấy — tín hiệu sớm của thay đổi taxonomy thượng nguồn.

Kết quả đã xác minh:

```
NAME_CONTRACT_TYPE   JS=0.0019  [none]
CODE_GENDER          JS=0.0044  [none]
NAME_FAMILY_STATUS   JS=0.0047  [none]  new=['Unknown'] (!)
ORGANIZATION_TYPE    JS=0.0089  [none]
WALLSMATERIAL_MODE   JS=0.0107  [none]
```

Tập test chứa hạng mục mới `'Unknown'` trong `NAME_FAMILY_STATUS` — drift mà PSI số **không thể thấy**, chính là lý do tồn tại của tầng hạng mục. (Lý thuyết: arXiv:2511.03807.)

### 10.3 Xếp hạng đặc trưng Kruskal–Wallis

Xếp hạng phi tham số sức mạnh dự báo của từng đặc trưng số — không giả định chuẩn tắc hay phương sai bằng nhau, phù hợp với đặc trưng tài chính lệch. Toàn bộ top-6 có p < 1e-16. Hai điểm `EXT_SOURCE` đứng đầu ở cả tương quan (EDA) lẫn Kruskal–Wallis — xác nhận chéo giữa hai phương pháp. (Phương pháp: Ashofteh & Bravo, 2021.)

### 10.4 Chính sách tái huấn luyện theo drift

Thay cho tái huấn luyện theo lịch cố định:

| Điều kiện | Hành động | Mức độ |
|-----------|-----------|--------|
| Bất kỳ đặc trưng nào PSI ≥ 0,25 | `RETRAIN_IMMEDIATE` | Cao |
| ≥ 3 đặc trưng PSI ∈ [0,10; 0,25) | `RETRAIN_SCHEDULED` | Trung bình |
| 1–2 đặc trưng PSI ∈ [0,10; 0,25) | `CONTINUE_MONITORING` (watchlist) | Thấp |
| Toàn bộ PSI < 0,10 | `CONTINUE_MONITORING` | Không |

Kết quả chạy thực trên VM (lưu tại `data/08_reporting/retraining_decision.json`):

```
✓ RETRAINING DECISION: CONTINUE_MONITORING
  Severity: NONE
  Reason: No drift detected.
```

---

## 11. Khởi động nhanh

### Yêu cầu

Python 3.10/3.12 · Java JDK 11 · Hadoop 3.3.6 (pseudo-distributed) · Docker · ≥ 4 GB RAM (đã kiểm chứng trên VM 3,8 GB + 2 GB swap).

### Cài đặt

```bash
git clone https://github.com/muahahaha55/Big_Data_Final_Exam.git
cd Big_Data_Final_Exam
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Dữ liệu

Tải CSV từ [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) vào `data/01_raw/`, rồi upload lên HDFS:

```bash
start-dfs.sh && start-yarn.sh
hdfs dfs -mkdir -p /user/$USER/credit_risk/raw
hdfs dfs -put data/01_raw/*.csv /user/$USER/credit_risk/raw/
```

### Pipeline batch

```bash
mlflow server --host 0.0.0.0 --port 5000 &

python pipelines/etl_pipeline.py                 # load + join + đặc trưng
python pipelines/training_pipeline.py            # GBT + log MLflow
python pipelines/risk_pipeline.py                # tier + EL + stress
python pipelines/monitoring_pipeline.py          # PSI + KS
python pipelines/feature_selection_pipeline.py   # Kruskal–Wallis
python pipelines/categorical_drift_pipeline.py   # JS divergence

# hoặc: make pipeline
```

### Pipeline streaming

```bash
make stack-up                                    # Kafka + Zookeeper qua Docker

kafka-topics.sh --create --topic loan-applications \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
kafka-topics.sh --create --topic scoring-results \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Terminal 1 — Spark Streaming scorer
python streaming/spark_streaming.py --output console --trigger 10
# Terminal 2 — producer
python streaming/kafka_producer.py --mode random --rate 3 --total 30
# Terminal 3 (tùy chọn) — consumer kết quả
python streaming/kafka_consumer.py --topic scoring-results
```

### Giao diện web

HDFS NameNode `:9870` · YARN RM `:8088` · Spark UI `:4040` · MLflow `:5000` · Kafka UI `:8080`

---

## 12. Kết quả thực nghiệm

| Hạng mục | Kết quả | Trạng thái |
|----------|---------|------------|
| ETL: 7 bảng → master parquet | 307.511 × ~200 cột | ✅ Xác minh trên HDFS |
| ROC-AUC | **0,7663** | ✅ MLflow |
| Ablation: tổng đóng góp của 5 bảng con | **+0,0106 ROC-AUC** | ✅ Báo cáo JSON |
| KS | **0,4016** (> ngưỡng 0,30) | ✅ MLflow |
| Cân chỉnh (mean PD vs thực tế) | 8,29% vs 8,2% | ✅ Xác minh |
| EL danh mục | 950,72 M | ✅ Báo cáo JSON |
| Vốn yêu cầu theo IRB (ước lượng) | 1.547,84 M | ✅ Báo cáo JSON |
| RWA / mật độ RWA | 19.348,03 M / 70,59% | ✅ Báo cáo JSON |
| 4 kịch bản stress | EL ×1,0 → ×5,66 | ✅ Báo cáo JSON |
| Streaming đầu-cuối (demo scorer) | 30 hồ sơ / cửa sổ 1 phút | ✅ Console |
| Quyết định drift | `CONTINUE_MONITORING` (không drift) | ✅ Báo cáo JSON |
| Đặc trưng Kruskal–Wallis đứng đầu | EXT_SOURCE_3 (H = 4.672,97) | ✅ MLflow |

### 12.1 Ablation theo từng bảng (tích lũy)

Câu hỏi: **join 7 bảng có thực sự đáng không?** Thêm từng bảng một, cùng split, cùng seed, cùng hyperparams, đo lại từ đầu mỗi bước. Mọi số dưới đây trích từ [`data/08_reporting/ablation_report.json`](data/08_reporting/ablation_report.json).

| # | Bước | Cột vào model | ROC-AUC | PR-AUC | KS | Δ ROC-AUC vs bước trước | Thời gian (s) |
|--:|------|--------------:|--------:|-------:|-----:|------------------------:|--------------:|
| 1 | `application_only` | 85 | 0,7546 | 0,2356 | 0,3775 | — | 350,7 |
| 2 | `+bureau` | 93 | 0,7562 | 0,2391 | 0,3806 | +0,0016 | 357,8 |
| 3 | `+previous_application` | 101 | 0,7621 | 0,2440 | 0,3884 | **+0,0059** | 1.124,5 |
| 4 | `+installments` | 107 | 0,7667 | 0,2504 | 0,3984 | **+0,0046** | 1.112,0 |
| 5 | `+credit_card` | 107 | 0,7667 | 0,2504 | 0,3984 | 0,0000 | 389,9 |
| 6 | `+pos_cash` | 112 | 0,7652 | 0,2505 | 0,3971 | −0,0015 | 359,5 |

**Tổng đóng góp của cả 5 bảng con: +0,0106 ROC-AUC, +0,0149 PR-AUC, +0,0196 KS** (bước 1 → bước 6). Mọi bước dùng đúng 261.691 dòng train và 45.820 dòng test — cùng một tập khách hàng, ID được ghi ra Parquet một lần ở pha precompute.

**Cách đọc bảng này cho đúng — bốn giới hạn:**

1. **Thứ tự thêm bảng là cố định** (`bureau → previous_application → installments → credit_card → pos_cash`), nên các con số là **đóng góp biên theo thứ tự đó**, không phải "giá trị nội tại" của từng bảng. Một bảng vào sau có thể hiện ra nhỏ chỉ vì bảng vào trước đã mang thông tin tương tự — ba bảng `installments`, `credit_card`, `pos_cash` đều mô tả lịch sử trả nợ qua `SK_ID_PREV` nên khả năng trùng lặp là cao. Đổi thứ tự sẽ ra bộ số khác. Thiết kế leave-one-out (`--mode loo`) trả lời câu hỏi khác — đóng góp **riêng có** của từng bảng — và chưa được chạy.
2. **Không so sánh được với §7.4 (0,7663).** Ablation chia 85/15 trên `SK_ID_CURR`; model ở §7.4 dùng split 3 phần 70/15/15 chia trên master. Tập test khác nhau và tập train lớn hơn 15 điểm phần trăm, nên việc 0,7667 ở bước 4 trông "giống" 0,7663 là **trùng hợp**, không phải xác nhận lẫn nhau.
3. **Một lần chạy, một seed, chưa có khoảng tin cậy.** GBT dùng `subsamplingRate = 0,8` nên bản thân nó có yếu tố ngẫu nhiên. **Chênh lệch dưới ~0,002 ROC-AUC chưa thể kết luận** — tức chỉ `+previous_application` (+0,0059) và `+installments` (+0,0046) là đóng góp đọc được chắc chắn; `+bureau` (+0,0016) và `+pos_cash` (−0,0015) đều nằm trong vùng chưa phân biệt được với nhiễu. Không có cơ sở để nói bảng nào "gây hại".
4. **`+credit_card` cho Δ = 0,0000 vì các đặc trưng của nó chưa bao giờ vào model**, không phải vì nó vô dụng: cả 6 cột bị loại do > 50% null (§6). Bảng này **chưa được kiểm chứng**.

Xếp hạng theo đóng góp biên (từ `marginal_ranking_by_delta_roc_auc` trong JSON): `previous_application` (+0,0059) → `installments` (+0,0046) → `bureau` (+0,0016) → `credit_card` (0,0000) → `pos_cash` (−0,0015).

**Kết luận có thể bảo vệ được:** join đa bảng **có** đem lại cải thiện, nhưng là **+0,0106 ROC-AUC**, và phần lớn đến từ **hai** bảng (`previous_application`, `installments`) chứ không phải cả năm. Đây là mức cải thiện nhỏ nhưng nhất quán trên cả ba chỉ số (ROC-AUC, PR-AUC, KS cùng tăng), đổi lại chi phí hạ tầng của việc đọc và gộp ~58,4M dòng.

Chạy lại: `make ablation-precompute` rồi `make ablation-step N=1` … `N=6` (mỗi bước một process; kết quả checkpoint sau từng bước nên dừng giữa chừng không mất gì).

---

Bằng chứng chính: 6 báo cáo JSON trong `data/08_reporting/` và báo cáo kỹ thuật trong `documents/`. Toàn bộ số liệu rủi ro (§8) lấy từ model `credit_risk_gbt` **version 2**, stage `Production`.

---

## 13. Hạn chế đã biết & Lộ trình

Liệt kê có chủ đích — mọi khẳng định phía trên đều được giới hạn trong phạm vi đã thực đo.

**Ràng buộc kỹ thuật**

- Triển khai pseudo-distributed một nút; khả năng scale-out được hỗ trợ về kiến trúc nhưng chưa kiểm chứng trên cụm nhiều worker.
- AQE / Kryo / Snappy được bật nhưng chưa benchmark định lượng (không có số trước/sau trên bộ dữ liệu này).
- Streaming chấm điểm bằng heuristic PD, không phải GBT đã huấn luyện (§9) — kiến trúc là thật, bước chấm điểm là placeholder chờ dư địa bộ nhớ.
- Quy tắc tier/decision/EL bị lặp code giữa batch và streaming (cùng ngưỡng, hai chỗ cài đặt) — chưa tách module `credit_risk/inference/` dùng chung.
- Đảm bảo phân phối là at-least-once, không phải exactly-once (§9.1).
- `bureau_balance` (27,3M dòng) chưa có aggregator, chưa đóng góp đặc trưng và **không nằm trong nghiên cứu ablation** (§6, §12.1) — ablation phủ 5 bảng con, không phải 6.
- `credit_card_balance` tạo 6 đặc trưng nhưng **cả 6 bị loại vì > 50% null**, nên đóng góp của bảng này **chưa được kiểm chứng** (§6). Cần cờ hiện diện + giá trị mặc định, hoặc hạ `missing_threshold`, mới đánh giá được.
- `scorecard.py` (WoE/IV) và `explainer.py` có trong package nhưng chưa pipeline nào gọi.
- **Class weighting khai trong config nhưng không được áp dụng.** `conf/base/config.yaml` đặt `models.imbalance.method: class_weight`, và `credit_risk/models/trainer.py` có hàm `add_class_weights()`, nhưng hàm đó **không được gọi ở đâu** và không classifier nào nhận `weightCol` — nên mọi model trong repo, kể cả Production (§7.4), được huấn luyện **không có** class weighting. Hệ quả cần biết khi đọc §7.4: Mean PD 8,29% ≈ tỷ lệ nền 8,2% **chính là vì** không có weighting — nếu bật cân bằng lớp 50/50 thì PD dự đoán sẽ bị đẩy lên rất cao và tính chất hiệu chuẩn này mất đi. Việc bật weighting là một quyết định mô hình riêng, có đánh đổi giữa thứ tự xếp hạng và mức hiệu chuẩn, chưa được thực hiện hay đo.

**Ràng buộc nghiên cứu**

- Một lần huấn luyện duy nhất — mọi chỉ số là ước lượng điểm, không có khoảng tin cậy cross-validation.
- **Ablation chưa có cross-validation.** Mỗi bước ở §12.1 là **một** lần chạy với **một** seed, trong khi GBT có `subsamplingRate = 0,8` nên kết quả có thành phần ngẫu nhiên. Hệ quả: chênh lệch dưới ~0,002 ROC-AUC (`+bureau` +0,0016, `+pos_cash` −0,0015) **không phân biệt được với nhiễu** và không được dùng để kết luận. Cần k-fold hoặc nhiều seed cho mỗi bước mới có khoảng dao động — xem T1.4 trong lộ trình.
- Ablation đo **đóng góp biên theo một thứ tự thêm bảng cố định**, nên không phải "giá trị nội tại" của từng bảng; thiết kế leave-one-out (`--mode loo`, đã có trong pipeline) chưa chạy.
- Vốn IRB (§8.2) là **ước lượng**, đứng trên bốn giả định cần nêu rõ khi bảo vệ số:
  - **Phân lớp tài sản** suy ra từ `NAME_CONTRACT_TYPE`: `Revolving loans` → QRRE (R = 0,04), còn lại → other retail (R theo §328, phụ thuộc PD). Đây là phép gán hợp lý theo dữ liệu có sẵn, không phải phân lớp do bộ phận rủi ro xác nhận.
  - **LGD = 0,45 là hằng số giả định** (tham số Foundation IRB của danh mục doanh nghiệp). Bán lẻ theo Basel bắt buộc A-IRB với LGD tự ước lượng — repo chưa có mô hình LGD.
  - **PD là point-in-time** trên một tập test, không phải PD bình quân qua chu kỳ (through-the-cycle) như IRB yêu cầu.
  - **EAD = `AMT_CREDIT` là proxy**, chưa có CCF (credit conversion factor) cho phần hạn mức chưa giải ngân.
  - Sàn PD 0,03% theo Basel **đã** được áp; không có điều chỉnh kỳ hạn vì bán lẻ không áp dụng.
- Stress test dùng hệ số nhân cố định; chưa có mô hình vệ tinh vĩ mô nối với GDP/thất nghiệp.
- Ngưỡng tái huấn luyện theo quy tắc kinh nghiệm Siddiqi, chưa dùng khoảng tin cậy bootstrap.

**Lộ trình**

1. Triển khai trên cụm Spark thật (YARN/K8s) để kiểm chứng scale-out
2. Benchmark định lượng AQE/Kryo
3. K-fold CV (hoặc nhiều seed) cho khoảng tin cậy — cho cả model chính và từng bước ablation
4. Chạy ablation leave-one-out (`--mode loo`) để có đóng góp riêng có, đối chiếu với đóng góp biên ở §12.1
5. Sửa `credit_card_balance` để các đặc trưng của nó vào được model (cờ hiện diện + giá trị mặc định)
6. Mô hình vệ tinh vĩ mô: PD × (a + b·ΔGDP + c·ΔThất nghiệp)
7. Phát hiện drift theo cửa sổ trượt (ngày/tuần)
8. Feature store (Feast)
9. Tách `credit_risk/inference/` dùng chung; thay heuristic streaming bằng `model.transform()` với GBT từ MLflow registry
10. Sink Kafka exactly-once (ghi idempotent theo `batchId` hoặc Kafka transactions); chuyển checkpoint lên HDFS
11. Aggregator nhóm DPD cho `bureau_balance`; nối `scorecard.py` (bảng IV) và `explainer.py` vào pipeline

---

## 14. Tài liệu tham khảo

**Học thuật**

1. Ashofteh, A. & Bravo, J. M. (2021). *A non-parametric-based computationally efficient approach for credit scoring.* DOI: [10.24433/CO.1963899.v1](https://doi.org/10.24433/CO.1963899.v1)
2. *Fair and Explainable Credit-Scoring under Concept Drift.* arXiv:2511.03807
3. Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley
4. Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II).*
5. Federal Reserve (2011). *SR 11-7: Guidance on Model Risk Management.*
6. WJARR (2025). *Machine learning for credit scoring and loan default prediction.*

**Kỹ thuật**

7. [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — Kaggle, 2018
8. [Apache Spark 3.5 Documentation](https://spark.apache.org/docs/3.5.0/) · [Structured Streaming Guide](https://spark.apache.org/docs/3.5.0/structured-streaming-programming-guide.html) · [Kafka Integration](https://spark.apache.org/docs/3.5.0/structured-streaming-kafka-integration.html)
9. [Apache Kafka Documentation](https://kafka.apache.org/documentation/) · [MLflow Documentation](https://mlflow.org/docs/latest/) · [HDFS Architecture](https://hadoop.apache.org/docs/r3.3.6/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
10. Karau, H. & Warren, R. (2017). *High Performance Spark.* O'Reilly