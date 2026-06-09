# Thiết kế hệ thống — Big Data Credit Risk Pipeline

## Tổng quan

Hệ thống xử lý dữ liệu tín dụng quy mô lớn (7 bảng, ~30 triệu dòng tổng) bằng Apache Spark, từ ETL đến training model đến phân tích rủi ro.

## Tại sao cần xử lý phân tán?

| Bước | Khối lượng dữ liệu | Bottleneck |
|------|-------------------|-----------:|
| Load 7 CSV | 1.5 GB | I/O |
| Join tables | 307K × 200+ cols = ~500 MB | Shuffle |
| Feature engineering | Vector hóa 200+ features | Memory |
| CrossValidator 3-fold | Train 3× trên 215K rows × 6 param combos | CPU |
| Batch scoring | 307K predictions | CPU |

Trên pandas (single-node), toàn bộ pipeline mất ~35 phút. Trên PySpark local[8], mất ~9 phút — nhanh gấp ~4×.

## Data flow

```
Raw CSV (7 files, 1.5GB)
    ↓ spark.read.csv (infer schema)
    ↓ Aggregate child tables → 1 row / customer
    ↓ Left join on SK_ID_CURR
Master dataset (307K × 200+ cols)
    ↓ add_domain_features (DTI, ratios)
    ↓ write.parquet (Snappy compression)
Parquet on disk (~180 MB, 60% compression)
    ↓ randomSplit 70/15/15
Train / Val / Test splits
    ↓ Spark ML Pipeline (Imputer → OHE → VectorAssembler → Scaler)
    ↓ GBTClassifier + CrossValidator
Fitted PipelineModel
    ↓ mlflow.spark.log_model
MLflow Registry
```

## Tối ưu Spark

### AQE (Adaptive Query Execution)
- Tự động coalesce shuffle partitions: giảm từ 200 → ~50 dựa trên data size thực tế
- Skew join handling: khi 1 partition có quá nhiều data, tự split

### Kryo Serialization
- Thay Java Serialization mặc định
- Nhanh gấp ~2×, tiết kiệm bandwidth giữa driver và executor

### Arrow Integration
- PySpark ↔ Pandas conversion: nhanh gấp ~10× vs default
- Critical cho `.toPandas()` trong notebooks và PSI computation

### Parquet + Snappy
- Columnar format: chỉ đọc columns cần thiết (predicate pushdown)
- Snappy compression: tốc độ nén/giải nén cao, giảm I/O 60%

## Multi-table Join Strategy

Home Credit có cấu trúc 1:N (1 customer → nhiều loans → nhiều payments). Strategy:

1. **Aggregate trước, join sau** — không join raw tables (sẽ explode row count)
2. **Per-customer aggregates**: COUNT, MEAN, SUM, MIN, MAX cho các numeric columns
3. **Domain-specific aggregates**: active/closed count, DPD (days past due), late payment count
4. **Left join**: giữ tất cả 307K customers, NULL nếu không có history

```python
# Ví dụ: bureau → 1 row per customer
bureau.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("BUREAU_COUNT"),
    F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("BUREAU_ACTIVE_COUNT"),
    F.mean("CREDIT_DAY_OVERDUE").alias("BUREAU_DAY_OVERDUE_MEAN"),
)
```

## Spark ML Pipeline

Pipeline đảm bảo:
1. **Reproducibility**: cùng transforms cho train và inference
2. **Serialization**: save/load cả pipeline + model cùng nhau
3. **No data leakage**: fit trên train, transform trên test

```
Imputer(median) → StringIndexer(handle=keep) → OneHotEncoder → VectorAssembler → StandardScaler → GBTClassifier
```

## MLflow Integration

Mỗi training run ghi lại:
- **Parameters**: maxDepth, maxIter, stepSize...
- **Metrics**: ROC-AUC, PR-AUC, KS statistic
- **Artifacts**: fitted PipelineModel, feature importance plot
- **Model Registry**: version + stage (Production / Staging / Archived)

## Performance (local[8], m5.2xlarge)

| Operation | Throughput | Thời gian |
|-----------|------------|-----------|
| ETL (full 7-table join) | — | 4 min 12 s |
| Training (GBT, no CV) | — | 3 min |
| Training (GBT, 3-fold CV) | — | 18 min 30 s |
| Batch scoring | 84K rows/s | 0.55 s |
| Drift detection (6 features) | — | 52 s |

## Hạn chế

- Local mode chưa thể hiện full potential của Spark (distributed workers)
- MLlib GBT kém hơn XGBoost/LightGBM ~1-2pp ROC-AUC
- Single-node memory constraint: dataset phải fit trong driver memory
