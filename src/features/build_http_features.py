
from pathlib import Path
import os

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# ==========================================
# PROJECT PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT_FILE = DATA_ROOT / "raw" / "r4.2" / "http.csv"

OUTPUT_DIR = DATA_ROOT / "processed" / "http_daily_features"
OUTPUT_FILE = OUTPUT_DIR / "http_daily_features.parquet"

TEMP_DIR = DATA_ROOT.parent / "temp" / "spark_http"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

# ==========================================
# SPARK CONFIGURATION
# ==========================================

spark = (
    SparkSession.builder
    .appName("UEBA_HTTP_Feature_Engineering")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.local.dir", str(TEMP_DIR))
    .config("spark.sql.csv.parser.columnPruning.enabled", "true")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # ======================================
    # EXPLICIT CSV SCHEMA
    # ======================================

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("user", StringType(), True),
        StructField("pc", StringType(), True),
        StructField("url", StringType(), True),
        StructField("content", StringType(), True),
    ])

    print("\n=== READING HTTP LOG ===")

    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .csv(str(INPUT_FILE))
    )

    # Read only fields required for features.
    # Do not carry the large content column.
    df = df.select("date", "user", "pc", "url")

    # ======================================
    # PREPROCESS
    # ======================================

    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("date"),
            "MM/dd/yyyy HH:mm:ss"
        )
    )

    df = df.filter(
        F.col("timestamp").isNotNull()
        & F.col("user").isNotNull()
        & (F.trim(F.col("user")) != "")
    )

    df = (
        df.withColumn("day", F.to_date("timestamp"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn(
            "day_of_week",
            F.dayofweek("timestamp")
        )
    )

    df = df.withColumn(
        "after_hours",
        (F.col("hour") < 9)
        | (F.col("hour") >= 17)
    )

    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(1, 7)
    )

    # HTTPS is a protocol indicator, not a
    # suspicious or safe website classification.
    df = df.withColumn(
        "uses_https",
        F.lower(F.col("url")).startswith("https://")
    )

    # ======================================
    # USER-DAY AGGREGATION
    # ======================================

    print("\n=== AGGREGATING HTTP FEATURES ===")

    daily = df.groupBy("user", "day").agg(

        F.count("*").alias(
            "http_request_count"
        ),

        F.sum(
            F.when(
                F.col("after_hours"), 1
            ).otherwise(0)
        ).alias(
            "after_hours_http_count"
        ),

        F.sum(
            F.when(
                F.col("is_weekend"), 1
            ).otherwise(0)
        ).alias(
            "weekend_http_count"
        ),

        F.sum(
            F.when(
                F.col("uses_https"), 1
            ).otherwise(0)
        ).alias(
            "https_request_count"
        ),

        F.countDistinct("pc").alias(
            "unique_http_pc_count"
        )
    )

    # ======================================
    # COLLECT AGGREGATED OUTPUT ONLY
    # ======================================

    # Do not call toPandas() on the raw log.
    # This collects only the grouped user-day
    # feature table after Spark processing.

    print("\n=== COLLECTING AGGREGATED FEATURES ===")

    local_df = daily.toPandas()

    print("HTTP user-days:", len(local_df))

    # ======================================
    # VALIDATION
    # ======================================

    assert len(local_df) > 0, "Empty HTTP feature table"

    assert len(local_df) <= 1_000_000, (
        "Aggregated table exceeds the expected "
        "size for local pandas collection"
    )

    assert not local_df[["user", "day"]].isna().any().any()

    assert not local_df.duplicated(
        ["user", "day"]
    ).any(), "Duplicate user-day records"

    assert not local_df.isna().any().any(), (
        "Missing feature values"
    )

    feature_cols = [
        col for col in local_df.columns
        if col not in ["user", "day"]
    ]

    assert (local_df[feature_cols] >= 0).all().all(), (
        "Negative feature count found"
    )

    assert (
        local_df["after_hours_http_count"]
        <= local_df["http_request_count"]
    ).all()

    assert (
        local_df["weekend_http_count"]
        <= local_df["http_request_count"]
    ).all()

    print("\n=== FEATURE SAMPLE ===")
    print(local_df.head(10).to_string(index=False))

    print("\n=== VALIDATION ===")
    print("User-days:", len(local_df))
    print("Unique users:", local_df["user"].nunique())
    print("Total HTTP events represented:",
          int(local_df["http_request_count"].sum()))
    print("Missing values:", int(local_df.isna().sum().sum()))
    print("Duplicate user-days:",
          int(local_df.duplicated(["user", "day"]).sum()))

    # ======================================
    # SAVE PARQUET
    # ======================================

    table = pa.Table.from_pandas(
        local_df,
        preserve_index=False
    )

    pq.write_table(
        table,
        OUTPUT_FILE,
        compression="snappy"
    )

    saved_rows = pq.ParquetFile(
        OUTPUT_FILE
    ).metadata.num_rows

    assert saved_rows == len(local_df)

    print("\n=== OUTPUT ===")
    print("Saved to:", OUTPUT_FILE)
    print("Verified rows:", saved_rows)

    print("\nHTTP feature engineering successful!")

finally:
    spark.stop()
