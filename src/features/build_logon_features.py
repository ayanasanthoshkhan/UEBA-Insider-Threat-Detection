
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = r"D:\UEBA_Capstone"

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "raw", "r4.2", "logon.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "logon_daily_features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "logon_daily_features.parquet"
)

TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["SPARK_LOCAL_DIRS"] = TEMP_DIR

# ==========================================
# 2. INITIALIZE APACHE SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("UEBA_Logon_Features")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    # ======================================
    # 3. LOAD CERT LOGON DATA
    # ======================================

    df = spark.read.csv(
        INPUT_PATH,
        header=True,
        inferSchema=False
    )

    print("\n=== DATASET LOADED ===")

    # ======================================
    # 4. PARSE TIMESTAMPS
    # ======================================

    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("date"),
            "MM/dd/yyyy HH:mm:ss"
        )
    )

    # Remove records with invalid timestamps
    df = df.filter(
        F.col("timestamp").isNotNull()
    )

    df = (
        df.withColumn("day", F.to_date("timestamp"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn(
            "day_of_week",
            F.dayofweek("timestamp")
        )
    )

    # ======================================
    # 5. DEFINE BEHAVIORAL INDICATORS
    # ======================================

    # Assumption: normal working hours 09:00-17:00
    df = df.withColumn(
        "after_hours",
        (F.col("hour") < 9) |
        (F.col("hour") >= 17)
    )

    # Spark: Sunday=1, Saturday=7
    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(1, 7)
    )

    # ======================================
    # 6. USER-DAY FEATURE ENGINEERING
    # ======================================

    daily_features = (
        df.groupBy("user", "day")
        .agg(

            F.sum(
                F.when(
                    F.col("activity") == "Logon", 1
                ).otherwise(0)
            ).alias("login_count"),

            F.sum(
                F.when(
                    F.col("activity") == "Logoff", 1
                ).otherwise(0)
            ).alias("logoff_count"),

            F.sum(
                F.when(
                    (F.col("activity") == "Logon") &
                    F.col("after_hours"),
                    1
                ).otherwise(0)
            ).alias("after_hours_login_count"),

            F.countDistinct("pc").alias(
                "unique_pc_count"
            ),

            F.sum(
                F.when(
                    (F.col("activity") == "Logon") &
                    F.col("is_weekend"),
                    1
                ).otherwise(0)
            ).alias("weekend_login_count")

        )
    )

    # ======================================
    # 7. DISPLAY RESULTS
    # ======================================

    print("\n=== DAILY LOGON FEATURES ===")
    daily_features.show(10, truncate=False)

    total_rows = daily_features.count()

    print("\nTotal user-day records:", total_rows)

    # ======================================
    # 8. SAVE USING PYARROW
    # ======================================

    # Only convert the aggregated feature table,
    # never the full raw CERT event dataset.

    print("\nConverting aggregated features...")

    local_features = daily_features.toPandas()

    table = pa.Table.from_pandas(
        local_features,
        preserve_index=False
    )

    pq.write_table(
        table,
        OUTPUT_FILE,
        compression="snappy"
    )

    print("\n=== PARQUET SAVED ===")
    print("File:", OUTPUT_FILE)
    print("Saved rows:", len(local_features))

    # ======================================
    # 9. VERIFY SAVED PARQUET
    # ======================================

    metadata = pq.ParquetFile(OUTPUT_FILE).metadata

    print("\n=== VERIFICATION ===")
    print("Parquet rows:", metadata.num_rows)

    assert metadata.num_rows == total_rows, (
        "Saved row count does not match!"
    )

    print("Parquet verification successful!")

finally:
    spark.stop()

print("\nUEBA logon feature engineering completed!")
