
from pathlib import Path
import os

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -----------------------------------------
# PROJECT PATHS
# -----------------------------------------

DATA_ROOT = Path(
    os.environ.get("UEBA_DATA_ROOT", r"D:\UEBA_Capstone\data")
)

INPUT_FILE = DATA_ROOT / "raw" / "r4.2" / "file.csv"
OUTPUT_DIR = DATA_ROOT / "processed" / "file_daily_features"
OUTPUT_FILE = OUTPUT_DIR / "file_daily_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# SPARK SESSION
# -----------------------------------------

spark = (
    SparkSession.builder
    .appName("UEBA_File_Feature_Engineering")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # -------------------------------------
    # LOAD RAW FILE LOG
    # -------------------------------------

    df = spark.read.csv(
        str(INPUT_FILE),
        header=True,
        inferSchema=False
    )

    # Keep only the columns needed for features.
    df = df.select("date", "user", "pc", "filename")

    print("\n=== FILE LOG SCHEMA ===")
    df.printSchema()

    print("\n=== FILE ACTIVITY COUNT ===")
    print("Events:", df.count())

    # -------------------------------------
    # PARSE DATE AND EXTRACT FEATURES
    # -------------------------------------

    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("date"),
            "MM/dd/yyyy HH:mm:ss"
        )
    )

    invalid_timestamps = df.filter(
        F.col("timestamp").isNull()
    ).count()

    print("Invalid timestamps:", invalid_timestamps)

    df = df.filter(F.col("timestamp").isNotNull())

    df = (
        df.withColumn("day", F.to_date("timestamp"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn(
            "extension",
            F.lower(
                F.regexp_extract(
                    F.col("filename"),
                    r"\.([^.]+)$",
                    1
                )
            )
        )
    )

    # Working-hours assumption: 09:00 to 17:00
    df = df.withColumn(
        "after_hours",
        (F.col("hour") < 9) |
        (F.col("hour") >= 17)
    )

    # Spark dayofweek: Sunday=1, Saturday=7
    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(1, 7)
    )

    # -------------------------------------
    # AGGREGATE BY USER-DAY
    # -------------------------------------

    daily = df.groupBy("user", "day").agg(
        F.count("*").alias("file_activity_count"),

        F.countDistinct("filename").alias(
            "unique_file_count"
        ),

        F.sum(
            F.when(
                F.col("after_hours"), 1
            ).otherwise(0)
        ).alias("after_hours_file_count"),

        F.sum(
            F.when(
                F.col("is_weekend"), 1
            ).otherwise(0)
        ).alias("weekend_file_count"),

        F.countDistinct("pc").alias(
            "unique_file_pc_count"
        ),

        F.sum(
            F.when(
                F.col("extension").isin(
                    "doc", "docx", "pdf"
                ),
                1
            ).otherwise(0)
        ).alias("document_file_count"),

        F.sum(
            F.when(
                F.col("extension").isin(
                    "zip", "rar", "7z"
                ),
                1
            ).otherwise(0)
        ).alias("archive_file_count")
    )

    # -------------------------------------
    # VERIFY AGGREGATED DATA
    # -------------------------------------

    print("\n=== DAILY FILE FEATURES ===")
    daily.show(10, truncate=False)

    total_rows = daily.count()
    print("Total file user-days:", total_rows)

    # Convert only the aggregated result to pandas.
    local_df = daily.toPandas()

    assert not local_df.duplicated(
        ["user", "day"]
    ).any(), "Duplicate user-day records found"

    assert not local_df.isna().any().any(), (
        "Missing values found in feature table"
    )

    assert len(local_df) == total_rows, (
        "Spark and pandas row counts do not match"
    )

    # -------------------------------------
    # SAVE PARQUET USING PYARROW
    # -------------------------------------

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

    assert saved_rows == total_rows, (
        "Saved Parquet row count does not match"
    )

    print("\n=== VERIFICATION ===")
    print("Saved to:", OUTPUT_FILE)
    print("Verified rows:", saved_rows)
    print("Missing values:", local_df.isna().sum().sum())
    print(
        "Duplicate user-days:",
        local_df.duplicated(["user", "day"]).sum()
    )

    print("\nFile feature engineering successful!")

finally:
    spark.stop()
