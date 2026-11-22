
from pathlib import Path
import os

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==========================================
# PROJECT PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT_FILE = DATA_ROOT / "raw" / "r4.2" / "email.csv"

OUTPUT_DIR = DATA_ROOT / "processed" / "email_daily_features"
OUTPUT_FILE = OUTPUT_DIR / "email_daily_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# START SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("UEBA_Email_Feature_Engineering")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # ======================================
    # READ EMAIL DATA
    # ======================================

    df = spark.read.csv(
        str(INPUT_FILE),
        header=True,
        inferSchema=False
    )

    # Retain only metadata needed for features.
    df = df.select(
        "date", "user", "pc",
        "to", "cc", "bcc",
        "size", "attachments"
    )

    print("\n=== EMAIL SCHEMA ===")
    df.printSchema()

    print("\n=== EMAIL EVENT COUNT ===")
    print("Total events:", df.count())

    # ======================================
    # CLEAN TIMESTAMPS AND NUMERIC FIELDS
    # ======================================

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
        .withColumn(
            "email_size",
            F.greatest(
                F.coalesce(
                    F.col("size").cast("long"),
                    F.lit(0)
                ),
                F.lit(0)
            )
        )
        .withColumn(
            "attachment_count",
            F.greatest(
                F.coalesce(
                    F.col("attachments").cast("long"),
                    F.lit(0)
                ),
                F.lit(0)
            )
        )
    )

    # ======================================
    # RECIPIENT COUNTS
    # ======================================

    # Recipient addresses are separated by ;
    # Empty or null recipient lists count as zero.

    def count_addresses(column_name):
        value = F.trim(
            F.coalesce(
                F.col(column_name),
                F.lit("")
            )
        )

        return F.when(
            value == "",
            F.lit(0)
        ).otherwise(
            F.size(
                F.filter(
                    F.split(value, ";"),
                    lambda address: F.trim(address) != ""
                )
            )
        )

    df = (
        df.withColumn(
            "to_count",
            count_addresses("to")
        )
        .withColumn(
            "cc_count",
            count_addresses("cc")
        )
        .withColumn(
            "bcc_count",
            count_addresses("bcc")
        )
    )

    df = df.withColumn(
        "recipient_count",
        F.col("to_count")
        + F.col("cc_count")
        + F.col("bcc_count")
    )

    # ======================================
    # TIME-BASED FLAGS
    # ======================================

    # Working-hours assumption: 09:00–17:00

    df = df.withColumn(
        "after_hours",
        (F.col("hour") < 9)
        | (F.col("hour") >= 17)
    )

    # Sunday=1, Saturday=7

    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(1, 7)
    )

    # ======================================
    # USER-DAY FEATURE ENGINEERING
    # ======================================

    daily = df.groupBy("user", "day").agg(

        F.count("*").alias(
            "email_activity_count"
        ),

        F.sum("recipient_count").alias(
            "email_recipient_count"
        ),

        F.sum(
            F.when(
                F.col("attachment_count") > 0,
                1
            ).otherwise(0)
        ).alias("email_with_attachment_count"),

        F.sum("attachment_count").alias(
            "email_attachment_count"
        ),

        F.sum("email_size").alias(
            "email_total_size"
        ),

        F.sum(
            F.when(
                F.col("after_hours"),
                1
            ).otherwise(0)
        ).alias("after_hours_email_count"),

        F.sum(
            F.when(
                F.col("is_weekend"),
                1
            ).otherwise(0)
        ).alias("weekend_email_count"),

        F.sum(
            F.when(
                F.col("bcc_count") > 0,
                1
            ).otherwise(0)
        ).alias("email_bcc_event_count"),

        F.countDistinct("pc").alias(
            "unique_email_pc_count"
        )
    )

    # ======================================
    # VALIDATE AGGREGATED FEATURES
    # ======================================

    print("\n=== DAILY EMAIL FEATURES ===")
    daily.show(10, truncate=False)

    total_rows = daily.count()

    print("\nTotal email user-days:", total_rows)

    # Protect local memory from unexpectedly
    # large aggregated outputs.
    if total_rows > 1_000_000:
        raise RuntimeError(
            "Aggregated output exceeds 1 million rows. "
            "Use a Spark-native export on Linux/cloud "
            "instead of collecting to pandas."
        )

    # Only aggregated data is collected.
    local_df = daily.toPandas()

    assert len(local_df) == total_rows
    assert not local_df.duplicated(
        ["user", "day"]
    ).any()
    assert not local_df.isna().any().any()

    # ======================================
    # SAVE TO PARQUET
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

    assert saved_rows == total_rows

    # ======================================
    # RESULTS
    # ======================================

    print("\n=== VERIFICATION ===")
    print("Saved rows:", saved_rows)
    print("Unique users:", local_df["user"].nunique())

    print(
        "Missing values:",
        local_df.isna().sum().sum()
    )

    print(
        "Duplicate user-days:",
        local_df.duplicated(["user", "day"]).sum()
    )

    print("\nSaved to:", OUTPUT_FILE)
    print("\nEmail feature engineering successful!")

finally:
    spark.stop()
