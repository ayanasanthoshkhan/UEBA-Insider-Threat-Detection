
from pathlib import Path
from datetime import datetime
import argparse
import csv
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
)

# ------------------------------------------
# Configuration
# ------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cores",
    type=int,
    choices=[1, 2, 4],
    required=True
)

parser.add_argument(
    "--run",
    type=int,
    default=1
)

args = parser.parse_args()

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT_FILE = (
    DATA_ROOT / "raw" / "r4.2" / "email.csv"
)

RESULT_DIR = DATA_ROOT / "benchmarks"

RESULT_FILE = (
    RESULT_DIR / "spark_email_benchmark.csv"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

input_mib = (
    INPUT_FILE.stat().st_size / (1024 * 1024)
)

print("\n=== BENCHMARK CONFIGURATION ===")
print("Dataset:", INPUT_FILE)
print("Size (MiB):", round(input_mib, 2))
print("Spark workers:", args.cores)
print("Run:", args.run)

# ------------------------------------------
# Spark configuration
# ------------------------------------------

spark = (
    SparkSession.builder
    .appName(
        f"UEBA_Email_Benchmark_{args.cores}"
    )
    .master(f"local[{args.cores}]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    # --------------------------------------
    # Raw email schema
    # --------------------------------------

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("user", StringType(), True),
        StructField("pc", StringType(), True),
        StructField("to", StringType(), True),
        StructField("cc", StringType(), True),
        StructField("bcc", StringType(), True),
        StructField("from", StringType(), True),
        StructField("size", StringType(), True),
        StructField("attachments", StringType(), True),
        StructField("content", StringType(), True),
    ])

    # File reading is lazy; actual data
    # processing happens during the action.

    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .csv(str(INPUT_FILE))
        .select(
            "date",
            "user",
            "pc",
            "size",
            "attachments"
        )
    )

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
    )

    df = df.withColumn(
        "day",
        F.to_date("timestamp")
    )

    df = df.withColumn(
        "after_hours",
        (F.hour("timestamp") < 9)
        | (F.hour("timestamp") >= 17)
    )

    df = df.withColumn(
        "email_size",
        F.col("size").cast("long")
    )

    df = df.withColumn(
        "attachment_count",
        F.col("attachments").cast("long")
    )

    # --------------------------------------
    # Daily feature aggregation
    # --------------------------------------

    daily = df.groupBy(
        "user", "day"
    ).agg(
        F.count("*").alias(
            "email_event_count"
        ),

        F.sum("email_size").alias(
            "total_email_size"
        ),

        F.sum("attachment_count").alias(
            "total_attachments"
        ),

        F.sum(
            F.when(
                F.col("after_hours"),
                1
            ).otherwise(0)
        ).alias("after_hours_count"),

        F.countDistinct("pc").alias(
            "unique_pc_count"
        )
    )

    # --------------------------------------
    # Timed execution
    # --------------------------------------

    print("\nRunning email benchmark...")

    start = time.perf_counter()

    result = daily.agg(
        F.sum("email_event_count").alias(
            "total_events"
        ),
        F.count("*").alias("user_days"),
        F.sum("total_email_size").alias(
            "total_email_bytes"
        ),
        F.sum("total_attachments").alias(
            "total_attachments"
        ),
        F.sum("after_hours_count").alias(
            "total_after_hours"
        ),
        F.sum("unique_pc_count").alias(
            "total_distinct_pc_per_user_day"
        )
    ).first()

    elapsed = time.perf_counter() - start

    events = int(result["total_events"])
    user_days = int(result["user_days"])

    events_per_second = events / elapsed
    mib_per_second = input_mib / elapsed

    print("\n=== EMAIL BENCHMARK RESULT ===")
    print("Cores:", args.cores)
    print("Run:", args.run)
    print("Events:", events)
    print("User-days:", user_days)
    print("Runtime (seconds):", round(elapsed, 3))
    print(
        "Events/second:",
        round(events_per_second, 2)
    )
    print(
        "MiB/second:",
        round(mib_per_second, 2)
    )

    # --------------------------------------
    # Save results
    # --------------------------------------

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": "email.csv",
        "cores": args.cores,
        "run": args.run,
        "input_mib": round(input_mib, 3),
        "events": events,
        "user_days": user_days,
        "runtime_seconds": round(elapsed, 4),
        "events_per_second": round(
            events_per_second, 2
        ),
        "mib_per_second": round(
            mib_per_second, 2
        ),
        "spark_version": spark.version,
    }

    file_exists = RESULT_FILE.exists()

    with open(
        RESULT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if (
            not file_exists
            or RESULT_FILE.stat().st_size == 0
        ):
            writer.writeheader()

        writer.writerow(row)

    print("\nSaved to:", RESULT_FILE)
    print("Email benchmark successful!")

finally:
    spark.stop()
