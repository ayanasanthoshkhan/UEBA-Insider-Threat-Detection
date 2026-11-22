
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

# ==========================================
# ARGUMENTS
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--size-mib",
    type=int,
    choices=[256, 512, 1024],
    required=True
)

parser.add_argument(
    "--cores",
    type=int,
    choices=[1, 2, 4],
    default=4
)

parser.add_argument(
    "--run",
    type=int,
    default=1
)

args = parser.parse_args()

# ==========================================
# PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

BENCHMARK_DIR = DATA_ROOT / "benchmarks"

INPUT = (
    BENCHMARK_DIR
    / "http_subsets"
    / f"http_{args.size_mib}mib.csv"
)

OUTPUT = (
    BENCHMARK_DIR
    / "spark_http_scaling.csv"
)

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

input_mib = INPUT.stat().st_size / (1024 * 1024)

EXPECTED_EVENTS = {
    256: 525559,
    512: 1051198,
    1024: 2102687,
}

print("\n=== HTTP SCALABILITY BENCHMARK ===")
print("Dataset:", INPUT.name)
print("Input size (MiB):", round(input_mib, 3))
print("Spark workers:", args.cores)
print("Trial:", args.run)

# ==========================================
# SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName(
        f"UEBA_HTTP_{args.size_mib}MiB"
    )
    .master(f"local[{args.cores}]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    # ======================================
    # HTTP CSV SCHEMA
    # ======================================

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("user", StringType(), True),
        StructField("pc", StringType(), True),
        StructField("url", StringType(), True),
        StructField("content", StringType(), True),
    ])

    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .csv(str(INPUT))
        .select("date", "user", "pc", "url")
    )

    # ======================================
    # FEATURE TRANSFORMATIONS
    # ======================================

    df = df.withColumn(
        "timestamp",
        F.try_to_timestamp(
            F.col("date"),
            F.lit("MM/dd/yyyy HH:mm:ss")
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
        "is_weekend",
        F.dayofweek("timestamp").isin(1, 7)
    )

    # ======================================
    # USER-DAY AGGREGATION
    # ======================================

    daily = df.groupBy(
        "user", "day"
    ).agg(
        F.count("*").alias(
            "http_request_count"
        ),

        F.sum(
            F.when(
                F.col("after_hours"), 1
            ).otherwise(0)
        ).alias("after_hours_http_count"),

        F.sum(
            F.when(
                F.col("is_weekend"), 1
            ).otherwise(0)
        ).alias("weekend_http_count"),

        F.countDistinct("pc").alias(
            "unique_http_pc_count"
        )
    )

    # ======================================
    # TIMED SPARK EXECUTION
    # ======================================

    print("\nProcessing HTTP events...")

    start = time.perf_counter()

    result = daily.agg(
        F.sum("http_request_count").alias(
            "total_events"
        ),
        F.count("*").alias("user_days"),
        F.sum("after_hours_http_count").alias(
            "total_after_hours"
        ),
        F.sum("weekend_http_count").alias(
            "total_weekend"
        ),
        F.sum("unique_http_pc_count").alias(
            "total_unique_pc_counts"
        )
    ).first()

    elapsed = time.perf_counter() - start

    events = int(result["total_events"])
    user_days = int(result["user_days"])

    # ======================================
    # VALIDATE EVENT COUNTS
    # ======================================

    expected = EXPECTED_EVENTS[args.size_mib]

    print("\n=== VALIDATION ===")
    print("Expected events:", expected)
    print("Spark processed events:", events)
    print("User-days:", user_days)

    if events != expected:
        raise ValueError(
            f"Event mismatch: expected {expected}, "
            f"processed {events}"
        )

    # ======================================
    # PERFORMANCE METRICS
    # ======================================

    events_per_sec = events / elapsed
    mib_per_sec = input_mib / elapsed

    print("\n=== BENCHMARK RESULT ===")
    print("Workers:", args.cores)
    print("Input (MiB):", round(input_mib, 3))
    print("Runtime (seconds):", round(elapsed, 3))
    print("Events/second:", round(events_per_sec, 2))
    print("MiB/second:", round(mib_per_sec, 2))

    # ======================================
    # SAVE BENCHMARK
    # ======================================

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": INPUT.name,
        "cores": args.cores,
        "run": args.run,
        "input_mib": round(input_mib, 3),
        "events": events,
        "user_days": user_days,
        "runtime_seconds": round(elapsed, 4),
        "events_per_second": round(
            events_per_sec, 2
        ),
        "mib_per_second": round(
            mib_per_sec, 2
        ),
        "spark_version": spark.version,
    }

    file_exists = OUTPUT.exists()

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if not file_exists or OUTPUT.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)

    print("\nSaved to:", OUTPUT)
    print("\nHTTP scalability benchmark successful!")

finally:
    spark.stop()
