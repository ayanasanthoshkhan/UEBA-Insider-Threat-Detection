
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

# ==========================================
# PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT_FILE = DATA_ROOT / "raw" / "r4.2" / "file.csv"

RESULT_DIR = DATA_ROOT / "benchmarks"
RESULT_FILE = RESULT_DIR / "spark_file_benchmark.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

input_bytes = INPUT_FILE.stat().st_size
input_mb = input_bytes / (1024 * 1024)

# ==========================================
# START SPARK
# ==========================================

print("\n=== BENCHMARK CONFIGURATION ===")
print("Spark workers:", args.cores)
print("Run:", args.run)
print("Dataset:", INPUT_FILE)
print("Input size (MiB):", round(input_mb, 2))

spark = (
    SparkSession.builder
    .appName(f"UEBA_Benchmark_{args.cores}_cores")
    .master(f"local[{args.cores}]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # ======================================
    # READ RAW FILE ACTIVITY
    # ======================================

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("user", StringType(), True),
        StructField("pc", StringType(), True),
        StructField("filename", StringType(), True),
        StructField("content", StringType(), True),
    ])

    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .csv(str(INPUT_FILE))
        .select("date", "user", "pc", "filename")
    )

    # Same transformations for all runs
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

    daily = df.groupBy("user", "day").agg(
        F.count("*").alias("event_count"),

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

    # ======================================
    # TIMED SPARK ACTION
    # ======================================

    print("\nRunning benchmark...")

    start = time.perf_counter()

    # Trigger Spark execution without
    # collecting the whole output table.
    result = daily.agg(
        F.sum("event_count").alias("total_events"),
        F.count("*").alias("user_days")
    ).first()

    elapsed = time.perf_counter() - start

    events = int(result["total_events"])
    user_days = int(result["user_days"])

    throughput = events / elapsed
    mb_per_second = input_mb / elapsed

    # ======================================
    # PRINT RESULTS
    # ======================================

    print("\n=== BENCHMARK RESULT ===")
    print("Cores:", args.cores)
    print("Run:", args.run)
    print("Events:", events)
    print("User-days:", user_days)
    print("Runtime (seconds):", round(elapsed, 3))
    print("Events/second:", round(throughput, 2))
    print("MiB/second:", round(mb_per_second, 2))

    # ======================================
    # SAVE CSV RESULT
    # ======================================

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": "file.csv",
        "cores": args.cores,
        "run": args.run,
        "input_mib": round(input_mb, 3),
        "events": events,
        "user_days": user_days,
        "runtime_seconds": round(elapsed, 4),
        "events_per_second": round(throughput, 2),
        "mib_per_second": round(mb_per_second, 2),
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

        if not file_exists or RESULT_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)

    print("\nSaved to:", RESULT_FILE)
    print("Benchmark run completed!")

finally:
    spark.stop()
