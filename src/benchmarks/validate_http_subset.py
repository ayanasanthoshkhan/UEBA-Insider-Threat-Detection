
from pathlib import Path
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
)

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT = (
    DATA_ROOT
    / "benchmarks"
    / "http_subsets"
    / "http_256mib.csv"
)

EXPECTED_LINES = 525559

schema = StructType([
    StructField("id", StringType(), True),
    StructField("date", StringType(), True),
    StructField("user", StringType(), True),
    StructField("pc", StringType(), True),
    StructField("url", StringType(), True),
    StructField("content", StringType(), True),
])

spark = (
    SparkSession.builder
    .appName("UEBA_HTTP_Subset_Validation")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .csv(str(INPUT))
        .select("id", "date", "user")
    )

    df = df.withColumn(
        "parsed_timestamp",
        F.try_to_timestamp(
            F.col("date"),
            F.lit("MM/dd/yyyy HH:mm:ss")
        )
    )

    result = df.agg(
        F.count("*").alias("rows"),
        F.sum(
            F.when(
                F.col("parsed_timestamp").isNull(),
                1
            ).otherwise(0)
        ).alias("invalid_timestamps"),
        F.sum(
            F.when(
                F.col("user").isNull(),
                1
            ).otherwise(0)
        ).alias("missing_users")
    ).first()

    print("\n=== HTTP SUBSET VALIDATION ===")
    print("Expected physical lines:", EXPECTED_LINES)
    print("Spark parsed rows:", result["rows"])
    print("Invalid timestamps:", result["invalid_timestamps"])
    print("Missing users:", result["missing_users"])

    assert result["rows"] == EXPECTED_LINES, (
        "Spark row count differs from copied line count"
    )

    assert result["invalid_timestamps"] == 0
    assert result["missing_users"] == 0

    print("\nHTTP subset validation successful!")

finally:
    spark.stop()
