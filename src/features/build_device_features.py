
import os
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = r"D:\UEBA_Capstone"

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "raw", "r4.2", "device.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "device_daily_features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "device_daily_features.parquet"
)

TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["SPARK_LOCAL_DIRS"] = TEMP_DIR

# ==========================================
# START APACHE SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("UEBA_Device_Features")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # ======================================
    # READ DEVICE LOG
    # ======================================

    df = spark.read.csv(
        INPUT_PATH,
        header=True,
        inferSchema=False
    )

    print("\n=== DEVICE SCHEMA ===")
    df.printSchema()

    print("\n=== ACTIVITY COUNTS ===")
    df.groupBy("activity").count().show()

    # ======================================
    # PARSE TIMESTAMPS
    # ======================================

    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("date"),
            "MM/dd/yyyy HH:mm:ss"
        )
    )

    invalid_count = df.filter(
        F.col("timestamp").isNull()
    ).count()

    print("Invalid timestamps:", invalid_count)

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

    # Assumed working hours: 09:00 to 17:00
    df = df.withColumn(
        "after_hours",
        (F.col("hour") < 9) |
        (F.col("hour") >= 17)
    )

    # Sunday=1, Saturday=7
    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin(1, 7)
    )

    # ======================================
    # USER-DAY DEVICE FEATURES
    # ======================================

    daily_features = (
        df.groupBy("user", "day")
        .agg(
            F.sum(
                F.when(
                    F.col("activity") == "Connect", 1
                ).otherwise(0)
            ).alias("usb_connect_count"),

            F.sum(
                F.when(
                    F.col("activity") == "Disconnect", 1
                ).otherwise(0)
            ).alias("usb_disconnect_count"),

            F.sum(
                F.when(
                    (F.col("activity") == "Connect") &
                    F.col("after_hours"), 1
                ).otherwise(0)
            ).alias("after_hours_usb_count"),

            F.sum(
                F.when(
                    (F.col("activity") == "Connect") &
                    F.col("is_weekend"), 1
                ).otherwise(0)
            ).alias("weekend_usb_count"),

            F.countDistinct("pc").alias(
                "unique_device_pc_count"
            )
        )
    )

    print("\n=== DAILY DEVICE FEATURES ===")
    daily_features.show(10, truncate=False)

    total_rows = daily_features.count()
    print("Total device user-day records:", total_rows)

    # ======================================
    # SAVE SMALL AGGREGATED TABLE
    # ======================================

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

    # ======================================
    # VERIFY OUTPUT
    # ======================================

    metadata = pq.ParquetFile(OUTPUT_FILE).metadata

    print("\n=== VERIFICATION ===")
    print("Saved Parquet rows:", metadata.num_rows)

    assert metadata.num_rows == total_rows

    print("Device feature engineering successful!")

finally:
    spark.stop()
