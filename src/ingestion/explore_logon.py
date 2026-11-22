
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = r"D:\UEBA_Capstone"
DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "r4.2", "logon.csv"
)
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["SPARK_LOCAL_DIRS"] = TEMP_DIR

spark = (
    SparkSession.builder
    .appName("CERT_Logon_EDA")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read the CERT logon data
    df = spark.read.csv(
        DATA_PATH,
        header=True,
        inferSchema=False
    )

    print("\n=== DATASET SCHEMA ===")
    df.printSchema()

    print("\n=== FIRST 5 RECORDS ===")
    df.show(5, truncate=False)

    print("\n=== TOTAL EVENTS ===")
    print(df.count())

    print("\n=== UNIQUE USERS ===")
    print(df.select("user").distinct().count())

    print("\n=== ACTIVITY TYPES ===")
    df.groupBy("activity").count().show()

    # Parse timestamp
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(
            F.col("date"),
            "MM/dd/yyyy HH:mm:ss"
        )
    )

    print("\n=== PARSED TIMESTAMPS ===")
    df.select("user", "timestamp", "activity").show(5)

    # Daily login activity
    daily_logins = (
        df.filter(F.col("activity") == "Logon")
        .withColumn("day", F.to_date("timestamp"))
        .groupBy("user", "day")
        .agg(F.count("*").alias("login_count"))
    )

    print("\n=== USER-DAILY LOGIN COUNTS ===")
    daily_logins.orderBy("user", "day").show(10)

finally:
    spark.stop()

print("\nCERT logon exploration completed!")
