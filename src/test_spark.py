
import os
from pyspark.sql import SparkSession

# Keep Spark temporary files on D:
TEMP_DIR = r"D:\UEBA_Capstone\temp"
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["SPARK_LOCAL_DIRS"] = TEMP_DIR

spark = (
    SparkSession.builder
    .appName("UEBA_CERT_r4_2_Test")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.local.dir", TEMP_DIR)
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

print("Spark version:", spark.version)

df = spark.range(1000)

print("Total records:", df.count())

spark.stop()

print("Spark test completed successfully!")
