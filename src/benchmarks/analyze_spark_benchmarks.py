
from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

BENCHMARK_DIR = DATA_ROOT / "benchmarks"
INPUT = BENCHMARK_DIR / "spark_file_benchmark.csv"

RESULTS = BENCHMARK_DIR / "analysis"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load measured benchmark results
df = pd.read_csv(INPUT)

df["cores"] = pd.to_numeric(df["cores"])
df["run"] = pd.to_numeric(df["run"])
df["runtime_seconds"] = pd.to_numeric(
    df["runtime_seconds"]
)

# Validate the experiment
assert set(df["cores"]) == {1, 2, 4}
assert df.groupby(["cores", "run"]).size().eq(1).all()
assert df.groupby("cores")["run"].nunique().ge(3).all()
assert df["events"].nunique() == 1
assert df["user_days"].nunique() == 1

# Summary statistics
summary = (
    df.groupby("cores")
    .agg(
        runs=("runtime_seconds", "count"),
        mean_runtime=("runtime_seconds", "mean"),
        std_runtime=("runtime_seconds", "std"),
        min_runtime=("runtime_seconds", "min"),
        max_runtime=("runtime_seconds", "max"),
    )
    .reset_index()
    .sort_values("cores")
)

baseline = summary.loc[
    summary["cores"] == 1,
    "mean_runtime"
].iloc[0]

summary["speedup"] = (
    baseline / summary["mean_runtime"]
)

summary["parallel_efficiency"] = (
    summary["speedup"] / summary["cores"]
)

summary["mean_events_per_sec"] = (
    df["events"].iloc[0] / summary["mean_runtime"]
)

summary["mean_mib_per_sec"] = (
    df["input_mib"].iloc[0]
    / summary["mean_runtime"]
)

summary_file = RESULTS / "spark_benchmark_summary.csv"
summary.to_csv(summary_file, index=False)

print("\n=== SPARK BENCHMARK SUMMARY ===")
print(summary.round(3).to_string(index=False))

# Runtime graph
plt.figure(figsize=(8, 5))
plt.errorbar(
    summary["cores"],
    summary["mean_runtime"],
    yerr=summary["std_runtime"],
    fmt="o-",
    capsize=5
)

plt.xlabel("Local Spark worker threads")
plt.ylabel("Mean runtime (seconds)")
plt.title("Apache Spark Local Scalability")
plt.xticks([1, 2, 4])
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "spark_runtime_comparison.png",
    dpi=300
)
plt.close()

# Speedup graph
plt.figure(figsize=(8, 5))

plt.plot(
    summary["cores"],
    summary["speedup"],
    "o-",
    label="Measured speedup"
)

plt.plot(
    summary["cores"],
    summary["cores"],
    "--",
    label="Ideal linear speedup"
)

plt.xlabel("Local Spark worker threads")
plt.ylabel("Speedup")
plt.title("Apache Spark Speedup")
plt.xticks([1, 2, 4])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "spark_speedup.png",
    dpi=300
)
plt.close()

print("\nSaved:")
print(summary_file)
print(RESULTS / "spark_runtime_comparison.png")
print(RESULTS / "spark_speedup.png")
print("\nBenchmark analysis successful!")
