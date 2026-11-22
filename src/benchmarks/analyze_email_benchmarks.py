
from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATH CONFIGURATION
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

BENCHMARK_DIR = DATA_ROOT / "benchmarks"

INPUT = BENCHMARK_DIR / "spark_email_benchmark.csv"

RESULTS = BENCHMARK_DIR / "email_analysis"
RESULTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD BENCHMARK RESULTS
# ==========================================

df = pd.read_csv(INPUT)

df["cores"] = pd.to_numeric(df["cores"])
df["run"] = pd.to_numeric(df["run"])
df["runtime_seconds"] = pd.to_numeric(
    df["runtime_seconds"]
)

# ==========================================
# VALIDATE EXPERIMENTS
# ==========================================

assert set(df["cores"]) == {1, 2, 4}, (
    "Expected benchmark configurations: 1, 2, 4"
)

assert df.groupby(
    ["cores", "run"]
).size().eq(1).all(), (
    "Duplicate benchmark runs detected"
)

assert df.groupby(
    "cores"
)["run"].nunique().ge(3).all(), (
    "At least three runs required per configuration"
)

assert df["events"].nunique() == 1, (
    "Event counts differ between runs"
)

assert df["user_days"].nunique() == 1, (
    "User-day counts differ between runs"
)

print("\n=== EMAIL BENCHMARK DATASET ===")
print("Total benchmark runs:", len(df))
print("Events processed:", df["events"].iloc[0])
print("User-days:", df["user_days"].iloc[0])

# ==========================================
# SUMMARY STATISTICS
# ==========================================

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

# Speedup relative to one worker
summary["speedup"] = (
    baseline / summary["mean_runtime"]
)

# Parallel efficiency
summary["parallel_efficiency"] = (
    summary["speedup"] / summary["cores"]
)

# Throughput calculated from mean runtime
summary["mean_events_per_sec"] = (
    df["events"].iloc[0] / summary["mean_runtime"]
)

summary["mean_mib_per_sec"] = (
    df["input_mib"].iloc[0]
    / summary["mean_runtime"]
)

# Percentage reduction in runtime
summary["runtime_reduction_pct"] = (
    (baseline - summary["mean_runtime"])
    / baseline * 100
)

# ==========================================
# SAVE SUMMARY CSV
# ==========================================

summary_file = (
    RESULTS / "spark_email_benchmark_summary.csv"
)

summary.to_csv(summary_file, index=False)

print("\n=== EMAIL SPARK BENCHMARK SUMMARY ===")
print(summary.round(3).to_string(index=False))

# ==========================================
# GRAPH 1: RUNTIME COMPARISON
# ==========================================

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
plt.title("Apache Spark Email Processing - Runtime")
plt.xticks([1, 2, 4])
plt.grid(True, alpha=0.3)
plt.tight_layout()

runtime_graph = (
    RESULTS / "spark_email_runtime_comparison.png"
)

plt.savefig(runtime_graph, dpi=300)
plt.close()

# ==========================================
# GRAPH 2: SPEEDUP
# ==========================================

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
plt.title("Apache Spark Email Processing - Speedup")
plt.xticks([1, 2, 4])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

speedup_graph = RESULTS / "spark_email_speedup.png"

plt.savefig(speedup_graph, dpi=300)
plt.close()

# ==========================================
# GRAPH 3: THROUGHPUT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    summary["cores"],
    summary["mean_events_per_sec"],
    "o-"
)

plt.xlabel("Local Spark worker threads")
plt.ylabel("Events processed per second")
plt.title("Apache Spark Email Processing - Throughput")
plt.xticks([1, 2, 4])
plt.grid(True, alpha=0.3)
plt.tight_layout()

throughput_graph = (
    RESULTS / "spark_email_throughput.png"
)

plt.savefig(throughput_graph, dpi=300)
plt.close()

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n=== SAVED REPORT FILES ===")
print(summary_file)
print(runtime_graph)
print(speedup_graph)
print(throughput_graph)

print("\nEmail benchmark analysis successful!")
