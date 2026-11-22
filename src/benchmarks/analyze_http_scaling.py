
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
INPUT = BENCHMARK_DIR / "spark_http_scaling.csv"

OUTPUT_DIR = BENCHMARK_DIR / "http_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD AND VALIDATE
# ==========================================

df = pd.read_csv(INPUT)

for col in [
    "input_mib",
    "cores",
    "run",
    "events",
    "user_days",
    "runtime_seconds",
]:
    df[col] = pd.to_numeric(df[col])

assert set(df["input_mib"].round().astype(int)) == {
    256, 512, 1024
}
assert set(df["cores"]) == {4}
assert df.groupby(["input_mib", "run"]).size().eq(1).all()
assert df.groupby("input_mib")["run"].nunique().ge(3).all()
assert df["runtime_seconds"].gt(0).all()

expected_events = {
    256: 525559,
    512: 1051198,
    1024: 2102687,
}

for size, events in expected_events.items():
    subset = df[df["input_mib"].round() == size]
    assert subset["events"].eq(events).all(), (
        f"Unexpected event count at {size} MiB"
    )
    assert subset["user_days"].nunique() == 1

# ==========================================
# SUMMARY STATISTICS
# ==========================================

summary = (
    df.groupby("input_mib")
    .agg(
        trials=("runtime_seconds", "count"),
        events=("events", "first"),
        user_days=("user_days", "first"),
        mean_runtime=("runtime_seconds", "mean"),
        std_runtime=("runtime_seconds", "std"),
        min_runtime=("runtime_seconds", "min"),
        max_runtime=("runtime_seconds", "max"),
    )
    .reset_index()
    .sort_values("input_mib")
)

summary["events_per_second"] = (
    summary["events"] / summary["mean_runtime"]
)

summary["mib_per_second"] = (
    summary["input_mib"] / summary["mean_runtime"]
)

base = summary.iloc[0]

summary["input_growth_factor"] = (
    summary["input_mib"] / base["input_mib"]
)

summary["runtime_growth_factor"] = (
    summary["mean_runtime"] / base["mean_runtime"]
)

OUTPUT_CSV = OUTPUT_DIR / "spark_http_scaling_summary.csv"
summary.to_csv(OUTPUT_CSV, index=False)

print("\n=== HTTP SCALABILITY SUMMARY ===")
print(summary.round(3).to_string(index=False))

# ==========================================
# GRAPH 1: MEAN RUNTIME
# ==========================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["input_mib"],
    summary["mean_runtime"],
    yerr=summary["std_runtime"],
    fmt="o-",
    capsize=5,
)

plt.xlabel("HTTP input size (MiB)")
plt.ylabel("Mean runtime (seconds)")
plt.title("Apache Spark HTTP Input-Size Scaling")
plt.xticks([256, 512, 1024])
plt.grid(True, alpha=0.3)
plt.tight_layout()

runtime_png = OUTPUT_DIR / "http_runtime_scaling.png"
plt.savefig(runtime_png, dpi=300)
plt.close()

# ==========================================
# GRAPH 2: THROUGHPUT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    summary["input_mib"],
    summary["mib_per_second"],
    "o-"
)

plt.xlabel("HTTP input size (MiB)")
plt.ylabel("Effective throughput (MiB/s)")
plt.title("Apache Spark HTTP Processing Throughput")
plt.xticks([256, 512, 1024])
plt.grid(True, alpha=0.3)
plt.tight_layout()

throughput_png = OUTPUT_DIR / "http_throughput_scaling.png"
plt.savefig(throughput_png, dpi=300)
plt.close()

# ==========================================
# GRAPH 3: INPUT GROWTH VS RUNTIME GROWTH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    summary["input_growth_factor"],
    summary["runtime_growth_factor"],
    "o-",
    label="Measured runtime growth"
)

plt.plot(
    summary["input_growth_factor"],
    summary["input_growth_factor"],
    "--",
    label="Proportional growth reference"
)

plt.xlabel("Input size relative to 256 MiB")
plt.ylabel("Runtime relative to 256 MiB")
plt.title("HTTP Workload Growth Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

growth_png = OUTPUT_DIR / "http_growth_comparison.png"
plt.savefig(growth_png, dpi=300)
plt.close()

print("\n=== SAVED FILES ===")
print(OUTPUT_CSV)
print(runtime_png)
print(throughput_png)
print(growth_png)

print("\nHTTP scalability analysis successful!")
