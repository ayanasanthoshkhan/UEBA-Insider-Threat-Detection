
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ==========================================
# PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT = (
    DATA_ROOT / "processed"
    / "integrated_features"
    / "ldap_enriched_features.parquet"
)

OUTPUT = (
    DATA_ROOT / "processed"
    / "integrated_features"
    / "user_baseline_features.parquet"
)

# ==========================================
# CONFIGURATION
# ==========================================

WINDOW = 30
MIN_HISTORY = 7

METRICS = [
    "login_count",
    "usb_connect_count",
    "file_activity_count",
    "email_activity_count",
    "http_request_count"
]

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_parquet(INPUT)

df["day"] = pd.to_datetime(df["day"])

df = df.sort_values(
    ["user", "day"]
).reset_index(drop=True)

assert not df.duplicated(
    ["user", "day"]
).any()

assert all(m in df.columns for m in METRICS)

print("\n=== INPUT ===")
print("User-days:", len(df))
print("Users:", df["user"].nunique())

# ==========================================
# HISTORY COVERAGE
# ==========================================

df["prior_user_days"] = (
    df.groupby("user").cumcount()
)

df["has_user_baseline"] = (
    df["prior_user_days"] >= MIN_HISTORY
).astype("int8")

# ==========================================
# HISTORICAL ROLLING BASELINES
# ==========================================

for metric in METRICS:

    print("Processing:", metric)

    grouped = df.groupby(
        "user", sort=False
    )[metric]

    # Shift excludes the current observation.
    # Rolling window uses the previous
    # 30 observed user-days, not calendar days.

    previous = grouped.shift(1)

    rolling = previous.groupby(
        df["user"], sort=False
    ).rolling(
        window=WINDOW,
        min_periods=MIN_HISTORY
    )

    mean = rolling.mean().reset_index(
        level=0, drop=True
    )

    std = rolling.std(
        ddof=0
    ).reset_index(
        level=0, drop=True
    )

    # Define features
    mean_col = f"{metric}_hist_mean"
    std_col = f"{metric}_hist_std"
    z_col = f"{metric}_hist_z"

    df[mean_col] = mean

    df[std_col] = std

    # Avoid division by zero and undefined
    # standardized values.
    safe_std = std.where(std > 0, np.nan)

    z_score = (
        (df[metric] - mean) / safe_std
    )

    # Missing baselines remain distinguishable
    # through has_user_baseline.
    df[z_col] = (
        z_score
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-20, 20)
    )

# ==========================================
# VALIDATE
# ==========================================

print("\n=== BASELINE COVERAGE ===")

print(
    df["has_user_baseline"]
    .value_counts()
    .to_string()
)

assert len(df) > 0

assert not df.duplicated(
    ["user", "day"]
).any()

for metric in METRICS:
    for suffix in ["hist_mean", "hist_std"]:
        col = f"{metric}_{suffix}"
        assert df.loc[
            df["has_user_baseline"] == 1,
            col
        ].notna().all()

print("\nBaseline feature sample:")

cols = [
    "user",
    "day",
    "file_activity_count",
    "file_activity_count_hist_mean",
    "file_activity_count_hist_z",
    "has_user_baseline"
]

print(
    df[cols].head(15).to_string(index=False)
)

# ==========================================
# SAVE DATA
# ==========================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Preserve missing historical statistics
# for rows without sufficient history.
# Downstream ML preprocessing will handle
# these missing values using training-only rules.

pq.write_table(
    pa.Table.from_pandas(
        df,
        preserve_index=False
    ),
    OUTPUT,
    compression="snappy"
)

saved_rows = pq.ParquetFile(
    OUTPUT
).metadata.num_rows

assert saved_rows == len(df)

print("\n=== OUTPUT ===")
print("Rows:", saved_rows)
print("Columns:", len(df.columns))
print("Saved to:", OUTPUT)
print("\nUser baseline engineering successful!")
