
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

BASE = DATA_ROOT / "processed" / "integrated_features"

INPUT = BASE / "user_baseline_features.parquet"
OUTPUT = BASE / "peer_baseline_features.parquet"

METRICS = [
    "login_count",
    "usb_connect_count",
    "file_activity_count",
    "email_activity_count",
    "http_request_count",
]

WINDOW = 30
MIN_HISTORY_DAYS = 7
MIN_GROUP_SIZE = 5

# ------------------------------------------
# Load historical user features
# ------------------------------------------

df = pd.read_parquet(INPUT)
df["day"] = pd.to_datetime(df["day"])

df = df.sort_values(
    ["user", "day"]
).reset_index(drop=True)

assert not df.duplicated(["user", "day"]).any()

print("\n=== INPUT ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------
# Historical department and role baselines
# ------------------------------------------

for group_col in ["department", "role"]:

    print(f"\nProcessing peer group: {group_col}")

    # Unknown groups should not be treated
    # as real comparable employee groups.

    valid = df.loc[
        df[group_col].notna()
        & df[group_col].ne("Unknown"),
        [group_col, "day", "user"] + METRICS
    ].copy()

    # Daily average among employees in a group.
    # Only group-days with at least 5 people
    # can contribute to the peer history.

    grouped = valid.groupby(
        [group_col, "day"],
        as_index=False
    ).agg(
        peer_n=("user", "nunique"),
        **{
            f"{metric}_daily_mean": (metric, "mean")
            for metric in METRICS
        }
    )

    grouped = grouped.loc[
        grouped["peer_n"] >= MIN_GROUP_SIZE
    ].copy()

    grouped = grouped.sort_values(
        [group_col, "day"]
    ).reset_index(drop=True)

    # Previous eligible group-days only.
    # Current-day peer behavior is not used.

    grouped["peer_history_days"] = (
        grouped.groupby(group_col).cumcount()
    )

    keep = [
        group_col,
        "day",
        "peer_history_days",
    ]

    for metric in METRICS:

        source = f"{metric}_daily_mean"

        history = grouped.groupby(
            group_col, sort=False
        )[source]

        mean_col = (
            f"{group_col}_{metric}_peer_mean"
        )

        grouped[mean_col] = history.transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    WINDOW,
                    min_periods=MIN_HISTORY_DAYS
                )
                .mean()
            )
        )

        keep.append(mean_col)

    df = df.merge(
        grouped[keep],
        on=[group_col, "day"],
        how="left",
        validate="many_to_one"
    )

    # Whether historical peer information
    # exists for this employee-day.

    valid_flag = f"has_{group_col}_peer_baseline"

    first_metric = (
        f"{group_col}_{METRICS[0]}_peer_mean"
    )

    df[valid_flag] = (
        df[first_metric].notna().astype("int8")
    )

    for metric in METRICS:

        peer_mean = (
            f"{group_col}_{metric}_peer_mean"
        )

        deviation = (
            f"{group_col}_{metric}_peer_difference"
        )

        # Difference from historical
        # group average (not a z-score).

        df[deviation] = (
            df[metric] - df[peer_mean]
        )

    print(
        "Peer baseline available:",
        int(df[valid_flag].sum())
    )

# ------------------------------------------
# Validation
# ------------------------------------------

assert len(df) == 330452, (
    "Unexpected change in user-day count"
)

assert not df.duplicated(
    ["user", "day"]
).any()

assert not df[["user", "day"]].isna().any().any()

print("\n=== PEER BASELINE COVERAGE ===")

for col in [
    "has_department_peer_baseline",
    "has_role_peer_baseline"
]:
    print(col, int(df[col].sum()))

print("\n=== OUTPUT DATA ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Missing peer features are intentional
# when groups or history are insufficient.
# ML preprocessing must handle them.

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

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

print("Saved rows:", saved_rows)
print("Saved to:", OUTPUT)
print("\nPeer baseline engineering successful!")
