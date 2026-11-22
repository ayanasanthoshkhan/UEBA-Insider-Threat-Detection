
from pathlib import Path
import os

import numpy as np
import pandas as pd

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

INPUT = (
    DATA_ROOT / "processed"
    / "integrated_features"
    / "peer_baseline_features.parquet"
)

METRICS = [
    "login_count",
    "usb_connect_count",
    "file_activity_count",
    "email_activity_count",
    "http_request_count",
]

WINDOW = 30
MIN_HISTORY = 7
MIN_GROUP_SIZE = 5

df = pd.read_parquet(INPUT)
df["day"] = pd.to_datetime(df["day"])

assert not df.duplicated(["user", "day"]).any()

print("\n=== PEER BASELINE TEMPORAL AUDIT ===")
print("Rows:", len(df))
print("Users:", df["user"].nunique())

for group_col in ["department", "role"]:

    print(f"\n=== CHECKING {group_col.upper()} ===")

    valid = df.loc[
        df[group_col].notna()
        & df[group_col].ne("Unknown"),
        [group_col, "day", "user"] + METRICS
    ].copy()

    # Reconstruct daily group averages.
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

    expected = grouped[[group_col, "day"]].copy()

    for metric in METRICS:

        source = f"{metric}_daily_mean"

        mean_col = (
            f"{group_col}_{metric}_peer_mean"
        )

        # Exclude current group-day.
        expected[mean_col] = (
            grouped.groupby(group_col)[source]
            .transform(
                lambda s: (
                    s.shift(1)
                    .rolling(
                        window=WINDOW,
                        min_periods=MIN_HISTORY
                    )
                    .mean()
                )
            )
        )

    # Merge independent calculations.
    check = df[
        ["user", "day", group_col]
        + [
            f"{group_col}_{m}_peer_mean"
            for m in METRICS
        ]
        + [f"has_{group_col}_peer_baseline"]
    ].merge(
        expected,
        on=[group_col, "day"],
        how="left",
        suffixes=("_saved", "_expected"),
        validate="many_to_one"
    )

    total_mismatches = 0

    for metric in METRICS:

        name = f"{group_col}_{metric}_peer_mean"

        saved = check[f"{name}_saved"].to_numpy(
            dtype=float
        )

        recomputed = check[
            f"{name}_expected"
        ].to_numpy(dtype=float)

        matches = np.isclose(
            saved,
            recomputed,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True
        )

        mismatches = int((~matches).sum())

        print(
            f"{metric}: {mismatches} mismatches"
        )

        total_mismatches += mismatches

    flag = f"has_{group_col}_peer_baseline"

    reference_mean = (
        f"{group_col}_{METRICS[0]}_peer_mean_expected"
    )

    expected_flag = (
        check[reference_mean].notna()
        .astype("int8")
    )

    flag_mismatches = int(
        (check[flag] != expected_flag).sum()
    )

    print("Flag mismatches:", flag_mismatches)

    assert total_mismatches == 0, (
        f"{group_col} historical means differ"
    )

    assert flag_mismatches == 0, (
        f"{group_col} baseline flags differ"
    )

    print(f"{group_col} audit passed!")

print("\n=== FINAL RESULT ===")
print("Historical peer means verified.")
print("Peer availability flags verified.")
print("No peer baseline mismatches detected.")
print("\nPeer baseline temporal audit successful!")
