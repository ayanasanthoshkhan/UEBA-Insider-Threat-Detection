
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

# Load saved features
df = pd.read_parquet(INPUT)

df["day"] = pd.to_datetime(df["day"])

df = df.sort_values(
    ["user", "day"]
).reset_index(drop=True)

assert not df.duplicated(["user", "day"]).any()

print("\n=== TEMPORAL BASELINE AUDIT ===")
print("User-days:", len(df))
print("Users:", df["user"].nunique())

# Independently recompute user history
# using strictly previous observed days.

for metric in METRICS:

    print(f"\nChecking: {metric}")

    history = (
        df.groupby("user")[metric]
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

    saved = df[f"{metric}_hist_mean"]

    matches = np.isclose(
        history.to_numpy(dtype=float),
        saved.to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )

    mismatches = int((~matches).sum())

    print("Mismatched user-days:", mismatches)

    assert mismatches == 0, (
        f"Historical mean mismatch: {metric}"
    )

# Verify warm-up periods
expected_prior = df.groupby(
    "user"
).cumcount()

assert (
    df["prior_user_days"].to_numpy()
    == expected_prior.to_numpy()
).all()

expected_flag = (
    expected_prior >= MIN_HISTORY
).astype("int8")

assert (
    df["has_user_baseline"].to_numpy()
    == expected_flag.to_numpy()
).all()

# Check invalid peer-baseline flags
for group in ["department", "role"]:

    flag = f"has_{group}_peer_baseline"

    means = [
        f"{group}_{metric}_peer_mean"
        for metric in METRICS
    ]

    assert not df.loc[
        df[flag] == 1, means
    ].isna().any().any()

    assert df.loc[
        df[group].eq("Unknown"), flag
    ].eq(0).all()

    print(
        f"\n{group} valid peer baselines:",
        int(df[flag].sum())
    )

print("\n=== AUDIT RESULT ===")
print("Historical user means verified.")
print("Warm-up flags verified.")
print("Peer-baseline availability checks passed.")
print(
    "Note: Peer historical values and all "
    "other leakage risks require separate review."
)
print("\nTemporal baseline audit successful!")
