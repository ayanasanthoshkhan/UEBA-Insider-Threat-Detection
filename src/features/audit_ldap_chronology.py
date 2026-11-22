
from pathlib import Path
import os

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

df = pd.read_parquet(INPUT)

df["day"] = pd.to_datetime(df["day"])
df["snapshot_month"] = pd.to_datetime(
    df["snapshot_month"]
)

# The preceding calendar month is used
# for every observed user-day.
expected_month = (
    df["day"].dt.to_period("M") - 1
).dt.to_timestamp()

print("\n=== LDAP CHRONOLOGY AUDIT ===")
print("User-days:", len(df))

print("\n=== SNAPSHOT TIMING ===")
future_snapshot = (
    df["snapshot_month"] >=
    df["day"].dt.to_period("M").dt.to_timestamp()
)

wrong_month = (
    df["snapshot_month"] != expected_month
)

print(
    "Future/current-month snapshots:",
    int(future_snapshot.sum())
)
print(
    "Unexpected snapshot months:",
    int(wrong_month.sum())
)

assert not future_snapshot.any(), (
    "Current or future LDAP snapshot found"
)

assert not wrong_month.any(), (
    "Unexpected LDAP snapshot month found"
)

# Check organizational values
print("\n=== UNKNOWN ORGANIZATIONAL VALUES ===")

for col in [
    "role",
    "department",
    "team",
    "supervisor"
]:
    count = int(df[col].eq("Unknown").sum())
    print(f"{col}: {count}")

# Confirm matching flags
matched = int(df["has_ldap_match"].sum())

print("\nLDAP matched user-days:", matched)
print("LDAP unmatched user-days:", len(df) - matched)

print("\nLDAP chronology audit successful!")
