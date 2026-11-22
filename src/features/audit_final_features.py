
from pathlib import Path
import os
import pandas as pd
import numpy as np

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

print("\n=== DATASET OVERVIEW ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique users:", df["user"].nunique())
print("Date range:", df["day"].min(), "to", df["day"].max())

print("\n=== DATA INTEGRITY ===")
print(
    "Duplicate user-days:",
    df.duplicated(["user", "day"]).sum()
)
print(
    "Missing user/day keys:",
    df[["user", "day"]].isna().sum().sum()
)

print("\n=== MISSING VALUES ===")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(
    ascending=False
)

print(
    missing.head(30).to_string()
    if not missing.empty
    else "No missing values"
)

print("\n=== CONSTANT COLUMNS ===")
constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=True) <= 1
]

for col in constant_cols:
    print(col)

print("\n=== NUMERIC FEATURE SUMMARY ===")
numeric = df.select_dtypes(include=np.number)

print(
    numeric.describe()
    .loc[["mean", "std", "min", "max"]]
    .T
    .head(40)
    .to_string()
)

print("\n=== BASELINE COVERAGE ===")
for col in [
    "has_user_baseline",
    "has_department_peer_baseline",
    "has_role_peer_baseline"
]:
    if col in df.columns:
        print(
            col,
            int(df[col].sum()),
            f"({df[col].mean() * 100:.2f}%)"
        )

print("\n=== POTENTIAL NON-MODEL COLUMNS ===")

non_model_columns = [
    "user",
    "day",
    "snapshot_month",
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor"
]

for col in non_model_columns:
    if col in df.columns:
        print(col)

print("\nFeature audit complete!")
