
from pathlib import Path
import os
import json
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

OUTPUT_DIR = (
    DATA_ROOT / "processed"
    / "feature_quality"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

# Columns used for identification or context
metadata = [
    "user",
    "day",
    "snapshot_month",
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor",
    "peer_history_days_x",
    "peer_history_days_y"
]

metadata = [c for c in metadata if c in df.columns]

# Constant features
constant = [
    c for c in df.columns
    if df[c].nunique(dropna=True) <= 1
]

# Candidate numerical features
numeric = df.select_dtypes(
    include="number"
).columns.tolist()

candidates = [
    c for c in numeric
    if c not in metadata
    and c not in constant
]

# Check suspected duplicate features
pairs = [
    ("file_activity_count", "unique_file_count"),
    ("unique_email_pc_count", "has_email_activity"),
    ("unique_http_pc_count", "has_http_activity")
]

duplicates = []

for a, b in pairs:
    if a in df.columns and b in df.columns:
        if df[a].equals(df[b]):
            duplicates.append([a, b])

# Missing value summary
missing = {
    c: int(df[c].isna().sum())
    for c in df.columns
    if df[c].isna().any()
}

report = {
    "rows": len(df),
    "total_columns": len(df.columns),
    "unique_users": int(df["user"].nunique()),
    "constant_columns": constant,
    "metadata_columns": metadata,
    "candidate_numeric_columns": candidates,
    "suspected_duplicate_pairs_confirmed": duplicates,
    "missing_values_by_column": missing,
    "notes": [
        "Candidates are not a finalized model feature set.",
        "Inspect and remove redundant features before training.",
        "Fit imputation and scaling using training data only.",
        "Preserve baseline availability flags.",
        "Review temporal leakage and labeling methodology.",
        "Exclude user identifiers and outcome/provenance fields."
    ]
}

OUTPUT = OUTPUT_DIR / "ml_feature_report.json"

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n=== ML FEATURE REPORT ===")
print("Rows:", report["rows"])
print("Total columns:", report["total_columns"])
print("Candidate numeric columns:", len(candidates))

print("\nConstant columns:")
print(constant)

print("\nConfirmed duplicate pairs:")
print(duplicates)

print("\nColumns with missing values:")
print(len(missing))

print("\nSaved report to:", OUTPUT)
print("\nML feature report generated successfully!")
