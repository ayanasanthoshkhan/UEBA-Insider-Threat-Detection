
from pathlib import Path
import os

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

PROCESSED = DATA_ROOT / "processed"

LOGON_DEVICE = (
    PROCESSED
    / "integrated_features"
    / "logon_device_features.parquet"
)

FILE_FEATURES = (
    PROCESSED
    / "file_daily_features"
    / "file_daily_features.parquet"
)

OUTPUT_DIR = PROCESSED / "integrated_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "logon_device_file_features.parquet"
)

# ==========================================
# LOAD FEATURE TABLES
# ==========================================

logon_device = pd.read_parquet(LOGON_DEVICE)
file_data = pd.read_parquet(FILE_FEATURES)

print("\n=== INPUT DATASETS ===")
print("Logon + Device:", len(logon_device))
print("File:", len(file_data))

# ==========================================
# STANDARDIZE JOIN KEYS
# ==========================================

keys = ["user", "day"]

for name, df in [
    ("logon_device", logon_device),
    ("file", file_data)
]:
    df["user"] = df["user"].astype("string").str.strip()
    df["day"] = pd.to_datetime(df["day"]).dt.date

    assert not df[keys].isna().any().any(), (
        f"{name}: missing join keys"
    )

    assert not df.duplicated(keys).any(), (
        f"{name}: duplicate user-day records"
    )

# ==========================================
# MERGE FEATURE TABLES
# ==========================================

merged = logon_device.merge(
    file_data,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True
)

print("\n=== JOIN COVERAGE ===")
print(merged["_merge"].value_counts())

# Keep track of original activity sources
merged["has_file_activity"] = (
    merged["_merge"] != "left_only"
).astype("int8")

merged = merged.drop(columns=["_merge"])

# ==========================================
# HANDLE MISSING ACTIVITY VALUES
# ==========================================

feature_columns = [
    column
    for column in merged.columns
    if column not in keys
]

merged[feature_columns] = (
    merged[feature_columns]
    .fillna(0)
    .astype("int64")
)

# ==========================================
# VALIDATION
# ==========================================

assert not merged.duplicated(keys).any()
assert not merged[keys].isna().any().any()
assert not merged.isna().any().any()

assert (
    merged["has_file_activity"].sum()
    == len(file_data)
), "File activity coverage mismatch"

assert len(merged) >= len(logon_device)
assert len(merged) >= len(file_data)

merged = (
    merged.sort_values(keys)
    .reset_index(drop=True)
)

print("\n=== FINAL INTEGRATED DATASET ===")
print("Total user-days:", len(merged))
print("Unique users:", merged["user"].nunique())
print("Total columns:", len(merged.columns))
print("Missing values:", merged.isna().sum().sum())
print("Duplicate user-days:", merged.duplicated(keys).sum())

print("\nFirst 5 records:")
print(merged.head())

# ==========================================
# SAVE PARQUET
# ==========================================

table = pa.Table.from_pandas(
    merged,
    preserve_index=False
)

pq.write_table(
    table,
    OUTPUT_FILE,
    compression="snappy"
)

saved_rows = pq.ParquetFile(
    OUTPUT_FILE
).metadata.num_rows

assert saved_rows == len(merged)

print("\n=== OUTPUT ===")
print("Saved to:", OUTPUT_FILE)
print("Verified rows:", saved_rows)

print("\nThree-source integration successful!")
