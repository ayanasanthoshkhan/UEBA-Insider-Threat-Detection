
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(r"D:\UEBA_Capstone\data\processed")

LOGON = (
    BASE / "logon_daily_features"
    / "logon_daily_features.parquet"
)
DEVICE = (
    BASE / "device_daily_features"
    / "device_daily_features.parquet"
)

OUTPUT_DIR = BASE / "integrated_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "logon_device_features.parquet"

# 1. Load the aggregated feature tables
logon = pd.read_parquet(LOGON)
device = pd.read_parquet(DEVICE)

print("Logon user-days:", len(logon))
print("Device user-days:", len(device))

# 2. Standardize keys
for df in (logon, device):
    df["user"] = df["user"].astype("string").str.strip()
    df["day"] = pd.to_datetime(df["day"]).dt.date

# 3. Validate keys before joining
keys = ["user", "day"]

for name, df in [("logon", logon), ("device", device)]:
    assert not df[keys].isna().any().any(), (
        f"{name} has missing user-day keys"
    )
    assert not df.duplicated(keys).any(), (
        f"{name} has duplicate user-day keys"
    )

# 4. Full outer join preserves days from either source
merged = logon.merge(
    device,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

print("\n=== JOIN COVERAGE ===")
print(merged["_merge"].value_counts())

# 5. Record source presence before filling missing counts
merged["has_logon_activity"] = (
    merged["_merge"] != "right_only"
).astype("int8")

merged["has_device_activity"] = (
    merged["_merge"] != "left_only"
).astype("int8")

merged = merged.drop(columns=["_merge"])

# 6. Missing activity counts become zero
feature_cols = [
    c for c in merged.columns
    if c not in keys
    and c not in [
        "has_logon_activity",
        "has_device_activity"
    ]
]

merged[feature_cols] = (
    merged[feature_cols]
    .fillna(0)
    .astype("int64")
)

# 7. Final validation
assert not merged.duplicated(keys).any()
assert not merged[keys].isna().any().any()

assert merged["has_logon_activity"].sum() == len(logon)
assert merged["has_device_activity"].sum() == len(device)

merged = merged.sort_values(keys).reset_index(drop=True)

# 8. Save output
table = pa.Table.from_pandas(
    merged,
    preserve_index=False
)

pq.write_table(
    table,
    OUTPUT,
    compression="snappy"
)

saved_rows = pq.ParquetFile(OUTPUT).metadata.num_rows

assert saved_rows == len(merged)

print("\n=== INTEGRATED DATASET ===")
print("Total unique user-days:", len(merged))
print("Unique users:", merged["user"].nunique())
print("Missing values:", merged.isna().sum().sum())
print("Duplicate user-days:", merged.duplicated(keys).sum())

print("\nFirst 5 records:")
print(merged.head())

print("\nSaved to:", OUTPUT)
print("Verified rows:", saved_rows)
print("\nIntegration successful!")
