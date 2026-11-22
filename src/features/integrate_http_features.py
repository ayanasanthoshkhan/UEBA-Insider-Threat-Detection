
from pathlib import Path
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

PROCESSED = DATA_ROOT / "processed"

EXISTING = (
    PROCESSED / "integrated_features"
    / "logon_device_file_email_features.parquet"
)

HTTP = (
    PROCESSED / "http_daily_features"
    / "http_daily_features.parquet"
)

OUTPUT = (
    PROCESSED / "integrated_features"
    / "all_activity_features.parquet"
)

# Read previously aggregated datasets
existing = pd.read_parquet(EXISTING)
http = pd.read_parquet(HTTP)

print("\n=== INPUT DATASETS ===")
print("Existing user-days:", len(existing))
print("HTTP user-days:", len(http))

keys = ["user", "day"]

# Standardize join keys and validate
for name, df in [("existing", existing), ("http", http)]:
    df["user"] = df["user"].astype("string").str.strip()
    df["day"] = pd.to_datetime(
        df["day"], errors="raise"
    ).dt.date

    assert not df[keys].isna().any().any(), (
        f"{name}: missing keys"
    )
    assert not df.duplicated(keys).any(), (
        f"{name}: duplicate user-days"
    )

# Prevent accidental duplicate feature columns
overlap = (
    set(existing.columns) & set(http.columns)
) - set(keys)

assert not overlap, f"Overlapping columns: {overlap}"

# Full outer join
merged = existing.merge(
    http,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True
)

print("\n=== JOIN COVERAGE ===")
print(merged["_merge"].value_counts())

# Preserve source-presence information
merged["has_http_activity"] = (
    merged["_merge"] != "left_only"
).astype("int8")

merged = merged.drop(columns=["_merge"])

# Zero means no recorded activity for that source
feature_cols = [
    c for c in merged.columns if c not in keys
]

merged[feature_cols] = (
    merged[feature_cols]
    .fillna(0)
    .astype("int64")
)

# Validation
assert not merged[keys].isna().any().any()
assert not merged.duplicated(keys).any()
assert not merged.isna().any().any()

assert merged["has_http_activity"].sum() == len(http)
assert len(merged) >= len(existing)
assert len(merged) >= len(http)

merged = merged.sort_values(keys).reset_index(drop=True)

print("\n=== FIVE-SOURCE RESULTS ===")
print("Total user-days:", len(merged))
print("Unique users:", merged["user"].nunique())
print("Total columns:", len(merged.columns))
print("Missing values:", merged.isna().sum().sum())
print("Duplicate user-days:", merged.duplicated(keys).sum())

print("\nHTTP feature distribution:")
print(
    merged[
        ["http_request_count", "https_request_count"]
    ].describe()
)

# Save to Parquet
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pq.write_table(
    pa.Table.from_pandas(
        merged, preserve_index=False
    ),
    OUTPUT,
    compression="snappy"
)

saved_rows = pq.ParquetFile(OUTPUT).metadata.num_rows
assert saved_rows == len(merged)

print("\n=== OUTPUT ===")
print("Saved to:", OUTPUT)
print("Verified rows:", saved_rows)
print("\nFive-source integration successful!")
