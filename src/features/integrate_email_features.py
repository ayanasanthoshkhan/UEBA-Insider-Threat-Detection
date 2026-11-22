
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

BASE = DATA_ROOT / "processed"

EXISTING = (
    BASE / "integrated_features"
    / "logon_device_file_features.parquet"
)

EMAIL = (
    BASE / "email_daily_features"
    / "email_daily_features.parquet"
)

OUTPUT = (
    BASE / "integrated_features"
    / "logon_device_file_email_features.parquet"
)

# Load already-aggregated data
existing = pd.read_parquet(EXISTING)
email = pd.read_parquet(EMAIL)

print("Existing user-days:", len(existing))
print("Email user-days:", len(email))

keys = ["user", "day"]

# Standardize keys and check uniqueness
for name, df in [("existing", existing), ("email", email)]:
    df["user"] = df["user"].astype("string").str.strip()
    df["day"] = pd.to_datetime(df["day"]).dt.date

    assert not df[keys].isna().any().any(), (
        f"{name}: missing keys"
    )
    assert not df.duplicated(keys).any(), (
        f"{name}: duplicate user-days"
    )

# Check for overlapping feature names
overlap = (
    set(existing.columns) & set(email.columns)
) - set(keys)

assert not overlap, f"Overlapping columns: {overlap}"

# Preserve days appearing in either input
merged = existing.merge(
    email,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True
)

print("\n=== JOIN COVERAGE ===")
print(merged["_merge"].value_counts())

# Source presence flags
merged["has_email_activity"] = (
    merged["_merge"] != "left_only"
).astype("int8")

merged = merged.drop(columns=["_merge"])

# Missing activity means zero recorded events
feature_cols = [c for c in merged.columns if c not in keys]

merged[feature_cols] = (
    merged[feature_cols]
    .fillna(0)
    .astype("int64")
)

# Validate integration
assert not merged[keys].isna().any().any()
assert not merged.duplicated(keys).any()
assert not merged.isna().any().any()
assert merged["has_email_activity"].sum() == len(email)

merged = merged.sort_values(keys).reset_index(drop=True)

print("\n=== INTEGRATED RESULTS ===")
print("Total user-days:", len(merged))
print("Unique users:", merged["user"].nunique())
print("Total columns:", len(merged.columns))
print("Missing values:", merged.isna().sum().sum())
print("Duplicate user-days:", merged.duplicated(keys).sum())

# Save and verify
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pq.write_table(
    pa.Table.from_pandas(merged, preserve_index=False),
    OUTPUT,
    compression="snappy"
)

saved_rows = pq.ParquetFile(OUTPUT).metadata.num_rows
assert saved_rows == len(merged)

print("\nSaved to:", OUTPUT)
print("Verified rows:", saved_rows)
print("\nFour-source integration successful!")
