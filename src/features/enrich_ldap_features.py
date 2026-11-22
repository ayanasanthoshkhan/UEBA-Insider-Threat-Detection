
from pathlib import Path
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ==========================================
# PROJECT PATHS
# ==========================================

DATA_ROOT = Path(
    os.environ.get(
        "UEBA_DATA_ROOT",
        r"D:\UEBA_Capstone\data"
    )
)

LDAP_DIR = DATA_ROOT / "raw" / "r4.2" / "LDAP"

INPUT_FILE = (
    DATA_ROOT / "processed"
    / "integrated_features"
    / "all_activity_features.parquet"
)

OUTPUT_FILE = (
    DATA_ROOT / "processed"
    / "integrated_features"
    / "ldap_enriched_features.parquet"
)

# ==========================================
# LOAD ACTIVITY FEATURES
# ==========================================

activity = pd.read_parquet(INPUT_FILE)

activity["user"] = (
    activity["user"].astype("string").str.strip()
)
activity["day"] = pd.to_datetime(
    activity["day"], errors="raise"
).dt.date

print("\n=== ACTIVITY DATA ===")
print("User-days:", len(activity))
print("Users:", activity["user"].nunique())
print("Columns:", len(activity.columns))

assert not activity[["user", "day"]].isna().any().any()
assert not activity.duplicated(["user", "day"]).any()

# ==========================================
# LOAD MONTHLY LDAP SNAPSHOTS
# ==========================================

ldap_files = sorted(LDAP_DIR.glob("????-??.csv"))

if not ldap_files:
    raise FileNotFoundError("No LDAP snapshots found")

ldap_tables = []

ldap_columns = [
    "user_id",
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor"
]

for path in ldap_files:

    snapshot = pd.read_csv(
        path,
        dtype="string",
        usecols=ldap_columns
    )

    snapshot["snapshot_month"] = path.stem

    snapshot = snapshot.rename(
        columns={"user_id": "user"}
    )

    snapshot["user"] = (
        snapshot["user"].str.strip()
    )

    ldap_tables.append(snapshot)

ldap = pd.concat(
    ldap_tables,
    ignore_index=True
)

print("\n=== LDAP DATA ===")
print("Snapshots:", len(ldap_files))
print("LDAP rows:", len(ldap))
print("Distinct LDAP users:", ldap["user"].nunique())

# ==========================================
# VALIDATE LDAP KEYS
# ==========================================

assert not ldap[
    ["user", "snapshot_month"]
].isna().any().any(), "Missing LDAP keys"

assert not ldap.duplicated(
    ["user", "snapshot_month"]
).any(), "Duplicate employee-month in LDAP"

# ==========================================
# TIME-AWARE LDAP MATCHING
# ==========================================

# Use previous month's snapshot, not future data.
# Example: January 2010 -> December 2009.

activity["snapshot_month"] = (
    pd.to_datetime(activity["day"])
    .dt.to_period("M")
    .sub(1)
    .astype(str)
)

enriched = activity.merge(
    ldap,
    on=["user", "snapshot_month"],
    how="left",
    validate="many_to_one",
    indicator=True
)

# ==========================================
# LDAP COVERAGE
# ==========================================

enriched["has_ldap_match"] = (
    enriched["_merge"] == "both"
).astype("int8")

enriched = enriched.drop(columns=["_merge"])

matched = int(enriched["has_ldap_match"].sum())
unmatched = len(enriched) - matched

print("\n=== LDAP MATCH COVERAGE ===")
print("Matched user-days:", matched)
print("Unmatched user-days:", unmatched)
print("Match rate (%):", round(
    matched / len(enriched) * 100, 2
))

print("\nCoverage by activity month:")

enriched["activity_month"] = (
    pd.to_datetime(enriched["day"])
    .dt.to_period("M")
    .astype(str)
)

print(
    enriched.groupby("activity_month")[
        "has_ldap_match"
    ].agg(["count", "sum", "mean"])
    .to_string()
)

# ==========================================
# HANDLE UNKNOWN ORGANIZATIONAL FIELDS
# ==========================================

org_columns = [
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor"
]

for column in org_columns:
    enriched[column] = (
        enriched[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown")
    )

# These categorical fields are metadata.
# Encode them later using training-only rules.
# Do not replace categories with numeric IDs here.

# ==========================================
# FINAL VALIDATION
# ==========================================

enriched = enriched.drop(
    columns=["activity_month"]
)

keys = ["user", "day"]

assert len(enriched) == len(activity), (
    "LDAP enrichment changed the row count"
)

assert not enriched.duplicated(keys).any(), (
    "Duplicate user-day records after LDAP join"
)

assert not enriched[keys].isna().any().any()

assert enriched["has_ldap_match"].isin(
    [0, 1]
).all()

# ==========================================
# SAVE PARQUET
# ==========================================

enriched = enriched.sort_values(
    keys
).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

pq.write_table(
    pa.Table.from_pandas(
        enriched,
        preserve_index=False
    ),
    OUTPUT_FILE,
    compression="snappy"
)

saved_rows = pq.ParquetFile(
    OUTPUT_FILE
).metadata.num_rows

assert saved_rows == len(activity)

# ==========================================
# RESULTS
# ==========================================

print("\n=== ENRICHED DATASET ===")
print("Rows:", len(enriched))
print("Unique users:", enriched["user"].nunique())
print("Columns:", len(enriched.columns))
print("Duplicate user-days:", enriched.duplicated(keys).sum())

print("\nTop departments:")
print(
    enriched["department"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\n=== OUTPUT ===")
print("Saved to:", OUTPUT_FILE)
print("Verified rows:", saved_rows)

print("\nLDAP enrichment successful!")
