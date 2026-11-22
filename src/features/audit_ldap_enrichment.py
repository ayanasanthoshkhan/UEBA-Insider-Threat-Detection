
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
    / "ldap_enriched_features.parquet"
)

df = pd.read_parquet(INPUT)

print("\n=== LDAP COVERAGE ===")
print("Total user-days:", len(df))
print("Matched:", int(df["has_ldap_match"].sum()))
print("Unmatched:", int((df["has_ldap_match"] == 0).sum()))

print("\n=== UNMATCHED LDAP RECORDS ===")
print(
    df.loc[
        df["has_ldap_match"] == 0,
        ["user", "day", "snapshot_month"]
    ].to_string(index=False)
)

print("\n=== UNKNOWN VALUES BY COLUMN ===")

org_cols = [
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor"
]

for col in org_cols:
    unknown = (
        df[col].astype("string").str.strip().eq("Unknown")
    )
    print(
        f"{col}: {int(unknown.sum()):,} "
        f"({unknown.mean() * 100:.2f}%)"
    )

print("\n=== UNKNOWN DEPARTMENT ANALYSIS ===")

unknown_dept = df[
    df["department"].astype("string") == "Unknown"
]

print("Unknown department user-days:", len(unknown_dept))
print("Distinct affected users:", unknown_dept["user"].nunique())

print("\nUnknown department by LDAP match:")
print(
    unknown_dept["has_ldap_match"]
    .value_counts()
    .to_string()
)

print("\nTop roles among Unknown departments:")
print(
    unknown_dept["role"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\n=== AUDIT COMPLETE ===")
