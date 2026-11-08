
import pandas as pd

path = (
    r"D:\UEBA_Capstone\data\processed"
    r"\logon_daily_features"
    r"\logon_daily_features.parquet"
)

df = pd.read_parquet(path)

print("=== FEATURE DATASET VALIDATION ===")

print("Total rows:", len(df))
print("Unique users:", df["user"].nunique())
print("Date range:", df["day"].min(), "to", df["day"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate user-day combinations:")
print(df.duplicated(subset=["user", "day"]).sum())

print("\nFeature summary:")
print(
    df[
        [
            "login_count",
            "logoff_count",
            "after_hours_login_count",
            "unique_pc_count",
            "weekend_login_count",
        ]
    ].describe()
)

assert not df[["user", "day"]].isnull().any().any()
assert df.duplicated(subset=["user", "day"]).sum() == 0

print("\nBasic validation passed!")
