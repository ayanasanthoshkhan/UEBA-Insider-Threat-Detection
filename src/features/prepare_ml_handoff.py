
from pathlib import Path
import os
import json

import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================

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

OUTPUT_DIR = DATA_ROOT / "processed" / "ml_handoff"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = OUTPUT_DIR / "ml_candidate_features.parquet"
METADATA_FILE = OUTPUT_DIR / "ml_context_metadata.parquet"
MANIFEST_FILE = OUTPUT_DIR / "feature_manifest.json"

# ==========================================
# LOAD AND VALIDATE
# ==========================================

df = pd.read_parquet(INPUT)
df["day"] = pd.to_datetime(df["day"])

assert len(df) == 330452
assert df["user"].nunique() == 1000
assert not df.duplicated(["user", "day"]).any()
assert not df[["user", "day"]].isna().any().any()

print("\n=== SOURCE DATASET ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==========================================
# FEATURE EXCLUSIONS
# ==========================================

# Identifiers and context are not model inputs.
context_cols = [
    "role",
    "business_unit",
    "functional_unit",
    "department",
    "team",
    "supervisor",
    "snapshot_month",
]

# Constant, redundant or audit-only columns.
excluded_numeric = {
    "has_logon_activity",
    "https_request_count",
    "unique_file_count",
    "unique_email_pc_count",
    "unique_http_pc_count",
    "has_ldap_match",
    "prior_user_days",
    "peer_history_days_x",
    "peer_history_days_y",
}

# Automatically detect other constants.
numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

constant_cols = {
    c for c in numeric_cols
    if df[c].nunique(dropna=True) <= 1
}

excluded_numeric.update(constant_cols)

feature_cols = [
    c for c in numeric_cols
    if c not in excluded_numeric
]

assert len(feature_cols) > 0

# ==========================================
# PREPARE CANDIDATE FEATURES
# ==========================================

# Preserve identifiers for labeling and
# chronological splits, not as predictors.

features = df[
    ["user", "day"] + feature_cols
].copy()

# Preserve missing historical values.
# Do NOT impute or normalize here.

assert len(features) == len(df)
assert not features.duplicated(
    ["user", "day"]
).any()

features.to_parquet(
    FEATURE_FILE,
    index=False
)

# ==========================================
# SEPARATE ORGANIZATIONAL CONTEXT
# ==========================================

available_context = [
    c for c in context_cols
    if c in df.columns
]

metadata = df[
    ["user", "day"] + available_context
].copy()

metadata.to_parquet(
    METADATA_FILE,
    index=False
)

# ==========================================
# FEATURE MANIFEST
# ==========================================

missing_counts = {
    c: int(features[c].isna().sum())
    for c in feature_cols
    if features[c].isna().any()
}

manifest = {
    "dataset": "CERT r4.2 UEBA",
    "granularity": "user-day",
    "rows": int(len(features)),
    "users": int(features["user"].nunique()),
    "date_min": str(features["day"].min().date()),
    "date_max": str(features["day"].max().date()),
    "feature_count": len(feature_cols),
    "candidate_features": feature_cols,
    "excluded_numeric_columns": sorted(
        excluded_numeric.intersection(df.columns)
    ),
    "context_columns": available_context,
    "missing_values": missing_counts,
    "modeling_rules": [
        "Candidate features are not final approved predictors.",
        "Join labels using user and day keys.",
        "Use chronological train/validation/test splits.",
        "Fit imputation on training data only.",
        "Fit scaling on training data only.",
        "Never use user or day as direct model predictors.",
        "Review behavioral and temporal leakage.",
        "Review historical LDAP membership assumptions.",
        "Keep baseline availability flags.",
        "Do not interpret missing baselines as zero.",
        "Evaluate label mapping and class imbalance."
    ]
}

with MANIFEST_FILE.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=2)

# ==========================================
# VERIFY SAVED FILES
# ==========================================

saved = pd.read_parquet(FEATURE_FILE)

assert len(saved) == len(df)
assert list(saved.columns) == [
    "user", "day"
] + feature_cols

print("\n=== ML HANDOFF SUMMARY ===")
print("User-days:", len(saved))
print("Candidate numeric features:", len(feature_cols))
print("Total handoff columns:", len(saved.columns))
print("Columns with missing values:", len(missing_counts))

print("\n=== EXCLUDED NUMERIC COLUMNS ===")
for col in sorted(excluded_numeric):
    if col in df.columns:
        print("-", col)

print("\n=== OUTPUT FILES ===")
print(FEATURE_FILE)
print(METADATA_FILE)
print(MANIFEST_FILE)

print("\nML handoff preparation successful!")
