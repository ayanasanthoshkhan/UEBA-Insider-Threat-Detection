
import pandas as pd
from pathlib import Path

# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading incident events and ML features...")

events = pd.read_csv(
    "work/r42_observables_parsed.csv",
    dtype=str
).fillna("")

features = pd.read_parquet(
    "data/ml_handoff/ml_candidate_features.parquet"
)

print("Incident events:", len(events))
print("ML feature rows:", len(features))


# ==================================================
# 2. NORMALIZE DATES
# ==================================================

events["day"] = pd.to_datetime(
    events["timestamp"],
    errors="coerce",
    format="mixed"
).dt.normalize()

features["day"] = pd.to_datetime(
    features["day"],
    errors="coerce"
).dt.normalize()

if events["day"].isna().any():
    raise ValueError("Some incident dates could not be parsed.")

if features["day"].isna().any():
    raise ValueError("Some feature dates could not be parsed.")


# ==================================================
# 3. SAVE SCENARIO 3 CROSS-ACCOUNT AUDIT
# ==================================================

cross_account = events[
    events["incident_user"] != events["logged_user"]
].copy()

cross_account.to_csv(
    "work/scenario3_cross_account_audit.csv",
    index=False
)

print("\nCross-account events:", len(cross_account))
print("Saved Scenario 3 audit.")


# ==================================================
# 4. CREATE POSITIVE USER-DAY LABELS
# ==================================================

# The CERT incident user is treated as the identified insider.
# Each incident user receives one positive label per incident day.

labels = (
    events.groupby(
        ["incident_user", "day"],
        as_index=False
    )
    .agg(
        incident_event_count=("event_id", "nunique"),
        scenario_count=("scenario", "nunique"),
        scenarios=(
            "scenario",
            lambda x: ",".join(sorted(set(x)))
        ),
        event_types=(
            "event_type",
            lambda x: ",".join(sorted(set(x)))
        )
    )
)

labels = labels.rename(
    columns={"incident_user": "user"}
)

labels["label"] = 1


# Count cross-account events separately.

cross_counts = (
    cross_account.groupby(
        ["incident_user", "day"]
    )
    .size()
    .reset_index(name="cross_account_event_count")
)

cross_counts = cross_counts.rename(
    columns={"incident_user": "user"}
)

labels = labels.merge(
    cross_counts,
    on=["user", "day"],
    how="left"
)

labels["cross_account_event_count"] = (
    labels["cross_account_event_count"]
    .fillna(0)
    .astype(int)
)


# ==================================================
# 5. CHECK USER-DAY ALIGNMENT
# ==================================================

feature_keys = set(
    zip(features["user"], features["day"])
)

label_keys = set(
    zip(labels["user"], labels["day"])
)

missing_keys = label_keys - feature_keys

if missing_keys:
    raise ValueError(
        f"{len(missing_keys)} incident user-day keys "
        "are missing from the feature dataset."
    )

if labels.duplicated(["user", "day"]).any():
    raise ValueError("Duplicate user-day labels detected.")

print("\nAll incident user-day keys matched.")


# ==================================================
# 6. MERGE LABELS WITH FEATURES
# ==================================================

dataset = features.merge(
    labels,
    on=["user", "day"],
    how="left",
    validate="one_to_one"
)

# No recorded incident observable = provisional label 0.
dataset["label"] = (
    dataset["label"]
    .fillna(0)
    .astype(int)
)

dataset["incident_event_count"] = (
    dataset["incident_event_count"]
    .fillna(0)
    .astype(int)
)

dataset["scenario_count"] = (
    dataset["scenario_count"]
    .fillna(0)
    .astype(int)
)

dataset["scenarios"] = (
    dataset["scenarios"].fillna("")
)

dataset["event_types"] = (
    dataset["event_types"].fillna("")
)

dataset["cross_account_event_count"] = (
    dataset["cross_account_event_count"]
    .fillna(0)
    .astype(int)
)


# ==================================================
# 7. FINAL VALIDATION
# ==================================================

if len(dataset) != len(features):
    raise ValueError("Unexpected change in dataset row count.")

if dataset["label"].isna().any():
    raise ValueError("Missing labels detected.")

if not set(dataset["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected label values detected.")

if dataset.duplicated(["user", "day"]).any():
    raise ValueError("Duplicate user-day rows detected.")


# ==================================================
# 8. SAVE OUTPUT FILES
# ==================================================

output_folder = Path("data/processed")
output_folder.mkdir(
    parents=True,
    exist_ok=True
)

labels.to_csv(
    output_folder / "ground_truth_user_day_labels.csv",
    index=False
)

dataset.to_parquet(
    output_folder / "ueba_labeled_dataset.parquet",
    index=False
)

cross_account.to_csv(
    "work/scenario3_cross_account_audit.csv",
    index=False
)


# ==================================================
# 9. DISPLAY FINAL RESULTS
# ==================================================

print("\n" + "=" * 60)
print("GROUND-TRUTH LABELING COMPLETE")
print("=" * 60)

print("\nLabel distribution:")
print(dataset["label"].value_counts().sort_index())

print("\nPositive user-days:",
      int((dataset["label"] == 1).sum()))

print("Negative user-days:",
      int((dataset["label"] == 0).sum()))

print(
    "\nPositive percentage:",
    round(dataset["label"].mean() * 100, 4),
    "%"
)

print(
    "\nUnique positive users:",
    labels["user"].nunique()
)

print(
    "Unique positive days:",
    labels["day"].nunique()
)

print(
    "\nCross-account events retained for audit:",
    len(cross_account)
)

print("\nFinal dataset shape:", dataset.shape)

print("\nSaved files:")
print("1. data/processed/ground_truth_user_day_labels.csv")
print("2. data/processed/ueba_labeled_dataset.parquet")
print("3. work/scenario3_cross_account_audit.csv")

print("\nAll validation checks passed.")
