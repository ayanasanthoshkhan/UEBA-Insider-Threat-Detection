
import pandas as pd
import numpy as np
import shap
import joblib

import matplotlib.pyplot as plt
from pathlib import Path

print("Loading model and dataset...")

# Load trained model
model = joblib.load(
    "models/ueba_random_forest.joblib"
)

# Load complete labeled dataset
df = pd.read_parquet(
    "data/processed/ueba_labeled_dataset.parquet"
)

# Exclude identifiers and label-derived metadata
excluded = [
    "user",
    "day",
    "label",
    "incident_event_count",
    "scenario_count",
    "scenarios",
    "event_types",
    "cross_account_event_count"
]

X = df.drop(
    columns=excluded,
    errors="ignore"
)

X = X.select_dtypes(include=np.number)
X = X.replace([np.inf, -np.inf], np.nan)

# Recreate the chronological training split
df["day"] = pd.to_datetime(df["day"])

unique_days = np.array(sorted(df["day"].unique()))
split_position = int(len(unique_days) * 0.80)
cutoff_date = pd.Timestamp(unique_days[split_position])

train_mask = df["day"] < cutoff_date
test_mask = df["day"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

# Remove features that were completely empty in training
empty_columns = X_train.columns[
    X_train.isna().all()
].tolist()

X_train = X_train.drop(columns=empty_columns)
X_test = X_test.drop(columns=empty_columns)

# Sample test records for SHAP
sample_size = min(500, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print("SHAP sample size:", len(X_sample))
print("Number of features:", X_sample.shape[1])

# Apply the already-trained imputer
imputer = model.named_steps["imputer"]

X_sample_imputed = imputer.transform(X_sample)

# Get the Random Forest classifier
classifier = model.named_steps["classifier"]

print("\nCalculating SHAP values...")
print("This may take a few minutes.")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(
    X_sample_imputed
)

# Support different SHAP output formats
if isinstance(shap_values, list):
    values = shap_values[1]
elif len(shap_values.shape) == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values

# Create output folder
Path("results/shap").mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# 1. GLOBAL FEATURE IMPORTANCE
# ==================================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    values,
    X_sample_imputed,
    feature_names=X_sample.columns.tolist(),
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "results/shap/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# 2. SHAP BAR PLOT
# ==================================================

print("Generating SHAP feature importance plot...")

shap.summary_plot(
    values,
    X_sample_imputed,
    feature_names=X_sample.columns.tolist(),
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "results/shap/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# 3. SAVE NUMERICAL SHAP IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_absolute_shap": np.abs(values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_absolute_shap",
    ascending=False
)

importance.to_csv(
    "results/shap/shap_importance.csv",
    index=False
)

print("\nTOP 15 SHAP FEATURES")
print(importance.head(15).to_string(index=False))

print("\nSHAP ANALYSIS COMPLETED")

print("\nSaved files:")
print("1. results/shap/shap_summary.png")
print("2. results/shap/shap_feature_importance.png")
print("3. results/shap/shap_importance.csv")
