
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD THE COMPLETE LABELED DATASET
# ==================================================

print("=" * 65)
print("LOADING LABELED DATASET")

df = pd.read_parquet(
    "data/processed/ueba_labeled_dataset.parquet"
)

print("Dataset shape:", df.shape)
print("\nActual label distribution:")
print(df["label"].value_counts())

print(
    "\nPositive percentage:",
    round(df["label"].mean() * 100, 4),
    "%"
)


# ==================================================
# 2. PREPARE FEATURES AND TARGET
# ==================================================

print("\n" + "=" * 65)
print("PREPARING MODEL FEATURES")

# Columns that must never be model predictors
excluded_columns = [
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
    columns=excluded_columns,
    errors="ignore"
)

y = df["label"].astype(int)

# Keep only numerical predictors
X = X.select_dtypes(include=np.number).copy()

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

print("Initial predictor count:", X.shape[1])

# Check for duplicate feature names
if X.columns.duplicated().any():
    raise ValueError("Duplicate predictor column names found.")

# ==================================================
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==================================================

print("\n" + "=" * 65)
print("CHRONOLOGICAL TRAIN/TEST SPLIT")

df["day"] = pd.to_datetime(df["day"])

unique_days = np.array(sorted(df["day"].unique()))

split_position = int(len(unique_days) * 0.80)

if split_position <= 0 or split_position >= len(unique_days):
    raise ValueError("Invalid chronological split.")

cutoff_date = pd.Timestamp(unique_days[split_position])

train_mask = df["day"] < cutoff_date
test_mask = df["day"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("Training cutoff date:", cutoff_date.date())
print("Training date range:",
      df.loc[train_mask, "day"].min().date(),
      "to",
      df.loc[train_mask, "day"].max().date())

print("Testing date range:",
      df.loc[test_mask, "day"].min().date(),
      "to",
      df.loc[test_mask, "day"].max().date())

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both train and test sets must contain positive "
        "and negative records. We need to adjust the "
        "chronological cutoff before training."
    )

if y_test.sum() < 10:
    raise ValueError(
        "The test set contains fewer than 10 positive "
        "records. We should inspect the split before "
        "reporting model performance."
    )


# ==================================================
# 4. REMOVE FEATURES EMPTY IN TRAINING DATA
# ==================================================

empty_columns = X_train.columns[
    X_train.isna().all()
].tolist()

if empty_columns:
    print("\nRemoving completely empty training features:")
    print(empty_columns)

    X_train = X_train.drop(columns=empty_columns)
    X_test = X_test.drop(columns=empty_columns)

print("\nFinal predictor count:", X_train.shape[1])


# ==================================================
# 5. BUILD RANDOM FOREST PIPELINE
# ==================================================

print("\n" + "=" * 65)
print("BUILDING RANDOM FOREST MODEL")

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    )
])

# ==================================================
# 6. TRAIN MODEL
# ==================================================

print("\nTraining Random Forest...")
print("Please wait. This may take a few minutes.")

model.fit(X_train, y_train)

print("Training completed!")


# ==================================================
# 7. MAKE PREDICTIONS
# ==================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==================================================
# 8. CALCULATE PERFORMANCE METRICS
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, y_probability
)

pr_auc = average_precision_score(
    y_test, y_probability
)

print("\nMODEL PERFORMANCE")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")


# ==================================================
# 9. CLASSIFICATION REPORT
# ==================================================

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ==================================================
# 10. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, y_pred)

print("CONFUSION MATRIX")
print(cm)

Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(6, 5))

plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Normal", "Incident"])
plt.yticks([0, 1], ["Normal", "Incident"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==================================================
# 11. FEATURE IMPORTANCE
# ==================================================

classifier = model.named_steps["classifier"]

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": classifier.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance.head(15).to_string(index=False))


# ==================================================
# 12. SAVE MODEL AND RESULTS
# ==================================================

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/ueba_random_forest.joblib"
)

metrics = {
    "model": "Random Forest",
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_positive": int(y_train.sum()),
    "testing_positive": int(y_test.sum()),
    "training_cutoff": str(cutoff_date.date()),
    "predictor_count": int(X_train.shape[1]),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc)
}

with open("results/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED")

print("\nSaved files:")
print("1. models/ueba_random_forest.joblib")
print("2. results/model_metrics.json")
print("3. results/confusion_matrix.png")
print("4. results/feature_importance.csv")

print("\nAll done!")
