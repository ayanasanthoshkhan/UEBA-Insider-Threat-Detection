
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="UEBA Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ UEBA Insider Threat Detection")
st.caption("CERT r4.2 | User Behaviour Analytics")

# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = Path("models/ueba_random_forest.joblib")
DATA_PATH = Path("data/processed/ueba_labeled_dataset.parquet")
METRICS_PATH = Path("results/model_metrics.json")
CM_PATH = Path("results/confusion_matrix.png")
SHAP_PATH = Path("results/shap/shap_summary.png")
SHAP_BAR_PATH = Path("results/shap/shap_feature_importance.png")
IMPORTANCE_PATH = Path("results/shap/shap_importance.csv")

# ==========================================
# LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

try:
    model = load_model()
    df = load_data()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Project Overview",
        "Model Performance",
        "Feature Importance",
        "User-Day Risk Explorer"
    ]
)

# ==========================================
# PROJECT OVERVIEW
# ==========================================

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        """
        This project uses User and Entity Behaviour Analytics
        (UEBA) to identify unusual employee activity.

        A Random Forest classifier is trained using behavioural
        features extracted from the CERT r4.2 dataset.

        SHAP is used to explain which behavioural features
        influence the model's predictions.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total User-Days", f"{len(df):,}")
    col2.metric("Unique Users", f"{df['user'].nunique():,}")
    col3.metric(
        "Behavioural Features",
        str(len(model.feature_names_in_))
        if hasattr(model, "feature_names_in_")
        else "69"
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.info(
        "A positive label represents a user-day with incident-derived "
        "evidence. A negative label means no incident observable was "
        "recorded for that user-day; it does not prove benign behaviour."
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")

    st.subheader("Additional Evaluation Metrics")

    col5, col6 = st.columns(2)

    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col6.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")

    st.subheader("Confusion Matrix")

    if CM_PATH.exists():
        st.image(str(CM_PATH), use_container_width=True)

    st.warning(
        "These are preliminary results from an incident-derived "
        "user-day classification experiment. Same-day feature and "
        "label overlap may inflate performance. They should not be "
        "interpreted as validated early-warning performance."
    )

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

elif page == "Feature Importance":

    st.header("Feature Importance")

    st.subheader("Random Forest Feature Importance")

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH)
        st.bar_chart(
            importance.head(15).set_index("feature")
        )

    st.subheader("SHAP Summary Plot")

    if SHAP_PATH.exists():
        st.image(str(SHAP_PATH), use_container_width=True)

    st.subheader("SHAP Feature Importance")

    if SHAP_BAR_PATH.exists():
        st.image(str(SHAP_BAR_PATH), use_container_width=True)

    if IMPORTANCE_PATH.exists():
        st.subheader("Top SHAP Features")
        st.dataframe(
            pd.read_csv(IMPORTANCE_PATH).head(15),
            use_container_width=True
        )

# ==========================================
# USER-DAY RISK EXPLORER
# ==========================================

elif page == "User-Day Risk Explorer":

    st.header("User-Day Risk Explorer")

    st.write(
        "Explore the model's predicted incident risk for a selected "
        "user-day in the evaluation dataset."
    )

    user_list = sorted(df["user"].dropna().unique())

    selected_user = st.selectbox(
        "Select User",
        user_list
    )

    user_data = df[df["user"] == selected_user].copy()

    user_data["day"] = pd.to_datetime(user_data["day"])

    selected_day = st.selectbox(
        "Select Date",
        sorted(user_data["day"].dt.strftime("%Y-%m-%d").unique())
    )

    selected_row = user_data[
        user_data["day"].dt.strftime("%Y-%m-%d") == selected_day
    ].iloc[[0]]

    excluded = [
        "user", "day", "label", "incident_event_count",
        "scenario_count", "scenarios", "event_types",
        "cross_account_event_count"
    ]

    X = selected_row.drop(
        columns=excluded,
        errors="ignore"
    )

    X = X.select_dtypes(include=np.number)

    probability = model.predict_proba(X)[0][1]

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Incident Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Recorded Ground-Truth Label",
        str(int(selected_row["label"].iloc[0]))
    )

    st.progress(float(probability))

    st.caption(
        "This probability is a model score, not a calibrated "
        "probability of future insider misconduct."
    )

    st.subheader("User-Day Behavioural Features")
    st.dataframe(
        X.T.rename(columns={X.index[0]: "Value"}),
        use_container_width=True
    )

st.sidebar.divider()
st.sidebar.caption("UEBA | CERT r4.2 | Member B ML Dashboard")
