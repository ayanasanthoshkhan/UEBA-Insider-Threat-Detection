from pathlib import Path
import os
import json
import pandas as pd
import streamlit as st
# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="UEBA | Insider Threat Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
DATA_DIR = (
    Path(
        os.environ.get(
            "UEBA_DATA_ROOT",
            r"D:\UEBA_Capstone\data"
        )
    )
    / "processed"
    / "ml_handoff"
)
FEATURE_FILE = DATA_DIR / "ml_candidate_features.parquet"
METADATA_FILE = DATA_DIR / "ml_context_metadata.parquet"
MANIFEST_FILE = DATA_DIR / "feature_manifest.json"
ACTIVITY_METRICS = {
    "Logins": "login_count",
    "USB connections": "usb_connect_count",
    "File activity": "file_activity_count",
    "Email activity": "email_activity_count",
    "HTTP requests": "http_request_count",
}
# ----------------------------------------
# DATA LOADING
# ----------------------------------------
@st.cache_data(show_spinner="Loading behavioral dataset...")
def load_data():
    features = pd.read_parquet(FEATURE_FILE)
    metadata = pd.read_parquet(METADATA_FILE)
    features["day"] = pd.to_datetime(features["day"])
    metadata["day"] = pd.to_datetime(metadata["day"])
    keys = ["user", "day"]
    if features.duplicated(keys).any():
        raise ValueError("Duplicate feature user-day keys")
    if metadata.duplicated(keys).any():
        raise ValueError("Duplicate metadata user-day keys")
    if not pd.MultiIndex.from_frame(
        features[keys]
    ).equals(pd.MultiIndex.from_frame(metadata[keys])):
        raise ValueError("Feature and metadata keys are not aligned")
    combined = features.merge(
        metadata,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return combined, manifest
# ----------------------------------------
# HEADER AND FILE CHECKS
# ----------------------------------------
st.title("🛡️ UEBA Insider Threat Analytics")
st.caption(
    "CERT r4.2 | Behavioral Analytics & "
    "Insider Threat Detection Research Prototype"
)
missing_files = [
    str(p)
    for p in [FEATURE_FILE, METADATA_FILE, MANIFEST_FILE]
    if not p.is_file()
]
if missing_files:
    st.error("Required handoff files were not found.")
    st.code("\n".join(missing_files))
    st.stop()
try:
    df, manifest = load_data()
except Exception as exc:
    st.error(f"Unable to load validated dataset: {exc}")
    st.stop()
available_metrics = {
    label: col
    for label, col in ACTIVITY_METRICS.items()
    if col in df.columns
}
if not available_metrics:
    st.error("No expected activity metrics found.")
    st.stop()
# ----------------------------------------
# SIDEBAR
# ----------------------------------------
with st.sidebar:
    st.header("UEBA Analytics")
    st.caption("Behavioral monitoring prototype")
    st.divider()
    st.subheader("Dataset")
    st.write("CERT Insider Threat r4.2")
    st.write("Granularity: User-day")
    st.write("Sources: Logon, USB, File, Email, HTTP")
    st.divider()
    st.subheader("System status")
    st.success("Behavioral data: Loaded")
    st.info("ML predictions: Awaiting integration")
    st.divider()
    st.caption(
        "Research use only. An anomaly does not "
        "prove malicious intent."
    )
# ----------------------------------------
# GLOBAL DATE-RANGE FILTER
# ----------------------------------------
min_date = df["day"].min().date()
max_date = df["day"].max().date()
with st.sidebar:
    st.divider()
    st.subheader("Date Range")
    selected_dates = st.date_input(
        "Select observation period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )
if len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()
start_date, end_date = selected_dates
# Filter all dashboard views consistently
df = df.loc[
    df["day"].dt.date.between(start_date, end_date)
].copy()
if df.empty:
    st.warning("No records found for the selected date range.")
    st.stop()
# ----------------------------------------

# CHART AGGREGATION CONTROL

# ----------------------------------------

with st.sidebar:
    st.divider()
    st.subheader("Chart Settings")
    chart_frequency = st.selectbox(
        "Time aggregation",
        ["Daily", "Weekly", "Monthly"],
        index=0,
    )


def aggregate_activity(data, columns):
    """Aggregate counts for display; do not recompute historical baselines."""
    daily = data.groupby("day")[columns].sum().sort_index()

    if chart_frequency == "Daily":
        return daily
    if chart_frequency == "Weekly":
        # Weeks start Monday; period may be partial at date-filter boundaries.
        return daily.resample("W-MON", closed="left", label="left").sum()
    return daily.resample("MS").sum()


# ----------------------------------------
# MAIN NAVIGATION
# ----------------------------------------
overview, activity, employee, models = st.tabs(
    [
        "Overview",
        "Activity Analytics",
        "Employee Explorer",
        "ML Integration",
    ]
)
# ----------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------
with overview:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{df['user'].nunique():,}")
    c2.metric("User-days", f"{len(df):,}")
    c3.metric("Activity Sources", len(available_metrics))
    c4.metric(
        "Candidate Features",
        manifest["feature_count"],
    )
    st.divider()
    st.subheader("Activity Timeline")
    metric_name = st.selectbox(
        "Select activity type",
        list(available_metrics.keys()),
        key="overview_metric",
    )
    selected_col = available_metrics[metric_name]
    activity_trend = aggregate_activity(df, [selected_col])

    st.line_chart(activity_trend)
    st.caption(
        "Activity totals across all recorded users at the selected time aggregation. "
        "These are descriptive counts, not detected attacks."
    )
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Coverage")
        st.write(
            "Start:",
            df["day"].min().date(),
        )
        st.write(
            "End:",
            df["day"].max().date(),
        )
        st.write(
            "Unique employees:",
            df["user"].nunique(),
        )
    with c2:
        st.subheader("Feature Readiness")
        st.write(
            "Candidate numerical features:",
            manifest["feature_count"],
        )
        st.write(
            "Features containing missing values:",
            len(manifest["missing_values"]),
        )
        st.write(
            "Dataset contains verified labels:",
            "No",
        )
# ----------------------------------------
# TAB 2: ACTIVITY ANALYTICS
# ----------------------------------------
with activity:
    st.subheader("Behavioral Activity Analytics")
    selected_metrics = st.multiselect(
        "Compare activity sources",
        options=list(available_metrics.keys()),
        default=["Logins", "USB connections"],
    )
    if selected_metrics:
        columns = [
            available_metrics[name]
            for name in selected_metrics
        ]
        daily_activity = aggregate_activity(df, columns)
        daily_activity = daily_activity.rename(
            columns={
                col: label
                for label, col in available_metrics.items()
            }
        )
        st.line_chart(daily_activity)
        st.caption(
            "Activity counts have different natural scales. "
            "Compare trends carefully."
        )
    else:
        st.info("Select an activity source to plot.")
    st.divider()
    st.subheader("Activity Summary")
    totals = []
    for label, col in available_metrics.items():
        totals.append({
            "Activity": label,
            "Total events": int(df[col].sum()),
            "Mean per user-day": round(
                float(df[col].mean()), 2
            ),
            "Active user-days": int(
                (df[col] > 0).sum()
            ),
        })
    st.dataframe(
        pd.DataFrame(totals),
        hide_index=True,
        width="stretch",
    )
# ----------------------------------------
# TAB 3: EMPLOYEE EXPLORER
# ----------------------------------------
with employee:
    st.subheader("Individual Employee Behavior")
    users = sorted(df["user"].dropna().unique())
    selected_user = st.selectbox(
        "Select employee ID",
        users,
    )
    user_df = (
        df.loc[df["user"] == selected_user]
        .sort_values("day")
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed Days", len(user_df))
    c2.metric(
        "Total Logins",
        int(user_df["login_count"].sum()),
    )
    c3.metric(
        "Total USB Connections",
        int(user_df["usb_connect_count"].sum()),
    )
    metric_name = st.selectbox(
        "Select behavior",
        list(available_metrics.keys()),
        key="employee_metric",
    )
    metric = available_metrics[metric_name]
    history_col = f"{metric}_hist_mean"
    plot_columns = [metric]
    if history_col in user_df.columns:
        plot_columns.append(history_col)
    st.subheader("Activity vs Historical Baseline")
    st.line_chart(
        user_df.set_index("day")[plot_columns]
    )
    st.caption(
        "The historical baseline uses preceding observed "
        "user-days. Missing warm-up values are not replaced "
        "with artificial zero baselines."
    )
    st.divider()
    st.subheader("Organizational Context")
    latest = user_df.iloc[-1]
    for col in ["role", "department", "team"]:
        if col in user_df.columns:
            st.write(
                f"{col.replace('_', ' ').title()}:",
                str(latest[col]),
            )
    st.subheader("Recent User-day Records")
    display_cols = [
        "day",
        "login_count",
        "usb_connect_count",
        "file_activity_count",
        "email_activity_count",
        "http_request_count",
    ]
    display_cols = [
        col for col in display_cols
        if col in user_df.columns
    ]
    st.dataframe(
        user_df[display_cols].tail(30),
        hide_index=True,
        width="stretch",
    )
# ----------------------------------------
# TAB 4: MODEL INTEGRATION
# ----------------------------------------
with models:
    st.subheader("Machine Learning Integration")
    st.info(
        "Awaiting Member B's verified CERT labels, "
        "trained models, prediction outputs and "
        "evaluation results."
    )
    st.markdown(
        """
        **Planned integration components**
        - Isolation Forest anomaly scores
        - Random Forest classification results
        - Precision, recall and PR-AUC
        - Confusion matrix
        - SHAP feature explanations
        - Suspicious user-day investigation table
        """
    )
    st.warning(
        "No model predictions are currently displayed. "
        "Observed activity should not be presented "
        "as confirmed insider threats."
    )
st.divider()
st.caption(
    "UEBA Research Prototype | CERT r4.2 Synthetic Dataset"
)
