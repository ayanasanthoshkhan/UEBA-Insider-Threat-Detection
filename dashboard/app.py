from pathlib import Path
import os
import json
import pandas as pd
import streamlit as st
# ----------------------------------------
# DESIGN SYSTEM — BLACK SOC CONSOLE
# ----------------------------------------
SVG_ICONS = {
    "shield": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2 20 5v6c0 5-3.2 8.6-8 11-4.8-2.4-8-6-8-11V5l8-3Z"/><path d="m8.7 12 2.2 2.2 4.6-4.7"/></svg>""",
    "activity": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12h4l3-7 5 14 3-7h5"/></svg>""",
    "users": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="9" cy="8" r="3"/><path d="M2.5 20v-2c0-3.3 3-5 6.5-5s6.5 1.7 6.5 5v2M17 5.4a3 3 0 0 1 0 5.7M17 13.3c2.9.2 4.5 2 4.5 4.7v2"/></svg>""",
    "database": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>""",
    "layers": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 2 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 17l9 5 9-5"/></svg>""",
    "network": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="2" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="16" y="16" width="6" height="6" rx="1"/><path d="M12 8v5M5 16v-3h14v3"/></svg>""",
}

SVG_ICONS.update({
    "grid": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>""",
    "compare": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h18M3 17h18"/><circle cx="8" cy="7" r="3" fill="#18140e"/><circle cx="16" cy="17" r="3" fill="#18140e"/></svg>""",
    "clock": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l4 2"/></svg>""",
    "cpu": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/></svg>""",
})

# SVGs are rendered explicitly, next to each native Streamlit navigation button.
# This avoids browser-dependent CSS masks and ensures every icon is visible.
NAV_ITEMS = [
    ("Overview", "grid", "overview"),
    ("Activity Analytics", "activity", "activity"),
    ("Employee Explorer", "users", "employees"),
    ("Compare Employees", "compare", "compare"),
    ("Department Explorer", "network", "departments"),
    ("Peer Baselines", "clock", "peers"),
    ("ML Integration", "cpu", "ml"),
]

DASHBOARD_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: #0C0A08 !important; color: #F5F0E6 !important;
}
[data-testid="stHeader"] { background: rgba(12,10,8,.97) !important; }
[data-testid="stAppDeployButton"] {display:none !important;}
[data-testid="stSidebar"] { background: #110E0B !important; border-right:1px solid #32291C !important; }
[data-testid="stSidebar"] * { color: #D2C5AD; }
[data-testid="stMainBlockContainer"] { padding-top: .65rem !important; padding-bottom:2rem; max-width: 1530px; }
h1, h2, h3, h4 { color:#F5F0E6 !important; letter-spacing:-.026em; font-weight:650 !important; }
h2, h3 {font-size:1.23rem !important;}
p,span,label {letter-spacing: .002em;}
[data-testid="stCaptionContainer"] p { color:#BDB09A !important; line-height:1.6 !important; }
[data-testid="stMetric"] { background:#17130E !important; border:1px solid #413423 !important;
  padding:17px 18px !important; border-radius:13px !important; min-height:125px; box-shadow:none !important; }
[data-testid="stMetricLabel"] p {color:#C5B49A !important; font-size:.77rem; font-weight:600; white-space:normal !important; overflow:visible !important; line-height:1.35 !important;}
[data-testid="stMetricValue"] {color:#F7F0E4 !important; font-variant-numeric:tabular-nums;}
[data-testid="stMetricDelta"] {color:#E0B96E !important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#413423 !important;}
[data-testid="stMarkdownContainer"] hr {border-color:#352C20;}
[data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div,
[data-testid="stDateInput"] > div {border-radius:10px;}
[data-testid="stDataFrame"] { border:1px solid #403321; border-radius:11px; overflow:hidden; }
[data-testid="stAlert"] {border-radius:11px !important;}
[data-testid="stButton"] button {border-radius:9px !important;}
/* Compact icon + native Streamlit button layout. SVGs are HTML, not CSS masks. */
[data-testid="stSidebar"] [class*="st-key-navrow_"] {
  margin:0 !important;
  padding:0 !important;
}
[data-testid="stSidebar"] [class*="st-key-navrow_"] [data-testid="stHorizontalBlock"] {
  gap:2px !important;
  align-items:center !important;
  flex-wrap:nowrap !important;
}
[data-testid="stSidebar"] [class*="st-key-navrow_"] [data-testid="stColumn"] {
  min-width:0 !important;
}
[data-testid="stSidebar"] [class*="st-key-navrow_"] [data-testid="stColumn"]:first-child {
  flex:0 0 29px !important;
  width:29px !important;
}
[data-testid="stSidebar"] [class*="st-key-navrow_"] [data-testid="stColumn"]:last-child {
  flex:1 1 auto !important;
  width:auto !important;
}
.ueba-nav-icon {height:31px;display:flex;align-items:center;justify-content:center;
  color:#B29B74;}
.ueba-nav-icon svg {width:17px;height:17px;display:block;}
.ueba-nav-icon.active {color:#E0B86D;}
[data-testid="stSidebar"] [class*="st-key-nav_"] button {
  display:flex !important; width:100% !important; min-height:33px !important;
  height:33px !important; justify-content:flex-start !important;
  align-items:center !important; padding:5px 10px !important;
  border:0 !important; border-left:2px solid transparent !important;
  border-radius:7px !important; background:transparent !important;
  color:#C5B7A0 !important; font-size:12px !important;
  font-weight:500 !important; text-align:left !important;
  box-shadow:none !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {
  background:#211B13 !important;color:#F2E3C7 !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] {
  background:#2C2418 !important;border-left-color:#D8B166 !important;
  color:#F4E5C9 !important;font-weight:650 !important;
}
.ueba-hero {display:flex;align-items:center;gap:17px;padding:17px 22px;
  border:1px solid #453823;border-radius:15px;background:#15110C;margin-bottom:10px;}
.ueba-hero-icon {display:flex;width:48px;height:48px;justify-content:center;align-items:center;
  color:#DBB66E;background:#241B10;border:1px solid #574329;border-radius:12px;flex-shrink:0;}
.ueba-hero-icon svg {width:26px;height:26px;}
.ueba-hero-copy {min-width:0;flex:1;}
.ueba-eyebrow {font-size:10px;color:#D7AB58;font-weight:700;letter-spacing:.17em;}
.ueba-hero-title {font-size:clamp(23px,2.5vw,32px);color:#F7F0E4;font-weight:680;line-height:1.2;
  letter-spacing:-.035em;margin-top:4px;}
.ueba-hero-sub {font-size:12px;color:#A79A86;margin-top:6px;}
.ueba-hero-status {font-size:10px;font-weight:700;color:#AACABB;border:1px solid #365445;
  background:#14201A;padding:8px 11px;border-radius:30px;white-space:nowrap;letter-spacing:.05em;}
.ueba-status-dot {display:inline-block;width:6px;height:6px;background:#96C4AC;border-radius:50%;margin-right:7px;}
.ueba-brand {display:flex;align-items:center;gap:11px;margin:6px 0 13px;}
.ueba-brand-icon {width:39px;height:39px;display:flex;align-items:center;justify-content:center;
  background:#261D10;border:1px solid #5D4728;border-radius:11px;color:#DBB66E;}
.ueba-brand-icon svg {width:21px;height:21px;}
.ueba-brand-name {color:#F5EEE2;font-size:14px;font-weight:750;letter-spacing:.055em;}
.ueba-brand-desc {color:#A99A81;font-size:9px;letter-spacing:.14em;margin-top:2px;}
.ueba-sidebar-section {font-size:9px;color:#AD9B7F;font-weight:750;letter-spacing:.18em;margin:11px 0 7px;}
.ueba-sidebar-note {font-size:11px;color:#9E947F;line-height:1.55;}
.ueba-sidebar-state {border:1px solid #2E4739;background:#162119;border-radius:9px;padding:8px 11px;
  color:#9DCFB8;font-size:10px;letter-spacing:.02em;}
.ueba-sidebar-state.muted {border-color:#463726;background:#211A12;color:#C6B598;margin-top:7px;}
.ueba-section {display:flex;align-items:center;gap:9px;margin:18px 0 14px;}
.ueba-section svg {width:18px;height:18px;color:#D5AB5F;}
.ueba-section span {font-size:19px;color:#F0E9DB;font-weight:660;letter-spacing:-.022em;}
.ueba-note {font-size:11px;color:#BBAE97;border-left:2px solid #9D783A;padding:8px 13px;
  background:#1D180F;border-radius:0 8px 8px 0;}
.ueba-footer {font-size:10px;letter-spacing:.07em;color:#9A8B73;border-top:1px solid #3A2E20;
  padding:19px 0;margin-top:27px;}
.ueba-kpi {height:152px; background:#17130E; border:1px solid #403320; border-radius:13px;
  padding:17px 17px 14px; display:flex; flex-direction:column; justify-content:space-between; gap:8px;
  transition:border-color .17s ease,transform .17s ease;}
.ueba-kpi:hover {border-color:#785A2D;transform:translateY(-1px);}
.ueba-kpi-top {display:flex;align-items:center;justify-content:space-between;gap:10px;}
.ueba-kpi-label {font-size:10px;letter-spacing:.07em;color:#C1AD90;font-weight:700;line-height:1.3;}
.ueba-kpi-icon {height:23px;width:23px;display:flex;color:#D9B46E;align-items:center;justify-content:center;}
.ueba-kpi-icon svg {width:19px;height:19px;}
.ueba-kpi-number {font-size:28px;line-height:1.12;color:#F6EEE1;font-weight:630;
  font-variant-numeric:tabular-nums;letter-spacing:-.03em;}
.ueba-kpi-sub {font-size:10px;color:#B3A38C;letter-spacing:.01em;line-height:1.35;min-width:0;flex:1;}
.ueba-kpi-bottom {display:flex;gap:6px;align-items:flex-end;justify-content:space-between;min-height:31px;}
.ueba-mini-trend {width:68px;height:26px;min-width:68px;flex:none;opacity:.95;}
@media (max-width:860px){.ueba-hero {flex-wrap:wrap;padding:14px;}.ueba-hero-status {margin-left:0;}}
</style>
"""



def section_heading(title, icon="activity"):
    """Plain inline SVG and text; no remote assets or JavaScript."""
    st.markdown(
        f'<div class="ueba-section">{SVG_ICONS[icon]}<span>{title}</span></div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="UEBA Analytics | Insider Threat Detection",
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
# The dashboard deliberately uses historical descriptive data, not live alerts.
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
st.markdown(
    f"""<div class="ueba-hero">
        <div class="ueba-hero-icon">{SVG_ICONS['shield']}</div>
        <div class="ueba-hero-copy">
          <div class="ueba-eyebrow">UEBA ANALYTICS / CERT R4.2</div>
          <div class="ueba-hero-title">Behavior intelligence</div>
          <div class="ueba-hero-sub">Historical user and entity behavior analytics · Research prototype</div>
        </div>
        <div class="ueba-hero-status"><span class="ueba-status-dot"></span> DATASET READY</div>
    </div>""",
    unsafe_allow_html=True,
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
# SIDEBAR — BRAND / NAVIGATION / DATA CONTROLS
# ----------------------------------------
with st.sidebar:
    st.markdown(
        f"""<div class="ueba-brand"><div class="ueba-brand-icon">{SVG_ICONS['shield']}</div>
        <div><div class="ueba-brand-name">UEBA ANALYTICS</div>
        <div class="ueba-brand-desc">BEHAVIOR INTELLIGENCE</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ueba-sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)
    if "ueba_page" not in st.session_state:
        st.session_state.ueba_page = "Overview"
    for nav_label, nav_icon, slug in NAV_ITEMS:
        selected = st.session_state.ueba_page == nav_label
        with st.container(key=f"navrow_{slug}"):
            icon_col, button_col = st.columns([1, 8], gap="small", vertical_alignment="center")
            with icon_col:
                st.markdown(
                    f'<div class="ueba-nav-icon {"active" if selected else ""}">'
                    f'{SVG_ICONS[nav_icon]}</div>',
                    unsafe_allow_html=True,
                )
            with button_col:
                if st.button(
                    nav_label,
                    key=f"nav_{slug}",
                    type="primary" if selected else "secondary",
                    width="stretch",
                ):
                    st.session_state.ueba_page = nav_label
                    st.rerun()
    page = st.session_state.ueba_page
    st.divider()
    st.markdown('<div class="ueba-sidebar-section">DATA STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="ueba-sidebar-state">●  DATA AVAILABLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="ueba-sidebar-state muted">○  MODEL OUTPUTS PENDING</div>', unsafe_allow_html=True)
    st.markdown('<div class="ueba-sidebar-note" style="margin-top:10px">CERT r4.2 · Synthetic events · User-day observations</div>', unsafe_allow_html=True)
# ----------------------------------------
# GLOBAL DATE-RANGE FILTER
# ----------------------------------------
min_date = df["day"].min().date()
max_date = df["day"].max().date()
with st.sidebar:
    st.divider()
    st.markdown('<div class="ueba-sidebar-section">DATE RANGE</div>', unsafe_allow_html=True)
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

# Optional organizational scope is a display filter only. Upstream historical
# and peer baselines must never be recomputed from the selected subset.
if "department" in df.columns:
    dept_options = sorted(
        df["department"].astype("string").fillna("Unknown")
        .str.strip().replace("", "Unknown").unique().tolist()
    )
    with st.sidebar:
        department_scope = st.selectbox(
            "Department scope", ["All departments"] + dept_options,
            index=0, key="global_department_scope",
        )
    if department_scope != "All departments":
        dept_values = (
            df["department"].astype("string").fillna("Unknown")
            .str.strip().replace("", "Unknown")
        )
        df = df.loc[dept_values.eq(department_scope)].copy()
        if df.empty:
            st.info("No user-day records match the selected department and dates.")
            st.stop()
# ----------------------------------------

# CHART AGGREGATION CONTROL

# ----------------------------------------

with st.sidebar:
    st.divider()
    st.markdown('<div class="ueba-sidebar-section">CHART SETTINGS</div>', unsafe_allow_html=True)
    chart_frequency = st.selectbox(
        "Time aggregation",
        ["Daily", "Weekly", "Monthly"],
        index=1,
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


def render_overview_activity_chart(activity_trend, metric_column, metric_label):
    """Historical activity only; no inferred risk scores or anomaly markers."""
    import altair as alt

    values = activity_trend[[metric_column]].reset_index()
    values.columns = ["date", "events"]
    values["date"] = pd.to_datetime(values["date"])
    values = values.sort_values("date")
    if values.empty:
        st.info("No events available for this period.")
        return

    # For daily view, a rolling 7-day mean smooths the expected weekend cycle.
    # For weekly/monthly views, the line is the aggregated count itself.
    values["moving_average"] = (
        values["events"].rolling(window=7, min_periods=1).mean()
        if chart_frequency == "Daily" else values["events"]
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False))
    y = alt.Y("events:Q", title="Events per period", scale=alt.Scale(zero=True))
    tips = [alt.Tooltip("date:T", title="Period", format="%d %b %Y"),
            alt.Tooltip("events:Q", title=metric_label, format=",.0f")]
    base = alt.Chart(values).encode(x=x)
    area = base.mark_area(
        color="#CBA253", opacity=0.13, interpolate="linear"
    ).encode(y=y)
    observed_line = base.mark_line(
        color="#DDB56C", strokeWidth=2, interpolate="linear"
    ).encode(y=y, tooltip=tips)
    chart = area + observed_line
    if chart_frequency == "Daily":
        moving = base.mark_line(
            color="#F2DEB0", strokeWidth=2, interpolate="linear", strokeDash=[5, 3]
        ).encode(
            y=alt.Y("moving_average:Q", title="Events per period"),
            tooltip=[alt.Tooltip("date:T", title="Day", format="%d %b %Y"),
                     alt.Tooltip("moving_average:Q", title="7-day mean", format=",.1f")],
        )
        chart = chart + moving
        st.caption("Gold: daily events · Pale dashed line: 7-day moving average. Initial window may be partial.")
    st.altair_chart(
        chart_style(chart, height=322).interactive(),
        width="stretch",
    )


def tiny_sparkline(series):
    """An actual observed count trend; no artificial percentage change."""
    values = pd.to_numeric(series.tail(24), errors="coerce").dropna().tolist()
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    width, height = 82, 30
    coords = []
    for i, value in enumerate(values):
        x = 1 + i * (width - 2) / (len(values) - 1)
        y = height / 2 if high == low else 2 + (high - value) * (height - 4) / (high - low)
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        '<svg class="ueba-mini-trend" viewBox="0 0 82 30" aria-hidden="true">'
        '<polyline points="' + " ".join(coords) + '" fill="none" '
        'stroke="#D8AE61" stroke-width="1.6" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>'
    )


def premium_kpi(label, value, icon, note, trend=None):
    """Render a factual metric card with a locally defined SVG icon."""
    from html import escape
    spark = tiny_sparkline(trend) if trend is not None else ""
    return (
        '<div class="ueba-kpi">'
        '<div class="ueba-kpi-top"><span class="ueba-kpi-label">' + escape(str(label)) +
        '</span><span class="ueba-kpi-icon">' + SVG_ICONS[icon] + '</span></div>'
        '<div class="ueba-kpi-number">' + escape(str(value)) + '</div>'
        '<div class="ueba-kpi-bottom"><span class="ueba-kpi-sub">' +
        escape(str(note)) + '</span>' + spark + '</div></div>'
    )


def chart_style(chart, height=340):
    """Uniform analytical styling using Altair, with no model-derived annotations."""
    return (
        chart.properties(height=height)
        .configure_axis(
            labelColor="#B7A78E", titleColor="#B7A78E", gridColor="#30291E",
            domain=False, tickColor="#56452A", labelFontSize=11,
        )
        .configure_legend(labelColor="#D4C4A9", titleColor="#D4C4A9", orient="bottom")
        .configure_view(strokeOpacity=0)
    )


def premium_series_chart(data, height=325, ylabel="Events per period"):
    """Plot one or more observed series with real dates and gaps retained."""
    import altair as alt
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "Events")
    frame = data.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    long = frame.reset_index().melt(id_vars="Date", var_name="Series", value_name="Value")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    # Keep missing dates as NaN so Altair breaks a line instead of
    # joining separate observation periods across an unobserved gap.
    if not long["Value"].notna().any():
        st.info("No observed values available for these chart settings.")
        return
    names = list(frame.columns)
    palette = ["#DEB769", "#EBD7A2", "#B17F41", "#7FA69A", "#A68CB5"]
    domain = [str(name) for name in names]
    colors = palette[:len(domain)]
    # Short, two-column legend avoids clipping while preserving exact series values.
    patterns = [[1, 0], [7, 3], [2, 3], [10, 4], [4, 3]]
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.35, point=False, interpolate="linear")
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Value:Q", title=ylabel, scale=alt.Scale(zero=True)),
            color=alt.Color(
                "Series:N", scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(orient="bottom", columns=2, labelLimit=240, title=None),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N", scale=alt.Scale(domain=domain, range=patterns),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=",.2f")],
        )
    )
    st.altair_chart(chart_style(chart, height), width="stretch")


# ----------------------------------------
# MAIN NAVIGATION
# ----------------------------------------
# Sidebar radio controls the visible page; no crowded horizontal tabs.
# Filters stay global and continue working for every page.

# ----------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------
if page == "Overview":
    section_heading("Dataset overview", "database")
    c1, c2, c3, c4 = st.columns(4)
    # Mini-trends reflect observed active users / observed user-day rows per day.
    # They are not model scores, forecasts or arbitrary KPI deltas.
    daily_active_users = df.groupby("day")["user"].nunique().sort_index()
    daily_userdays = df.groupby("day").size().sort_index()
    with c1:
        st.markdown(premium_kpi("EMPLOYEES", f"{df['user'].nunique():,}",
                                "users", "Active users · 24 days",
                                daily_active_users), unsafe_allow_html=True)
    with c2:
        st.markdown(premium_kpi("USER-DAY RECORDS", f"{len(df):,}",
                                "database", "Daily records · 24 days",
                                daily_userdays), unsafe_allow_html=True)
    with c3:
        st.markdown(premium_kpi("ACTIVITY SOURCES", f"{len(available_metrics):02d}",
                                "activity", "Logon · USB · File · Email · HTTP"), unsafe_allow_html=True)
    with c4:
        st.markdown(premium_kpi("CANDIDATE FEATURES", f"{manifest['feature_count']:,}",
                                "layers", "Prepared for model evaluation"), unsafe_allow_html=True)
    st.divider()
    st.subheader("Activity timeline")
    metric_name = st.selectbox(
        "Select activity type",
        list(available_metrics.keys()),
        key="overview_metric",
    )
    selected_col = available_metrics[metric_name]
    activity_trend = aggregate_activity(df, [selected_col])

    render_overview_activity_chart(activity_trend, selected_col, metric_name)
    st.caption(
        "Event totals across observed user-days in the selected scope and time aggregation. "
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
if page == "Activity Analytics":
    section_heading("Activity analytics", "activity")
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
        premium_series_chart(daily_activity)
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
if page == "Employee Explorer":
    section_heading("Employee explorer", "users")
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
    premium_series_chart(user_df.set_index("day")[plot_columns])
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
# TAB 4: EMPLOYEE-TO-EMPLOYEE COMPARISON
# ----------------------------------------
if page == "Compare Employees":
    section_heading("Compare employees", "users")
    st.caption(
        "Compare two employees over the same selected observation period. "
        "These are descriptive activity measurements, not threat predictions."
    )

    compare_users = sorted(df["user"].dropna().unique().tolist())
    if len(compare_users) < 2:
        st.info("At least two employees are required for comparison.")
    else:
        select_left, select_right = st.columns(2)
        with select_left:
            employee_a = st.selectbox(
                "Employee A", compare_users, index=0, key="compare_employee_a"
            )
        with select_right:
            employee_b = st.selectbox(
                "Employee B", compare_users,
                index=1, key="compare_employee_b"
            )

        if employee_a == employee_b:
            st.warning("Select two different employees to compare their behavior.")
        else:
            metric_label = st.selectbox(
                "Activity to compare",
                list(available_metrics.keys()),
                key="compare_metric",
            )
            metric_col = available_metrics[metric_label]

            a_df = df.loc[df["user"] == employee_a].sort_values("day")
            b_df = df.loc[df["user"] == employee_b].sort_values("day")

            # Counts are only compared on observed user-days. Missing observations
            # must not be silently converted into zero activity.
            def employee_summary(user_data, user_id):
                observed_days = len(user_data)
                total = user_data[metric_col].sum()
                return {
                    "Employee": user_id,
                    "Observed user-days": observed_days,
                    f"Total {metric_label.lower()}": int(total),
                    "Mean events per observed user-day": (
                        round(float(total) / observed_days, 2)
                        if observed_days else float("nan")
                    ),
                    "Days with activity": int((user_data[metric_col] > 0).sum()),
                }

            st.subheader("Activity comparison")
            trends = []
            for user_id, user_data in [(employee_a, a_df), (employee_b, b_df)]:
                series = aggregate_activity(user_data, [metric_col])[metric_col]
                # Aggregation must not fill periods without observed user-days.
                # Align counts to the observed date periods rather than a
                # continuous resample range spanning gaps in employee activity.
                if chart_frequency == "Weekly":
                    observed_periods = (
                        user_data["day"].dt.to_period("W-SUN")
                        .dt.start_time.unique()
                    )
                elif chart_frequency == "Monthly":
                    observed_periods = (
                        user_data["day"].dt.to_period("M")
                        .dt.start_time.unique()
                    )
                else:
                    observed_periods = user_data["day"].unique()
                series = series.reindex(pd.DatetimeIndex(observed_periods))
                trends.append(series.rename(str(user_id)))

            chart_data = pd.concat(trends, axis=1).sort_index()
            premium_series_chart(chart_data)
            st.caption(
                f"{chart_frequency} {metric_label.lower()} totals for each employee. "
                "A gap indicates no observed user-day in that period, not zero activity. "
                "Unequal observation coverage should be considered before comparison."
            )

            st.subheader("Comparison summary")
            st.dataframe(
                pd.DataFrame([
                    employee_summary(a_df, employee_a),
                    employee_summary(b_df, employee_b),
                ]),
                hide_index=True,
                width="stretch",
            )

            st.subheader("Organizational context")
            context_rows = []
            for user_id, user_data in [(employee_a, a_df), (employee_b, b_df)]:
                latest = user_data.iloc[-1]
                context_rows.append({
                    "Employee": user_id,
                    "Role": latest.get("role", "Unavailable"),
                    "Department": latest.get("department", "Unavailable"),
                    "Team": latest.get("team", "Unavailable"),
                    "Latest observed date": latest["day"].date(),
                })
            st.dataframe(
                pd.DataFrame(context_rows),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Organizational attributes are taken from each employee's "
                "latest observed user-day within the selected date range. "
                "Comparisons do not imply that either employee is anomalous."
            )

# ----------------------------------------
# TAB 5: DEPARTMENT EXPLORER
# ----------------------------------------
if page == "Department Explorer":
    section_heading("Department explorer", "network")
    st.caption(
        "LDAP department assignments are read from the existing user-day "
        "metadata. All results respect the global date filter."
    )

    if "department" not in df.columns:
        st.warning("Department metadata is unavailable in this handoff file.")
    else:
        # Treat missing/unknown assignments transparently; never discard them
        # when calculating a department comparison.
        org = df.copy()
        org["department_display"] = (
            org["department"].astype("string").str.strip()
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        dept_names = sorted(org["department_display"].unique().tolist())
        selected_department = st.selectbox(
            "Department", dept_names, key="department_select"
        )
        dept_df = org.loc[
            org["department_display"] == selected_department
        ]
        c1, c2, c3 = st.columns(3)
        c1.metric("Distinct employees", f"{dept_df['user'].nunique():,}")
        c2.metric("Observed user-days", f"{len(dept_df):,}")
        c3.metric("Department login events", f"{int(dept_df['login_count'].sum()):,}")

        chosen_label = st.selectbox(
            "Activity to compare", list(available_metrics),
            key="department_metric"
        )
        chosen_col = available_metrics[chosen_label]

        st.subheader("Activity trend for selected department")
        # Period totals are divided by the number of observed user-day rows.
        # This avoids presenting larger departments as inherently more active.
        totals_by_period = aggregate_activity(dept_df, [chosen_col])
        row_counts = aggregate_activity(
            dept_df.assign(_observation=1), ["_observation"]
        )["_observation"]
        trend = totals_by_period[chosen_col].div(
            row_counts.replace(0, float("nan"))
        ).rename(f"Mean {chosen_label.lower()} per user-day")
        premium_series_chart(trend, ylabel="Mean events per observed user-day")
        st.caption(
            "Mean count per observed user-day for each selected chart period; "
            "not a risk or anomaly score. The first/last period may be partial."
        )

        st.subheader("Compare departments")
        grouped = org.groupby("department_display", dropna=False)
        comparison = grouped.agg(
            employees=("user", "nunique"),
            user_days=("user", "size"),
            total_events=(chosen_col, "sum"),
        ).reset_index().rename(columns={
            "department_display": "Department",
            "employees": "Employees",
            "user_days": "User-days",
            "total_events": "Total events",
        })
        comparison[f"Mean {chosen_label.lower()} per user-day"] = (
            comparison["Total events"] / comparison["User-days"]
        ).round(2)
        comparison = comparison.sort_values(
            f"Mean {chosen_label.lower()} per user-day", ascending=False
        )
        import altair as alt
        measure = f"Mean {chosen_label.lower()} per user-day"
        horizontal = (
            alt.Chart(comparison)
            .mark_bar(cornerRadiusEnd=4, color="#CFA75D")
            .encode(
                x=alt.X(f"{measure}:Q", title="Mean events per observed user-day"),
                y=alt.Y("Department:N", sort="-x", title=None, axis=alt.Axis(labelLimit=275, labelFontSize=11)),
                tooltip=["Department:N", alt.Tooltip(f"{measure}:Q", format=".2f"),
                         "Employees:Q", "User-days:Q"],
            )
            .properties(height=max(320, 25 * len(comparison)))
        )
        st.altair_chart(chart_style(horizontal, height=max(350, 29 * len(comparison))), width="stretch")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            "Departments are compared on mean activity per observed user-day. "
            "Unknown assignments are retained, and an employee can appear "
            "in multiple departments over time."
        )

# ----------------------------------------
# TAB 6: HISTORICAL PEER BASELINE COMPARISON
# ----------------------------------------
if page == "Peer Baselines":
    section_heading("Historical peer baselines", "activity")
    st.caption(
        "Compare observed user-day activity with precomputed historical "
        "baselines for the employee, their role and their department. "
        "These are descriptive comparisons, not anomaly classifications."
    )

    peer_users = sorted(df["user"].dropna().unique().tolist())
    if not peer_users:
        st.info("No employee records available for the selected dates.")
    else:
        left, right = st.columns(2)
        with left:
            peer_user = st.selectbox(
                "Employee ID", peer_users, key="peer_employee"
            )
        with right:
            peer_metric_label = st.selectbox(
                "Activity to examine",
                list(available_metrics.keys()),
                key="peer_metric",
            )

        peer_metric = available_metrics[peer_metric_label]
        # Exact field names originate in the validated PySpark handoff.
        baseline_columns = {
            "Observed activity": peer_metric,
            "Own historical mean": f"{peer_metric}_hist_mean",
            "Role peer historical mean": f"role_{peer_metric}_peer_mean",
            "Department peer historical mean": f"department_{peer_metric}_peer_mean",
        }
        required = list(baseline_columns.values())
        missing = [column for column in required if column not in df.columns]

        if missing:
            st.error(
                "Expected precomputed baseline columns are absent: "
                + ", ".join(missing)
            )
        else:
            person = (
                df.loc[df["user"] == peer_user]
                .sort_values("day")
                .copy()
            )
            # Preserve warm-up NaNs. Do not recompute baselines after filtering.
            for column in required:
                person[column] = pd.to_numeric(person[column], errors="coerce")

            observed_days = len(person)
            own_days = int(person[baseline_columns["Own historical mean"]].notna().sum())
            role_days = int(person[baseline_columns["Role peer historical mean"]].notna().sum())
            department_days = int(
                person[baseline_columns["Department peer historical mean"]].notna().sum()
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observed user-days", f"{observed_days:,}")
            c2.metric("Own baseline available", f"{own_days:,}")
            c3.metric("Role baseline available", f"{role_days:,}")
            c4.metric("Department baseline available", f"{department_days:,}")

            st.subheader("Observed activity and historical references")
            show_series = st.multiselect(
                "Show series",
                options=list(baseline_columns.keys()),
                default=list(baseline_columns.keys()),
                key="peer_series",
            )
            if show_series:
                chart_data = person.set_index("day")[[
                    baseline_columns[name] for name in show_series
                ]].rename(columns={
                    baseline_columns[name]: name for name in show_series
                })
                chart_data = chart_data.rename(columns={
                    "Observed activity": "Observed",
                    "Own historical mean": "Own history",
                    "Role peer historical mean": "Role peers",
                    "Department peer historical mean": "Department peers",
                })
                premium_series_chart(
                    chart_data,
                    height=355,
                    ylabel="Events per observed user-day",
                )
            else:
                st.info("Select at least one series to display the chart.")

            st.caption(
                "Values are event counts per observed user-day. Own, role and "
                "department historical baselines were calculated upstream "
                "using earlier data; missing warm-up baselines remain missing. "
                "This chart stays at daily granularity even when the global "
                "aggregate-chart setting is Weekly or Monthly."
            )

            st.subheader("Baseline coverage")
            coverage = pd.DataFrame([
                {"Reference": "Own historical mean", "Available days": own_days},
                {"Reference": "Role peer historical mean", "Available days": role_days},
                {"Reference": "Department peer historical mean", "Available days": department_days},
            ])
            coverage["Coverage (%)"] = (
                100 * coverage["Available days"] / observed_days
            ).round(1)
            st.dataframe(coverage, hide_index=True, width="stretch")

            st.subheader("Recent observed days and baseline differences")
            comparison = pd.DataFrame({
                "Date": person["day"].dt.date,
                "Observed": person[peer_metric],
                "Own historical mean": person[baseline_columns["Own historical mean"]],
                "Role peer mean": person[baseline_columns["Role peer historical mean"]],
                "Department peer mean": person[baseline_columns["Department peer historical mean"]],
            })
            for label, baseline in [
                ("Minus own baseline", "Own historical mean"),
                ("Minus role peer mean", "Role peer mean"),
                ("Minus department peer mean", "Department peer mean"),
            ]:
                comparison[label] = comparison["Observed"] - comparison[baseline]
            st.dataframe(
                comparison.tail(30).round(2),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Difference = observed event count minus the corresponding "
                "precomputed historical mean. A positive difference is not "
                "proof of malicious activity. Role and department assignments "
                "can change over time; peer references are row-specific."
            )
            if own_days == 0 and role_days == 0 and department_days == 0:
                st.warning(
                    "No historical reference values are available for this "
                    "employee within the selected date range."
                )

# ----------------------------------------
# TAB 7: MODEL INTEGRATION
# ----------------------------------------
if page == "ML Integration":
    section_heading("Machine learning integration", "layers")
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
st.markdown(
    '<div class="ueba-footer">UEBA ANALYTICS / CERT r4.2 &nbsp;·&nbsp; '
    'OFFLINE RESEARCH CONSOLE &nbsp;·&nbsp; NOT FOR OPERATIONAL THREAT DECISIONS</div>',
    unsafe_allow_html=True,
)
