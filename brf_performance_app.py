"""
BRF Referral Performance Dashboard
----------------------------------
Upload two files:
  1. User Data export (.xlsx)       — Client Id, Total Deposit, First Deposit,
                                       Last Transaction, Trade Dates
  2. Broker Referrals export (.csv) — Broker Client ID, Referred User Client ID,
                                       Referred User Created At (Broker Name optional,
                                       used for nicer referrer labels if present)

The app merges User Data ⋈ Broker Referrals for IB-children deposit/trade
performance, and separately keeps the cleaned Broker Referral file on its
own for broker-count / coverage stats — a broker (or their referral) still
counts there even if the referred client never made it into User_Data.
The two views are combined on the shared Month/Date/Week bucket via an
outer join, so a period present in only one source still shows up with
0s instead of being silently dropped.

No CRM/Trading Customers data is used anywhere in this version.

Run with:
    streamlit run brf_performance_app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="BRF Referral Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stMetric {
        background-color: rgba(151, 166, 195, 0.08);
        border: 1px solid rgba(151, 166, 195, 0.2);
        padding: 16px 12px 8px 12px;
        border-radius: 10px;
    }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: -0.5px; }
    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    /* Give charts breathing room from the tab bar above them so a
       previous tab's chart title can never visually bleed through. */
    div[data-testid="stTabs"] { margin-top: 6px; }
    div[data-testid="stTabs"] div[data-baseweb="tab-panel"] { padding-top: 14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
REQUIRED_USER_COLS = ["Client Id", "Total Deposit", "First Deposit", "Last Transaction", "Trade Dates"]
# Real export header is: Broker ID, Broker Name, Broker Client ID, Broker Referral Code,
# Broker Referral Count, Referred User ID, Referred User Name, Referred User Client ID,
# Referred User Status, Referred User Balance, Referred User Created At.
# Only these three are required; "Broker Name" is picked up if present (for nicer labels)
# but the app doesn't hard-fail without it.
REQUIRED_BRF_COLS = ["Broker Client ID", "Referred User Client ID", "Referred User Created At"]
OPTIONAL_BRF_COLS = ["Broker Name"]


@st.cache_data(show_spinner=False)
def load_user_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_brf_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def parse_referred_date(series: pd.Series) -> pd.Series:
    """Try the expected dd/mm/yyyy HH:MM format first, fall back to a
    flexible parse so the app doesn't hard-fail on slightly different
    export formats."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")
    if parsed.isna().mean() > 0.3:  # format guess was probably wrong
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


@st.cache_data(show_spinner=False)
def build_merged(user_bytes: bytes, brf_bytes: bytes):
    """Load, validate, and merge the two files. Returns:
      - merged: User Data ⋈ Broker Referrals (inner join) — the basis for
        IB-children deposit/trade performance.
      - brf_clean: the cleaned Broker Referrals file on its own, BEFORE the
        merge — the basis for broker-file / coverage stats, since a broker's
        referral still counts toward "brokers in the file" even if that
        referred client never shows up in User_Data.
    Aggregation is intentionally kept out of this cached step so sidebar
    filters can be applied cheaply afterwards without re-reading/re-merging."""
    user_data = load_user_data(user_bytes)
    brf = load_brf_data(brf_bytes)

    missing_user = [c for c in REQUIRED_USER_COLS if c not in user_data.columns]
    missing_brf = [c for c in REQUIRED_BRF_COLS if c not in brf.columns]
    if missing_user or missing_brf:
        return None, None, missing_user, missing_brf

    user_data = user_data[REQUIRED_USER_COLS].copy()

    brf_cols = REQUIRED_BRF_COLS + [c for c in OPTIONAL_BRF_COLS if c in brf.columns]
    brf = brf[brf_cols].copy()
    brf = brf.rename(columns={
        "Broker Client ID": "Referrer Client ID",
        "Referred User Client ID": "Client Id",
        "Broker Name": "Referrer Name",
    })
    if "Referrer Name" not in brf.columns:
        # Older exports without a Broker Name column — fall back to the ID itself.
        brf["Referrer Name"] = brf["Referrer Client ID"]

    brf["Referred User Created At Parsed"] = parse_referred_date(brf["Referred User Created At"])
    brf["Month"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m")
    brf["Date"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m-%d")
    # Week bucket (Monday-start) for a less cluttered daily view
    brf["Week"] = brf["Referred User Created At Parsed"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.strftime("%Y-%m-%d") if pd.notna(p) else None
    )

    merged = pd.merge(user_data, brf, on="Client Id", how="inner")
    return merged, brf, [], []


def aggregate(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=[group_col, "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"])
    g = merged.groupby(group_col, as_index=False)
    agg = g.agg(
        Ib_Children=("Client Id", "count"),
        Deposited=("Total Deposit", lambda x: (x > 0).sum()),
        Trade_Count=("Last Transaction", lambda x: x.notna().sum()),
    ).astype({"Ib_Children": int, "Deposited": int, "Trade_Count": int})
    agg["deposit%"] = round(agg["Deposited"] / agg["Ib_Children"].replace(0, np.nan) * 100, 2)
    agg["trade%"] = round(agg["Trade_Count"] / agg["Deposited"].replace(0, np.nan) * 100, 2)
    return agg[[group_col, "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]].sort_values(group_col)


def safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator * 100, rounded, with 0 (not NaN/inf) wherever
    the denominator is 0."""
    denom = denominator.replace(0, np.nan)
    return (numerator / denom * 100).round(2).fillna(0)


def aggregate_brokers(brf_frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Count of distinct brokers (Referrer Client ID) active in each period,
    bucketed by the same Referred User Created At-derived Month/Date/Week
    used everywhere else. Takes the Broker Referral file directly (not the
    IB-performance merge), so a broker still counts even if their referred
    client never made it into User_Data."""
    cols = [group_col, "Total Brokers"]
    if brf_frame.empty:
        return pd.DataFrame(columns=cols)
    g = brf_frame.groupby(group_col)["Referrer Client ID"].nunique().reset_index(name="Total Brokers")
    return g[cols].sort_values(group_col)


def combine_frames(base: pd.DataFrame, brokers: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Outer-join the BRF/User_Data-based performance table with the
    broker-activity counts on the shared time bucket (Month/Date/Week), so
    a period present in only one source still shows up (with 0s, not a
    silently dropped row) — mirroring the day_merge_day / merge_month logic
    in the underlying notebook."""
    combined = pd.merge(base, brokers, on=group_col, how="outer")

    numeric_cols = ["Ib_Children", "Deposited", "Trade_Count", "Total Brokers"]
    for col in numeric_cols:
        if col not in combined.columns:
            combined[col] = 0
    combined[numeric_cols] = combined[numeric_cols].fillna(0).astype(int)

    combined["deposit%"] = safe_pct(combined["Deposited"], combined["Ib_Children"])
    combined["trade%"] = safe_pct(combined["Trade_Count"], combined["Deposited"])

    ordered = [group_col, "Total Brokers", "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]
    return combined[[c for c in ordered if c in combined.columns]].sort_values(group_col).reset_index(drop=True)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def kpi_delta(series: pd.Series):
    if len(series) == 0:
        return None, None
    latest = series.iloc[-1]
    prev = series.iloc[-2] if len(series) > 1 else None
    delta = None if prev is None else round(latest - prev, 2)
    return latest, delta


# ----------------------------------------------------------------------
# Sidebar — uploads
# ----------------------------------------------------------------------
st.sidebar.title("📁 Data Upload")
st.sidebar.caption("Upload both files to run the analysis.")

user_file = st.sidebar.file_uploader(
    "User Data export (.xlsx)",
    type=["xlsx", "xls"],
    help="e.g. users-all-compiled.xlsx — must contain Client Id, Total Deposit, "
         "First Deposit, Last Transaction, Trade Dates",
)
brf_file = st.sidebar.file_uploader(
    "Broker Referrals export (.csv)",
    type=["csv"],
    help="e.g. brokers-with-referrals.csv — must contain Broker Client ID, "
         "Referred User Client ID, Referred User Created At (Broker Name is "
         "optional but gives nicer referrer labels if present)",
)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("📊 BRF Referral Performance Dashboard")
st.caption("Track IB-children onboarding, deposit conversion, and trade activation — monthly & daily.")

if not user_file or not brf_file:
    st.info("👈 Upload both the **User Data (.xlsx)** and **Broker Referrals (.csv)** files in the sidebar to get started.")
    st.stop()

with st.spinner("Merging files..."):
    merged_raw, brf_clean_raw, missing_user, missing_brf = build_merged(user_file.getvalue(), brf_file.getvalue())

if missing_user or missing_brf:
    if missing_user:
        st.error(f"User Data file is missing required column(s): {', '.join(missing_user)}")
    if missing_brf:
        st.error(f"Broker Referrals file is missing required column(s): {', '.join(missing_brf)}")
    st.stop()

if merged_raw.empty:
    st.warning("No matching Client Ids found between User Data and Broker Referrals after the merge. Double-check the uploads.")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar — filters (Referrer / Child Client Id)
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🔎 Filters")

referrer_options = sorted(brf_clean_raw["Referrer Client ID"].dropna().unique().tolist())
referrer_name_map = (
    brf_clean_raw.drop_duplicates(subset=["Referrer Client ID"])
    .set_index("Referrer Client ID")["Referrer Name"]
    .to_dict()
)
child_options = sorted(merged_raw["Client Id"].dropna().unique().tolist())

selected_referrers = st.sidebar.multiselect(
    "Referrer Client ID",
    options=referrer_options,
    default=[],
    format_func=lambda rid: f"{referrer_name_map.get(rid, rid)} ({rid})",
    help="Leave empty to include all referrers.",
)
selected_children = st.sidebar.multiselect(
    "Client Id (child)",
    options=child_options,
    default=[],
    help="Leave empty to include all referred children.",
)

# ----------------------------------------------------------------------
# Sidebar — date range (applies to every table, chart, and KPI below)
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("📅 Date Range")

all_dates = pd.to_datetime(
    pd.concat([merged_raw["Date"], brf_clean_raw["Date"]], ignore_index=True),
    errors="coerce",
).dropna()

if not all_dates.empty:
    min_date, max_date = all_dates.min().date(), all_dates.max().date()
    date_range = st.sidebar.date_input(
        "Filter by date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filters everything below by Referred User Created At — the KPIs, both "
             "trend tabs, and the raw-data downloads at the bottom.",
    )
    # st.date_input can momentarily return a single date while the user is still
    # picking the second end of the range — fall back to the full span in that case.
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = None, None

st.sidebar.caption(
    f"{start_date:%b %d, %Y} → {end_date:%b %d, %Y}" if start_date and end_date else "No dated rows found."
)


def filter_by_date(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Restrict a frame to the sidebar's selected date range, on its own
    Date column (Referred User Created At for both merged and brf_clean)."""
    if start_date is None or end_date is None or df.empty or date_col not in df.columns:
        return df
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    mask = (parsed.dt.date >= start_date) & (parsed.dt.date <= end_date)
    return df[mask]


st.sidebar.divider()
st.sidebar.caption("Built for Nexora BRF / IB-children performance tracking.")

merged = merged_raw.copy()
if selected_referrers:
    merged = merged[merged["Referrer Client ID"].isin(selected_referrers)]
if selected_children:
    merged = merged[merged["Client Id"].isin(selected_children)]
merged = filter_by_date(merged)

# The Broker Referral file, filtered the same way, kept separate from `merged`
# (the IB-performance join) — this is the basis for the broker-file / coverage
# stats, so a broker still counts even if their referred client never made it
# into User_Data.
brf_clean = brf_clean_raw.copy()
if selected_referrers:
    brf_clean = brf_clean[brf_clean["Referrer Client ID"].isin(selected_referrers)]
if selected_children:
    brf_clean = brf_clean[brf_clean["Client Id"].isin(selected_children)]
brf_clean = filter_by_date(brf_clean)

st.sidebar.caption(
    f"Showing **{merged['Client Id'].nunique():,}** of {merged_raw['Client Id'].nunique():,} children matched to IB performance · "
    f"**{brf_clean['Referrer Client ID'].nunique():,}** of {brf_clean_raw['Referrer Client ID'].nunique():,} brokers in the referral file."
)

if merged.empty:
    st.warning("No rows match the current filters. Try widening the date range or clearing the Referrer / Client Id filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------
# Build the combined Month / Date / Week tables:
# BRF+User_Data performance  ⋈  broker-file activity
# ----------------------------------------------------------------------
result_monthly = combine_frames(aggregate(merged, "Month"), aggregate_brokers(brf_clean, "Month"), "Month")
result_daily = combine_frames(aggregate(merged, "Date"), aggregate_brokers(brf_clean, "Date"), "Date")
result_weekly = combine_frames(aggregate(merged, "Week"), aggregate_brokers(brf_clean, "Week"), "Week")

# ----------------------------------------------------------------------
# KPI dashboard — two separate, clearly-labeled sections so IB
# performance and broker-file stats never get visually mixed together.
# ----------------------------------------------------------------------

# --- Section 1: IB Children Performance (User Data ⋈ Broker Referrals) ---
st.subheader("📊 IB Children Performance")
st.caption("From User Data ⋈ Broker Referrals — deposit and trade activation of referred children.")

total_children = int(merged["Client Id"].nunique())
total_deposited = int((merged["Total Deposit"] > 0).sum())
total_traded = int(merged["Last Transaction"].notna().sum())
overall_deposit_pct = round(total_deposited / total_children * 100, 2) if total_children else 0
overall_trade_pct = round(total_traded / total_deposited * 100, 2) if total_deposited else 0

latest_month_children, delta_children = kpi_delta(result_monthly["Ib_Children"])
latest_month_dep_pct, delta_dep_pct = kpi_delta(result_monthly["deposit%"])
latest_month_trade_pct, delta_trade_pct = kpi_delta(result_monthly["trade%"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Total IB Children", f"{total_children:,}",
    help="merged['Client Id'].nunique() — distinct referred children present in both "
         "User Data and Broker Referrals (after sidebar filters).",
)
k2.metric(
    "Deposited", f"{total_deposited:,}",
    help="Count of those children with Total Deposit > 0.",
)
k3.metric(
    "Overall Deposit %", f"{overall_deposit_pct}%",
    help="Deposited ÷ Total IB Children × 100.",
)
k4.metric(
    "Traded (of Deposited)", f"{total_traded:,}",
    help="Count of children with a non-null Last Transaction — note this is measured "
         "across ALL children, not only the deposited ones.",
)
k5.metric(
    "Overall Trade %", f"{overall_trade_pct}%",
    help="Traded ÷ Deposited × 100 (NOT ÷ Total IB Children).",
)

st.markdown("")
k6, k7, k8 = st.columns(3)
k6.metric(
    "Latest Month — New Children",
    f"{int(latest_month_children):,}" if latest_month_children is not None else "—",
    delta=f"{int(delta_children):+,}" if delta_children is not None else None,
    help="Ib_Children from the most recent row of the Monthly table; delta vs. the prior month.",
)
k7.metric(
    "Latest Month — Deposit %",
    f"{latest_month_dep_pct}%" if latest_month_dep_pct is not None else "—",
    delta=f"{delta_dep_pct:+}%" if delta_dep_pct is not None else None,
    help="deposit% from the most recent month; delta vs. the prior month.",
)
k8.metric(
    "Latest Month — Trade %",
    f"{latest_month_trade_pct}%" if latest_month_trade_pct is not None else "—",
    delta=f"{delta_trade_pct:+}%" if delta_trade_pct is not None else None,
    help="trade% from the most recent month; delta vs. the prior month.",
)

st.divider()

# --- Section 2: Broker File Coverage (Broker Referral file vs. User_Data match) ---
st.subheader("🔗 Broker File Coverage")
st.caption("Purely broker/referral-file stats — how many brokers exist and how much of the file's referral volume is matched into User_Data. No CRM data involved.")

total_distinct_brokers = int(brf_clean["Referrer Client ID"].nunique())
broker_file_referrals = int(brf_clean["Client Id"].nunique())
match_rate_pct = round(total_children / broker_file_referrals * 100, 2) if broker_file_referrals else 0

k9, k10, k11, k12 = st.columns(4)
k9.metric(
    "Total Distinct Brokers", f"{total_distinct_brokers:,}",
    help="brf_clean['Referrer Client ID'].nunique() — every unique broker in the Broker "
         "Referral file (after filters), regardless of whether their referred client "
         "appears in User_Data.",
)
k10.metric(
    "Broker File Referrals", f"{broker_file_referrals:,}",
    help="brf_clean['Client Id'].nunique() — distinct referred clients logged in the "
         "Broker Referral file.",
)
k11.metric(
    "Matched to IB Children", f"{total_children:,}",
    help="Same figure as Total IB Children above — those Broker File Referrals that "
         "also appear in User_Data (and so have measurable deposit/trade performance).",
)
k12.metric(
    "Match Rate %", f"{match_rate_pct}%",
    help=f"Matched to IB Children ÷ Broker File Referrals × 100 — {total_children:,} "
         f"of {broker_file_referrals:,}. Below 100% means the broker file has referrals "
         f"that never showed up in User_Data.",
)

st.divider()

# ----------------------------------------------------------------------
# Trend charts
# ----------------------------------------------------------------------
tab_monthly, tab_daily, tab_brokers = st.tabs(["📅 Monthly Trends", "📆 Daily / Weekly Trends", "📡 Broker Activity"])

with tab_monthly:
    st.subheader("New IB Children vs Deposited (Monthly)")
    fig = go.Figure()
    fig.add_bar(x=result_monthly["Month"], y=result_monthly["Ib_Children"], name="IB Children", marker_color="#6C8EF5")
    fig.add_bar(x=result_monthly["Month"], y=result_monthly["Deposited"], name="Deposited", marker_color="#38C793")
    fig.update_layout(
        barmode="group", xaxis_title="Month", yaxis_title="Count",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="monthly_bar_chart")

    st.subheader("Deposit % & Trade % Trend (Monthly)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=result_monthly["Month"], y=result_monthly["deposit%"],
                               mode="lines+markers", name="Deposit %", line=dict(color="#38C793", width=3)))
    fig2.add_trace(go.Scatter(x=result_monthly["Month"], y=result_monthly["trade%"],
                               mode="lines+markers", name="Trade %", line=dict(color="#F5A623", width=3)))
    fig2.update_layout(
        xaxis_title="Month", yaxis_title="%",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True, key="monthly_pct_chart")

    st.subheader("Trade Count (Monthly)")
    fig3 = px.bar(result_monthly, x="Month", y="Trade_Count", color_discrete_sequence=["#6C8EF5"])
    fig3.update_layout(margin=dict(t=20, b=20), xaxis_title="Month", yaxis_title="Trade Count")
    st.plotly_chart(fig3, use_container_width=True, key="monthly_trade_count_chart")

    st.subheader("Monthly Result Table")
    st.dataframe(result_monthly, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Monthly Table (CSV)",
        data=df_to_csv_bytes(result_monthly),
        file_name="brf_monthly_performance.csv",
        mime="text/csv",
        key="monthly_download",
    )

with tab_daily:
    granularity = st.radio(
        "Granularity", ["Daily", "Weekly"], horizontal=True,
        help="Weekly rolls the daily figures up by ISO week (Mon–Sun) — much easier to read over a long date range.",
        key="daily_granularity",
    )
    view_df, x_col = (result_daily, "Date") if granularity == "Daily" else (result_weekly, "Week")

    if granularity == "Daily" and len(view_df) > 1:
        dates_sorted = sorted(pd.to_datetime(view_df["Date"]).unique())
        min_d, max_d = dates_sorted[0].date(), dates_sorted[-1].date()
        date_range_local = st.slider(
            "Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d),
            format="YYYY-MM-DD", key="daily_date_range",
        )
        mask = (pd.to_datetime(view_df["Date"]).dt.date >= date_range_local[0]) & \
               (pd.to_datetime(view_df["Date"]).dt.date <= date_range_local[1])
        view_df = view_df[mask]

    st.subheader(f"New IB Children vs Deposited ({granularity})")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=view_df[x_col], y=view_df["Ib_Children"],
                               mode="lines+markers", name="IB Children",
                               line=dict(color="#6C8EF5", width=2), marker=dict(size=5)))
    fig4.add_trace(go.Scatter(x=view_df[x_col], y=view_df["Deposited"],
                               mode="lines+markers", name="Deposited",
                               line=dict(color="#38C793", width=2), marker=dict(size=5)))
    fig4.update_layout(
        xaxis_title=x_col, yaxis_title="Count",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig4, use_container_width=True, key="daily_count_chart")

    st.subheader(f"Deposit % & Trade % Trend ({granularity})")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=view_df[x_col], y=view_df["deposit%"],
                               mode="lines+markers", name="Deposit %",
                               line=dict(color="#38C793", width=2), marker=dict(size=5)))
    fig5.add_trace(go.Scatter(x=view_df[x_col], y=view_df["trade%"],
                               mode="lines+markers", name="Trade %",
                               line=dict(color="#F5A623", width=2), marker=dict(size=5)))
    fig5.update_layout(
        xaxis_title=x_col, yaxis_title="%",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig5, use_container_width=True, key="daily_pct_chart")

    st.subheader(f"{granularity} Result Table")
    st.dataframe(view_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Download {granularity} Table (CSV)",
        data=df_to_csv_bytes(view_df),
        file_name=f"brf_{granularity.lower()}_performance.csv",
        mime="text/csv",
        key="daily_download",
    )

with tab_brokers:
    st.subheader("Broker Activity vs. IB Performance")
    st.caption("Total Brokers is from the Broker Referral file itself; Ib_Children/Deposited are from the User Data ⋈ Broker Referrals merge.")
    broker_granularity = st.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], horizontal=True, key="broker_granularity",
    )
    if broker_granularity == "Monthly":
        bdf, xcol = result_monthly, "Month"
    elif broker_granularity == "Weekly":
        bdf, xcol = result_weekly, "Week"
    else:
        bdf, xcol = result_daily, "Date"

    fig6 = go.Figure()
    fig6.add_bar(x=bdf[xcol], y=bdf["Ib_Children"], name="IB Children", marker_color="#6C8EF5")
    fig6.add_bar(x=bdf[xcol], y=bdf["Deposited"], name="Deposited", marker_color="#38C793")
    fig6.add_trace(go.Scatter(
        x=bdf[xcol], y=bdf["Total Brokers"], name="Total Brokers", mode="lines+markers",
        line=dict(color="#F5A623", width=3), yaxis="y2",
    ))
    fig6.update_layout(
        barmode="group", xaxis_title=xcol, yaxis_title="Count",
        yaxis2=dict(title="Brokers", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig6, use_container_width=True, key="broker_activity_chart")

    st.subheader(f"{broker_granularity} Broker & Performance Table")
    st.dataframe(bdf, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Download {broker_granularity} Broker Table (CSV)",
        data=df_to_csv_bytes(bdf),
        file_name=f"brf_{broker_granularity.lower()}_broker_activity.csv",
        mime="text/csv",
        key="broker_download",
    )

st.divider()
with st.expander("🔍 View merged raw data (User Data ⋈ BRF)"):
    st.dataframe(merged, use_container_width=True)
    st.download_button(
        "⬇️ Download Merged Raw Data (CSV)",
        data=df_to_csv_bytes(merged),
        file_name="brf_merged_raw.csv",
        mime="text/csv",
        key="raw_download",
    )

with st.expander("🔍 View filtered Broker Referral file"):
    st.dataframe(brf_clean, use_container_width=True)
    st.download_button(
        "⬇️ Download Broker Referral Data (CSV)",
        data=df_to_csv_bytes(brf_clean),
        file_name="brf_broker_file_filtered.csv",
        mime="text/csv",
        key="brf_download",
    )
