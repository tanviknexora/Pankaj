"""
BRF Referral Performance Dashboard
----------------------------------
Upload three files:
  1. User Data export (.xlsx)          — Client Id, Total Deposit, First Deposit,
                                          Last Transaction, Trade Dates
  2. Broker Referrals export (.csv)    — Referrer Client ID, Client Id,
                                          Referred User Created At
  3. Trading Customers export (.csv)   — Name, Client ID, Status, Manager,
                                          First Trade, Referred By

The app merges User Data ⋈ Broker Referrals for IB-children deposit/trade
performance, and separately reads the Trading Customers (CRM) export for
Total/Pullback/Sales referral counts and broker activity — then combines
all three on the shared Month/Date/Week bucket (outer join, so a period
present in only one source still shows up with 0s instead of being
silently dropped, matching the merge_month / day_merge_day logic used in
the underlying notebook).

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
REQUIRED_BRF_COLS = ["Referrer Client ID", "Client Id", "Referred User Created At"]
REQUIRED_CUSTOMERS_COLS = ["Name", "Client ID", "Status", "Manager", "First Trade", "Referred By"]
DEFAULT_PULLBACK_MANAGERS = {"parth", "swathi", "rajinder"}


@st.cache_data(show_spinner=False)
def load_user_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_brf_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_customers_data(file_bytes: bytes) -> pd.DataFrame:
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
    """Load, validate, and merge the two files. Returns the merged raw
    frame (unfiltered) plus any missing-column errors. Aggregation is
    intentionally kept out of this cached step so sidebar filters can be
    applied cheaply afterwards without re-reading/re-merging the files."""
    user_data = load_user_data(user_bytes)
    brf = load_brf_data(brf_bytes)

    missing_user = [c for c in REQUIRED_USER_COLS if c not in user_data.columns]
    missing_brf = [c for c in REQUIRED_BRF_COLS if c not in brf.columns]
    if missing_user or missing_brf:
        return None, missing_user, missing_brf

    user_data = user_data[REQUIRED_USER_COLS].copy()
    brf = brf[REQUIRED_BRF_COLS].copy()

    brf["Referred User Created At Parsed"] = parse_referred_date(brf["Referred User Created At"])
    brf["Month"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m")
    brf["Date"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m-%d")
    # Week bucket (Monday-start) for a less cluttered daily view
    brf["Week"] = brf["Referred User Created At Parsed"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.strftime("%Y-%m-%d") if pd.notna(p) else None
    )

    merged = pd.merge(user_data, brf, on="Client Id", how="inner")
    return merged, [], []


@st.cache_data(show_spinner=False)
def build_customers(customers_bytes: bytes):
    """Load + clean the Trading Customers export (the CRM's own referral
    record). Only rows with a non-empty 'Referred By' are kept, matching
    the notebook. Manager -> Pullback/Sales classification is intentionally
    left out of this cached step since the manager list is user-editable
    in the sidebar and shouldn't require re-reading the file to change."""
    customers = load_customers_data(customers_bytes)
    missing = [c for c in REQUIRED_CUSTOMERS_COLS if c not in customers.columns]
    if missing:
        return None, missing

    customers = customers[customers["Referred By"].notna()].copy()
    customers = customers[["Name", "Client ID", "Status", "Manager", "First Trade"]].copy()
    parsed = pd.to_datetime(customers["First Trade"], errors="coerce")
    customers["Month"] = parsed.dt.strftime("%Y-%m")
    customers["Date"] = parsed.dt.strftime("%Y-%m-%d")
    customers["Week"] = parsed.dt.to_period("W-SUN").apply(
        lambda p: p.start_time.strftime("%Y-%m-%d") if pd.notna(p) else None
    )
    return customers, []


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


def aggregate_referrals(customers: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Pullback vs Sales referral counts straight from the Trading Customers
    (CRM) export — this is the 'ground truth' referral total, independent
    of whether BRF/User_Data happen to have caught up to the same dates."""
    cols = [group_col, "Total Referrals", "Pullback Referrals", "Sales Referrals"]
    if customers.empty:
        return pd.DataFrame(columns=cols)
    pivot = (
        customers.groupby([group_col, "Type"])["Client ID"]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=["Pullback", "Sales"], fill_value=0)
        .rename(columns={"Pullback": "Pullback Referrals", "Sales": "Sales Referrals"})
        .reset_index()
    )
    pivot["Total Referrals"] = pivot["Pullback Referrals"] + pivot["Sales Referrals"]
    return pivot[cols].sort_values(group_col)


def aggregate_brokers(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Count of distinct brokers (Referrer Client ID) who were active in
    each period, bucketed by the same Referred User Created At-derived
    Month/Date/Week used everywhere else — i.e. 'brokers at the time'."""
    cols = [group_col, "Total Brokers"]
    if merged.empty:
        return pd.DataFrame(columns=cols)
    g = merged.groupby(group_col)["Referrer Client ID"].nunique().reset_index(name="Total Brokers")
    return g[cols].sort_values(group_col)


def combine_frames(base: pd.DataFrame, referrals: pd.DataFrame, brokers: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Outer-join the BRF/User_Data-based performance table with the CRM
    referral counts and the broker-activity counts on the shared time
    bucket (Month/Date/Week), so a period present in only one source still
    shows up (with 0s, not a silently dropped row) — mirroring the
    day_merge_day / merge_month logic in the notebook."""
    combined = pd.merge(referrals, base, on=group_col, how="outer")
    combined = pd.merge(combined, brokers, on=group_col, how="outer")

    numeric_cols = ["Ib_Children", "Deposited", "Trade_Count",
                     "Total Referrals", "Pullback Referrals", "Sales Referrals", "Total Brokers"]
    for col in numeric_cols:
        if col not in combined.columns:
            combined[col] = 0
    combined[numeric_cols] = combined[numeric_cols].fillna(0).astype(int)

    # BRF/User_Data-based rates (original basis: Ib_Children / Deposited)
    combined["deposit%"] = safe_pct(combined["Deposited"], combined["Ib_Children"])
    combined["trade%"] = safe_pct(combined["Trade_Count"], combined["Deposited"])
    # Team split, as a % of the CRM's own Total Referrals
    combined["Pullback %"] = safe_pct(combined["Pullback Referrals"], combined["Total Referrals"])
    combined["Sales %"] = safe_pct(combined["Sales Referrals"], combined["Total Referrals"])
    # Deposit/trade rates re-based on Total Referrals (CRM ground truth) instead of Ib_Children
    combined["deposit% (of Referrals)"] = safe_pct(combined["Deposited"], combined["Total Referrals"])
    combined["trade% (of Referrals)"] = safe_pct(combined["Trade_Count"], combined["Total Referrals"])

    ordered = [group_col, "Total Referrals", "Pullback Referrals", "Sales Referrals", "Pullback %", "Sales %",
               "Total Brokers", "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%",
               "deposit% (of Referrals)", "trade% (of Referrals)"]
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
    help="e.g. brokers-with-referrals.csv — must contain Referrer Client ID, "
         "Client Id, Referred User Created At",
)
customers_file = st.sidebar.file_uploader(
    "Trading Customers export (.csv)",
    type=["csv"],
    help="e.g. trading-customers.csv — must contain Name, Client ID, Status, "
         "Manager, First Trade, Referred By",
)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("📊 BRF Referral Performance Dashboard")
st.caption("Track IB-children onboarding, deposit conversion, and trade activation — monthly & daily.")

if not user_file or not brf_file or not customers_file:
    st.info("👈 Upload the **User Data (.xlsx)**, **Broker Referrals (.csv)**, and "
            "**Trading Customers (.csv)** files in the sidebar to get started.")
    st.stop()

with st.spinner("Merging files..."):
    merged_raw, missing_user, missing_brf = build_merged(user_file.getvalue(), brf_file.getvalue())
    customers_raw, missing_customers = build_customers(customers_file.getvalue())

if missing_user or missing_brf or missing_customers:
    if missing_user:
        st.error(f"User Data file is missing required column(s): {', '.join(missing_user)}")
    if missing_brf:
        st.error(f"Broker Referrals file is missing required column(s): {', '.join(missing_brf)}")
    if missing_customers:
        st.error(f"Trading Customers file is missing required column(s): {', '.join(missing_customers)}")
    st.stop()

if merged_raw.empty:
    st.warning("No matching Client Ids found between User Data and Broker Referrals after the merge. Double-check the uploads.")
    st.stop()

if customers_raw.empty:
    st.warning("No rows in the Trading Customers file have a non-empty 'Referred By' — "
               "Total Referrals / Pullback / Sales figures will all show as 0.")

# ----------------------------------------------------------------------
# Sidebar — filters (Referrer / Child Client Id)
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🔎 Filters")

referrer_options = sorted(merged_raw["Referrer Client ID"].dropna().unique().tolist())
child_options = sorted(merged_raw["Client Id"].dropna().unique().tolist())

selected_referrers = st.sidebar.multiselect(
    "Referrer Client ID",
    options=referrer_options,
    default=[],
    help="Leave empty to include all referrers.",
)
selected_children = st.sidebar.multiselect(
    "Client Id (child)",
    options=child_options,
    default=[],
    help="Leave empty to include all referred children.",
)

merged = merged_raw.copy()
if selected_referrers:
    merged = merged[merged["Referrer Client ID"].isin(selected_referrers)]
if selected_children:
    merged = merged[merged["Client Id"].isin(selected_children)]

st.sidebar.caption(
    f"Showing **{merged['Client Id'].nunique():,}** of {merged_raw['Client Id'].nunique():,} children · "
    f"**{merged['Referrer Client ID'].nunique():,}** of {merged_raw['Referrer Client ID'].nunique():,} brokers."
)

if merged.empty:
    st.warning("No rows match the current filters. Try clearing the Referrer / Client Id filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar — team classification (Pullback vs Sales managers)
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🧑‍💼 Team Classification")

manager_options = sorted(customers_raw["Manager"].dropna().unique().tolist())
default_pullback = [m for m in manager_options if str(m).strip().lower() in DEFAULT_PULLBACK_MANAGERS]

pullback_managers = st.sidebar.multiselect(
    "Managers classified as Pullback",
    options=manager_options,
    default=default_pullback,
    help="Everyone else in the Manager column is classified as Sales. "
         "Edit this list to reclassify without touching code.",
)

st.sidebar.divider()
st.sidebar.caption("Built for Nexora BRF / IB-children performance tracking.")

customers_typed = customers_raw.copy()
customers_typed["Type"] = customers_typed["Manager"].isin(pullback_managers).map({True: "Pullback", False: "Sales"})

# Apply the same Referrer / Client Id filters to the Trading Customers data.
# The CRM export has no Referrer column of its own, so map child -> referrer
# via the (unfiltered) BRF data first.
referrer_lookup = (
    merged_raw.drop_duplicates(subset=["Client Id"]).set_index("Client Id")["Referrer Client ID"]
)
customers_filtered = customers_typed.copy()
if selected_children:
    customers_filtered = customers_filtered[customers_filtered["Client ID"].isin(selected_children)]
if selected_referrers:
    mapped_referrer = customers_filtered["Client ID"].map(referrer_lookup)
    customers_filtered = customers_filtered[mapped_referrer.isin(selected_referrers)]

# ----------------------------------------------------------------------
# Build the combined Month / Date / Week tables:
# BRF+User_Data performance  ⋈  CRM referral counts (Pullback/Sales)  ⋈  broker activity
# ----------------------------------------------------------------------
result_monthly = combine_frames(
    aggregate(merged, "Month"),
    aggregate_referrals(customers_filtered, "Month"),
    aggregate_brokers(merged, "Month"),
    "Month",
)
result_daily = combine_frames(
    aggregate(merged, "Date"),
    aggregate_referrals(customers_filtered, "Date"),
    aggregate_brokers(merged, "Date"),
    "Date",
)
result_weekly = combine_frames(
    aggregate(merged, "Week"),
    aggregate_referrals(customers_filtered, "Week"),
    aggregate_brokers(merged, "Week"),
    "Week",
)

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------
total_children = int(merged["Client Id"].nunique())
total_deposited = int((merged["Total Deposit"] > 0).sum())
total_traded = int(merged["Last Transaction"].notna().sum())
overall_deposit_pct = round(total_deposited / total_children * 100, 2) if total_children else 0
overall_trade_pct = round(total_traded / total_deposited * 100, 2) if total_deposited else 0

latest_month_children, delta_children = kpi_delta(result_monthly["Ib_Children"])
latest_month_dep_pct, delta_dep_pct = kpi_delta(result_monthly["deposit%"])
latest_month_trade_pct, delta_trade_pct = kpi_delta(result_monthly["trade%"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total IB Children", f"{total_children:,}")
k2.metric("Deposited", f"{total_deposited:,}", help="Clients with Total Deposit > 0")
k3.metric("Overall Deposit %", f"{overall_deposit_pct}%")
k4.metric("Traded (of Deposited)", f"{total_traded:,}")
k5.metric("Overall Trade %", f"{overall_trade_pct}%")

st.markdown("")
k6, k7, k8 = st.columns(3)
k6.metric(
    "Latest Month — New Children",
    f"{int(latest_month_children):,}" if latest_month_children is not None else "—",
    delta=f"{int(delta_children):+,}" if delta_children is not None else None,
)
k7.metric(
    "Latest Month — Deposit %",
    f"{latest_month_dep_pct}%" if latest_month_dep_pct is not None else "—",
    delta=f"{delta_dep_pct:+}%" if delta_dep_pct is not None else None,
)
k8.metric(
    "Latest Month — Trade %",
    f"{latest_month_trade_pct}%" if latest_month_trade_pct is not None else "—",
    delta=f"{delta_trade_pct:+}%" if delta_trade_pct is not None else None,
)

st.markdown("")
total_referrals_overall = int(customers_filtered["Client ID"].nunique())
total_pullback_overall = int((customers_filtered["Type"] == "Pullback").sum())
total_sales_overall = int((customers_filtered["Type"] == "Sales").sum())
overall_pullback_pct = round(total_pullback_overall / total_referrals_overall * 100, 2) if total_referrals_overall else 0
overall_sales_pct = round(total_sales_overall / total_referrals_overall * 100, 2) if total_referrals_overall else 0
total_brokers_overall = int(merged["Referrer Client ID"].nunique())

k9, k10, k11, k12 = st.columns(4)
k9.metric(
    "Total Referrals (CRM)", f"{total_referrals_overall:,}",
    help="Trading Customers rows with a non-empty 'Referred By', after Referrer/Client Id filters.",
)
k10.metric(
    "Total Brokers", f"{total_brokers_overall:,}",
    help="Unique Referrer Client IDs active over the current filtered date range.",
)
k11.metric(
    "Pullback %", f"{overall_pullback_pct}%",
    help=f"{total_pullback_overall:,} of {total_referrals_overall:,} referrals",
)
k12.metric(
    "Sales %", f"{overall_sales_pct}%",
    help=f"{total_sales_overall:,} of {total_referrals_overall:,} referrals",
)

st.divider()

# ----------------------------------------------------------------------
# Trend charts
# ----------------------------------------------------------------------
tab_monthly, tab_daily, tab_team = st.tabs(["📅 Monthly Trends", "📆 Daily / Weekly Trends", "🧑‍💼 Team & Brokers"])

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
        date_range = st.slider(
            "Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d),
            format="YYYY-MM-DD", key="daily_date_range",
        )
        mask = (pd.to_datetime(view_df["Date"]).dt.date >= date_range[0]) & \
               (pd.to_datetime(view_df["Date"]).dt.date <= date_range[1])
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

with tab_team:
    st.subheader("Referrals by Team vs. Broker Activity")
    team_granularity = st.radio(
        "Granularity", ["Monthly", "Weekly", "Daily"], horizontal=True, key="team_granularity",
    )
    if team_granularity == "Monthly":
        tdf, xcol = result_monthly, "Month"
    elif team_granularity == "Weekly":
        tdf, xcol = result_weekly, "Week"
    else:
        tdf, xcol = result_daily, "Date"

    fig6 = go.Figure()
    fig6.add_bar(x=tdf[xcol], y=tdf["Pullback Referrals"], name="Pullback Referrals", marker_color="#6C8EF5")
    fig6.add_bar(x=tdf[xcol], y=tdf["Sales Referrals"], name="Sales Referrals", marker_color="#F5A623")
    fig6.add_trace(go.Scatter(
        x=tdf[xcol], y=tdf["Total Brokers"], name="Total Brokers", mode="lines+markers",
        line=dict(color="#38C793", width=3), yaxis="y2",
    ))
    fig6.update_layout(
        barmode="stack", xaxis_title=xcol, yaxis_title="Referrals",
        yaxis2=dict(title="Brokers", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig6, use_container_width=True, key="team_broker_chart")

    st.subheader(f"Pullback % / Sales % of Total Referrals ({team_granularity})")
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=tdf[xcol], y=tdf["Pullback %"], mode="lines+markers",
                               name="Pullback %", line=dict(color="#6C8EF5", width=3)))
    fig7.add_trace(go.Scatter(x=tdf[xcol], y=tdf["Sales %"], mode="lines+markers",
                               name="Sales %", line=dict(color="#F5A623", width=3)))
    fig7.update_layout(
        xaxis_title=xcol, yaxis_title="%",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig7, use_container_width=True, key="team_pct_chart")

    st.subheader(f"Deposit % / Trade % — of Total Referrals ({team_granularity})")
    st.caption(
        "These re-base deposit%/trade% on the CRM's Total Referrals instead of Ib_Children — "
        "useful when the BRF/User_Data export is a few days stale relative to the Trading "
        "Customers export, since Total Referrals doesn't depend on that join at all."
    )
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=tdf[xcol], y=tdf["deposit% (of Referrals)"], mode="lines+markers",
                               name="Deposit % (of Referrals)", line=dict(color="#38C793", width=2)))
    fig8.add_trace(go.Scatter(x=tdf[xcol], y=tdf["trade% (of Referrals)"], mode="lines+markers",
                               name="Trade % (of Referrals)", line=dict(color="#F5A623", width=2)))
    fig8.update_layout(
        xaxis_title=xcol, yaxis_title="%",
        legend=dict(orientation="h", y=1.12), margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig8, use_container_width=True, key="team_referral_pct_chart")

    st.subheader(f"{team_granularity} Team & Broker Table")
    st.dataframe(tdf, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Download {team_granularity} Team/Broker Table (CSV)",
        data=df_to_csv_bytes(tdf),
        file_name=f"brf_{team_granularity.lower()}_team_broker.csv",
        mime="text/csv",
        key="team_download",
    )

    st.divider()
    with st.expander("👤 Referrals by raw Manager name (before Pullback/Sales grouping)"):
        manager_breakdown = (
            customers_filtered.groupby(["Manager", "Type"], as_index=False)["Client ID"]
            .count()
            .rename(columns={"Client ID": "Referrals"})
            .sort_values("Referrals", ascending=False)
        )
        st.dataframe(manager_breakdown, use_container_width=True, hide_index=True)

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

with st.expander("🔍 View filtered Trading Customers data (CRM referrals)"):
    st.dataframe(customers_filtered, use_container_width=True)
    st.download_button(
        "⬇️ Download Trading Customers Data (CSV)",
        data=df_to_csv_bytes(customers_filtered),
        file_name="brf_trading_customers_filtered.csv",
        mime="text/csv",
        key="customers_download",
    )
