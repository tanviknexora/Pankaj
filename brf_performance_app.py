"""
BRF Referral Performance Dashboard
----------------------------------
Upload the User Data export (xlsx) and the Broker Referrals export (csv),
and this app will merge them, compute IB-children deposit/trade performance,
show KPIs + trend charts, and let you download the aggregated tables.

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
# Real Broker Referrals export header is: Broker ID, Broker Name, Broker Client ID,
# Broker Referral Code, Broker Referral Count, Referred User ID, Referred User Name,
# Referred User Client ID, Referred User Status, Referred User Balance,
# Referred User Created At. Only these three are required to run; "Broker Name" is
# picked up if present (for nicer referrer labels) but isn't required.
REQUIRED_BRF_COLS = ["Broker Client ID", "Referred User Client ID", "Referred User Created At"]
OPTIONAL_BRF_COLS = ["Broker Name"]


@st.cache_data(show_spinner=False)
def load_user_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_brf_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def parse_referred_date(series: pd.Series) -> pd.Series:
    """Try known 'Referred User Created At' export formats explicitly, in
    order, and use whichever matches the most rows. Explicit formats never
    guess which number is the day vs. the month, so this is safe even when
    day and month are both <=12 (the classic ambiguous case).

    Two real export formats have been seen in practice:
      - "2026-09-05 11:11:27"  (ISO, %Y-%m-%d %H:%M:%S)
      - "03/09/2026 18:20"     (dd/mm/yyyy %H:%M)

    IMPORTANT: a naive fallback of pd.to_datetime(..., dayfirst=True) with
    no explicit format is NOT safe. dayfirst=True forces a day/month swap
    on any ambiguous pair of numbers -- including in an otherwise-
    unambiguous ISO "YYYY-MM-DD" string, e.g. it silently turns
    2026-09-05 into 2026-05-09. Only as an absolute last resort (no known
    format reaches even 50% match) do we fall back to a plain, non-dayfirst
    flexible parse, which is far safer for ISO-like input.
    """
    candidate_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]
    best_parsed, best_score = None, -1.0
    for fmt in candidate_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        score = parsed.notna().mean()
        if score > best_score:
            best_parsed, best_score = parsed, score
        if score >= 0.98:
            break

    if best_score < 0.5:
        best_parsed = pd.to_datetime(series, errors="coerce")
    return best_parsed


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
        return None, None, missing_user, missing_brf

    user_data = user_data[REQUIRED_USER_COLS].copy()
    # Defensive normalization: trim stray whitespace on the join key so an
    # otherwise-matching Client Id (e.g. "A201974" vs "A201974 ") doesn't
    # silently fail to match and get miscounted as "never joined User_Data".
    user_data["Client Id"] = user_data["Client Id"].astype(str).str.strip()

    brf_cols = REQUIRED_BRF_COLS + [c for c in OPTIONAL_BRF_COLS if c in brf.columns]
    brf = brf[brf_cols].copy()
    brf = brf.rename(columns={
        "Broker Client ID": "Referrer Client ID",
        "Referred User Client ID": "Client Id",
        "Broker Name": "Referrer Name",
    })
    brf["Client Id"] = brf["Client Id"].astype(str).str.strip()

    brf["Referred User Created At Parsed"] = parse_referred_date(brf["Referred User Created At"])
    brf["Month"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m")
    brf["Date"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m-%d")
    # Week bucket (Monday-start) for a less cluttered daily view
    brf["Week"] = brf["Referred User Created At Parsed"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.strftime("%Y-%m-%d") if pd.notna(p) else None
    )

    merged = pd.merge(user_data, brf, on="Client Id", how="inner")
    return merged, brf, [], []


def aggregate(brf_filtered: pd.DataFrame, merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Ib_Children is the TRUE referred-child count for each Month/Date/Week,
    taken straight from the Broker Referral file — NOT restricted to only
    those who also matched into User_Data. Deposited/Trade_Count can only
    ever be computed from User_Data (that's the only place deposit/trade
    figures exist), so those still come from `merged`; deposit%/trade% are
    then computed against the TRUE Ib_Children total, not a matched-only
    subset, so they're not artificially inflated."""
    cols = [group_col, "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]
    if brf_filtered.empty:
        return pd.DataFrame(columns=cols)

    total_children = (
        brf_filtered.groupby(group_col, as_index=False)["Client Id"]
        .nunique()
        .rename(columns={"Client Id": "Ib_Children"})
    )

    if merged.empty:
        perf = pd.DataFrame(columns=[group_col, "Deposited", "Trade_Count"])
    else:
        perf = merged.groupby(group_col, as_index=False).agg(
            Deposited=("Total Deposit", lambda x: (x > 0).sum()),
            Trade_Count=("Last Transaction", lambda x: x.notna().sum()),
        )

    agg = pd.merge(total_children, perf, on=group_col, how="left")
    for col in ["Ib_Children", "Deposited", "Trade_Count"]:
        agg[col] = agg[col].fillna(0).astype(int)
    agg["deposit%"] = round(agg["Deposited"] / agg["Ib_Children"].replace(0, np.nan) * 100, 2)
    agg["trade%"] = round(agg["Trade_Count"] / agg["Deposited"].replace(0, np.nan) * 100, 2)
    return agg[[group_col, "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]].sort_values(group_col)


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
         "Referred User Client ID, Referred User Created At",
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
    merged_raw, brf_raw, missing_user, missing_brf = build_merged(user_file.getvalue(), brf_file.getvalue())

if missing_user or missing_brf:
    if missing_user:
        st.error(f"User Data file is missing required column(s): {', '.join(missing_user)}")
    if missing_brf:
        st.error(f"Broker Referrals file is missing required column(s): {', '.join(missing_brf)}")
    st.stop()

if merged_raw.empty:
    st.warning("No matching Client Ids found between the two files after the merge. Double-check the uploads.")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar — filters (Referrer / Child Client Id)
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🔎 Filters")

referrer_options = sorted(brf_raw["Referrer Client ID"].dropna().unique().tolist())
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

# The Broker Referral file, filtered the same way but kept separate from
# `merged` — this is the true broker-file count (e.g. 414 distinct brokers,
# 2,857 referred clients), independent of whether a referred client also
# has a row in User_Data.
brf_filtered = brf_raw.copy()
if selected_referrers:
    brf_filtered = brf_filtered[brf_filtered["Referrer Client ID"].isin(selected_referrers)]
if selected_children:
    brf_filtered = brf_filtered[brf_filtered["Client Id"].isin(selected_children)]

st.sidebar.caption(
    f"Showing **{merged['Client Id'].nunique():,}** of {merged_raw['Client Id'].nunique():,} children matched to User_Data · "
    f"**{brf_filtered['Referrer Client ID'].nunique():,}** of {brf_raw['Referrer Client ID'].nunique():,} brokers in the referral file."
)
st.sidebar.divider()
st.sidebar.caption("Built for Nexora BRF / IB-children performance tracking.")

if merged.empty:
    st.warning("No rows match the current filters. Try clearing the Referrer / Client Id filters in the sidebar.")
    st.stop()

result_monthly = aggregate(brf_filtered, merged, "Month")
result_daily = aggregate(brf_filtered, merged, "Date")
result_weekly = aggregate(brf_filtered, merged, "Week")

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------
total_children = int(brf_filtered["Client Id"].nunique())
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
    help="Distinct referred clients in the Broker Referral file — the true total, "
         "not restricted to only those who also have a row in User_Data.",
)
k2.metric(
    "Deposited", f"{total_deposited:,}",
    help="Of those, clients with Total Deposit > 0 in User_Data. Can only be counted "
         "for children who have a matching User_Data row, since that's the only place "
         "deposit figures exist.",
)
k3.metric(
    "Overall Deposit %", f"{overall_deposit_pct}%",
    help="Deposited ÷ Total IB Children × 100.",
)
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
total_distinct_brokers = int(brf_filtered["Referrer Client ID"].nunique())
k9, _, _, _, _ = st.columns(5)
k9.metric(
    "Total Brokers", f"{total_distinct_brokers:,}",
    help="Unique Referrer Client IDs in the Broker Referral file (after filters).",
)

st.divider()

# ----------------------------------------------------------------------
# Trend charts
# ----------------------------------------------------------------------
tab_monthly, tab_daily = st.tabs(["📅 Monthly Trends", "📆 Daily / Weekly Trends"])

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
