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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
REQUIRED_USER_COLS = ["Client Id", "Total Deposit", "First Deposit", "Last Transaction", "Trade Dates"]
REQUIRED_BRF_COLS = ["Referrer Client ID", "Client Id", "Referred User Created At"]


@st.cache_data(show_spinner=False)
def load_user_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df


@st.cache_data(show_spinner=False)
def load_brf_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def parse_referred_date(series: pd.Series) -> pd.Series:
    """Try the expected dd/mm/yyyy HH:MM format first, fall back to a
    flexible parse so the app doesn't hard-fail on slightly different
    export formats."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")
    if parsed.isna().mean() > 0.3:  # format guess was probably wrong
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


@st.cache_data(show_spinner=False)
def process(user_bytes: bytes, brf_bytes: bytes):
    user_data = load_user_data(user_bytes)
    brf = load_brf_data(brf_bytes)

    missing_user = [c for c in REQUIRED_USER_COLS if c not in user_data.columns]
    missing_brf = [c for c in REQUIRED_BRF_COLS if c not in brf.columns]
    if missing_user or missing_brf:
        return None, None, None, missing_user, missing_brf

    user_data = user_data[REQUIRED_USER_COLS].copy()
    brf = brf[REQUIRED_BRF_COLS].copy()

    brf["Referred User Created At Parsed"] = parse_referred_date(brf["Referred User Created At"])
    brf["Month"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m")
    brf["Date"] = brf["Referred User Created At Parsed"].dt.strftime("%Y-%m-%d")

    merged = pd.merge(user_data, brf, on="Client Id", how="inner")

    # Monthly aggregation
    g = merged.groupby("Month", as_index=False)
    agg = g.agg(
        Ib_Children=("Client Id", "count"),
        Deposited=("Total Deposit", lambda x: (x > 0).sum()),
        Trade_Count=("Last Transaction", lambda x: x.notna().sum()),
    ).astype({"Ib_Children": int, "Deposited": int, "Trade_Count": int})
    agg["deposit%"] = round(agg["Deposited"] / agg["Ib_Children"].replace(0, np.nan) * 100, 2)
    agg["trade%"] = round(agg["Trade_Count"] / agg["Deposited"].replace(0, np.nan) * 100, 2)
    result_monthly = agg[["Month", "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]].sort_values("Month")

    # Daily aggregation
    d = merged.groupby("Date", as_index=False)
    agg_1 = d.agg(
        Ib_Children=("Client Id", "count"),
        Deposited=("Total Deposit", lambda x: (x > 0).sum()),
        Trade_Count=("Last Transaction", lambda x: x.notna().sum()),
    )
    agg_1["deposit%"] = round(agg_1["Deposited"] / agg_1["Ib_Children"].replace(0, np.nan) * 100, 2)
    agg_1["trade%"] = round(agg_1["Trade_Count"] / agg_1["Deposited"].replace(0, np.nan) * 100, 2)
    result_daily = agg_1[["Date", "Ib_Children", "Deposited", "deposit%", "Trade_Count", "trade%"]].sort_values("Date")

    return merged, result_monthly, result_daily, [], []


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def kpi_delta(series: pd.Series):
    """Return (latest value, delta vs previous period) for a numeric series."""
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

st.sidebar.divider()
st.sidebar.caption("Built for Nexora BRF / IB-children performance tracking.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("📊 BRF Referral Performance Dashboard")
st.caption("Track IB-children onboarding, deposit conversion, and trade activation — monthly & daily.")

if not user_file or not brf_file:
    st.info("👈 Upload both the **User Data (.xlsx)** and **Broker Referrals (.csv)** files in the sidebar to get started.")
    st.stop()

with st.spinner("Merging and crunching numbers..."):
    merged, result_monthly, result_daily, missing_user, missing_brf = process(
        user_file.getvalue(), brf_file.getvalue()
    )

if missing_user or missing_brf:
    if missing_user:
        st.error(f"User Data file is missing required column(s): {', '.join(missing_user)}")
    if missing_brf:
        st.error(f"Broker Referrals file is missing required column(s): {', '.join(missing_brf)}")
    st.stop()

if merged.empty:
    st.warning("No matching Client Ids found between the two files after the merge. Double-check the uploads.")
    st.stop()

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

st.divider()

# ----------------------------------------------------------------------
# Trend charts
# ----------------------------------------------------------------------
tab_monthly, tab_daily = st.tabs(["📅 Monthly Trends", "📆 Daily Trends"])

with tab_monthly:
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_bar(x=result_monthly["Month"], y=result_monthly["Ib_Children"], name="IB Children", marker_color="#6C8EF5")
        fig.add_bar(x=result_monthly["Month"], y=result_monthly["Deposited"], name="Deposited", marker_color="#38C793")
        fig.update_layout(
            barmode="group", title="New IB Children vs Deposited (Monthly)",
            xaxis_title="Month", yaxis_title="Count", legend=dict(orientation="h", y=1.15),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=result_monthly["Month"], y=result_monthly["deposit%"],
                                   mode="lines+markers", name="Deposit %", line=dict(color="#38C793", width=3)))
        fig2.add_trace(go.Scatter(x=result_monthly["Month"], y=result_monthly["trade%"],
                                   mode="lines+markers", name="Trade %", line=dict(color="#F5A623", width=3)))
        fig2.update_layout(
            title="Deposit % & Trade % Trend (Monthly)",
            xaxis_title="Month", yaxis_title="%", legend=dict(orientation="h", y=1.15),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(result_monthly, x="Month", y="Trade_Count", title="Trade Count (Monthly)",
                   color_discrete_sequence=["#6C8EF5"])
    fig3.update_layout(margin=dict(t=60, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Monthly Result Table")
    st.dataframe(result_monthly, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Monthly Table (CSV)",
        data=df_to_csv_bytes(result_monthly),
        file_name="brf_monthly_performance.csv",
        mime="text/csv",
    )

with tab_daily:
    c1, c2 = st.columns(2)

    with c1:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=result_daily["Date"], y=result_daily["Ib_Children"],
                                   mode="lines", name="IB Children", line=dict(color="#6C8EF5", width=2)))
        fig4.add_trace(go.Scatter(x=result_daily["Date"], y=result_daily["Deposited"],
                                   mode="lines", name="Deposited", line=dict(color="#38C793", width=2)))
        fig4.update_layout(
            title="New IB Children vs Deposited (Daily)",
            xaxis_title="Date", yaxis_title="Count", legend=dict(orientation="h", y=1.15),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=result_daily["Date"], y=result_daily["deposit%"],
                                   mode="lines", name="Deposit %", line=dict(color="#38C793", width=2)))
        fig5.add_trace(go.Scatter(x=result_daily["Date"], y=result_daily["trade%"],
                                   mode="lines", name="Trade %", line=dict(color="#F5A623", width=2)))
        fig5.update_layout(
            title="Deposit % & Trade % Trend (Daily)",
            xaxis_title="Date", yaxis_title="%", legend=dict(orientation="h", y=1.15),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Daily Result Table")
    st.dataframe(result_daily, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Daily Table (CSV)",
        data=df_to_csv_bytes(result_daily),
        file_name="brf_daily_performance.csv",
        mime="text/csv",
    )

st.divider()
with st.expander("🔍 View merged raw data (User Data ⋈ BRF)"):
    st.dataframe(merged, use_container_width=True)
    st.download_button(
        "⬇️ Download Merged Raw Data (CSV)",
        data=df_to_csv_bytes(merged),
        file_name="brf_merged_raw.csv",
        mime="text/csv",
    )
