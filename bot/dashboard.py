# dashboard.py
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DB_PATH

"""
#########################################

Running the Dashboard

cd ~/weather-arb-bot/bot
streamlit run dashboard.py --server.port 8501

#########################################
"""

st.set_page_config(
    page_title="Weather Arb Dashboard",
    page_icon="n",
    layout="wide",
)

st.title("Weather Arbitrage Bot Dashboard")

@st.cache_data(ttl=60)  # Refresh every 60 seconds
def load_signals() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_positions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM positions ORDER BY entry_time DESC", conn)
    conn.close()
    return df


# Load data
signals_df = load_signals()
positions_df = load_positions()

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_pnl = positions_df["pnl"].sum() if not positions_df.empty else 0
    st.metric("Cumulative P&L", f"${total_pnl:.2f}")
with col2:
    open_count = len(positions_df[positions_df["status"] == "open"]) if not positions_df.empty else 0
    st.metric("Open Positions", open_count)
with col3:
    total_signals = len(signals_df)
    st.metric("Total Signals", total_signals)
with col4:
    if not signals_df.empty and len(signals_df) > 0:
        avg_ev = signals_df["ev"].mean()
        st.metric("Avg Signal EV", f"{avg_ev:.3f}")
    else:
        st.metric("Avg Signal EV", "N/A")

st.divider()

# Cumulative P&L chart
st.subheader("Cumulative P&L Over Time")
if not positions_df.empty and "pnl" in positions_df.columns:
    closed = positions_df[positions_df["status"] == "closed"].copy()
    if not closed.empty:
        closed = closed.sort_values("exit_time")
        closed["cumulative_pnl"] = closed["pnl"].cumsum()
        fig = px.line(
            closed,
            x="exit_time",
            y="cumulative_pnl",
            title="Cumulative P&L (Closed Positions)",
            labels={"exit_time": "Date", "cumulative_pnl": "Cumulative P&L ($)"},
        )
        fig.update_traces(line_color="#00CC96")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed positions yet")
else:
    st.info("No position data available")

# Model calibration chart
st.subheader("Model Calibration")
st.markdown("*Bars show average actual outcome for each model probability bucket. Perfect calibration = bars match the diagonal.*")

if not signals_df.empty and "model_p" in signals_df.columns:
    # Bin model predictions
    executed = signals_df[signals_df["executed"] == 1].copy()
    if not executed.empty and "outcome" in executed.columns:
        executed["p_bucket"] = pd.cut(executed["model_p"], bins=10, labels=False)
        calibration = executed.groupby("p_bucket").agg(
            avg_model_p=("model_p", "mean"),
            avg_outcome=("outcome", "mean"),
            count=("outcome", "count"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=calibration["avg_model_p"],
            y=calibration["avg_outcome"],
            name="Actual Frequency",
            marker_color="steelblue",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Perfect Calibration",
            line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(xaxis_title="Model Probability", yaxis_title="Actual Outcome Rate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough resolved trades for calibration chart")

# Recent signals table
st.subheader("Recent Signals")
if not signals_df.empty:
    display_cols = ["timestamp", "question", "recommended_side", "model_p", "market_p", "ev", "kelly_size", "executed"]
    available = [c for c in display_cols if c in signals_df.columns]
    st.dataframe(
        signals_df[available].head(20),
        use_container_width=True,
        column_config={
            "ev": st.column_config.NumberColumn("EV", format="%.3f"),
            "model_p": st.column_config.NumberColumn("Model P", format="%.3f"),
            "market_p": st.column_config.NumberColumn("Market P", format="%.3f"),
        }
    )

# Open positions
st.subheader("Open Positions")
if not positions_df.empty:
    open_pos = positions_df[positions_df["status"] == "open"]
    if not open_pos.empty:
        st.dataframe(open_pos, use_container_width=True)
    else:
        st.info("No open positions")
else:
    st.info("No position data")

# Auto-refresh
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Data source: {DB_PATH} | Auto-refreshes every 60 seconds")