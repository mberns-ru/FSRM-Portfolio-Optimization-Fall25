import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Random Forest Portfolio – Results",
    layout="wide",
)

st.title("🌲 Random Forest Portfolio – Backtest Results")


@st.cache_data(show_spinner=False)
def load_results_from_bytes(uploaded_file) -> dict:
    """Load pickled RF results."""
    return pickle.load(uploaded_file)


# ====== File upload ======

st.sidebar.header("Load Saved RF Results")

uploaded = st.sidebar.file_uploader(
    "Upload results pickle (.pkl) from _randomforest.py",
    type=["pkl"],
)

if uploaded is None:
    st.info("👈 Upload a `randomforest_results_*.pkl` file created by `_randomforest.py`.")
    st.stop()

results = load_results_from_bytes(uploaded)

ml_curve = results["ml_curve"]
bench_curve = results["bench_curve"]
trade_log = results["trade_log"]
stats_port = results["stats_port"]
stats_bench = results["stats_bench"]
train_scores = results["train_scores"]
tickers = results.get("tickers", [])
start = results.get("start", "")
end = results.get("end", "")
train_end_date = results.get("train_end_date", "")
backtest_start = results.get("backtest_start", "")
start_value = results.get("start_value", 1000.0)

# ====== Top-level info ======

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data & Backtest")
    st.markdown(f"- **Tickers**: {', '.join(tickers)}")
    st.markdown(f"- **Price sample**: `{start}` → `{end}`")
    st.markdown(f"- **Train end date**: `{train_end_date}`")
    st.markdown(f"- **Backtest start**: `{backtest_start}`")
    st.markdown(f"- **Initial capital**: `${start_value:,.0f}`")

with col2:
    st.subheader("Random Forest Model")
    st.markdown(
        f"- **Train R²** (on test split after {train_end_date}): "
        f"`{train_scores.get('r2', np.nan):.4f}`"
    )
    st.markdown(
        f"- **Test RMSE**: `{train_scores.get('rmse', np.nan):.6f}`"
    )
    st.markdown(
        """
        - **Features**: short/long momentum & volatility  
        - **Signals**: daily, top-5 stocks by predicted next-day return  
        """
    )

# ====== 1. Equity curves RF vs SPY ======

st.markdown("## 1. Equity Curves – RF Strategy vs S&P 500 (SPY)")

eq_df = pd.concat(
    [ml_curve.rename("RF Strategy"), bench_curve.rename("SPY")],
    axis=1,
).reset_index().rename(columns={"index": "Date"})

eq_long = eq_df.melt("Date", var_name="Series", value_name="Equity")

chart_equity = (
    alt.Chart(eq_long)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Equity:Q", title="Portfolio value ($)", scale=alt.Scale(nice=True)),
        color=alt.Color("Series:N", title="Strategy"),
        tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=".2f")],
    )
    .properties(height=350)
    .interactive()
)

st.altair_chart(chart_equity, use_container_width=True)

st.markdown(
    f"**Final RF value**: `${ml_curve.iloc[-1]:,.2f}` &nbsp;&nbsp; "
    f"**Final SPY value**: `${bench_curve.iloc[-1]:,.2f}`"
)

# ====== 2. Performance metrics ======

st.markdown("## 2. Performance Metrics")

metrics_rows = [
    {"portfolio": "RF Strategy", **stats_port},
    {"portfolio": "SPY", **stats_bench},
]
metrics_df = pd.DataFrame(metrics_rows)

metrics_display = metrics_df.copy()
for col in ["Annualized return", "Annualized vol", "Max drawdown"]:
    if col in metrics_display.columns:
        metrics_display[col] = metrics_display[col].map(
            lambda x: f"{x:.2%}" if pd.notnull(x) else ""
        )
if "Sharpe (rf=0)" in metrics_display.columns:
    metrics_display["Sharpe (rf=0)"] = metrics_display["Sharpe (rf=0)"].map(
        lambda x: f"{x:.2f}" if pd.notnull(x) else ""
    )

st.dataframe(metrics_display, use_container_width=True)

st.markdown(
    """
    - **Annualized return / vol** computed from daily log-returns.  
    - **Sharpe (rf=0)** uses a zero risk-free rate for simplicity.  
    - **Max drawdown** is the worst peak-to-trough loss over the backtest.
    """
)

# ====== 3. Trade log ======

st.markdown("## 3. Trade Log (Daily Selections)")

st.caption(
    "Each row shows the ML-selected portfolio for that day along with which "
    "stocks were bought, sold, or held relative to the previous day."
)

# Optional: show only the last N rows by default
max_rows = st.slider("Rows to display", min_value=50, max_value=len(trade_log), value=min(200, len(trade_log)))
st.dataframe(
    trade_log.tail(max_rows),
    use_container_width=True,
)
