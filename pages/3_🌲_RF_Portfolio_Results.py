import glob
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

RESULTS_DIR = "results"

st.set_page_config(
    page_title="Random Forest Portfolio – Results",
    layout="wide",
)

st.title("🌲 Random Forest Portfolio – Backtest Results (2024–2025)")


@st.cache_data(show_spinner=False)
def load_results_from_bytes(uploaded_file) -> dict:
    return pickle.load(uploaded_file)


@st.cache_data(show_spinner=False)
def load_results_from_path(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_latest_rf_results():
    pattern = os.path.join(RESULTS_DIR, "randomforest_results_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ====== File upload OR auto-load latest ======
st.sidebar.header("Load Saved RF Results")

uploaded = st.sidebar.file_uploader(
    "Upload results pickle (.pkl) from _randomforest.py",
    type=["pkl"],
)

latest_path = None
if uploaded is None:
    latest_path = find_latest_rf_results()

if uploaded is not None:
    results = load_results_from_bytes(uploaded)
    st.sidebar.success(f"Using uploaded file: `{uploaded.name}`")
elif latest_path is not None:
    results = load_results_from_path(latest_path)
    st.sidebar.success(
        f"Auto-loaded latest RF results: `{os.path.basename(latest_path)}`"
    )
else:
    st.info(
        "👈 Upload a `randomforest_results_*.pkl` file, "
        "or run a backtest to create one in `results/`."
    )
    st.stop()

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
    st.markdown(f"- **Backtest start**: `{backtest_start}` (RF test window)")
    st.markdown(f"- **Initial capital**: `${start_value:,.0f}`")

with col2:
    st.subheader("Random Forest Model")
    st.markdown(
        f"- **Test R²** (post {train_end_date}): "
        f"`{train_scores.get('r2', np.nan):.4f}`"
    )
    st.markdown(
        f"- **Test RMSE**: `{train_scores.get('rmse', np.nan):.6f}`"
    )
    st.markdown(
        """
        - **Features**: short/long momentum & volatility  
        - **Signals**: daily, top-5 stocks by predicted next-day return  
        - **Backtest window**: 2024–2025  
        """
    )

# ====== 1. Equity curves RF vs SPY ======
st.markdown("## 1. Equity Curves – RF Strategy vs SPY")

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

# ====== 2. Portfolio Weights – Testing Window (2024–current) ======
st.markdown("## 2. Portfolio Weights – Testing Window (2024–current)")

tl = trade_log.copy()
tl["date"] = pd.to_datetime(tl["date"])
tl = tl.set_index("date").sort_index()
tl = tl.loc["2024-01-01":]

all_dates = tl.index
all_tickers = sorted({tk for lst in tl["chosen_stocks"] for tk in lst})
weights_df = pd.DataFrame(0.0, index=all_dates, columns=all_tickers)

for dt, row in tl.iterrows():
    chosen = row["chosen_stocks"]
    if not chosen:
        continue
    w = 1.0 / len(chosen)
    weights_df.loc[dt, chosen] = w

weights_2m = (
    weights_df
    .resample("2M")
    .last()
    .dropna(how="all")
)

weights_2m_table = weights_2m.copy()
weights_2m_table.index = weights_2m_table.index.strftime("%Y-%m-%d")

st.markdown("### 2.1 Weight Table (Bi-monthly, RF Test Window)")
st.dataframe(
    weights_2m_table.style.format("{:.2%}"),
    use_container_width=True,
)

st.markdown("### 2.2 Weight Composition Chart (Bi-monthly)")

weights_long = (
    weights_2m
    .reset_index()
    .melt("date", var_name="Ticker", value_name="Weight")
    .rename(columns={"date": "Date"})
)

chart_weights = (
    alt.Chart(weights_long)
    .mark_area()
    .encode(
        x=alt.X("Date:T", title="Signal Date"),
        y=alt.Y("Weight:Q", stack="normalize", title="Weight"),
        color=alt.Color("Ticker:N", title="Ticker"),
        tooltip=[
            "Date:T",
            "Ticker:N",
            alt.Tooltip("Weight:Q", format=".2%"),
        ],
    )
    .properties(height=350)
    .interactive()
)
st.altair_chart(chart_weights, use_container_width=True)

# ====== 3. Test Returns every 2 Months (2024–current) ======
st.markdown("## 3. Test Returns – Every 2 Months (2024–current)")

ml_ret = ml_curve.pct_change().dropna()
spy_ret = bench_curve.pct_change().dropna()

ml_ret_24 = ml_ret.loc["2024-01-01":]
spy_ret_24 = spy_ret.loc["2024-01-01":]

def two_month_ret(series):
    return (1.0 + series).prod() - 1.0

rf_2m = ml_ret_24.resample("2M").apply(two_month_ret)
spy_2m = spy_ret_24.resample("2M").apply(two_month_ret)

ret_2m_df = pd.concat(
    [rf_2m.rename("RF Strategy"), spy_2m.rename("SPY")],
    axis=1
).dropna(how="all")

ret_2m_long = (
    ret_2m_df
    .reset_index()
    .melt("index", var_name="Series", value_name="Return")
    .rename(columns={"index": "Date"})
)

chart_ret_2m = (
    alt.Chart(ret_2m_long)
    .mark_bar()
    .encode(
        x=alt.X("Date:T", title="Period Start"),
        y=alt.Y("Return:Q", title="2-month return", axis=alt.Axis(format="%")),
        color=alt.Color("Series:N", title="Series"),
        tooltip=[
            "Date:T",
            "Series:N",
            alt.Tooltip("Return:Q", format=".2%"),  # 2 decimal percent
        ],
    )
    .properties(height=350)
    .interactive()
)
st.altair_chart(chart_ret_2m, use_container_width=True)

st.dataframe(
    ret_2m_df.style.format("{:.2%}"),
    use_container_width=True,
)

# ====== 4. Performance metrics ======
st.markdown("## 4. Performance Metrics")

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

# ====== 5. Trade log ======
st.markdown("## 5. Trade Log (Daily Selections)")

st.caption(
    "Each row shows the ML-selected portfolio for that day along with which "
    "stocks were bought, sold, or held relative to the previous day."
)

trade_log_24 = trade_log.copy()
trade_log_24["date"] = pd.to_datetime(trade_log_24["date"])
trade_log_24 = trade_log_24[trade_log_24["date"] >= "2024-01-01"]

max_rows = st.slider(
    "Rows to display",
    min_value=50,
    max_value=len(trade_log_24),
    value=min(200, len(trade_log_24)),
)
st.dataframe(
    trade_log_24.tail(max_rows),
    use_container_width=True,
)
