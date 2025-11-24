import glob
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import yfinance as yf

RESULTS_DIR = "results"

st.set_page_config(
    page_title="CatBoost Portfolio – Results",
    layout="wide",
)

st.title("🐈 CatBoost Portfolio – Backtest Results")


@st.cache_data(show_spinner=False)
def load_results_from_bytes(uploaded_file) -> dict:
    return pickle.load(uploaded_file)


@st.cache_data(show_spinner=False)
def load_results_from_path(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_latest_catboost_results():
    # Look for files like: catboost_results_YYYYMMDD_HHMMSS.pkl
    if not os.path.isdir(RESULTS_DIR):
        return None
    candidates = [
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("catboost_results_") and f.endswith(".pkl")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    rows = []
    for split, models in metrics_dict.items():
        for model_name, vals in models.items():
            row = {"split": split, "model": model_name}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def two_month_ret(df: pd.DataFrame) -> pd.Series:
    return (1.0 + df).prod() - 1.0


# ====== File upload OR auto-load latest ======
st.sidebar.header("Load Saved CatBoost Results")

uploaded = st.sidebar.file_uploader(
    "Upload results pickle (.pkl) from _catboost.py",
    type=["pkl"],
)

latest_path = None
if uploaded is None:
    latest_path = find_latest_catboost_results()

if uploaded is not None:
    results = load_results_from_bytes(uploaded)
    st.sidebar.success(f"Using uploaded file: `{uploaded.name}`")
elif latest_path is not None:
    results = load_results_from_path(latest_path)
    st.sidebar.success(
        f"Auto-loaded latest CatBoost results: `{os.path.basename(latest_path)}`"
    )
else:
    st.info(
        "👈 Upload a CatBoost `.pkl` file created by `_catboost.py`, "
        "or run a backtest to create one in `results/`."
    )
    st.stop()

# ---------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------
prices = results.get("prices")
monthly = results["monthly"]
weights = results["weights"]
monthly_train = results["monthly_train"]
monthly_test = results["monthly_test"]
metrics = results["metrics"]

tickers = results.get(
    "tickers",
    list(prices.columns) if prices is not None else list(weights.columns),
)
start = results.get(
    "start",
    str(prices.index[0].date()) if prices is not None else "",
)
end = results.get(
    "end",
    str(prices.index[-1].date()) if prices is not None else "",
)

model_name = results.get("model_name", "CatBoost")

train_start = results.get(
    "train_start",
    str(monthly_train.index.min().date()),
)
train_end = results.get(
    "train_end",
    str(monthly_train.index.max().date()),
)
test_start = results.get(
    "test_start",
    str(monthly_test.index.min().date()),
)
test_end = results.get(
    "test_end",
    str(monthly_test.index.max().date()),
)

initial_investment = float(results.get("initial_investment", 1000.0))

# For weights / trades
weights_test = weights.loc[test_start:test_end].sort_index()

# ====== Top-level info ======
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Backtest Information")
    st.markdown(f"- **Tickers**: {', '.join(tickers)}")
    st.markdown(f"- **Price sample**: `{start}` → `{end}`")
    st.markdown(f"- **Train window**: `{train_start}` → `{train_end}`")
    st.markdown(f"- **Test window**: `{test_start}` → `{test_end}`")

with col_info2:
    st.subheader("Model Implementation")
    st.markdown(f"- **Model**: `{model_name}`")
    st.markdown("- **Rebalance frequency**: as set in training script")
    st.markdown("- **Objective**: Mean-variance optimization on predicted returns")
    st.markdown(
        f"- **Initial test capital**: `${int(initial_investment):,}`"
    )

# ====== 1. Equity curves: ML vs SPY in 2024–2025 ======
st.markdown(
    f"## 1. 2024–2025 Equity Curves – ML Strategy vs SPY "
    f"(initial = ${int(initial_investment):,})"
)

eq_df = pd.DataFrame(index=monthly_test.index)

# ML equity from monthly_test["ML_Opt"]
if "ML_Opt" in monthly_test.columns:
    ml_equity = initial_investment * (1.0 + monthly_test["ML_Opt"]).cumprod()
    eq_df[f"ML_Opt_${int(initial_investment)}"] = ml_equity

# SPY benchmark
spy_series = None
spy_cols = [c for c in monthly_test.columns if "SPY" in c.upper()]
if spy_cols:
    spy_ret = monthly_test[spy_cols[0]]
    spy_series = initial_investment * (1.0 + spy_ret).cumprod()
else:
    spy_px = yf.download(
        "SPY",
        start=test_start,
        end=(pd.to_datetime(test_end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"].dropna()
    if not spy_px.empty:
        spy_equity_daily = initial_investment * spy_px / spy_px.iloc[0]
        spy_series = spy_equity_daily.resample("M").last()
        spy_series = spy_series.reindex(eq_df.index).ffill()

if spy_series is not None:
    eq_df[f"SPY_${int(initial_investment)}"] = spy_series

eq_df = eq_df.dropna(how="all")

eq_long = (
    eq_df.reset_index()
    .rename(columns={"index": "Date"})
    .melt("Date", var_name="Series", value_name="Equity")
)

chart_equity = (
    alt.Chart(eq_long)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(
            "Equity:Q",
            title="Portfolio value ($)",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("Series:N", title="Series"),
        tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=".2f")],
    )
    .properties(height=350)
    .interactive()
)
st.altair_chart(chart_equity, use_container_width=True)

# ====== 2. Portfolio Weights – Testing Window (bi-monthly) ======
st.markdown("## 2. Portfolio Weights – Bi-monthly Snapshots (Test Window)")

weights_test_bimonth = weights_test.iloc[::2]
weights_test_bimonth.index.name = "Date"

st.markdown("### 2.1 Weight Table (Bi-monthly, Test Window)")
st.dataframe(
    weights_test_bimonth.style.format("{:.2%}"),
    use_container_width=True,
)

st.markdown("### 2.2 Weight Composition Chart (Bi-monthly)")

weights_long = (
    weights_test_bimonth
    .reset_index()
    .melt("Date", var_name="Ticker", value_name="Weight")
)

chart_weights = (
    alt.Chart(weights_long)
    .mark_area()
    .encode(
        x=alt.X("Date:T", title="Rebalance Date"),
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

# ====== 3. Trade Log – reconstructed from weights ======
st.markdown("## 3. Trade Log (Monthly Selections)")

st.caption(
    "Each row shows the active ML portfolio at that rebalance date "
    "and which tickers were bought, sold, or held."
)

rows = []
prev_nonzero = set()

for dt, row in weights_test.iterrows():
    current_nonzero = set(row[row > 0].index.tolist())
    chosen = sorted(current_nonzero)
    buys = sorted(current_nonzero - prev_nonzero)
    sells = sorted(prev_nonzero - current_nonzero)
    holds = sorted(current_nonzero & prev_nonzero)
    rows.append(
        {
            "date": dt,
            "chosen_stocks": chosen,
            "buys": buys,
            "sells": sells,
            "holds": holds,
        }
    )
    prev_nonzero = current_nonzero

trade_log = pd.DataFrame(rows)
trade_log["date"] = pd.to_datetime(trade_log["date"])

max_rows = st.slider(
    "Rows to display",
    min_value=10,
    max_value=len(trade_log),
    value=min(50, len(trade_log)),
)
st.dataframe(
    trade_log.tail(max_rows),
    use_container_width=True,
)

# ====== 4. Test Returns every 2 Months (2024–2025) ======
st.markdown("## 4. Test Returns – Every 2 Months (2024–2025)")

test_2024 = monthly_test.loc["2024-01-01":]

cols_for_bar = [c for c in ["ML_Opt", "EqualWeight"] if c in test_2024.columns]
returns_2m = (
    test_2024[cols_for_bar]
    .resample("2M")
    .apply(two_month_ret)
    .dropna(how="all")
)

returns_2m_long = (
    returns_2m.reset_index()
    .melt("Date", var_name="Strategy", value_name="Return")
)

chart_ret_2m = (
    alt.Chart(returns_2m_long)
    .mark_bar()
    .encode(
        x=alt.X("Date:T", title="Period Start"),
        y=alt.Y(
            "Return:Q",
            title="2-month return",
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color("Strategy:N", title="Strategy"),
        tooltip=[
            "Date:T",
            "Strategy:N",
            alt.Tooltip("Return:Q", format=".2%"),
        ],
    )
    .properties(height=350)
    .interactive()
)
st.altair_chart(chart_ret_2m, use_container_width=True)

st.dataframe(
    returns_2m.style.format("{:.2%}"),
    use_container_width=True,
)

# ====== 5. Performance metrics ======
st.markdown("## 5. Performance Metrics")

metrics_df = format_metrics_table(metrics)
metrics_display = metrics_df.copy()

for col in ["total_return", "ann_return", "ann_vol", "max_drawdown"]:
    if col in metrics_display.columns:
        metrics_display[col] = metrics_display[col].map(
            lambda x: f"{x:.2%}" if pd.notnull(x) else ""
        )
if "sharpe" in metrics_display.columns:
    metrics_display["sharpe"] = metrics_display["sharpe"].map(
        lambda x: f"{x:.2f}" if pd.notnull(x) else ""
    )

st.dataframe(metrics_display, use_container_width=True)

st.markdown(
    """
- **total_return**: cumulative return over the period.  
- **ann_return**: annualized return.  
- **ann_vol**: annualized volatility.  
- **sharpe**: annualized Sharpe ratio.  
- **max_drawdown**: maximum drawdown from peak to trough.
"""
)
