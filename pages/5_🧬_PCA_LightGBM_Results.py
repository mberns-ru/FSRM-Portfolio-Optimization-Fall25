import glob
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

RESULTS_DIR = "results"

st.set_page_config(
    page_title="PCA + LightGBM Portfolio – Results",
    layout="wide",
)

st.title("🧬 PCA + LightGBM Portfolio – Backtest Results (Masuda-inspired)")


@st.cache_data(show_spinner=False)
def load_results_from_bytes(uploaded_file) -> dict:
    return pickle.load(uploaded_file)


@st.cache_data(show_spinner=False)
def load_results_from_path(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_latest_pl_results():
    pattern = os.path.join(RESULTS_DIR, "pca_lightgbm_results_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    rows = []
    for split, models in metrics_dict.items():
        for model_name, vals in models.items():
            row = {"split": split, "model": model_name}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


# ====== File upload OR auto-load latest ======
st.sidebar.header("Load Saved PCA+LightGBM Results")

uploaded = st.sidebar.file_uploader(
    "Upload results pickle (.pkl) from _pca_lightgbm.py",
    type=["pkl"],
)

latest_path = None
if uploaded is None:
    latest_path = find_latest_pl_results()

if uploaded is not None:
    results = load_results_from_bytes(uploaded)
    st.sidebar.success(f"Using uploaded file: `{uploaded.name}`")
elif latest_path is not None:
    results = load_results_from_path(latest_path)
    st.sidebar.success(
        f"Auto-loaded latest PCA+LGBM results: `{os.path.basename(latest_path)}`"
    )
else:
    st.info(
        "👈 Upload a `.pkl` file created by `_pca_lightgbm.py`, "
        "or run a backtest to create one in `results/`."
    )
    st.stop()

monthly = results["monthly"]
weights = results["weights"]
monthly_train = results["monthly_train"]
monthly_test = results["monthly_test"]
weights_2025_bimonth = results["weights_2025_bimonth"]
equity_compare_2025 = results["equity_compare_2025"]
metrics = results["metrics"]
use_lightgbm = results.get("use_lightgbm", False)
prices = results.get("prices", None)
tickers = results.get("tickers", list(weights.columns))
start = results.get("start", "")
end = results.get("end", "")

# ====== Top-level info ======
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Backtest Information")
    st.markdown(f"- **Tickers**: {', '.join(map(str, tickers))}")
    st.markdown(f"- **Price sample**: `{start}` → `{end}`")
    st.markdown(
        f"- **Train window**: {monthly_train.index.min().date()} → "
        f"{monthly_train.index.max().date()}"
    )
    st.markdown(
        f"- **Test window**: {monthly_test.index.min().date()} → "
        f"{monthly_test.index.max().date()}"
    )

with col_info2:
    st.subheader("Model Implementation")
    st.markdown(f"- **Uses LightGBM**: `{use_lightgbm}`")
    st.markdown("- **Dimensionality reduction**: PCA on standardized features")
    st.markdown("- **Rebalance frequency**: Monthly")
    st.markdown("- **Portfolio construction**: Mean-variance + cluster penalties")


# ====== 1. Equity Curves – PCA+LGBM vs SPY (2025, $1000) ======
st.markdown("## 1. Equity Curves – PCA+LGBM vs SPY (2025, $1000)")

eq_df = equity_compare_2025.copy()
eq_df = eq_df.rename(
    columns={
        "ML_Opt_$1000": "PCA+LGBM Strategy",
        "SPY_$1000": "SPY",
    }
)

eq_long = (
    eq_df.reset_index()
    .rename(columns={"index": "Date"})
    .melt("Date", var_name="Series", value_name="Equity")
)

chart_eq = (
    alt.Chart(eq_long)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Equity:Q", title="Portfolio Value ($)"),
        color=alt.Color("Series:N", title="Series"),
        tooltip=[
            "Date:T",
            "Series:N",
            alt.Tooltip("Equity:Q", format=",.0f"),
        ],
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(chart_eq, use_container_width=True)

# ====== 2. Portfolio Weights – Bi-monthly Snapshots (2025) ======
st.markdown("## 2. Portfolio Weights – Bi-monthly Snapshots (2025)")

weights_test_bimonth = weights_2025_bimonth.copy()
weights_test_bimonth.index.name = "Date"

weights_table = weights_test_bimonth.copy()
weights_table.index = weights_table.index.strftime("%Y-%m-%d")

st.markdown("### 2.1 Weight Table (Bi-monthly, Test Window)")
st.dataframe(
    weights_table.style.format("{:.2%}"),
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


# ====== 3. Trade Log – RF-style (Monthly Selections) ======
st.markdown("## 3. Trade Log (Monthly Selections)")

st.caption(
    "Approximate trade log reconstructed from monthly weights: "
    "which assets are in the portfolio, and which are bought/sold at each rebalance."
)

weights_test = weights.loc["2024-01-01":"2025-12-31"].sort_index()

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

def two_month_ret(df_slice: pd.DataFrame) -> pd.Series:
    return (1.0 + df_slice).prod() - 1.0

returns_2m = (
    test_2024[["ML_Opt", "EqualWeight"]]
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
        y=alt.Y("Return:Q", title="2-month return"),
        color=alt.Color("Strategy:N", title="Strategy"),
        tooltip=[
            "Date:T",
            "Strategy:N",
            alt.Tooltip("Return:Q", format=".2%"),
        ],
    )
    .properties(height=300)
)

st.altair_chart(chart_ret_2m, use_container_width=True)

returns_2m_table = returns_2m.copy()
returns_2m_table.index = returns_2m_table.index.strftime("%Y-%m-%d")
st.dataframe(
    returns_2m_table.style.format("{:.2%}"),
    use_container_width=True,
)


# ====== 5. Performance Metrics – Train vs Test ======
st.markdown("## 5. Performance Metrics – Train vs Test")

metrics_df = format_metrics_table(metrics)

if not metrics_df.empty:
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
else:
    st.info("No metrics found in the results file.")
