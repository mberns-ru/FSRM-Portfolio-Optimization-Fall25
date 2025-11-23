import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


st.set_page_config(
    page_title="Gradient Boost Portfolio – Results",
    layout="wide",
)


st.title("📈 Gradient Boost Portfolio – Backtest Results")


@st.cache_data(show_spinner=False)
def load_results_from_bytes(uploaded_file) -> dict:
    """
    Load pickled results from an uploaded file.
    """
    return pickle.load(uploaded_file)


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Convert nested metrics dict {"train": {"ML_Opt": {...}, ...}, "test": {...}}
    into a tidy DataFrame.
    """
    rows = []
    for split, models in metrics_dict.items():
        for model_name, vals in models.items():
            row = {"split": split, "model": model_name}
            row.update(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Pretty formatting: percentages instead of raw decimals for returns
    for col in ["total_return", "ann_return", "ann_vol", "max_drawdown"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "sharpe" in df.columns:
        df["sharpe"] = df["sharpe"].astype(float)
    return df


# ====== File upload ======

st.sidebar.header("Load Saved Results")

uploaded = st.sidebar.file_uploader(
    "Upload results pickle (.pkl) from gradientboost_train.py",
    type=["pkl"],
)

if uploaded is None:
    st.info("👈 Upload a results `.pkl` file created by `gradientboost_train.py` to see the charts.")
    st.stop()

results = load_results_from_bytes(uploaded)

monthly = results["monthly"]
weights = results["weights"]
monthly_train = results["monthly_train"]
monthly_test = results["monthly_test"]
weights_2025_bimonth = results["weights_2025_bimonth"]
equity_2025 = results["equity_2025"]
metrics = results["metrics"]
tickers = results.get("tickers", list(weights.columns))


# ====== Top-level info ======

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.subheader("Backtest Information")
    st.markdown(f"- **Tickers**: {', '.join(tickers)}")
    st.markdown(f"- **Price sample**: {results['start']} → {results['end']}")
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
    st.markdown(f"- **Uses XGBoost (GPU)**: `{results.get('use_xgb', False)}`")
    st.markdown("- **Rebalance frequency**: Monthly")
    st.markdown("- **Objective**: Mean-variance optimization on predicted returns")


# ====== 1. Equity curves: ML vs S&P 500 in 2025 ======

st.markdown("## 1. 2025 Equity Curves – ML Strategy vs S&P 500 ($1000 start)")

eq_df = equity_2025.reset_index().rename(columns={"index": "Date"})
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


# ====== 2. Weights every 2 months in 2025 ======

st.markdown("## 2. Portfolio Weights (Every 2 Months in 2025)")

# Table of weights (every 2 months)
st.markdown("### 2.1 Weight Table (Bi-monthly 2025)")

weights_table = weights_2025_bimonth.copy()
weights_table.index = weights_table.index.strftime("%Y-%m-%d")
st.dataframe(
    weights_table.style.format("{:.2%}"),
    use_container_width=True,
)

# Stacked area chart of weights over time (bi-monthly sampling)
st.markdown("### 2.2 Weight Composition Chart")

weights_long = (
    weights_2025_bimonth
    .reset_index()
    .rename(columns={"Date": "Date"})
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


# ====== 3. Training / Testing Metrics ======

st.markdown("## 3. Performance Metrics")

metrics_df = format_metrics_table(metrics)

# Show a nicer formatted table
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
- **sharpe**: annualized Sharpe ratio (using the same risk-free rate as in training).  
- **max_drawdown**: maximum drawdown from peak to trough.
"""
)
