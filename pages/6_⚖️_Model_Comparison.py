import glob
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

RESULTS_DIR = "results"

st.set_page_config(
    page_title="⚖️ Model Comparison – GBM vs RF vs PCA+LGBM",
    layout="wide",
)

st.title("⚖️ Model Comparison – GBM vs RF vs PCA+LGBM")


@st.cache_data(show_spinner=False)
def load_latest(prefix: str):
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_path = max(files, key=os.path.getmtime)
    with open(latest_path, "rb") as f:
        results = pickle.load(f)
    return latest_path, results


def two_month_ret(df):
    return (1.0 + df).prod() - 1.0


# ---- Load each model's latest results ----
gbm_loaded = load_latest("gradientboost_results")
rf_loaded = load_latest("randomforest_results")
pl_loaded = load_latest("pca_lightgbm_results")

status_cols = st.columns(3)
with status_cols[0]:
    if gbm_loaded is None:
        st.error("No GBM results found in `results/`.")
    else:
        path, _ = gbm_loaded
        st.success(f"GBM: `{os.path.basename(path)}`")

with status_cols[1]:
    if rf_loaded is None:
        st.error("No RF results found in `results/`.")
    else:
        path, _ = rf_loaded
        st.success(f"RF: `{os.path.basename(path)}`")

with status_cols[2]:
    if pl_loaded is None:
        st.error("No PCA+LGBM results found in `results/`.")
    else:
        path, _ = pl_loaded
        st.success(f"PCA+LGBM: `{os.path.basename(path)}`")

if any(x is None for x in [gbm_loaded, rf_loaded, pl_loaded]):
    st.info("Run all three models (or upload their results) before using this page.")
    st.stop()

gbm_path, gbm_res = gbm_loaded
rf_path, rf_res = rf_loaded
pl_path, pl_res = pl_loaded

# ----------------------------------------------------------
# 1. Bi-monthly returns (2024–2025) – ML_Opt comparison
# ----------------------------------------------------------
st.markdown("## 1. Bi-monthly Returns (2024–2025) – ML_Opt Strategies")

returns_long_list = []

for label, res in [
    ("GBM", gbm_res),
    ("Random Forest", rf_res),
    ("PCA+LGBM", pl_res),
]:
    monthly_test = res["monthly_test"]
    test_2024 = monthly_test.loc["2024-01-01":"2025-12-31"]

    if "ML_Opt" not in test_2024.columns:
        continue

    r2m = (
        test_2024[["ML_Opt"]]
        .resample("2M")
        .apply(two_month_ret)
        .dropna(how="all")
    )
    tmp = (
        r2m.reset_index()
        .rename(columns={"ML_Opt": "Return"})
    )
    tmp["Model"] = label
    returns_long_list.append(tmp)

if returns_long_list:
    returns_2m_all = pd.concat(returns_long_list, ignore_index=True)

    chart_2m = (
        alt.Chart(returns_2m_all)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title="Period Start"),
            y=alt.Y("Return:Q", title="2-month return", axis=alt.Axis(format="%")),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                "Date:T",
                "Model:N",
                alt.Tooltip("Return:Q", format=".2%"),
            ],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_2m, use_container_width=True)

    # Table view
    table_2m = returns_2m_all.copy()
    table_2m["Return"] = table_2m["Return"].map(lambda x: f"{x:.2%}")
    st.dataframe(table_2m, use_container_width=True)
else:
    st.info("No ML_Opt returns found for the models.")


# ----------------------------------------------------------
# 2. 2024–2025 Equity Curves – All Models vs SPY
# ----------------------------------------------------------
st.markdown("## 2. 2024–2025 Equity Curves – All Models vs SPY")

def get_equity_df(label, res):
    # Prefer new 'equity_test'
    if "equity_test" in res:
        df = res["equity_test"].copy()
    elif "equity_2025" in res:
        df = res["equity_2025"].copy()
    elif "equity_compare_2025" in res:
        df = res["equity_compare_2025"].copy()
    else:
        return None

    # Try to rename the ML series to something standard
    ml_cols = [c for c in df.columns if "ML" in c or "Opt" in c]
    if ml_cols:
        ml_col = ml_cols[0]
    else:
        # fall back to any non-SPY column
        ml_col_candidates = [c for c in df.columns if "SPY" not in c]
        if not ml_col_candidates:
            return None
        ml_col = ml_col_candidates[0]

    new_cols = {}
    new_cols[ml_col] = f"{label}_ML"
    for c in df.columns:
        if "SPY" in c:
            new_cols[c] = "SPY"
    df = df.rename(columns=new_cols)
    return df[["SPY", f"{label}_ML"]] if "SPY" in df.columns else df[[f"{label}_ML"]]

eq_pieces = []
for label, res in [
    ("GBM", gbm_res),
    ("RF", rf_res),
    ("PCA+LGBM", pl_res),
]:
    df = get_equity_df(label, res)
    if df is not None:
        eq_pieces.append(df)

if not eq_pieces:
    st.info("Could not assemble any equity curves for comparison.")
else:
    eq_merged = pd.concat(eq_pieces, axis=1)
    eq_merged = eq_merged.loc[:, ~eq_merged.columns.duplicated()]

    eq_long = (
        eq_merged.reset_index()
        .rename(columns={"index": "Date"})
        .melt("Date", var_name="Series", value_name="Equity")
    )

    chart_eq = (
        alt.Chart(eq_long)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[
                "Date:T",
                "Series:N",
                alt.Tooltip("Equity:Q", format=",.0f"),
            ],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_eq, use_container_width=True)


# ----------------------------------------------------------
# 3. Test-period metrics comparison (2024–2025)
# ----------------------------------------------------------
st.markdown("## 3. Test-period Metrics (2024–2025) – ML_Opt")

rows = []
for label, res in [
    ("GBM", gbm_res),
    ("Random Forest", rf_res),
    ("PCA+LGBM", pl_res),
]:
    metrics = res.get("metrics", {})
    test_metrics = metrics.get("test", {})
    ml_metrics = test_metrics.get("ML_Opt", None)
    if ml_metrics is None:
        continue
    row = {"Model": label}
    row.update(ml_metrics)
    rows.append(row)

if rows:
    mdf = pd.DataFrame(rows)

    # Pretty formatting
    for col in ["total_return", "ann_return", "ann_vol", "max_drawdown"]:
        if col in mdf.columns:
            mdf[col] = mdf[col].map(lambda x: f"{x:.2%}")
    if "sharpe" in mdf.columns:
        mdf["sharpe"] = mdf["sharpe"].map(lambda x: f"{x:.2f}")

    st.dataframe(mdf, use_container_width=True)
    st.markdown(
        """
- **total_return**: cumulative return over the test window (2024–2025).  
- **ann_return**: annualized return over the test window.  
- **ann_vol**: annualized volatility.  
- **sharpe**: annualized Sharpe ratio (using the risk-free rate you set).  
- **max_drawdown**: worst peak-to-trough loss in the test period.  
"""
    )
else:
    st.info("No test-period ML_Opt metrics found for comparison.")
