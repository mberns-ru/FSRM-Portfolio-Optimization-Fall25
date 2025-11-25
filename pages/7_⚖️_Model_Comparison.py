import glob
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

RESULTS_DIR = "results"

st.set_page_config(
    page_title="⚖️ Model Comparison – GBM vs RF vs PCA+LGBM vs CatBoost",
    layout="wide",
)

st.title("⚖️ Model Comparison – GBM vs RF vs PCA+LGBM vs CatBoost")


def load_latest(prefix: str):
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_path = max(files, key=os.path.getmtime)
    with open(latest_path, "rb") as f:
        results = pickle.load(f)
    return latest_path, results


def two_month_ret(returns: pd.Series) -> float:
    """Cumulative return over the period from a series of periodic returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (1.0 + returns).prod() - 1.0


def compute_bimonthly_from_monthlies(series: pd.Series) -> pd.DataFrame:
    """
    Take a monthly-return series with a DatetimeIndex and return a DataFrame
    with bi-monthly cumulative returns and a representative Date (end of the
    second month in each 2-month block).

    We group months as:
        Jan+Feb, Mar+Apr, May+Jun, Jul+Aug, Sep+Oct, Nov+Dec
    for each year.
    """
    s = series.dropna().copy()
    if s.empty:
        return pd.DataFrame(columns=["Date", "Return"])

    periods = s.index.to_period("M")
    years = periods.year
    # bi = 1..6 within each year: (Jan+Feb)=1, (Mar+Apr)=2, ...
    bi = ((periods.month - 1) // 2) + 1

    grouped = s.groupby([years, bi]).apply(two_month_ret)

    # Build label date = end of the second month in each 2-month block
    idx_years = grouped.index.get_level_values(0)
    idx_bi = grouped.index.get_level_values(1)
    months = idx_bi * 2  # 1→2(Feb), 2→4(Apr), ..., 6→12(Dec)

    dates = pd.to_datetime(
        {"year": idx_years, "month": months, "day": 1}
    ) + pd.offsets.MonthEnd(0)

    out = pd.DataFrame(
        {"Date": dates, "Return": grouped.values}
    ).sort_values("Date")
    return out


# ---- Load each model's latest results ----
gbm_loaded = load_latest("gradientboost_results")
rf_loaded = load_latest("randomforest_results")
pl_loaded = load_latest("pca_lightgbm_results")
cb_loaded = load_latest("catboost_results")

status_cols = st.columns(4)
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

with status_cols[3]:
    if cb_loaded is None:
        st.error("No CatBoost results found in `results/`.")
    else:
        path, _ = cb_loaded
        st.success(f"CatBoost: `{os.path.basename(path)}`")

# stop if any missing
if any(x is None for x in [gbm_loaded, rf_loaded, pl_loaded, cb_loaded]):
    st.info("Run all four models (or upload their results) before using this page.")
    st.stop()

# unpack
gbm_path, gbm_res = gbm_loaded
rf_path, rf_res = rf_loaded
pl_path, pl_res = pl_loaded
cb_path, cb_res = cb_loaded


# ----------------------------------------------------------
# 1. Bi-monthly Returns (2024–2025) – Models + S&P 500
# ----------------------------------------------------------
st.markdown("## Bi-monthly Returns (2024–2025)")

returns_long_list: list[pd.DataFrame] = []

# ---- ML models' bi-monthly returns (from monthly_test["ML_Opt"]) ----
for label, res in [
    ("GBM", gbm_res),
    ("Random Forest", rf_res),
    ("PCA+LGBM", pl_res),
    ("CatBoost", cb_res),
]:
    monthly_test: pd.DataFrame = res["monthly_test"]
    # clamp to 2024–2025
    test_2024 = monthly_test.loc["2024-01-01":"2025-12-31"]

    if "ML_Opt" not in test_2024.columns:
        continue

    df_model = compute_bimonthly_from_monthlies(test_2024["ML_Opt"])
    if df_model.empty:
        continue
    df_model["Model"] = label
    returns_long_list.append(df_model)

# ---- Add S&P 500 (SPY) bi-monthly returns from equity_test ----
# Use the first results dict that has equity_test with a SPY column
for res in [gbm_res, rf_res, pl_res, cb_res]:
    eq = res.get("equity_test", None)
    if eq is None:
        continue

    spy_cols = [c for c in eq.columns if "SPY" in c.upper()]
    if not spy_cols:
        continue

    spy_eq = eq[spy_cols[0]].loc["2024-01-01":"2025-12-31"]
    if spy_eq.empty:
        continue

    spy_ret_m = spy_eq.pct_change().dropna()
    df_spy = compute_bimonthly_from_monthlies(spy_ret_m)
    if df_spy.empty:
        continue
    df_spy["Model"] = "S&P 500 (SPY)"
    returns_long_list.append(df_spy)
    break  # only need to get SPY once

# ---- Plot everything ----
if returns_long_list:
    returns_2m_all = pd.concat(returns_long_list, ignore_index=True)

    base = alt.Chart(returns_2m_all).encode(
        x=alt.X("Date:T", title="Period start"),
        y=alt.Y(
            "Return:Q",
            title="2-month return",
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color("Model:N", title="Model"),
    )

    chart_2m = (
        base.mark_line(point=True)
        .encode(
            tooltip=[
                alt.Tooltip("Date:T", title="Period start"),
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Return:Q", title="2-month return", format=".2%"),
            ]
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(chart_2m, use_container_width=True)
else:
    st.info("No ML_Opt or SPY returns found for the selected models.")


# ----------------------------------------------------------
# 2. 2024–2025 Equity Curves – All Models vs SPY
# ----------------------------------------------------------
st.markdown("## Equity Curves (2024–2025)")


def get_equity_df(label: str, res: dict) -> pd.DataFrame | None:
    # Prefer new 'equity_test'
    if "equity_test" in res:
        df = res["equity_test"].copy()
    elif "equity_2025" in res:
        df = res["equity_2025"].copy()
    elif "equity_compare_2025" in res:
        df = res["equity_compare_2025"].copy()
    else:
        return None

    ml_cols = [c for c in df.columns if "ML" in c or "Opt" in c]
    if ml_cols:
        ml_col = ml_cols[0]
    else:
        # fallback to any non-SPY column
        ml_col_candidates = [c for c in df.columns if "SPY" not in c]
        if not ml_col_candidates:
            return None
        ml_col = ml_col_candidates[0]

    new_cols = {ml_col: f"{label}_ML"}
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
    ("CatBoost", cb_res),
]:
    df_eq = get_equity_df(label, res)
    if df_eq is not None:
        eq_pieces.append(df_eq)

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
            color=alt.Color("Series:N", title="Model"),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Equity:Q", format=",.0f")],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart_eq, use_container_width=True)


# ----------------------------------------------------------
# 3. Test-period metrics comparison (2024–2025)
# ----------------------------------------------------------
st.markdown("## Model Testing Metrics")

rows = []
for label, res in [
    ("GBM", gbm_res),
    ("Random Forest", rf_res),
    ("PCA+LGBM", pl_res),
    ("CatBoost", cb_res),
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
- **total_return**: cumulative return  
- **ann_return**: annualized return  
- **ann_vol**: annualized volatility  
- **sharpe**: annualized Sharpe ratio  
- **max_drawdown**: worst peak-to-trough loss  
"""
    )
else:
    st.info("No test-period ML_Opt metrics found for comparison.")
