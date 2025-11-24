import os
from datetime import date

import numpy as np
import streamlit as st

# Try to import the model scripts
try:
    import _gradientboost as gb
except Exception:
    gb = None

try:
    import _randomforest as rf
except Exception:
    rf = None

try:
    import _pca_lightgbm as pl
except Exception:
    pl = None

try:
    import _catboost as cb
except Exception:
    cb = None

st.set_page_config(
    page_title="🧪 Run & Save ML Portfolio Backtests",
    layout="wide",
)

st.title("🧪 Run & Save ML Portfolio Backtests")

st.markdown(
    """
Use this page to **generate new results files** for your four models:

- 📈 Gradient Boost (XGBoost / GBM)  
- 🌲 Random Forest  
- 🧬 PCA + LightGBM  
- 🐈 CatBoost  

On this page you change only the **finance-related knobs**:

- Data range (start / end)
- Train vs test split (years)
- Rebalance frequency (monthly vs quarterly)
- Risk aversion λ
- Transaction costs (bps)
- Risk-free rate
- Initial investment for test-window equity curves

ML hyperparameters stay fixed inside the model scripts.
"""
)


def date_to_str(d):
    return d.strftime("%Y-%m-%d") if d is not None else None


def year_from_str(s, default_year):
    try:
        return int(s[:4])
    except Exception:
        return default_year


# ----------------------------------------------------------
# Common controls helper
# ----------------------------------------------------------
def finance_controls(defaults, prefix: str):
    """
    Build the common finance controls and return a dict of values.
    `defaults` is a dict with keys:
    start, end, train_start, train_end, test_start, test_end,
    lambda_risk, tc_bps, rf_rate, rebal_freq, initial_investment
    """
    with st.form(f"{prefix}_form"):
        col_dates, col_split, col_finance = st.columns(3)

        # --- Data range ---
        with col_dates:
            start_date = st.date_input(
                "Price history start",
                value=defaults["start_date"],
            )
            end_date = st.date_input(
                "Price history end",
                value=defaults["end_date"],
            )

            rebal_freq_label = st.selectbox(
                "Rebalance frequency",
                options=["Monthly", "Quarterly"],
                index=0 if defaults["rebal_freq"] == "M" else 1,
            )
            rebal_freq = "M" if rebal_freq_label == "Monthly" else "Q"

        # --- Train / Test split (by year) ---
        with col_split:
            st.markdown("**Train / Test split (years)**")
            train_start_year = st.number_input(
                "Train start year",
                min_value=1990,
                max_value=2100,
                value=defaults["train_start_year"],
                step=1,
            )
            train_end_year = st.number_input(
                "Train end year",
                min_value=1990,
                max_value=2100,
                value=defaults["train_end_year"],
                step=1,
            )
            test_start_year = st.number_input(
                "Test start year",
                min_value=1990,
                max_value=2100,
                value=defaults["test_start_year"],
                step=1,
            )
            test_end_year = st.number_input(
                "Test end year",
                min_value=1990,
                max_value=2100,
                value=defaults["test_end_year"],
                step=1,
            )

        # --- Finance parameters ---
        with col_finance:
            lambda_risk = st.slider(
                "Risk aversion (λ)",
                min_value=0.5,
                max_value=20.0,
                value=float(defaults["lambda_risk"]),
                step=0.5,
            )
            tc_bps = st.slider(
                "Transaction cost (bps per $ turnover)",
                min_value=0,
                max_value=50,
                value=int(defaults["tc_bps"]),
                step=1,
            )
            rf_rate = st.slider(
                "Risk-free annual rate (for Sharpe)",
                min_value=0.0,
                max_value=0.05,
                value=float(defaults["rf_rate"]),
                step=0.0025,
            )
            initial_investment = st.number_input(
                "Initial investment for test equity curves",
                min_value=100.0,
                max_value=100000.0,
                value=float(defaults["initial_investment"]),
                step=100.0,
            )

        run_button = st.form_submit_button(f"🚀 Run {prefix} backtest")

    # Construct split dates
    train_start = f"{int(train_start_year)}-01-01"
    train_end = f"{int(train_end_year)}-12-31"
    test_start = f"{int(test_start_year)}-01-01"
    test_end = f"{int(test_end_year)}-12-31"

    return {
        "run": run_button,
        "start_date": start_date,
        "end_date": end_date,
        "rebal_freq": rebal_freq,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "lambda_risk": float(lambda_risk),
        "tc_bps": int(tc_bps),
        "rf_rate": float(rf_rate),
        "initial_investment": float(initial_investment),
    }


# ==========================================================
# Model choice
# ==========================================================
model_choice = st.selectbox(
    "Choose which model to run",
    options=[
        "Gradient Boost (XGBoost / GBM)",
        "Random Forest",
        "PCA + LightGBM",
        "CatBoost",
    ],
)

# ==========================================================
# Gradient Boost
# ==========================================================
if model_choice.startswith("Gradient"):

    if gb is None:
        st.error("Could not import `_gradientboost.py`.")
        st.stop()

    st.subheader("📈 Gradient Boost Settings")

    default_start = getattr(gb, "START", "2010-01-01")
    default_end = getattr(gb, "END", None)
    default_train_start = getattr(gb, "TRAIN_START", "2010-01-01")
    default_train_end = getattr(gb, "TRAIN_END", "2023-12-31")
    default_test_start = getattr(gb, "TEST_START", "2024-01-01")
    default_test_end = getattr(gb, "TEST_END", "2025-12-31")

    start_default_date = date.fromisoformat(default_start)
    end_default_date = date.today() if default_end in (None, "") else date.fromisoformat(default_end)

    defaults = {
        "start_date": start_default_date,
        "end_date": end_default_date,
        "rebal_freq": getattr(gb, "REBAL_FREQ", "M"),
        "train_start_year": year_from_str(default_train_start, 2010),
        "train_end_year": year_from_str(default_train_end, 2023),
        "test_start_year": year_from_str(default_test_start, 2024),
        "test_end_year": year_from_str(default_test_end, 2025),
        "lambda_risk": getattr(gb, "LAMBDA_RISK", 5.0),
        "tc_bps": getattr(gb, "TC_BPS", 5),
        "rf_rate": getattr(gb, "RISK_FREE_ANNUAL", 0.015),
        "initial_investment": getattr(gb, "INITIAL_INVESTMENT", 1000.0),
    }

    controls = finance_controls(defaults, prefix="GBM")

    if controls["run"]:
        # Push settings into module
        gb.START = date_to_str(controls["start_date"])
        gb.END = date_to_str(controls["end_date"])
        gb.REBAL_FREQ = controls["rebal_freq"]
        gb.TRAIN_START = controls["train_start"]
        gb.TRAIN_END = controls["train_end"]
        gb.TEST_START = controls["test_start"]
        gb.TEST_END = controls["test_end"]
        gb.LAMBDA_RISK = controls["lambda_risk"]
        gb.TC_BPS = controls["tc_bps"]
        gb.RISK_FREE_ANNUAL = controls["rf_rate"]
        gb.INITIAL_INVESTMENT = controls["initial_investment"]

        progress = st.progress(0, text="Downloading prices…")
        prices = gb.download_prices(gb.TICKERS, gb.START, gb.END)
        progress.progress(5, text="Running backtest…")

        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(pct, text=f"Running GBM backtest… ({int(frac*100)}%)")
                yield x

        gb.tqdm = streamlit_tqdm  # type: ignore[assignment]

        results = gb.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = gb.save_results(results, gb.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved GBM results to: `{out_path}`")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download GBM results",
                data=f,
                file_name=os.path.basename(out_path),
                mime="application/octet-stream",
            )

# ==========================================================
# Random Forest
# ==========================================================
elif model_choice.startswith("Random"):

    if rf is None or gb is None:
        st.error("Could not import `_randomforest.py` / `_gradientboost.py`.")
        st.stop()

    st.subheader("🌲 Random Forest Settings")

    default_start = getattr(rf, "START", getattr(gb, "START", "2010-01-01"))
    default_end = getattr(rf, "END", getattr(gb, "END", None))
    default_train_start = getattr(gb, "TRAIN_START", "2010-01-01")
    default_train_end = getattr(gb, "TRAIN_END", "2023-12-31")
    default_test_start = getattr(gb, "TEST_START", "2024-01-01")
    default_test_end = getattr(gb, "TEST_END", "2025-12-31")

    start_default_date = date.fromisoformat(default_start)
    end_default_date = date.today() if default_end in (None, "") else date.fromisoformat(default_end)

    defaults = {
        "start_date": start_default_date,
        "end_date": end_default_date,
        "rebal_freq": getattr(gb, "REBAL_FREQ", "M"),
        "train_start_year": year_from_str(default_train_start, 2010),
        "train_end_year": year_from_str(default_train_end, 2023),
        "test_start_year": year_from_str(default_test_start, 2024),
        "test_end_year": year_from_str(default_test_end, 2025),
        "lambda_risk": getattr(gb, "LAMBDA_RISK", 5.0),
        "tc_bps": getattr(gb, "TC_BPS", 5),
        "rf_rate": getattr(gb, "RISK_FREE_ANNUAL", 0.015),
        "initial_investment": getattr(gb, "INITIAL_INVESTMENT", 1000.0),
    }

    controls = finance_controls(defaults, prefix="RF")

    if controls["run"]:
        gb.START = rf.START = date_to_str(controls["start_date"])
        gb.END = rf.END = date_to_str(controls["end_date"])
        gb.REBAL_FREQ = rf.REBAL_FREQ = controls["rebal_freq"]
        gb.TRAIN_START = controls["train_start"]
        gb.TRAIN_END = controls["train_end"]
        gb.TEST_START = controls["test_start"]
        gb.TEST_END = controls["test_end"]
        gb.LAMBDA_RISK = rf.LAMBDA_RISK = controls["lambda_risk"]
        gb.TC_BPS = rf.TC_BPS = controls["tc_bps"]
        gb.RISK_FREE_ANNUAL = rf.RISK_FREE_ANNUAL = controls["rf_rate"]
        gb.INITIAL_INVESTMENT = rf.INITIAL_INVESTMENT = controls["initial_investment"]

        progress = st.progress(0, text="Downloading prices…")
        prices = rf.download_prices(rf.TICKERS, rf.START, rf.END)
        progress.progress(5, text="Running RF backtest…")

        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(pct, text=f"Running RF backtest… ({int(frac*100)}%)")
                yield x

        gb.tqdm = streamlit_tqdm  # RF uses gb.run_backtest internally

        results = rf.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = rf.save_results(results, rf.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved RF results to: `{out_path}`")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download RF results",
                data=f,
                file_name=os.path.basename(out_path),
                mime="application/octet-stream",
            )

# ==========================================================
# PCA + LightGBM
# ==========================================================
elif model_choice.startswith("PCA"):

    if pl is None or gb is None:
        st.error("Could not import `_pca_lightgbm.py` / `_gradientboost.py`.")
        st.stop()

    st.subheader("🧬 PCA + LightGBM Settings")

    default_start = getattr(pl, "START", getattr(gb, "START", "2010-01-01"))
    default_end = getattr(pl, "END", getattr(gb, "END", None))
    default_train_start = getattr(gb, "TRAIN_START", "2010-01-01")
    default_train_end = getattr(gb, "TRAIN_END", "2023-12-31")
    default_test_start = getattr(gb, "TEST_START", "2024-01-01")
    default_test_end = getattr(gb, "TEST_END", "2025-12-31")

    start_default_date = date.fromisoformat(default_start)
    end_default_date = date.today() if default_end in (None, "") else date.fromisoformat(default_end)

    defaults = {
        "start_date": start_default_date,
        "end_date": end_default_date,
        "rebal_freq": getattr(gb, "REBAL_FREQ", "M"),
        "train_start_year": year_from_str(default_train_start, 2010),
        "train_end_year": year_from_str(default_train_end, 2023),
        "test_start_year": year_from_str(default_test_start, 2024),
        "test_end_year": year_from_str(default_test_end, 2025),
        "lambda_risk": getattr(gb, "LAMBDA_RISK", 5.0),
        "tc_bps": getattr(gb, "TC_BPS", 5),
        "rf_rate": getattr(gb, "RISK_FREE_ANNUAL", 0.015),
        "initial_investment": getattr(gb, "INITIAL_INVESTMENT", 1000.0),
    }

    controls = finance_controls(defaults, prefix="PCA_LGBM")

    if controls["run"]:
        pl.START = date_to_str(controls["start_date"])
        pl.END = date_to_str(controls["end_date"])
        pl.REBAL_FREQ = controls["rebal_freq"]
        gb.TRAIN_START = pl.TRAIN_START = controls["train_start"]
        gb.TRAIN_END = pl.TRAIN_END = controls["train_end"]
        gb.TEST_START = pl.TEST_START = controls["test_start"]
        gb.TEST_END = pl.TEST_END = controls["test_end"]
        gb.LAMBDA_RISK = pl.LAMBDA_RISK = controls["lambda_risk"]
        gb.TC_BPS = pl.TC_BPS = controls["tc_bps"]
        gb.RISK_FREE_ANNUAL = controls["rf_rate"]
        gb.INITIAL_INVESTMENT = pl.INITIAL_INVESTMENT = controls["initial_investment"]

        progress = st.progress(0, text="Downloading prices…")
        prices = pl.download_prices(pl.TICKERS, pl.START, pl.END)
        progress.progress(5, text="Running PCA + LightGBM backtest…")

        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(pct, text=f"Running PCA+LGBM backtest… ({int(frac*100)}%)")
                yield x

        pl.tqdm = streamlit_tqdm  # type: ignore[assignment]

        results = pl.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = pl.save_results(results, pl.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved PCA+LGBM results to: `{out_path}`")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download PCA+LGBM results",
                data=f,
                file_name=os.path.basename(out_path),
                mime="application/octet-stream",
            )

# ==========================================================
# CatBoost
# ==========================================================
else:

    if cb is None or gb is None:
        st.error("Could not import `_catboost.py` / `_gradientboost.py`.")
        st.stop()

    st.subheader("🐈 CatBoost Settings")

    default_start = getattr(cb, "START", getattr(gb, "START", "2010-01-01"))
    default_end = getattr(cb, "END", getattr(gb, "END", None))
    default_train_start = getattr(gb, "TRAIN_START", "2010-01-01")
    default_train_end = getattr(gb, "TRAIN_END", "2023-12-31")
    default_test_start = getattr(gb, "TEST_START", "2024-01-01")
    default_test_end = getattr(gb, "TEST_END", "2025-12-31")

    start_default_date = date.fromisoformat(default_start)
    end_default_date = date.today() if default_end in (None, "") else date.fromisoformat(default_end)

    defaults = {
        "start_date": start_default_date,
        "end_date": end_default_date,
        "rebal_freq": getattr(gb, "REBAL_FREQ", "M"),
        "train_start_year": year_from_str(default_train_start, 2010),
        "train_end_year": year_from_str(default_train_end, 2023),
        "test_start_year": year_from_str(default_test_start, 2024),
        "test_end_year": year_from_str(default_test_end, 2025),
        "lambda_risk": getattr(gb, "LAMBDA_RISK", 5.0),
        "tc_bps": getattr(gb, "TC_BPS", 5),
        "rf_rate": getattr(gb, "RISK_FREE_ANNUAL", 0.015),
        "initial_investment": getattr(gb, "INITIAL_INVESTMENT", 1000.0),
    }

    controls = finance_controls(defaults, prefix="CatBoost")

    if controls["run"]:
        # CatBoost module piggybacks on gb.run_backtest, so keep both in sync
        gb.START = cb.START = date_to_str(controls["start_date"])
        gb.END = cb.END = date_to_str(controls["end_date"])
        gb.REBAL_FREQ = cb.REBAL_FREQ = controls["rebal_freq"]
        gb.TRAIN_START = cb.TRAIN_START = controls["train_start"]
        gb.TRAIN_END = cb.TRAIN_END = controls["train_end"]
        gb.TEST_START = cb.TEST_START = controls["test_start"]
        gb.TEST_END = cb.TEST_END = controls["test_end"]
        gb.LAMBDA_RISK = cb.LAMBDA_RISK = controls["lambda_risk"]
        gb.TC_BPS = cb.TC_BPS = controls["tc_bps"]
        gb.RISK_FREE_ANNUAL = cb.RISK_FREE_ANNUAL = controls["rf_rate"]
        gb.INITIAL_INVESTMENT = cb.INITIAL_INVESTMENT = controls["initial_investment"]

        progress = st.progress(0, text="Downloading prices…")
        prices = cb.download_prices(cb.TICKERS, cb.START, cb.END)
        progress.progress(5, text="Running CatBoost backtest…")

        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(pct, text=f"Running CatBoost backtest… ({int(frac*100)}%)")
                yield x

        # CatBoost module monkey-patches gb.make_gbm inside cb.run_backtest,
        # so wiring tqdm into gb is enough
        gb.tqdm = streamlit_tqdm  # type: ignore[assignment]

        results = cb.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = cb.save_results(results, cb.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved CatBoost results to: `{out_path}`")
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download CatBoost results",
                data=f,
                file_name=os.path.basename(out_path),
                mime="application/octet-stream",
            )
