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
    from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
except Exception:
    rf = None
    SKRandomForestRegressor = None  # type: ignore

try:
    import _pca_lightgbm as pl
except Exception:
    pl = None


st.set_page_config(
    page_title="🧪 Run Backtests",
    layout="wide",
)

st.title("🧪 Run & Save ML Portfolio Backtests")

st.markdown(
    """
Use this page to **generate new results files** for your models.

1. Pick a model and adjust its key parameters.  
2. Click **Run backtest**.  
3. The script will download data, train, backtest, and save a timestamped
   `.pkl` in the `results/` folder.  
4. You can also download that file directly and upload it on the results pages.
"""
)

model_choice = st.selectbox(
    "Choose which model to run",
    options=[
        "Gradient Boost (XGBoost / GBM)",
        "Random Forest",
        "PCA + LightGBM (Masuda)",
    ],
)


def date_to_str(d):
    if d is None:
        return None
    return d.strftime("%Y-%m-%d")


# =============================================================
# Gradient Boost model controls
# =============================================================
if model_choice.startswith("Gradient"):

    if gb is None:
        st.error(
            "Could not import `_gradientboost.py`. "
            "Make sure the file exists and has no import errors."
        )
        st.stop()

    st.subheader("Gradient Boost Backtest Settings")

    default_start = getattr(gb, "START", "2010-01-01")
    default_end = getattr(gb, "END", None)
    default_lambda = float(getattr(gb, "LAMBDA_RISK", 5.0))

    try:
        start_default_date = date.fromisoformat(default_start)
    except Exception:
        start_default_date = date(2010, 1, 1)

    if default_end in (None, ""):
        end_default_date = date.today()
    else:
        try:
            end_default_date = date.fromisoformat(default_end)
        except Exception:
            end_default_date = date.today()

    with st.form("gbm_form"):
        col_dates, col_params = st.columns(2)

        with col_dates:
            start_date = st.date_input(
                "Start date",
                value=start_default_date,
                help="Start of price history to download.",
            )
            end_date = st.date_input(
                "End date",
                value=end_default_date,
                help="End of price history (usually today).",
            )
            lambda_risk = st.slider(
                "Risk aversion (λ)",
                min_value=0.5,
                max_value=20.0,
                value=default_lambda,
                step=0.5,
                help="Higher λ puts more weight on risk minimization.",
            )

        with col_params:
            st.markdown("**XGBoost / GBM hyperparameters**")
            n_estimators = st.slider("n_estimators", 50, 800, 400, step=50)
            max_depth = st.slider("max_depth", 2, 10, 3)
            learning_rate = st.slider(
                "learning_rate", 0.01, 0.30, 0.05, step=0.01
            )
            subsample = st.slider("subsample", 0.3, 1.0, 0.9, step=0.05)
            colsample_bytree = st.slider(
                "colsample_bytree", 0.3, 1.0, 0.9, step=0.05
            )

        run_gbm = st.form_submit_button("🚀 Run Gradient Boost backtest")

    if run_gbm:
        # Patch global settings used in the module
        gb.LAMBDA_RISK = float(lambda_risk)
        gb.START = date_to_str(start_date)
        gb.END = date_to_str(end_date)

        # Patch the model factory so run_backtest uses our hyperparameters
        def custom_make_gbm():
            # If XGBoost is available in the module, use it; otherwise fallback to sklearn GBM
            if getattr(gb, "USE_XGB", False):
                try:
                    from xgboost import XGBRegressor  # type: ignore

                    try:
                        return XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            tree_method="hist",
                            device="cuda",
                            random_state=7,
                        )
                    except TypeError:
                        # Older versions without 'device' argument
                        return XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            tree_method="gpu_hist",
                            random_state=7,
                        )
                except Exception:
                    pass  # fall through to sklearn GBM

            # CPU-only fallback
            try:
                from sklearn.ensemble import GradientBoostingRegressor
            except Exception:
                raise RuntimeError("Neither XGBoost nor sklearn GBM is available.")

            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=7,
            )

        gb.make_gbm = custom_make_gbm  # type: ignore[assignment]

        # ---- Progress bar wired to gb.tqdm ----
        progress = st.progress(0, text="Downloading prices…")
        prices = gb.download_prices(gb.TICKERS, gb.START, gb.END)
        progress.progress(5, text="Initializing backtest…")

        # Wrap tqdm inside the GBM module so we can drive the Streamlit progress bar
        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(
                    pct,
                    text=f"Running Gradient Boost backtest… ({int(100*frac)}%)",
                )
                yield x

        gb.tqdm = streamlit_tqdm  # type: ignore[assignment]

        results = gb.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = gb.save_results(results, gb.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved Gradient Boost results to: `{out_path}`")
        try:
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download GBM results file",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="application/octet-stream",
                )
        except Exception:
            st.info(
                "The file has been saved on disk in the `results/` folder. "
                "If the download button does not work, you can fetch it manually."
            )

# =============================================================
# Random Forest model controls
# =============================================================
elif model_choice.startswith("Random"):

    if rf is None:
        st.error(
            "Could not import `_randomforest.py`. "
            "Make sure the file exists and has no import errors."
        )
        st.stop()

    st.subheader("Random Forest Backtest Settings")

    default_start = getattr(rf, "START", "2010-01-01")
    default_end = getattr(rf, "END", None)
    default_train_end = getattr(rf, "TRAIN_END_DATE", "2023-12-31")
    default_backtest_start = getattr(rf, "BACKTEST_START", "2024-01-02")
    default_top_k = int(getattr(rf, "TOP_K", 5))
    default_start_value = float(getattr(rf, "START_VALUE", 1000.0))

    try:
        start_default_date = date.fromisoformat(default_start)
    except Exception:
        start_default_date = date(2010, 1, 1)

    if not default_end:
        end_default_date = date.today()
    else:
        try:
            end_default_date = date.fromisoformat(default_end)
        except Exception:
            end_default_date = date.today()

    try:
        train_end_default = date.fromisoformat(default_train_end)
    except Exception:
        train_end_default = date(2023, 12, 31)

    try:
        bt_start_default = date.fromisoformat(default_backtest_start)
    except Exception:
        bt_start_default = date(2024, 1, 2)

    with st.form("rf_form"):
        col_dates, col_params = st.columns(2)

        with col_dates:
            start_date = st.date_input(
                "Start date",
                value=start_default_date,
                help="Start of price history to download.",
            )
            end_date = st.date_input(
                "End date",
                value=end_default_date,
                help="End of price history (usually today).",
            )
            train_end = st.date_input(
                "Train end date",
                value=train_end_default,
                help="Last date included in the training split.",
            )
            backtest_start = st.date_input(
                "Backtest start date",
                value=bt_start_default,
                help="First date used in the RF strategy backtest.",
            )

        with col_params:
            top_k = st.slider(
                "Top-K stocks per day",
                min_value=1,
                max_value=20,
                value=default_top_k,
            )
            n_estimators = st.slider("n_estimators", 50, 800, 300, step=50)
            max_depth = st.slider("max_depth", 2, 20, 6)
            start_value = st.number_input(
                "Initial capital ($)",
                min_value=100.0,
                max_value=1000000.0,
                value=default_start_value,
                step=100.0,
            )

        run_rf = st.form_submit_button("🚀 Run Random Forest backtest")

    if run_rf:
        if hasattr(rf, "START"):
            rf.START = date_to_str(start_date)
        if hasattr(rf, "END"):
            rf.END = date_to_str(end_date)
        if hasattr(rf, "TRAIN_END_DATE"):
            rf.TRAIN_END_DATE = date_to_str(train_end)
        if hasattr(rf, "BACKTEST_START"):
            rf.BACKTEST_START = date_to_str(backtest_start)
        if hasattr(rf, "TOP_K"):
            rf.TOP_K = int(top_k)
        if hasattr(rf, "START_VALUE"):
            rf.START_VALUE = float(start_value)

        # Pass RF hyperparameters to the module so build_results can see them
        rf.N_ESTIMATORS = int(n_estimators)
        rf.MAX_DEPTH = int(max_depth)

        progress = st.progress(0, text="Downloading prices…")
        # Include SPY benchmark along with the stock universe
        all_tickers = list(dict.fromkeys(rf.TICKERS + [rf.BENCH_TICKER]))
        prices = rf.download_prices(all_tickers, rf.START, rf.END)

        progress.progress(40, text="Running Random Forest backtest…")
        results = rf.build_results(prices)

        progress.progress(80, text="Saving results…")
        out_path = rf.save_results(results, rf.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved Random Forest results to: `{out_path}`")
        try:
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download RF results file",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="application/octet-stream",
                )
        except Exception:
            st.info(
                "The file has been saved on disk in the `results/` folder. "
                "If the download button does not work, you can fetch it manually."
            )

# =============================================================
# PCA + LightGBM model controls
# =============================================================
else:
    if pl is None:
        st.error(
            "Could not import `_pca_lightgbm.py`. "
            "Make sure the file exists and has no import errors."
        )
        st.stop()

    st.subheader("PCA + LightGBM Backtest Settings")

    default_start = getattr(pl, "START", "2010-01-01")
    default_end = getattr(pl, "END", None)

    try:
        start_default_date = date.fromisoformat(default_start)
    except Exception:
        start_default_date = date(2010, 1, 1)

    if default_end in (None, ""):
        end_default_date = date.today()
    else:
        try:
            end_default_date = date.fromisoformat(default_end)
        except Exception:
            end_default_date = date.today()

    with st.form("pca_lgbm_form"):
        start_date = st.date_input(
            "Start date",
            value=start_default_date,
            help="Start of price history to download.",
        )
        end_date = st.date_input(
            "End date",
            value=end_default_date,
            help="End of price history (usually today).",
        )

        run_pl = st.form_submit_button("🚀 Run PCA + LightGBM backtest")

    if run_pl:
        pl.START = date_to_str(start_date)
        pl.END = date_to_str(end_date)

        progress = st.progress(0, text="Downloading prices…")
        prices = pl.download_prices(pl.TICKERS, pl.START, pl.END)
        progress.progress(5, text="Initializing backtest…")

        # Wrap tqdm inside the PCA+LGBM module
        def streamlit_tqdm(iterable, **kwargs):
            seq = list(iterable)
            total = len(seq)
            if total == 0:
                return seq
            for i, x in enumerate(seq):
                frac = (i + 1) / total
                pct = 5 + int(90 * frac)
                progress.progress(
                    pct,
                    text=f"Running PCA + LightGBM backtest… ({int(100*frac)}%)",
                )
                yield x

        pl.tqdm = streamlit_tqdm  # type: ignore[assignment]

        results = pl.build_results(prices)
        progress.progress(98, text="Saving results…")
        out_path = pl.save_results(results, pl.RESULTS_DIR)
        progress.progress(100, text="Done!")

        st.success(f"Saved PCA + LightGBM results to: `{out_path}`")
        try:
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PCA+LGBM results file",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="application/octet-stream",
                )
        except Exception:
            st.info(
                "The file has been saved on disk in the `results/` folder. "
                "If the download button does not work, you can fetch it manually."
            )
