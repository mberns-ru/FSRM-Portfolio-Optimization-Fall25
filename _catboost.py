import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostRegressor  # <-- CatBoost instead of RandomForest

# Reuse the full Masuda-style pipeline from _gradientboost
import _gradientboost as gb

np.random.seed(7)

# Re-export key settings so the rest of the app can treat CatBoost like GBM
TICKERS = gb.TICKERS
START = gb.START
END = gb.END
REBAL_FREQ = gb.REBAL_FREQ

TRAIN_START = getattr(gb, "TRAIN_START", "2010-01-01")
TRAIN_END   = getattr(gb, "TRAIN_END", "2023-12-31")
TEST_START  = getattr(gb, "TEST_START", "2024-01-01")
TEST_END    = getattr(gb, "TEST_END", "2025-12-31")
INITIAL_INVESTMENT = getattr(gb, "INITIAL_INVESTMENT", 1000.0)

LAMBDA_RISK = gb.LAMBDA_RISK
TC_BPS = gb.TC_BPS
RISK_FREE_ANNUAL = gb.RISK_FREE_ANNUAL
RESULTS_DIR = gb.RESULTS_DIR

download_prices = gb.download_prices
make_feature_table = gb.make_feature_table
cluster_labels = gb.cluster_labels
optimize_weights = gb.optimize_weights
minvar_weights = gb.minvar_weights
meanvar_weights = gb.meanvar_weights
compute_perf_metrics = gb.compute_perf_metrics

# ---- CatBoost hyperparameters (can be overridden from Streamlit) ----
N_ESTIMATORS_DEFAULT = 400    # CatBoost calls this "iterations"
MAX_DEPTH_DEFAULT = 6
LEARNING_RATE_DEFAULT = 0.05


def make_cb() -> CatBoostRegressor:
    """
    Factory for CatBoostRegressor, using module-level globals
    if they are set (so Streamlit can override them).
    """
    iterations = globals().get("N_ESTIMATORS", N_ESTIMATORS_DEFAULT)
    depth = globals().get("MAX_DEPTH", MAX_DEPTH_DEFAULT)
    lr = globals().get("LEARNING_RATE", LEARNING_RATE_DEFAULT)

    return CatBoostRegressor(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(lr),
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )


def run_backtest(prices: pd.DataFrame, store_weights: bool = False):
    """
    Run the full Masuda-style *monthly* backtest but with CatBoost
    in place of Gradient Boost.

    We temporarily monkey-patch gb.make_gbm so that the GBM pipeline
    uses CatBoostRegressor under the hood, then restore it.
    """
    old_make_gbm = gb.make_gbm

    def cb_factory():
        return make_cb()

    gb.make_gbm = cb_factory
    try:
        monthly, weights = gb.run_backtest(prices, store_weights=store_weights)
    finally:
        gb.make_gbm = old_make_gbm

    return monthly, weights


def build_results(prices: pd.DataFrame) -> dict:
    """
    Runs the full CatBoost backtest (using the GBM Masuda pipeline),
    builds SPY benchmark and metrics, and returns a dict suitable for saving.
    """
    monthly, weights = run_backtest(prices, store_weights=True)

    # ---- Train / Test split ----
    monthly_train = monthly.loc[TRAIN_START:TRAIN_END]
    monthly_test = monthly.loc[TEST_START:TEST_END]

    # Test-window bi-monthly weights
    weights_test = weights.loc[TEST_START:TEST_END]
    weights_test_bimonth = weights_test.sort_index().iloc[::2]

    # ---- SPY benchmark equity over full test window ----
    spy_start = TEST_START
    spy_end = (pd.to_datetime(TEST_END) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    bench_px = yf.download(
        "SPY",
        start=spy_start,
        end=spy_end,
        auto_adjust=True,
        progress=False,
    )["Close"].dropna()

    bench_initial = bench_px.iloc[0]
    bench_equity = INITIAL_INVESTMENT * bench_px / bench_initial
    bench_equity_m = bench_equity.resample("M").last()
    bench_equity_m = bench_equity_m.loc[monthly_test.index.min():monthly_test.index.max()]
    bench_equity_m.name = f"SPY_${int(INITIAL_INVESTMENT)}"

    # CatBoost ML_Opt equity over test window
    ml_equity_test = INITIAL_INVESTMENT * (1.0 + monthly_test["ML_Opt"]).cumprod()
    ml_equity_test.index = monthly_test.index
    ml_equity_test.name = f"CB_ML_Opt_${int(INITIAL_INVESTMENT)}"

    equity_test = pd.concat([ml_equity_test, bench_equity_m], axis=1).dropna()

    # 2025-only version kept for compatibility
    ml_ret_2025 = monthly.loc["2025-01-01":"2025-12-31"]["ML_Opt"]
    if not ml_ret_2025.empty:
        ml_equity_2025 = INITIAL_INVESTMENT * (1.0 + ml_ret_2025).cumprod()
        ml_equity_2025.name = f"CB_ML_Opt_${int(INITIAL_INVESTMENT)}"

        bench_equity_2025 = bench_equity.loc["2025-01-01":"2025-12-31"].resample("M").last()
        bench_equity_2025.name = f"SPY_${int(INITIAL_INVESTMENT)}"
        equity_2025 = pd.concat([ml_equity_2025, bench_equity_2025], axis=1).dropna()
    else:
        equity_2025 = equity_test.copy()

    metrics = {
        "train": {
            "ML_Opt": compute_perf_metrics(monthly_train["ML_Opt"]),
            "EqualWeight": compute_perf_metrics(monthly_train["EqualWeight"]),
        },
        "test": {
            "ML_Opt": compute_perf_metrics(monthly_test["ML_Opt"]),
            "EqualWeight": compute_perf_metrics(monthly_test["EqualWeight"]),
        },
    }

    results = {
        "prices": prices,
        "monthly": monthly,
        "weights": weights,
        "monthly_train": monthly_train,
        "monthly_test": monthly_test,
        "weights_test_bimonth": weights_test_bimonth,
        "weights_2025_bimonth": weights.loc["2025-01-01":"2025-12-31"].sort_index().iloc[::2],
        "equity_test": equity_test,
        "equity_2025": equity_2025,
        "metrics": metrics,
        "tickers": TICKERS,
        "start": START,
        "end": END,
        "model_name": "CatBoost",
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "initial_investment": float(INITIAL_INVESTMENT),
    }
    return results


def save_results(results: dict, directory: str = RESULTS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(directory, f"catboost_results_{ts}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    return fname


def main():
    print("Downloading prices…")
    px = download_prices(TICKERS, START, END)
    print(f"Got {px.shape[1]} tickers, {px.shape[0]} rows of prices.")

    print("Running CatBoost monthly backtest (Masuda-style)…")
    results = build_results(px)

    out_path = save_results(results, RESULTS_DIR)
    print(f"Saved CatBoost results to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
