import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(7)

# =====================
# User Settings
# =====================
# Core investable universe (shared with Gradient Boost & PCA-LGBM models)
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JNJ", "JPM", "XOM",
    "V", "PG", "MA", "HD", "CVX", "UNH", "MRK", "KO", "PEP", "ABBV",
    "AVGO", "NFLX", "ADBE", "ORCL", "CSCO",
]

# Separate benchmark index
BENCH_TICKER = "SPY"

START = "2010-01-01"
END = None  # None = today

# RF feature + split settings
FEATURE_COLS = ["mom_short", "mom_long", "vol"]
TRAIN_END_DATE = "2023-12-31"        # train on 2010–2023
BACKTEST_START = "2024-01-02"        # **test ONLY on 2024–2025**
BACKTEST_END = "2025-12-31"
TOP_K = 5
LOOKBACK_FEATURE = 60

RISK_FREE_ANNUAL = 0.015
START_VALUE = 1000.0
RESULTS_DIR = "results"

# Default RF hyperparameters (can be overridden from Streamlit by setting
# module globals N_ESTIMATORS and MAX_DEPTH before calling build_results)
N_ESTIMATORS_DEFAULT = 300
MAX_DEPTH_DEFAULT = 6


# =====================
# Helpers
# =====================

def download_prices(tickers, start, end):
    px = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").fillna(method="ffill").dropna(how="any", axis=1)
    return px


def make_supervised_panel(rets, lookback_short=5, lookback_long=20):
    """Panel with one row per (date, ticker) and target = next-day return."""
    frames = []
    for tk in rets.columns:
        r = rets[tk]
        df = pd.DataFrame(index=r.index)
        df["ticker"] = tk
        df["ret"] = r
        df["mom_short"] = r.rolling(lookback_short).sum()
        df["mom_long"] = r.rolling(lookback_long).sum()
        df["vol"] = r.rolling(lookback_long).std()
        df["target"] = r.shift(-1)   # next-day return

        df = df.dropna()
        frames.append(df)

    panel = pd.concat(frames).sort_index()
    return panel


def backtest_ml_strategy(stock_rets, bench_rets, model,
                         start_date="2024-01-02", top_k=5, lookback=60):
    """
    Daily ML strategy: each day, predict next-day returns for all stocks,
    go equal-weight into the top_k predictions, compare with benchmark.
    Backtest begins at `start_date` and ends at `BACKTEST_END`.
    """
    dates = stock_rets.index
    dates = dates[(dates >= start_date) & (dates <= BACKTEST_END)]

    if len(dates) < 2:
        raise ValueError("Not enough dates in backtest window.")

    portfolio_vals = [1.0]
    bench_vals = [1.0]
    bt_dates = [dates[0]]

    trade_log = []
    last_holdings = set()

    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]

        # Build features for ALL stocks at this date
        X_today = []
        tickers_today = []

        for tk in stock_rets.columns:
            hist = stock_rets[tk].loc[:date].iloc[-(lookback + 1):]
            if len(hist) < (lookback + 1):
                continue

            r = hist
            mom_short = r.iloc[-5:].sum()
            mom_long = r.iloc[-20:].sum()
            vol = r.iloc[-20:].std()

            X_today.append([mom_short, mom_long, vol])
            tickers_today.append(tk)

        X_today = np.array(X_today)
        if len(tickers_today) == 0:
            portfolio_vals.append(portfolio_vals[-1])
            bench_vals.append(bench_vals[-1] * (1 + bench_rets.loc[next_date]))
            bt_dates.append(next_date)
            trade_log.append(
                {
                    "date": date,
                    "chosen_stocks": [],
                    "buys": [],
                    "sells": [],
                    "holds": [],
                }
            )
            continue

        preds = model.predict(X_today)
        order = np.argsort(preds)[::-1]
        chosen = [tickers_today[i] for i in order[:top_k]]

        tomorrow_rets = stock_rets.loc[next_date, chosen]
        port_ret = tomorrow_rets.mean()

        bench_ret = bench_rets.loc[next_date]

        # Update portfolio and benchmark
        portfolio_vals.append(portfolio_vals[-1] * (1 + port_ret))
        bench_vals.append(bench_vals[-1] * (1 + bench_ret))
        bt_dates.append(next_date)

        # Simple trade log
        chosen_set = set(chosen)
        buys = sorted(chosen_set - last_holdings)
        sells = sorted(last_holdings - chosen_set)
        holds = sorted(chosen_set & last_holdings)

        trade_log.append(
            {
                "date": date,
                "chosen_stocks": chosen,
                "buys": buys,
                "sells": sells,
                "holds": holds,
            }
        )
        last_holdings = chosen_set

    curve_port = pd.Series(portfolio_vals, index=bt_dates)
    curve_bench = pd.Series(bench_vals, index=bt_dates)
    trade_log_df = pd.DataFrame(trade_log)
    return curve_port, curve_bench, trade_log_df


def performance_stats(curve, periods_per_year=252):
    rets = np.log(curve / curve.shift(1)).dropna()
    if len(rets) == 0:
        return {
            "Annualized return": np.nan,
            "Annualized vol": np.nan,
            "Sharpe (rf=0)": np.nan,
            "Max drawdown": np.nan,
        }
    mu_ann = rets.mean() * periods_per_year
    vol_ann = rets.std() * np.sqrt(periods_per_year)
    sharpe = mu_ann / vol_ann if vol_ann > 0 else np.nan
    max_dd = (curve / curve.cummax() - 1).min()
    return {
        "Annualized return": float(mu_ann),
        "Annualized vol": float(vol_ann),
        "Sharpe (rf=0)": float(sharpe),
        "Max drawdown": float(max_dd),
    }


# =====================
# Build results for Streamlit
# =====================

def build_results(prices: pd.DataFrame) -> dict:
    """
    Build a results dictionary for the RF strategy.

    - Train on daily supervised panel from 2010-01-01 up to TRAIN_END_DATE (inclusive).
    - Backtest daily RF strategy ONLY on 2024-01-02 to 2025-12-31.
    - Compare against a buy-and-hold benchmark index (BENCH_TICKER, e.g. SPY).
    """
    # Daily log returns for all downloaded assets (investable + benchmark)
    returns = np.log(prices / prices.shift(1)).dropna()

    # Restrict returns to a stable window
    returns = returns.loc[START:BACKTEST_END]

    # Detect benchmark column
    benchmark_candidates = ["SPY", "S&P 500", "^GSPC", BENCH_TICKER]
    benchmark_col = None
    for c in benchmark_candidates:
        if c in returns.columns:
            benchmark_col = c
            break
    if benchmark_col is None:
        raise ValueError("Could not find benchmark column (SPY / ^GSPC) in data.")

    # ML trades all non-benchmark tickers
    trade_cols = [c for c in returns.columns if c != benchmark_col]
    trade_rets = returns[trade_cols]
    bench_rets = returns[benchmark_col]

    # Supervised panel for RF
    panel = make_supervised_panel(trade_rets)

    # Train / test split (on panel dates)
    panel_train = panel.loc[:TRAIN_END_DATE]
    panel_test = panel.loc[TRAIN_END_DATE:]

    X_train = panel_train[FEATURE_COLS].values
    y_train = panel_train["target"].values
    X_test = panel_test[FEATURE_COLS].values
    y_test = panel_test["target"].values

    # Allow Streamlit to override these via module globals
    n_est = globals().get("N_ESTIMATORS", N_ESTIMATORS_DEFAULT)
    max_depth_val = globals().get("MAX_DEPTH", MAX_DEPTH_DEFAULT)

    model = RandomForestRegressor(
        n_estimators=int(n_est),
        max_depth=int(max_depth_val),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Backtest strategy vs benchmark, ONLY from BACKTEST_START to BACKTEST_END
    port_curve, bench_curve, trades = backtest_ml_strategy(
        trade_rets,
        bench_rets,
        model,
        start_date=BACKTEST_START,
        top_k=TOP_K,
        lookback=LOOKBACK_FEATURE,
    )

    # Restrict curves explicitly to [BACKTEST_START, BACKTEST_END]
    port_curve = port_curve.loc[BACKTEST_START:BACKTEST_END]
    bench_curve = bench_curve.loc[BACKTEST_START:BACKTEST_END]

    # Performance stats (daily)
    stats_port = performance_stats(port_curve)
    stats_bench = performance_stats(bench_curve)

    # Convert to dollar curves starting at START_VALUE
    ml_dollars = START_VALUE * (port_curve / port_curve.iloc[0])
    bench_dollars = START_VALUE * (bench_curve / bench_curve.iloc[0])
    ml_dollars.name = "RF_$"
    bench_dollars.name = "Benchmark_$"

    # Simplified trade log for dashboard (all in test window 2024–2025)
    trade_log_simple = trades[["date", "chosen_stocks", "buys", "sells", "holds"]].copy()

    results = {
        "prices": prices,
        "returns": returns,
        "benchmark_col": benchmark_col,
        "trade_cols": trade_cols,
        "feature_cols": FEATURE_COLS,
        "train_end_date": TRAIN_END_DATE,
        "backtest_start": BACKTEST_START,
        "backtest_end": BACKTEST_END,
        "train_scores": {"r2": float(r2), "rmse": float(rmse)},
        "stats_port": stats_port,
        "stats_bench": stats_bench,
        "ml_curve": ml_dollars,
        "bench_curve": bench_dollars,
        "trade_log": trade_log_simple,
        "start": START,
        "end": END,
        "tickers": TICKERS,
        "start_value": START_VALUE,
    }
    return results


def save_results(results: dict, directory: str = RESULTS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(directory, f"randomforest_results_{ts}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    return fname


def main():
    print("Downloading prices…")
    # Download both investable tickers and the benchmark index
    all_tickers = list(dict.fromkeys(TICKERS + [BENCH_TICKER]))
    px = download_prices(all_tickers, START, END)
    print(f"Got {px.shape[1]} tickers (including benchmark), {px.shape[0]} rows.")

    print("Training RF model & running backtest (2024–2025)…")
    results = build_results(px)

    ml_curve = results["ml_curve"]
    bench_curve = results["bench_curve"]
    print(f"Final RF portfolio value: ${ml_curve.iloc[-1]:.2f}")
    print(f"Final benchmark value:    ${bench_curve.iloc[-1]:.2f}")

    out_path = save_results(results, RESULTS_DIR)
    print(f"Saved RF results to: {out_path}")


if __name__ == "__main__":
    main()
