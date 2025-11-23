import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans  # not actually used here but fine to keep
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(7)

# =====================
# User Settings
# =====================
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JNJ", "JPM", "XOM",
    "V", "PG", "MA", "HD", "CVX", "UNH", "MRK", "KO", "PEP", "ABBV",
    "AVGO", "NFLX", "ADBE", "ORCL", "CSCO", "SPY"
]

START = "2010-01-01"
END = None  # None = today

REBAL_FREQ = "D"       # daily ML signals / trading
LOOKBACK_DAYS = 252    # not used directly here but kept for parity
TRAIN_MIN_MONTHS = 36  # idem

# Model / features
FEATURE_COLS = ["mom_short", "mom_long", "vol"]
TRAIN_END_DATE = "2023-12-31"   # train 2010–2023, test 2024–2025
BACKTEST_START = "2019-01-02"   # when to start trading in the backtest
TOP_K = 5                       # number of stocks each day
LOOKBACK_FEATURE = 60           # days of history needed for features

# Backtest settings
RISK_FREE_ANNUAL = 0.015  # if you ever want to adjust Sharpe
START_VALUE = 1000.0      # $1,000 initial capital for curves

RESULTS_DIR = "results"   # where we save pickled results


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
                         start_date="2019-01-02", top_k=5, lookback=60):
    """
    Daily ML strategy: each day, predict next-day returns for all stocks,
    go equal-weight into the top_k predictions, compare with SPY benchmark.
    """
    dates = stock_rets.index
    dates = dates[dates >= start_date]

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

        if len(X_today) == 0:
            continue

        X_today = np.array(X_today)
        preds = model.predict(X_today)

        # Select top-K by predicted return
        order = np.argsort(preds)[::-1]
        k = min(top_k, len(order))
        chosen_idx = order[:k]
        chosen_tks = [tickers_today[j] for j in chosen_idx]
        weights = {tk: 1.0 / k for tk in chosen_tks}

        current_holdings = set(chosen_tks)
        buys = list(current_holdings - last_holdings)
        sells = list(last_holdings - current_holdings)
        holds = list(current_holdings & last_holdings)

        # Realized next-day portfolio return
        r_p = 0.0
        for tk in chosen_tks:
            r_p += weights[tk] * stock_rets.loc[next_date, tk]
        r_bench = bench_rets.loc[next_date]

        portfolio_vals.append(portfolio_vals[-1] * np.exp(r_p))
        bench_vals.append(bench_vals[-1] * np.exp(r_bench))
        bt_dates.append(next_date)

        trade_log.append(
            {
                "date": date,
                "chosen_stocks": chosen_tks,
                "buys": buys,
                "sells": sells,
                "holds": holds,
            }
        )

        last_holdings = current_holdings

    port_curve = pd.Series(portfolio_vals, index=bt_dates, name="ML_Portfolio")
    bench_curve = pd.Series(bench_vals, index=bt_dates, name="SPY")
    trade_log_df = pd.DataFrame(trade_log)

    return port_curve, bench_curve, trade_log_df


def performance_stats(curve, periods_per_year=252):
    rets = np.log(curve / curve.shift(1)).dropna()
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
    # Daily log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    # Detect SPY / S&P 500 column
    benchmark_candidates = ["SPY", "S&P 500", "^GSPC"]
    benchmark_col = None
    for c in benchmark_candidates:
        if c in returns.columns:
            benchmark_col = c
            break
    if benchmark_col is None:
        raise ValueError("Could not find SPY / S&P 500 column in data.")

    # ML trades all tickers (including SPY)
    trade_cols = list(returns.columns)
    trade_rets = returns[trade_cols]
    bench_rets = returns[benchmark_col]

    # Supervised panel for RF
    panel = make_supervised_panel(trade_rets)

    # Train / test split (on panel dates)
    train = panel.loc[:TRAIN_END_DATE]
    test = panel.loc[TRAIN_END_DATE:]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Backtest strategy vs SPY (from BACKTEST_START onward)
    port_curve, bench_curve, trades = backtest_ml_strategy(
        trade_rets,
        bench_rets,
        model,
        start_date=BACKTEST_START,
        top_k=TOP_K,
        lookback=LOOKBACK_FEATURE,
    )

    # Performance stats
    stats_port = performance_stats(port_curve)
    stats_bench = performance_stats(bench_curve)

    # Convert to dollar curves starting at $1000
    ml_dollars = START_VALUE * (port_curve / port_curve.iloc[0])
    spy_dollars = START_VALUE * (bench_curve / bench_curve.iloc[0])
    ml_dollars.name = "ML_$"
    spy_dollars.name = "SPY_$"

    # Keep a simplified trade log for the dashboard
    trade_log_simple = trades[["date", "chosen_stocks", "buys", "sells", "holds"]].copy()

    results = {
        "prices": prices,
        "returns": returns,
        "benchmark_col": benchmark_col,
        "trade_cols": trade_cols,
        "feature_cols": FEATURE_COLS,
        "train_end_date": TRAIN_END_DATE,
        "backtest_start": BACKTEST_START,
        "train_scores": {"r2": float(r2), "rmse": float(rmse)},
        "stats_port": stats_port,
        "stats_bench": stats_bench,
        "ml_curve": ml_dollars,
        "bench_curve": spy_dollars,
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
    px = download_prices(TICKERS, START, END)
    print(f"Got {px.shape[1]} tickers, {px.shape[0]} rows.")

    # Optional: keep a CSV around
    px.to_csv("price_data_rf.csv")

    print("Training RF model & running backtest…")
    results = build_results(px)

    # Quick CLI summary
    ml_curve = results["ml_curve"]
    bench_curve = results["bench_curve"]
    print(f"Final ML portfolio value: ${ml_curve.iloc[-1]:.2f}")
    print(f"Final SPY value:          ${bench_curve.iloc[-1]:.2f}")

    out_path = save_results(results, RESULTS_DIR)
    print(f"Saved RF results to: {out_path}")

    # Optional quick matplotlib plot
    plt.figure(figsize=(10, 5))
    plt.plot(ml_curve.index, ml_curve.values, label="RF Strategy", linewidth=2)
    plt.plot(bench_curve.index, bench_curve.values, label="S&P 500 (SPY)", linewidth=2)
    plt.title("Portfolio Value: RF Strategy vs S&P 500 (Starting at $1,000)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
