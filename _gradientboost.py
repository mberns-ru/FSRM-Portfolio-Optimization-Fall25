import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.cluster import KMeans  # needed for cluster_labels
import warnings

# progress bar (tqdm); safe fallback if not installed
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# (optional but recommended on Windows to avoid MKL/KMeans warning)
os.environ["OMP_NUM_THREADS"] = "1"

# Silence the repeating KMeans + XGBoost device warnings
warnings.filterwarnings(
    "ignore",
    message=".*KMeans is known to have a memory leak on Windows with MKL.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Device is changed from GPU to CPU as we couldn't find any available GPU on the system.*",
)

# Fallback if xgboost not installed
try:
    from xgboost import XGBRegressor

    def make_gbm():
        """
        Create a GPU-accelerated gradient boosting model if possible.
        Falls back to CPU hist if GPU options not supported.
        """
        try:
            # Newer xgboost versions: device param + hist
            return XGBRegressor(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device="cuda",
                random_state=7,
            )
        except TypeError:
            # Older xgboost versions: gpu_hist tree method
            return XGBRegressor(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="gpu_hist",
                random_state=7,
            )

    USE_XGB = True

except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    def make_gbm():
        """
        CPU-only fallback (sklearn GradientBoostingRegressor).
        No GPU acceleration, but same interface.
        """
        return GradientBoostingRegressor(
            random_state=7,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
        )

    USE_XGB = False

# =====================
# User Settings
# =====================

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JNJ", "JPM", "XOM",
    "V", "PG", "MA", "HD", "CVX", "UNH", "MRK", "KO", "PEP", "ABBV",
    "AVGO", "NFLX", "ADBE", "ORCL", "CSCO",
]

# Train+backtest from 2010 onwards
START = "2010-01-01"
END = None  # None = today

REBAL_FREQ = "M"         # Monthly rebalance; we'll *report* every 2 months
LOOKBACK_DAYS = 252      # ~1Y lookback for features/covariance
TRAIN_MIN_MONTHS = 36    # Require ~3 years before first trade

# Optimization hyperparameters (simple mean-variance)
LAMBDA_RISK = 5.0        # risk aversion
N_CLUSTERS = 5           # for clustering

# Backtest settings
TC_BPS = 5               # transaction cost (bps) per $ turnover (e.g., 5 = 0.05%)
RISK_FREE_ANNUAL = 0.015 # for Sharpe if you want to compute it later

RESULTS_DIR = "results"  # where we save pickled results


# =====================
# Helper Functions
# =====================

def download_prices(tickers, start, end):
    """Download adjusted close prices with yfinance."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def make_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Build rolling-return / vol / momentum features per ticker."""
    rets = prices.pct_change()
    feats = {}

    for t in prices.columns:
        r = rets[t]
        p = prices[t]

        feats[f"ret21_{t}"] = r.rolling(21).mean()
        feats[f"ret63_{t}"] = r.rolling(63).mean()
        feats[f"ret252_{t}"] = r.rolling(252).mean()

        feats[f"vol21_{t}"] = r.rolling(21).std()
        feats[f"vol63_{t}"] = r.rolling(63).std()

        feats[f"mom63_{t}"] = p.pct_change(63)

    return pd.DataFrame(feats)


def cluster_labels(corr_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    n_assets = corr_matrix.shape[0]
    n_clusters = max(2, min(n_clusters, n_assets))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    return kmeans.fit_predict(corr_matrix)


def minvar_weights(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    ones = np.ones(n)
    Sigma_reg = Sigma + 1e-6 * np.eye(n)
    inv = np.linalg.pinv(Sigma_reg)
    w = inv @ ones
    w = np.maximum(w, 0)
    if w.sum() == 0:
        return np.ones(n) / n
    return w / w.sum()


def meanvar_weights(mu: np.ndarray, Sigma: np.ndarray, lam: float = 5.0) -> np.ndarray:
    Sigma_reg = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    inv = np.linalg.pinv(Sigma_reg)
    w = inv @ mu / max(lam, 1e-6)
    w = np.maximum(w, 0)
    if w.sum() == 0:
        return np.ones_like(w) / len(w)
    return w / w.sum()


def optimize_weights(mu_pred: np.ndarray,
                     Sigma: np.ndarray,
                     labs: np.ndarray,
                     lam: float = 5.0,
                     gamma: float = 2.0) -> np.ndarray:
    # Simplified: pure mean-variance on predicted returns
    return meanvar_weights(mu_pred, Sigma, lam=lam)


def compute_perf_metrics(returns: pd.Series,
                         risk_free_annual: float = RISK_FREE_ANNUAL,
                         periods_per_year: int = 12) -> dict:
    """
    Compute a few basic performance metrics for a series of periodic returns.
    """
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    total_ret = (1.0 + returns).prod() - 1.0
    n_periods = len(returns)

    ann_return = (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    rf_per_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = ann_return - risk_free_annual
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown from cumulative equity
    equity_curve = (1.0 + returns).cumprod()
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    max_dd = drawdown.min()

    return {
        "total_return": total_ret,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# =====================
# Backtest Core
# =====================

def run_backtest(prices: pd.DataFrame, store_weights: bool = False):
    prices = prices.dropna(how="all")
    rets_daily = prices.pct_change().dropna()
    feats = make_feature_table(prices)

    # Monthly endpoints for rebalancing
    rebal_dates = prices.resample(REBAL_FREQ).last().index
    rebal_dates = rebal_dates[
        rebal_dates >= (prices.index[0] + pd.Timedelta(days=LOOKBACK_DAYS))
    ]
    first_trade_dates = rebal_dates[TRAIN_MIN_MONTHS:]

    strat_monthly, eqw_monthly, minv_monthly, mv_monthly = [], [], [], []
    dates_out, turnovers = [], []
    prev_w = None
    weights_list = []

    names = rets_daily.columns.tolist()

    # >>> progress bar here <<<
    for d0 in tqdm(first_trade_dates, desc="Backtest", unit="month"):
        idx = rebal_dates.get_loc(d0)
        if idx + 1 >= len(rebal_dates):
            break
        d1 = rebal_dates[idx + 1]

        # Lookback window for covariance & features
        win_mask = (
            (rets_daily.index <= d0) &
            (rets_daily.index >= d0 - pd.Timedelta(days=LOOKBACK_DAYS))
        )
        if win_mask.sum() < 60:
            continue

        Sigma_full = rets_daily.loc[win_mask].cov().values
        corr_full = rets_daily.loc[win_mask].corr().values

        # Next period (target returns)
        target_mask = (rets_daily.index > d0) & (rets_daily.index <= d1)
        if target_mask.sum() == 0:
            continue
        y_next = rets_daily.loc[target_mask].add(1).prod() - 1.0

        # Build one-row feature matrix X_row for valid assets at time d0
        X_row, valid_assets = [], []
        for tkr in names:
            cols = [c for c in feats.columns if c.endswith("_" + tkr)]
            if len(cols) == 0:
                continue
            feat_sub = feats.loc[:d0, cols]
            if feat_sub.empty:
                continue
            xj = feat_sub.iloc[-1]
            if xj.isna().any():
                continue
            X_row.append(xj.values)
            valid_assets.append(tkr)

        if len(valid_assets) < max(8, len(names) // 3):
            continue

        X_row = np.vstack(X_row)
        y_vec = y_next.reindex(valid_assets).values

        # Historical endpoints for training up to 7 years
        past_endpoints = rebal_dates[
            (rebal_dates < d0) &
            (rebal_dates >= d0 - relativedelta(years=7))
        ]
        X_hist, y_hist = [], []

        for dd in past_endpoints:
            i = rebal_dates.get_loc(dd)
            if i + 1 >= len(rebal_dates):
                continue
            dd_next = rebal_dates[i + 1]

            win_target = (rets_daily.index > dd) & (rets_daily.index <= dd_next)
            if win_target.sum() == 0:
                continue
            yi_series = rets_daily.loc[win_target].add(1).prod() - 1.0

            Xi, vi = [], []
            for tkr in names:
                cols = [c for c in feats.columns if c.endswith("_" + tkr)]
                if len(cols) == 0:
                    continue
                feat_sub = feats.loc[:dd, cols]
                if feat_sub.empty:
                    continue
                row = feat_sub.iloc[-1]
                if row.isna().any():
                    continue
                Xi.append(row.values)
                vi.append(tkr)
            if len(Xi) == 0:
                continue

            Xi = np.vstack(Xi)
            yi = yi_series.reindex(vi).values
            mask = ~np.isnan(yi)
            if mask.sum() == 0:
                continue
            X_hist.append(Xi[mask])
            y_hist.append(yi[mask])

        if len(X_hist) < 6:
            continue

        X_hist = np.vstack(X_hist)
        y_hist = np.concatenate(y_hist)

        # Standardize features
        mu_f = X_hist.mean(axis=0)
        sd_f = X_hist.std(axis=0) + 1e-12
        X_hist_z = (X_hist - mu_f) / sd_f
        X_row_z = (X_row - mu_f) / sd_f

        # ---- GPU-friendly Gradient Boosting ----
        gbr = make_gbm()
        gbr.fit(X_hist_z, y_hist)
        mu_pred_assets = gbr.predict(X_row_z)

        # Subset covariance to valid assets
        idx_valid = [names.index(a) for a in valid_assets]
        Sigma_v = Sigma_full[np.ix_(idx_valid, idx_valid)]
        corr_v = corr_full[np.ix_(idx_valid, idx_valid)]

        ncl = int(min(N_CLUSTERS, max(2, len(valid_assets) // 4)))
        labs = cluster_labels(corr_v, ncl)

        w_valid = optimize_weights(
            mu_pred_assets,
            Sigma_v,
            labs,
            lam=LAMBDA_RISK,
            gamma=2.0,
        )

        w = pd.Series(0.0, index=names)
        w.loc[valid_assets] = w_valid
        if w.sum() > 0:
            w /= w.sum()
        else:
            w[:] = 1.0 / len(w)

        # Realized returns over [d0, d1]
        month_path = rets_daily.loc[target_mask]
        r_realized = (month_path.add(1).prod() - 1.0).reindex(names).fillna(0.0)

        turnover = w.abs().sum() if prev_w is None else (w - prev_w).abs().sum()
        tc = (TC_BPS / 10000.0) * turnover

        strat_monthly.append((w * r_realized).sum() - tc)
        dates_out.append(d1)
        turnovers.append(turnover)
        prev_w = w.copy()

        if store_weights:
            weights_list.append(w.copy())

        # Benchmarks
        w_eq = pd.Series(0.0, index=names)
        w_eq.loc[valid_assets] = 1.0 / len(valid_assets)
        eqw_monthly.append((w_eq * r_realized).sum())

        w_minv_v = minvar_weights(Sigma_v)
        w_minv = pd.Series(0.0, index=names)
        w_minv.loc[valid_assets] = w_minv_v / w_minv_v.sum()
        minv_monthly.append((w_minv * r_realized).sum())

        hist_window = rets_daily.loc[rets_daily.index <= d0].iloc[-252:]
        mu_hist_all = hist_window.mean()
        mu_hist_v = mu_hist_all.reindex(valid_assets).values
        w_mv_v = meanvar_weights(mu_hist_v, Sigma_v, lam=LAMBDA_RISK)
        w_mv = pd.Series(0.0, index=names)
        w_mv.loc[valid_assets] = w_mv_v / max(w_mv_v.sum(), 1e-12)
        mv_monthly.append((w_mv * r_realized).sum())

    idx = pd.DatetimeIndex(dates_out, name="Date")

    strat = pd.Series(strat_monthly, index=idx, name="ML_Opt")
    eqw = pd.Series(eqw_monthly, index=idx, name="EqualWeight")
    minv = pd.Series(minv_monthly, index=idx, name="MinVar")
    mv = pd.Series(mv_monthly, index=idx, name="MeanVar")

    strat.attrs["turnover"] = np.mean(turnovers) if turnovers else np.nan

    monthly_df = pd.concat([strat, eqw, minv, mv], axis=1).dropna(how="all")

    if store_weights:
        weights_df = pd.DataFrame(weights_list, index=idx)
        weights_df.index.name = "Date"
        return monthly_df, weights_df

    return monthly_df, None


def build_results(prices: pd.DataFrame) -> dict:
    """
    Runs the full backtest, builds benchmark and metrics,
    and returns a dict suitable for saving to disk and reloading in Streamlit.

    Train window: 2010–2023
    Test window:  2024–2025
    Benchmark:    SPY ($1000 buy-and-hold from 2025-01-01)
    """
    monthly, weights = run_backtest(prices, store_weights=True)

    # ---- Train / Test split ----
    monthly_train = monthly.loc["2010-01-01":"2023-12-31"]
    monthly_test = monthly.loc["2024-01-01":"2025-12-31"]

    # Portfolio composition: every 2 months in 2025
    weights_2025 = weights.loc["2025-01-01":"2025-12-31"]
    weights_2025_bimonth = weights_2025.sort_index().iloc[::2]

    # SPY benchmark: $1000 invested on 2025-01-01
    bench_px = yf.download(
        "SPY",
        start="2025-01-01",
        end="2026-01-01",
        auto_adjust=True,
        progress=False,
    )["Close"].dropna()

    bench_initial = bench_px.iloc[0]
    bench_equity = 1000.0 * bench_px / bench_initial
    bench_equity_m = bench_equity.resample("M").last()
    bench_equity_m.name = "SPY_$1000"

    # Only 2025 ML returns for the equity curve vs SPY
    ml_ret_2025 = monthly.loc["2025-01-01":"2025-12-31"]["ML_Opt"]
    ml_equity = 1000.0 * (1.0 + ml_ret_2025).cumprod()
    ml_equity.name = "ML_Opt_$1000"

    equity_compare_2025 = pd.concat([ml_equity, bench_equity_m], axis=1).dropna()

    # Performance metrics (train = 2010–2023, test = 2024–2025)
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
        "weights_2025_bimonth": weights_2025_bimonth,
        "equity_2025": equity_compare_2025,
        "metrics": metrics,
        "tickers": TICKERS,
        "start": START,
        "end": END,
        "use_xgb": USE_XGB,
    }
    return results

def save_results(results: dict, directory: str = RESULTS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(directory, f"gradientboost_results_{ts}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    return fname


if __name__ == "__main__":
    print("Downloading prices…")
    px = download_prices(TICKERS, START, END)
    print(f"Got {px.shape[1]} tickers, {px.shape[0]} rows of prices.")

    print("Running backtest (this can take a bit)…")
    results_dict = build_results(px)

    out_path = save_results(results_dict, RESULTS_DIR)
    print(f"Saved results to: {out_path}")
    print("Done.")
