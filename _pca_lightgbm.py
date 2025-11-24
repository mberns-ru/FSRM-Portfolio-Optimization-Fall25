import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import _gradientboost as gb

# Try LightGBM; fall back if missing
try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

from _gradientboost import (  # reuse universe & utilities
    TICKERS,
    START,
    END,
    REBAL_FREQ,
    LOOKBACK_DAYS,
    TRAIN_MIN_MONTHS,
    TC_BPS,
    N_CLUSTERS,
    LAMBDA_RISK,
    RESULTS_DIR as GBM_RESULTS_DIR,
    download_prices,
    make_feature_table,
    cluster_labels,
    optimize_weights,
    minvar_weights,
    meanvar_weights,
    compute_perf_metrics,
)

TRAIN_START = getattr(gb, "TRAIN_START", "2010-01-01")
TRAIN_END   = getattr(gb, "TRAIN_END", "2023-12-31")
TEST_START  = getattr(gb, "TEST_START", "2024-01-01")
TEST_END    = getattr(gb, "TEST_END", "2025-12-31")
INITIAL_INVESTMENT = getattr(gb, "INITIAL_INVESTMENT", 1000.0)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

RESULTS_DIR = "results"
MODEL_TAG = "PCA_LightGBM_Masuda"


def make_pca_lightgbm(n_features: int):
    """
    PCA (up to 6 components) + LightGBM (or GBM fallback).
    """
    n_components = min(6, max(1, n_features))

    if HAS_LIGHTGBM:
        base = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="regression",
            random_state=7,
            verbose=-1,  # quiet
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor

        base = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            random_state=7,
        )

    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components)),
            ("gbm", base),
        ]
    )


def run_backtest(prices: pd.DataFrame, store_weights: bool = False):
    """
    Same monthly MV+clustering framework as _gradientboost,
    but alpha model is PCA + LightGBM.
    """
    prices = prices.dropna(how="all")
    rets_daily = prices.pct_change().dropna()
    feats = make_feature_table(prices)

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

    for d0 in tqdm(first_trade_dates, desc="Backtest (PCA+LGBM)", unit="month"):
        idx = rebal_dates.get_loc(d0)
        if idx + 1 >= len(rebal_dates):
            break
        d1 = rebal_dates[idx + 1]

        win_mask = (
            (rets_daily.index <= d0)
            & (rets_daily.index >= d0 - pd.Timedelta(days=LOOKBACK_DAYS))
        )
        if win_mask.sum() < 60:
            continue

        Sigma_full = rets_daily.loc[win_mask].cov().values
        corr_full = rets_daily.loc[win_mask].corr().values

        target_mask = (rets_daily.index > d0) & (rets_daily.index <= d1)
        if target_mask.sum() == 0:
            continue
        y_next = rets_daily.loc[target_mask].add(1).prod() - 1.0

        # One-row feature matrix for valid assets at d0
        X_row, valid_assets = [], []
        for tkr in names:
            cols = [c for c in feats.columns if c.endswith("_" + tkr)]
            if not cols:
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

        # Historical training set (monthly endpoints up to ~7 years)
        past_endpoints = rebal_dates[
            (rebal_dates < d0) & (rebal_dates >= d0 - relativedelta(years=7))
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
                if not cols:
                    continue
                feat_sub = feats.loc[:dd, cols]
                if feat_sub.empty:
                    continue
                row = feat_sub.iloc[-1]
                if row.isna().any():
                    continue
                Xi.append(row.values)
                vi.append(tkr)
            if not Xi:
                continue

            Xi = np.vstack(Xi)
            yi = yi_series.reindex(vi).values
            mask = ~np.isnan(yi)
            if not mask.any():
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

        model = make_pca_lightgbm(n_features=X_hist_z.shape[1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(X_hist_z, y_hist)
        mu_pred_assets = model.predict(X_row_z)

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

        month_path = rets_daily.loc[target_mask]
        r_realized = (month_path.add(1).prod() - 1.0).reindex(names).fillna(0.0)

        turnover = w.abs().sum() if prev_w is None else (w - prev_w).abs().sum()
        tc = (TC_BPS / 10000.0) * turnover

        strat_monthly.append((w * r_realized).sum() - tc)
        dates_out.append(d1)
        turnovers.append(turnover)
        prev_w = w.copy()

        if store_weights:
            w.name = d1
            weights_list.append(w.copy())

        # Benchmarks (equal weight / minvar / meanvar)
        w_eq = pd.Series(0.0, index=names)
        w_eq.loc[valid_assets] = 1.0 / len(valid_assets)
        eqw_monthly.append((w_eq * r_realized).sum())

        w_minv_v = minvar_weights(Sigma_v)
        w_minv = pd.Series(0.0, index=names)
        w_minv.loc[valid_assets] = w_minv_v / max(w_minv_v.sum(), 1e-12)
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
    Train: TRAIN_START–TRAIN_END
    Test:  TEST_START–TEST_END
    Benchmark: SPY (equity scaled to INITIAL_INVESTMENT from TEST_START)
    """
    monthly, weights = run_backtest(prices, store_weights=True)

    monthly_train = monthly.loc[TRAIN_START:TRAIN_END]
    monthly_test = monthly.loc[TEST_START:TEST_END]

    weights_test = weights.loc[TEST_START:TEST_END]
    weights_test_bimonth = weights_test.sort_index().iloc[::2]

    bench_px = yf.download(
        "SPY",
        start=TEST_START,
        end=(pd.to_datetime(TEST_END) + relativedelta(days=10)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"].dropna()

    bench_initial = bench_px.iloc[0]
    bench_equity = INITIAL_INVESTMENT * bench_px / bench_initial
    bench_equity_m = bench_equity.resample("M").last()
    bench_equity_m = bench_equity_m.loc[monthly_test.index.min():monthly_test.index.max()]
    bench_equity_m.name = f"SPY_${int(INITIAL_INVESTMENT)}"

    ml_equity_test = INITIAL_INVESTMENT * (1.0 + monthly_test["ML_Opt"]).cumprod()
    ml_equity_test.index = monthly_test.index
    ml_equity_test.name = f"ML_Opt_${int(INITIAL_INVESTMENT)}"

    equity_test = pd.concat([ml_equity_test, bench_equity_m], axis=1).dropna()

    # 2025-only, for backward compatibility with older pages
    ml_ret_2025 = monthly.loc["2025-01-01":"2025-12-31"]["ML_Opt"]
    if not ml_ret_2025.empty:
        ml_equity_2025 = INITIAL_INVESTMENT * (1.0 + ml_ret_2025).cumprod()
        ml_equity_2025.name = f"ML_Opt_${int(INITIAL_INVESTMENT)}"
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
        "model": MODEL_TAG,
        "prices": prices,
        "monthly": monthly,
        "weights": weights,
        "monthly_train": monthly_train,
        "monthly_test": monthly_test,
        "metrics": metrics,
        "weights_test_bimonth": weights_test_bimonth,
        "weights_2025_bimonth": weights.loc["2025-01-01":"2025-12-31"].sort_index().iloc[::2],
        "equity_test": equity_test,
        "equity_compare_2025": equity_2025,
        "use_lightgbm": HAS_LIGHTGBM,
        "tickers": TICKERS,
        "start": START,
        "end": END,
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
    fname = os.path.join(directory, f"pca_lightgbm_results_{ts}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    return fname


if __name__ == "__main__":
    print("Downloading prices…")
    px = download_prices(TICKERS, START, END)
    print(f"Got {px.shape[1]} tickers, {px.shape[0]} rows of prices.")

    print("Running PCA+LightGBM backtest (this can take a bit)…")
    results_dict = build_results(px)

    out_path = save_results(results_dict, RESULTS_DIR)
    print(f"Saved results to: {out_path}")
    print("Done.")
