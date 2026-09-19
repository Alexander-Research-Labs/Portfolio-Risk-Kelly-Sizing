import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import config


def _cache_path(ticker):
    return os.path.join(config.PRICE_CACHE_DIR, ticker.replace(".", "_") + ".csv")


def fetch_prices(ticker, force=False):
    path = _cache_path(ticker)
    if not force and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty and (date.today() - df.index[-1].date()).days < 3:
            return df["close"]

    end = date.today()
    start = end - timedelta(days=int(config.LOOKBACK_YEARS * 365.25) + 10)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "close"

    os.makedirs(config.PRICE_CACHE_DIR, exist_ok=True)
    close.to_frame().to_csv(path)
    return close


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def latest_price(ticker):
    return float(fetch_prices(ticker).iloc[-1])


def detect_jumps(returns):
    threshold = config.JUMP_THRESHOLD_STD * returns.std()
    jump_mask = returns.abs() > threshold
    return returns[jump_mask], returns[~jump_mask]


def calibrate_gbm(diffusive_returns):
    mu = float(diffusive_returns.mean() * config.TRADING_DAYS_PER_YEAR)
    sigma = float(diffusive_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))
    return mu, sigma


def calibrate_jump_rate(returns, jump_returns):
    years = len(returns) / config.TRADING_DAYS_PER_YEAR
    return len(jump_returns) / years if years > 0 else 0.0


def calibrate_pooled_jump_size(all_jump_returns):
    pooled = np.concatenate(all_jump_returns) if all_jump_returns else np.array([0.0])
    mu_j = float(pooled.mean())
    sigma_j = float(pooled.std()) if len(pooled) > 1 else 1e-6
    return mu_j, sigma_j


def realized_variance_series(returns):
    rolling_std = returns.rolling(config.HESTON_ROLLING_WINDOW).std().dropna()
    return rolling_std ** 2 * config.TRADING_DAYS_PER_YEAR


def calibrate_heston(returns):
    v = realized_variance_series(returns)
    v_t = v.iloc[1:].values
    v_lag = v.iloc[:-1].values
    if len(v_t) < 30:
        raise ValueError("Not enough history to fit Heston AR(1)")

    b, a = np.polyfit(v_lag, v_t, 1)
    residuals = v_t - (a + b * v_lag)

    kappa = (1 - b) / config.DT
    theta = a / (1 - b) if abs(1 - b) > 1e-8 else float(v.mean())
    kappa = max(kappa, 1e-4)
    theta = max(theta, 1e-6)

    mean_v = float(v.mean())
    resid_var = float(np.var(residuals))
    xi_sq = resid_var / max(mean_v * config.DT, 1e-12)
    xi = float(np.sqrt(max(xi_sq, 1e-8)))

    aligned_returns = returns.reindex(v.index).iloc[1:]
    dv = v.diff().reindex(aligned_returns.index)
    valid = dv.notna() & aligned_returns.notna()
    if valid.sum() > 10:
        rho = float(np.corrcoef(aligned_returns[valid], dv[valid])[0, 1])
    else:
        rho = 0.0

    feller_satisfied = bool(2 * kappa * theta > xi ** 2)
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "xi": xi,
        "rho": rho,
        "v0": float(v.iloc[-1]),
        "feller_satisfied": feller_satisfied,
    }


def calibrate_student_t(all_standardized_returns):
    pooled = np.concatenate(all_standardized_returns)
    nu, _, _ = stats.t.fit(pooled, floc=0, fscale=1)
    return float(max(nu, 2.1))


def calibrate_universe(tickers):
    prices_by_ticker = {}
    returns_by_ticker = {}
    for t in tickers:
        prices_by_ticker[t] = fetch_prices(t)
        returns_by_ticker[t] = log_returns(prices_by_ticker[t])

    common_index = None
    for r in returns_by_ticker.values():
        common_index = r.index if common_index is None else common_index.intersection(r.index)
    returns_aligned = pd.DataFrame({t: r.reindex(common_index) for t, r in returns_by_ticker.items()}).dropna()

    per_ticker = {}
    all_jump_returns = []
    all_standardized = []
    for t in tickers:
        r = returns_by_ticker[t]
        jump_returns, diffusive_returns = detect_jumps(r)
        mu, sigma = calibrate_gbm(diffusive_returns)
        lam = calibrate_jump_rate(r, jump_returns)
        heston = calibrate_heston(r)
        all_jump_returns.append(jump_returns.values)
        standardized = (r - r.mean()) / r.std()
        all_standardized.append(standardized.values)
        per_ticker[t] = {"mu": mu, "sigma": sigma, "lambda": lam, "heston": heston}

    mu_j, sigma_j = calibrate_pooled_jump_size(all_jump_returns)
    nu = calibrate_student_t(all_standardized)
    correlation_matrix = returns_aligned.corr()

    for t in tickers:
        per_ticker[t]["mu_j"] = mu_j
        per_ticker[t]["sigma_j"] = sigma_j

    return {
        "tickers": tickers,
        "per_ticker": per_ticker,
        "nu": nu,
        "correlation_matrix": correlation_matrix.to_dict(),
        "calibrated_at": date.today().isoformat(),
    }


def save_params(params, path=None):
    path = path or config.PARAMS_CACHE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=1)


def load_params(path=None):
    path = path or config.PARAMS_CACHE_PATH
    with open(path) as f:
        return json.load(f)
