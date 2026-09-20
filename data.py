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


def fetch_prices_batch(tickers, force=False):
    result = {}
    needed = []
    for t in tickers:
        path = _cache_path(t)
        if not force and os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty and (date.today() - df.index[-1].date()).days < 3:
                result[t] = df["close"]
                continue
        needed.append(t)

    if needed:
        end = date.today()
        start = end - timedelta(days=int(config.LOOKBACK_YEARS * 365.25) + 10)
        raw = yf.download(needed, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
        os.makedirs(config.PRICE_CACHE_DIR, exist_ok=True)
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for t in needed:
            close = raw[t]["Close"] if is_multi else raw["Close"]
            close = close.dropna()
            if close.empty:
                raise ValueError(f"No price data returned for {t}")
            close.name = "close"
            close.to_frame().to_csv(_cache_path(t))
            result[t] = close

    return result


def fetch_prices(ticker, force=False):
    return fetch_prices_batch([ticker], force=force)[ticker]


def common_index(series_dict):
    idx = None
    for s in series_dict.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    return idx


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def latest_price(ticker):
    return float(fetch_prices(ticker, force=True).iloc[-1])


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
    pooled = np.concatenate(all_jump_returns) if all_jump_returns else np.array([])
    if len(pooled) == 0:
        return 0.0, 1e-6
    mu_j = float(pooled.mean())
    sigma_j = float(pooled.std()) if len(pooled) > 1 else 1e-6
    return mu_j, sigma_j


def detect_shared_jump_days(jump_days_by_ticker, aligned_index, min_tickers):
    shared_count = pd.Series(0, index=aligned_index)
    for jump_index in jump_days_by_ticker.values():
        hits = jump_index.intersection(aligned_index)
        shared_count.loc[hits] += 1
    return set(shared_count[shared_count >= min_tickers].index)


def realized_variance_series(returns):
    rolling_std = returns.rolling(config.HESTON_ROLLING_WINDOW).std().dropna()
    return rolling_std ** 2 * config.TRADING_DAYS_PER_YEAR


def calibrate_heston(returns, ticker=None):
    v = realized_variance_series(returns)
    v_t = v.iloc[1:].values
    v_lag = v.iloc[:-1].values
    if len(v_t) < 30:
        raise ValueError("Not enough history to fit Heston AR(1)")

    b, a = np.polyfit(v_lag, v_t, 1)
    residuals = v_t - (a + b * v_lag)

    kappa_raw = (1 - b) / config.DT
    theta_raw = a / (1 - b) if abs(1 - b) > 1e-8 else float(v.mean())
    kappa = max(kappa_raw, 1e-4)
    theta = max(theta_raw, 1e-6)
    if kappa_raw <= 0 or theta_raw <= 0:
        label = ticker or "ticker"
        print(f"WARNING: {label} Heston AR(1) fit was non-stationary or gave invalid theta "
              f"(kappa_raw={kappa_raw:.4g}, theta_raw={theta_raw:.4g}); clamped to floor values")

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


def calibrate_from_returns(returns_by_ticker, tickers):
    aligned_index = common_index(returns_by_ticker)
    returns_aligned = pd.DataFrame({t: r.reindex(aligned_index) for t, r in returns_by_ticker.items()}).dropna()
    years = len(aligned_index) / config.TRADING_DAYS_PER_YEAR

    jump_returns_by_ticker = {}
    diffusive_by_ticker = {}
    for t in tickers:
        jump_returns, diffusive_returns = detect_jumps(returns_by_ticker[t])
        jump_returns_by_ticker[t] = jump_returns
        diffusive_by_ticker[t] = diffusive_returns

    shared_days = detect_shared_jump_days(
        {t: jr.index for t, jr in jump_returns_by_ticker.items()}, aligned_index, config.SYSTEMIC_JUMP_MIN_TICKERS
    )
    lambda_market = len(shared_days) / years if years > 0 else 0.0

    per_ticker = {}
    all_jump_returns = []
    all_standardized = []
    for t in tickers:
        jump_returns = jump_returns_by_ticker[t]
        diffusive_returns = diffusive_by_ticker[t]
        mu, sigma = calibrate_gbm(diffusive_returns)
        idio_jump_count = len(jump_returns.index.difference(shared_days))
        lam_idio = idio_jump_count / years if years > 0 else 0.0
        heston = calibrate_heston(diffusive_returns, ticker=t)
        all_jump_returns.append(jump_returns.values)
        standardized = (diffusive_returns - diffusive_returns.mean()) / diffusive_returns.std()
        all_standardized.append(standardized.values)
        per_ticker[t] = {"mu": mu, "sigma": sigma, "lambda_idio": lam_idio, "heston": heston}

    mu_j, sigma_j = calibrate_pooled_jump_size(all_jump_returns)
    nu = calibrate_student_t(all_standardized)
    correlation_matrix = returns_aligned.tail(config.CORRELATION_WINDOW_DAYS).corr()

    for t in tickers:
        per_ticker[t]["mu_j"] = mu_j
        per_ticker[t]["sigma_j"] = sigma_j
        per_ticker[t]["lambda_market"] = lambda_market

    return {
        "tickers": tickers,
        "per_ticker": per_ticker,
        "nu": nu,
        "correlation_matrix": correlation_matrix.to_dict(),
        "calibrated_at": date.today().isoformat(),
    }


def calibrate_universe(tickers):
    prices_by_ticker = fetch_prices_batch(tickers)
    returns_by_ticker = {t: log_returns(prices_by_ticker[t]) for t in tickers}
    return calibrate_from_returns(returns_by_ticker, tickers)


def save_params(params, path=None):
    path = path or config.PARAMS_CACHE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=1)


def load_params(path=None):
    path = path or config.PARAMS_CACHE_PATH
    with open(path) as f:
        return json.load(f)
