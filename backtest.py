import json

import numpy as np
import pandas as pd

import config
import data
import stress_test


def run_backtest(holdings_path=None, years=None, step_days=None, n_paths=None, n_seeds=None):
    years = years if years is not None else config.BACKTEST_YEARS
    step_days = step_days if step_days is not None else config.BACKTEST_STEP_DAYS
    n_paths = n_paths if n_paths is not None else config.BACKTEST_N_PATHS
    n_seeds = n_seeds if n_seeds is not None else config.BACKTEST_N_SEEDS

    holdings = stress_test.load_holdings(holdings_path)
    weights, _, _ = stress_test.portfolio_weights(holdings)
    tickers = sorted(weights.keys())
    weights_vec = np.array([weights[t] for t in tickers])

    prices_by_ticker = data.fetch_prices_batch(tickers)
    returns_by_ticker = {t: data.log_returns(prices_by_ticker[t]) for t in tickers}
    aligned_index = data.common_index(returns_by_ticker)

    cutoff = aligned_index[-1] - pd.Timedelta(days=int(years * 365.25))
    candidate_dates = [d for d in aligned_index if d >= cutoff]
    test_dates = candidate_dates[::step_days]

    results = {c: {"breaches": 0, "total": 0} for c in config.VAR_CONFIDENCE_LEVELS}

    for as_of in test_dates:
        history_idx = aligned_index[aligned_index <= as_of]
        future_idx = aligned_index[aligned_index > as_of][:config.HORIZON_DAYS]
        if len(history_idx) < config.BACKTEST_MIN_HISTORY_DAYS or len(future_idx) < config.HORIZON_DAYS:
            continue

        truncated_returns = {t: returns_by_ticker[t].reindex(history_idx).dropna() for t in tickers}
        try:
            params = data.calibrate_from_returns(truncated_returns, tickers)
            inputs = stress_test.build_seed_invariant_inputs(tickers, params, config.HORIZON_DAYS)
        except Exception:
            continue

        pooled = np.concatenate([
            stress_test.simulate_one_seed(tickers, inputs, weights_vec, config.HORIZON_DAYS, n_paths, seed)
            for seed in range(n_seeds)
        ])

        actual_log_returns = np.array([returns_by_ticker[t].reindex(future_idx).sum() for t in tickers])
        actual_simple_returns = np.exp(actual_log_returns) - 1
        actual_portfolio_return = float(np.dot(weights_vec, actual_simple_returns))

        for c in config.VAR_CONFIDENCE_LEVELS:
            var, _ = stress_test.var_cvar(pooled, c)
            results[c]["total"] += 1
            if -actual_portfolio_return > var:
                results[c]["breaches"] += 1

    summary = {}
    for c, r in results.items():
        summary[c] = {
            "tests": r["total"],
            "breaches": r["breaches"],
            "breach_rate": (r["breaches"] / r["total"]) if r["total"] else None,
            "expected_breach_rate": round(1 - c, 4),
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(run_backtest(), indent=1))
