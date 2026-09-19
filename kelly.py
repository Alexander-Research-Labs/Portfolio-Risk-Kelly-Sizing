import json

import numpy as np

import config
import data
import stress_test


def historical_returns_frame(tickers):
    frames = {t: data.log_returns(data.fetch_prices(t)) for t in tickers}
    common_index = None
    for r in frames.values():
        common_index = r.index if common_index is None else common_index.intersection(r.index)
    return frames, common_index


def portfolio_mu_sigma(weights):
    tickers = list(weights.keys())
    frames, common_index = historical_returns_frame(tickers)
    returns_df = np.column_stack([frames[t].reindex(common_index).values for t in tickers])
    w = np.array([weights[t] for t in tickers])

    mu_vec = returns_df.mean(axis=0) * config.TRADING_DAYS_PER_YEAR
    cov = np.cov(returns_df, rowvar=False) * config.TRADING_DAYS_PER_YEAR

    mu_p = float(w @ mu_vec)
    sigma_p_sq = float(w @ cov @ w)
    portfolio_daily_returns = returns_df @ w
    return mu_p, sigma_p_sq, portfolio_daily_returns, common_index


def candidate_mu_sigma_rho(candidate, weights):
    mu_p, sigma_p_sq, portfolio_daily_returns, common_index = portfolio_mu_sigma(weights)
    candidate_returns = data.log_returns(data.fetch_prices(candidate)).reindex(common_index)

    valid = ~np.isnan(candidate_returns.values) & ~np.isnan(portfolio_daily_returns)
    mu_c = float(candidate_returns.values[valid].mean() * config.TRADING_DAYS_PER_YEAR)
    sigma_c = float(candidate_returns.values[valid].std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))
    rho_pc = float(np.corrcoef(candidate_returns.values[valid], portfolio_daily_returns[valid])[0, 1])

    sigma_p = float(np.sqrt(sigma_p_sq))
    return mu_p, sigma_p, mu_c, sigma_c, rho_pc


def solve_kelly(mu_p, sigma_p, mu_c, sigma_c, rho_pc):
    sigma2 = np.array([
        [sigma_p ** 2, rho_pc * sigma_p * sigma_c],
        [rho_pc * sigma_p * sigma_c, sigma_c ** 2],
    ])
    mu2 = np.array([mu_p, mu_c])
    f_star = np.linalg.solve(sigma2, mu2)
    return float(f_star[0]), float(f_star[1])


def size_candidate(candidate, weights, total_value, fraction=None):
    fraction = fraction if fraction is not None else config.KELLY_FRACTION
    mu_p, sigma_p, mu_c, sigma_c, rho_pc = candidate_mu_sigma_rho(candidate, weights)
    f_star_p, f_star_c = solve_kelly(mu_p, sigma_p, mu_c, sigma_c, rho_pc)

    f_half = fraction * f_star_c
    capped = f_half > config.KELLY_MAX_F or f_half < 0
    f_used = float(np.clip(f_half, 0, config.KELLY_MAX_F))
    suggested_dollars = f_used * total_value

    return {
        "candidate": candidate,
        "mu_portfolio": mu_p, "sigma_portfolio": sigma_p,
        "mu_candidate": mu_c, "sigma_candidate": sigma_c,
        "rho_portfolio_candidate": rho_pc,
        "f_star_candidate": f_star_c,
        "f_half_kelly": f_half,
        "f_used": f_used,
        "capped": capped,
        "suggested_dollars": suggested_dollars,
    }


def validate_with_stress_test(sizing_result, holdings, params, candidate_price):
    baseline_weights, _, total_value = stress_test.portfolio_weights(holdings)
    baseline_summary = stress_test.run_stress_test(baseline_weights, params)

    added_shares = sizing_result["suggested_dollars"] / candidate_price
    new_total = total_value + sizing_result["suggested_dollars"]

    new_weights = {t: (w * total_value) / new_total for t, w in baseline_weights.items()}
    candidate = sizing_result["candidate"]
    new_weights[candidate] = new_weights.get(candidate, 0.0) + sizing_result["suggested_dollars"] / new_total

    if candidate not in params["per_ticker"]:
        raise ValueError(f"{candidate} not in calibrated params - run data.calibrate_universe with it included first")

    after_summary = stress_test.run_stress_test(new_weights, params)

    return {
        "before": baseline_summary,
        "after": after_summary,
        "added_shares": added_shares,
        "new_weights": new_weights,
    }


def run(candidate, holdings_path=None, params_path=None, fraction=None):
    holdings = stress_test.load_holdings(holdings_path)
    weights, prices, total_value = stress_test.portfolio_weights(holdings)
    params = data.load_params(params_path)

    sizing = size_candidate(candidate, weights, total_value, fraction)
    candidate_price = data.latest_price(candidate)
    validation = validate_with_stress_test(sizing, holdings, params, candidate_price)

    return {"sizing": sizing, "validation": validation}


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    if not ticker:
        print("Usage: python kelly.py TICKER")
        sys.exit(1)
    result = run(ticker)
    print(json.dumps(result, indent=1))
