import json

import numpy as np

import config
import data
import stress_test


def portfolio_mu_sigma(weights, params, risk_free_rate=0.0):
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    mu_vec = np.array([params["per_ticker"][t]["mu"] - risk_free_rate for t in tickers])
    sigma_vec = np.array([params["per_ticker"][t]["sigma"] for t in tickers])
    try:
        corr = np.array([[params["correlation_matrix"][ti][tj] for tj in tickers] for ti in tickers])
    except KeyError as missing:
        raise ValueError(
            f"No correlation data for {missing} across the held portfolio. "
            f"Re-run calibration with all held tickers sharing enough overlapping history."
        ) from None
    cov = np.outer(sigma_vec, sigma_vec) * corr

    mu_p = float(w @ mu_vec)
    sigma_p_sq = float(w @ cov @ w)
    return mu_p, sigma_p_sq, tickers, w, sigma_vec


def candidate_mu_sigma_rho(candidate, weights, params, risk_free_rate=0.0):
    mu_p, sigma_p_sq, tickers, w, sigma_vec = portfolio_mu_sigma(weights, params, risk_free_rate)

    mu_c = params["per_ticker"][candidate]["mu"] - risk_free_rate
    sigma_c = params["per_ticker"][candidate]["sigma"]

    corr_row = params["correlation_matrix"].get(candidate, {})
    rho_to_each = np.array([corr_row.get(t, np.nan) for t in tickers])
    if np.any(np.isnan(rho_to_each)):
        missing = [t for t, r in zip(tickers, rho_to_each) if np.isnan(r)]
        raise ValueError(
            f"No correlation data between {candidate} and: {', '.join(missing)}. "
            f"Re-run calibration with all tickers sharing enough overlapping history."
        )

    sigma_p = float(np.sqrt(sigma_p_sq))
    cov_candidate_portfolio = float(np.sum(w * rho_to_each * sigma_c * sigma_vec))
    rho_pc = cov_candidate_portfolio / (sigma_c * sigma_p) if sigma_c > 0 and sigma_p > 0 else 0.0

    return mu_p, sigma_p, mu_c, sigma_c, rho_pc


def solve_kelly(mu_p, sigma_p, mu_c, sigma_c, rho_pc):
    sigma2 = np.array([
        [sigma_p ** 2, rho_pc * sigma_p * sigma_c],
        [rho_pc * sigma_p * sigma_c, sigma_c ** 2],
    ])
    mu2 = np.array([mu_p, mu_c])
    f_star = np.linalg.solve(sigma2, mu2)
    return float(f_star[0]), float(f_star[1])


def size_candidate(candidate, weights, total_value, params, fraction=None, risk_free_rate=None):
    fraction = fraction if fraction is not None else config.KELLY_FRACTION
    if risk_free_rate is None:
        risk_free_rate = data.risk_free_rate()
    mu_p, sigma_p, mu_c, sigma_c, rho_pc = candidate_mu_sigma_rho(candidate, weights, params, risk_free_rate)
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


def validate_with_stress_test(sizing_result, weights, total_value, params, candidate_price):
    added_shares = sizing_result["suggested_dollars"] / candidate_price
    new_total = total_value + sizing_result["suggested_dollars"]

    new_weights = {t: (w * total_value) / new_total for t, w in weights.items()}
    candidate = sizing_result["candidate"]
    new_weights[candidate] = new_weights.get(candidate, 0.0) + sizing_result["suggested_dollars"] / new_total

    before_10_day = stress_test.run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS)
    after_10_day = stress_test.run_stress_test(new_weights, params, horizon_days=config.HORIZON_DAYS)
    before_1_year = stress_test.run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS_LONG)
    after_1_year = stress_test.run_stress_test(new_weights, params, horizon_days=config.HORIZON_DAYS_LONG)

    return {
        "before_10_day": before_10_day,
        "after_10_day": after_10_day,
        "before_1_year": before_1_year,
        "after_1_year": after_1_year,
        "added_shares": added_shares,
        "new_weights": new_weights,
        "funding_assumption": "new_capital",
    }


def run(candidate, holdings_path=None, params_path=None, fraction=None):
    candidate = candidate.upper()
    holdings = stress_test.load_holdings(holdings_path)
    weights, prices, total_value = stress_test.portfolio_weights(holdings)
    params = data.load_params(params_path)

    if candidate not in params["per_ticker"]:
        raise ValueError(f"{candidate} not in calibrated params - run data.calibrate_universe with it included first")

    sizing = size_candidate(candidate, weights, total_value, params, fraction)
    candidate_price = data.latest_price(candidate)
    validation = validate_with_stress_test(sizing, weights, total_value, params, candidate_price)

    return {"sizing": sizing, "validation": validation}


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    if not ticker:
        print("Usage: python kelly.py TICKER")
        sys.exit(1)
    result = run(ticker)
    print(json.dumps(result, indent=1))
