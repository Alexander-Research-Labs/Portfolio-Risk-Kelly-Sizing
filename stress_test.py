import json
import sys

import numpy as np

import config
import data


def load_holdings(path=None):
    path = path or config.HOLDINGS_PATH
    with open(path) as f:
        return json.load(f)


def portfolio_weights(holdings):
    if not holdings:
        raise ValueError("No holdings to simulate. Run setup_portfolio.py first, or add entries to holdings.json.")
    tickers = sorted({h["ticker"] for h in holdings})
    prices = {t: data.latest_price(t) for t in tickers}
    values = {t: 0.0 for t in tickers}
    for h in holdings:
        values[h["ticker"]] += h["shares"] * prices[h["ticker"]]
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Portfolio total value is zero or negative - check share counts in holdings.json.")
    weights = {t: v / total for t, v in values.items()}
    return weights, prices, total


def build_correlation_cholesky(tickers, correlation_matrix):
    n = len(tickers)
    corr = np.eye(n)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            corr[i, j] = correlation_matrix[ti][tj]
    corr = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    return np.linalg.cholesky(corr_psd)


def require_ticker_coverage(tickers, params):
    missing = [t for t in tickers if t not in params["per_ticker"]]
    if missing:
        raise ValueError(
            f"Missing calibration for: {', '.join(missing)}. Run setup_portfolio.py or "
            f"data.calibrate_universe([...]) with these tickers included."
        )


def build_seed_invariant_inputs(tickers, params, horizon_days):
    chol_calm = build_correlation_cholesky(tickers, params["correlation_matrix"])
    stress_corr = params.get("stress_correlation_matrix", params["correlation_matrix"])
    chol_stress = build_correlation_cholesky(tickers, stress_corr)
    nu = params["nu"]
    t_scale = np.sqrt(nu / (nu - 2))

    mu = np.array([params["per_ticker"][t]["mu"] for t in tickers])
    lam_idio = np.array([params["per_ticker"][t]["lambda_idio"] for t in tickers])
    lam_market = params["per_ticker"][tickers[0]]["lambda_market"]
    mu_j = params["per_ticker"][tickers[0]]["mu_j"]
    sigma_j = params["per_ticker"][tickers[0]]["sigma_j"]
    k_comp = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    lam_total = lam_idio + lam_market

    kappa = np.array([params["per_ticker"][t]["heston"]["kappa"] for t in tickers])
    theta = np.array([params["per_ticker"][t]["heston"]["theta"] for t in tickers])
    xi = np.array([params["per_ticker"][t]["heston"]["xi"] for t in tickers])
    rho = np.array([params["per_ticker"][t]["heston"]["rho"] for t in tickers])
    v0 = np.array([params["per_ticker"][t]["heston"]["v0"] for t in tickers])

    dt = config.DT
    return {
        "chol_calm": chol_calm, "chol_stress": chol_stress, "nu": nu, "t_scale": t_scale,
        "mu_dt": (mu * dt)[None, :], "half_dt": 0.5 * dt,
        "lam_idio_dt": (lam_idio * dt)[None, :], "lam_market_dt": lam_market * dt,
        "lam_k_comp_dt": (lam_total * k_comp * dt)[None, :],
        "mu_j": mu_j, "sigma_j": sigma_j,
        "kappa_dt": (kappa * dt)[None, :], "theta": theta[None, :], "xi": xi[None, :], "rho": rho[None, :],
        "v0": v0, "dt": dt,
    }


def simulate_one_seed(tickers, inputs, weights_vec, horizon_days, n_paths, seed):
    rng = np.random.default_rng(seed)
    n = len(tickers)
    v = np.tile(inputs["v0"], (n_paths, 1))
    log_cum = np.zeros((n_paths, n))
    dt = inputs["dt"]

    for _ in range(horizon_days):
        v_now = v
        v_now_floored = np.clip(v_now, 0, None)
        stress_ratio = (v_now_floored / inputs["theta"]).mean(axis=1)
        stress_weight = np.clip((stress_ratio - 1) / config.CORRELATION_STRESS_RATIO_SPAN, 0, 1)

        z_indep = rng.standard_normal((n_paths, n))
        z_calm = z_indep @ inputs["chol_calm"].T
        z_stress = z_indep @ inputs["chol_stress"].T
        z_return = (1 - stress_weight)[:, None] * z_calm + stress_weight[:, None] * z_stress
        w_chi2 = rng.chisquare(inputs["nu"], size=n_paths)
        t_return = (z_return * np.sqrt(inputs["nu"] / w_chi2)[:, None]) / inputs["t_scale"]

        zv_indep = rng.standard_normal((n_paths, n))
        rho = inputs["rho"]
        zv = rho * z_return + np.sqrt(np.clip(1 - rho ** 2, 0, None)) * zv_indep

        n_jumps_idio = rng.poisson(np.broadcast_to(inputs["lam_idio_dt"], (n_paths, n)))
        z_jump_idio = rng.standard_normal((n_paths, n))
        idio_jump_contribution = n_jumps_idio * inputs["mu_j"] + inputs["sigma_j"] * np.sqrt(n_jumps_idio) * z_jump_idio

        n_jumps_market = rng.poisson(inputs["lam_market_dt"], size=n_paths)
        z_jump_market = rng.standard_normal((n_paths, n))
        market_jump_contribution = (
            n_jumps_market[:, None] * inputs["mu_j"]
            + inputs["sigma_j"] * np.sqrt(n_jumps_market)[:, None] * z_jump_market
        )

        jump_contribution = idio_jump_contribution + market_jump_contribution

        drift = inputs["mu_dt"] - inputs["half_dt"] * v_now_floored - inputs["lam_k_comp_dt"]
        diffusion = np.sqrt(v_now_floored * dt) * t_return
        log_cum += drift + diffusion + jump_contribution

        v = v_now + inputs["kappa_dt"] * (inputs["theta"] - v_now) + inputs["xi"] * np.sqrt(v_now_floored * dt) * zv
        v = np.clip(v, 0, None)

    simple_returns = np.exp(log_cum) - 1
    portfolio_returns = simple_returns @ weights_vec
    return portfolio_returns


def var_cvar(portfolio_returns, confidence):
    var = -np.percentile(portfolio_returns, 100 * (1 - confidence))
    tail = portfolio_returns[portfolio_returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def run_stress_test(weights, params, horizon_days=None, n_paths=None, n_seeds=None):
    horizon_days = horizon_days or config.HORIZON_DAYS
    n_paths = n_paths or config.N_SIMULATION_PATHS
    n_seeds = n_seeds or config.N_SIMULATION_SEEDS

    tickers = list(weights.keys())
    require_ticker_coverage(tickers, params)
    weights_vec = np.array([weights[t] for t in tickers])
    inputs = build_seed_invariant_inputs(tickers, params, horizon_days)

    horizon_label = "1-year" if horizon_days == config.HORIZON_DAYS_LONG else f"{horizon_days}-day"
    results = {c: [] for c in config.VAR_CONFIDENCE_LEVELS}
    for seed in range(n_seeds):
        print(f"\r{horizon_label} horizon: seed {seed + 1}/{n_seeds}", end="", flush=True, file=sys.stderr)
        portfolio_returns = simulate_one_seed(tickers, inputs, weights_vec, horizon_days, n_paths, seed)
        for c in config.VAR_CONFIDENCE_LEVELS:
            var, cvar = var_cvar(portfolio_returns, c)
            results[c].append((var, cvar))
    print(file=sys.stderr)

    summary = {}
    for c, pairs in results.items():
        vars_ = np.array([p[0] for p in pairs])
        cvars_ = np.array([p[1] for p in pairs])
        summary[c] = {
            "var_mean": float(vars_.mean()), "var_std": float(vars_.std()),
            "cvar_mean": float(cvars_.mean()), "cvar_std": float(cvars_.std()),
        }
    return summary


def run_from_holdings(holdings_path=None, params_path=None):
    holdings = load_holdings(holdings_path)
    weights, prices, total_value = portfolio_weights(holdings)
    params = data.load_params(params_path)
    summary_10_day = run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS)
    summary_1_year = run_stress_test(weights, params, horizon_days=config.HORIZON_DAYS_LONG)
    return {
        "weights": weights, "prices": prices, "total_value": total_value,
        "var_cvar_10_day": summary_10_day, "var_cvar_1_year": summary_1_year,
    }


if __name__ == "__main__":
    result = run_from_holdings()
    print(json.dumps(result, indent=1))
