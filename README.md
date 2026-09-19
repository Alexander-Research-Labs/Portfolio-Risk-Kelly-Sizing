# Portfolio Risk & Kelly Sizing

Monte Carlo stress-testing and Kelly-optimal position sizing for the actual held portfolio. Buy-only extension of the GBM/Merton/Heston/Student's-t research model — no rebalancing, no ERC.

## Setup

```
pip install -r requirements.txt
python setup_portfolio.py
```

This prompts for current holdings and candidate tickers, writes `holdings.json`/`candidates.json`, and runs the initial calibration.

## Usage

```
python stress_test.py          # VaR/CVaR on the current portfolio
python kelly.py TICKER         # size a candidate, re-validate VaR/CVaR with it added
```

Re-run `setup_portfolio.py` (or `data.calibrate_universe([...])` directly) whenever holdings or candidates change, so the calibration cache in `state/` covers every ticker being simulated.

## Files

- `holdings.json` — actual positions, `[{"ticker": "AAPL", "shares": 40}, ...]`
- `candidates.json` — tickers being considered
- `config.py` — every tunable parameter (horizon, path count, seed count, jump threshold, Kelly fraction and cap)
- `data.py` — price fetch/cache, jump/Heston/Student's-t calibration from 5 years of daily history
- `stress_test.py` — the Monte Carlo engine (Bates dynamics + shared-mixing-variable multivariate-t) and empirical VaR/CVaR on actual portfolio weights
- `kelly.py` — 2-asset continuous Kelly reduction (portfolio-as-one-asset vs. candidate), half-Kelly haircut, hard cap at `f_used <= 1.0`, then re-validated by re-running the full simulation with the candidate added

## Notes

- `f_star` from `kelly.py` is a closed-form Gaussian estimate. It knows nothing about the jumps, stochastic vol, or fat tails the simulation models — that's why every sizing run re-checks VaR/CVaR before/after through the actual simulation rather than trusting the formula alone.
- Jumps are simulated independently per asset, not as shared market-wide co-jumps.
- Heston variance is floored at zero each step (full truncation), since the Feller condition isn't guaranteed to hold for every ticker.
- `state/` holds cached prices and the last calibration; delete it to force a full refetch.
