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
python backtest.py             # walk-forward check of whether the VaR estimates actually hold up
python publish.py              # re-run everything and publish it to the ALR admin panel
```

Re-run `setup_portfolio.py` (or `data.calibrate_universe([...])` directly) whenever holdings or candidates change, so the calibration cache in `state/` covers every ticker being simulated.

`publish.py` needs `ALR_ADMIN_EMAIL`/`ALR_ADMIN_PASSWORD` in a local `.env` file (copy `.env.example`) — it signs in as that account and posts the results to the admin quant page, which just displays whatever was last published. Nothing runs live on the site itself.

## Files

- `holdings.json` — actual positions, `[{"ticker": "AAPL", "shares": 40}, ...]`
- `candidates.json` — tickers being considered
- `config.py` — every tunable parameter (horizon, path count, seed count, jump threshold, Kelly fraction and cap)
- `data.py` — price fetch/cache, jump/Heston/Student's-t calibration from 5 years of daily history
- `stress_test.py` — the Monte Carlo engine (Bates dynamics + shared-mixing-variable multivariate-t) and empirical VaR/CVaR on actual portfolio weights, at both a 10-day and 1-year horizon
- `kelly.py` — 2-asset continuous Kelly reduction (portfolio-as-one-asset vs. candidate), half-Kelly haircut, hard cap at `f_used <= 1.0`, then re-validated by re-running the full simulation with the candidate added
- `backtest.py` — walks backward through history, recalibrating with only the data available at each past date (no lookahead) and checking whether the realized return over the following 10 days actually breached the predicted VaR, at the rate the confidence level implies
- `publish.py` — re-runs the stress test and every candidate's sizing, then posts the bundled result to the ALR admin panel

## Notes

- `f_star` from `kelly.py` is a closed-form Gaussian estimate. It knows nothing about the jumps, stochastic vol, or fat tails the simulation models — that's why every sizing run re-checks VaR/CVaR before/after through the actual simulation rather than trusting the formula alone.
- Jumps are split into two components: an idiosyncratic rate per ticker, and a shared "market jump" rate calibrated from how often multiple tickers jumped on the same historical day. A market jump hits every holding simultaneously in the simulation; idiosyncratic jumps stay independent per ticker.
- The correlation matrix is calibrated from a rolling 90-day window (`CORRELATION_WINDOW_DAYS`), not the full 5-year history, so it reflects more recent co-movement rather than a stale multi-year average.
- Heston variance is floored at zero each step (full truncation), since the Feller condition isn't guaranteed to hold for every ticker — this is a standard, widely-used technique for this situation, not a shortcut unique to this tool.
- VaR/CVaR describe a confidence level, not a hard ceiling; losses beyond the reported number remain possible by construction.
- `state/` holds cached prices and the last calibration; delete it to force a full refetch.
