## Set up + Usage

```
pip install -r requirements.txt
python setup_portfolio.py
```
```
python setup_portfolio.py      - Sets up portfolio
python stress_test.py          - VaR/CVaR on the current portfolio
python kelly.py [TICKER]       - size a candidate, re-validate VaR/CVaR with it added
python backtest.py             - walk-forward check. 
```

## Data
Yahoo Finance

-> Daily Close prices

-> 5 years of data

-> Cached unless 3 days old

## Output
stress_test.py and kelly.py print JSON. These commands runn in the folder's command terminal.
```
`python stress_test.py` — the Monte Carlo engine (Bates dynamics + shared-mixing-variable multivariate-t) and empirical VaR/CVaR on actual portfolio weights, at both a 10-day and 1-year horizon
`python kelly.py` — 2-asset continuous Kelly reduction (portfolio-as-one-asset vs. candidate), half-Kelly haircut, hard cap at `f_used <= 1.0`, then re-validated by re-running the full simulation with the candidate added
`python backtest.py` — walks backward through history, recalibrating with only the data available at each past date (no lookahead) and checking whether the realized return over the following 10 days actually breached the predicted VaR, at the rate the confidence level implies
```
