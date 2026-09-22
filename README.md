# Portfolio Risk Management 
Discalimer: Informational and educational purposes only. Nothing in this repository, including its VaR/CVaR output, Kelly sizing suggestions, or any other result, is a recommendation, offer, or solicitation to buy, sell, or hold any security, and none of it should be construed as investment, financial, legal, or tax advice. This is a personal project, not a product of a registered investment adviser, broker-dealer, or fiduciary. The code and its output are provided "AS IS," without warranty of any kind: simulated results depend entirely on model assumptions and calibration choices, may contain errors, and are not guaranteed to reflect real-world outcomes, past or future. Anyone using this code is solely responsible for their own investment decisions and should consult a licensed financial professional before acting on anything produced by it.

## Purpose

The purpose of this repository is to showcase my design of stress-testing my portfolio and to determine the most optimal quantity of an equity to buy with AI-assisted code.

## Backstory + Functionality

This project started when I wanted to solve the problem of bleeding money as a value investor. I wanted a way to have peace of mind and estimate how much I could lose, within a given confidence level, over the next 10 days as well as over the next year. I designed a model that uses geometric Brownian motion with modified parameters (jump-diffusion, stochastic volatility, and Student's t distribution returns) because stocks aren't random walks but rather dynamic prices influenced by humanistic patterns and irrationality (I guess the two are inherently the same but I digress). The combined model is essentially a Bates model (minus me adding Student's t shocks on top of it).

Before the Monte Carlo simulation is run, it pulls 5 years of daily price history for every ticker added to the watchlist. The simulation is Monte Carlo style and generates 500,000 simulated joint paths across all holdings simultaneously, per seed. This is run for two time horizons: the next 10 days as well as the next year. Each day simulated, every path gets a continuous drift-and-volatility move using that day's simulated Heston volatility, a correlated Student's t random shock, and a chance of a jump; either idiosyncratic to that ticker, or a shared market-wide jump that hits every holding at once, calibrated from how often multiple tickers historically jumped on the same day. That correlation isn't a fixed rate either. It blends between a calm and a stress regime each simulated day, to simulate real-world market conditions where stocks that normally aren't strongly correlated tend to move together during choppy markets. This runs independently across 20 seeds (10,000,000 total paths). When finished it calculates VaR and CVaR (Value at Risk and Conditional Value at Risk) at 95% and 99% confidence levels. 

To size new positions, it uses the continuous Kelly criterion and treats the portfolio as one existing asset and the candidate equity as a second, to find the optimal size that maximizes long-run growth given the candidate equity's expected return, volatility, and correlation to the portfolio. Since the closed-form Kelly formula doesn't account for jumps or fat tails (unlike the simulation itself), the design takes half of the recommendation given and sets a hard limit of zero leverage. It then runs the full simulation again to confirm the VaR/CVaR before reporting findings.

To check whether the model's confidence levels are actually correct, it backtests itself. It walks backward through history, refitting using only the data that would've been available at each past date (To remove lookahead bias) and predict what the 10-day VaR would've been. Finally it checks how often the real 10 day outcome actually exceeded that prediction, compared to how often it should have. (A well-calibrated 95% VaR should be breached about 5% of the time historically; 99% VaR about 1% of the time.)

The general equation for continuous Kelly criterion is as follows:

```
f* = μ / σ²
```

The equation used in my design is:

```
f* = Σ⁻¹ · μ
```

Where:

```
f*  = [f_p*, f_c*]                                           - the optimal capital sizing for each of the two "assets" listed above
μ   = [μ_p, μ_c]                                             - expected annual return of each

                       | Portfolio      Candidate Equity |
Σ  =      Portfolio    | σ_p²           ρ·σ_p·σ_c        |   - the 2×2 covariance matrix
     Candidate Equity  | ρ·σ_p·σ_c      σ_c²             |
```

Variables are as follows:

- **μ_p** — the existing portfolio's expected annual return
- **σ_p** — the existing portfolio's annual volatility
- **μ_c** — the candidate equity's expected annual return
- **σ_c** — the candidate equity's annual volatility
- **ρ** — the correlation between the portfolio's returns and the candidate's returns
- **f_c\*** — what fraction of total capital to put into the candidate to maximize long-run (geometric) growth rate

To remove leverage, the variable `KELLY_MAX_F = 1.0` locks the position size to no more than 100% of the total portfolio's size. The same clamp also floors the size at 0%, so the tool can never suggest shorting a candidate either.


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

## Known Flaws/Limitations
No backtesting for CVaR since there are too few data points.
Variance should mathematically never go negative, but the math used to simulate it accidentally produces a negative number. Instead of a more complex fix, the tool just changes it to zero whenever that happens. Not a unique issue to this tool and not something that needs fixing.
VaR/CVaR describe a confidence level, losses beyond the reported number remain possible by construction.

## Output
stress_test.py and kelly.py print JSON. These commands runn in the folder's command terminal.
```
`python stress_test.py` — the Monte Carlo engine (Bates dynamics + shared-mixing-variable multivariate-t) and empirical VaR/CVaR on actual portfolio weights, at both a 10-day and 1-year horizon
`python kelly.py` — 2-asset continuous Kelly reduction (portfolio-as-one-asset vs. candidate), half-Kelly haircut, hard cap at `f_used <= 1.0`, then re-validated by re-running the full simulation with the candidate added
`python backtest.py` — walks backward through history, recalibrating with only the data available at each past date (no lookahead) and checking whether the realized return over the following 10 days actually breached the predicted VaR, at the rate the confidence level implies
```
## Other

This research project was made after my research project on the repository "Geometric-Brownian-Motion-for-VaR-and-Equal-Risk-Contribution," so excuse the "out of order" feel and similar study topic and method it might have. I may or may not release that project.

Originally built for my private portfolio tools.
