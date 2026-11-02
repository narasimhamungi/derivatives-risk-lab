# derivatives-risk-lab

[![CI](https://github.com/narasimhamungi/derivatives-risk-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/narasimhamungi/derivatives-risk-lab/actions/workflows/ci.yml)

European option pricing three ways (Black-Scholes-Merton, a binomial tree, Monte Carlo) with every Greek tested against an independent estimate; a delta-hedging simulator whose P&L is split into gamma, theta, carry, rebalancing mismatch, costs and a residual; and an options book on the eight stocks of [market-risk-lab](https://github.com/narasimhamungi/market-risk-lab), measured with sensitivities, a spot-by-volatility scenario ladder and a VaR backtest that runs on market-risk-lab's own engine and data. The VaR backtest compares four methods arranged so that two pairs each isolate one modelling choice: full revaluation against a linear P&L, and the empirical distribution against a normal one.

This is a self-initiated applied finance/analytics portfolio project. It shows the pricing and risk mechanics behind an options book, not front-office trading or structuring experience: there are no option quotes, no volatility surface, no trade capture and no real positions. It has no connection to any employer, vendor or product.

## Run

Requirements: Python 3.12 or later and a clone of market-risk-lab next to this one.

```
pip install -e ../market-risk-lab -e ".[dev]"
derivlab check              # the three pricers against the closed form, about a second
derivlab run --synthetic    # the whole pipeline on invented returns, no database, about 15 seconds
derivlab run                # the same on the marketdata-lakehouse gold layer
```

`derivlab run` reads prices through market-risk-lab's `pull_prices` and `build_returns`, so it needs the lakehouse gold layer in Postgres and uses market-risk-lab's connection settings: `MRL_DSN`, or `MDL_DB_HOST`, `MDL_DB_PORT`, `MDL_DB_NAME`, `MDL_DB_USER` and `MDL_DB_PASSWORD` (defaults: the local development values published in marketdata-lakehouse). It reads only; nothing is written to the database. Outputs go to `outputs/` (synthetic runs to `outputs/synthetic/`, which is not committed): `report.md`, `summary.json`, nine CSVs (one lists every price-validation warning) and five charts.

```
pytest -q
```

The unit tests need nothing else. `tests/test_integration_pg.py` needs a Postgres database whose name ends in `_test` (set `MDL_DB_NAME`): it creates the lakehouse gold tables, fills them with seeded synthetic prices and runs the lakehouse path end to end. It refuses any other database because it truncates those tables. CI runs both on Python 3.12 and 3.14 against Postgres 16, with market-risk-lab pinned to commit `76e8776`.

## Results that do not depend on market data

Everything in this section comes from the configuration and seeded simulation, so `derivlab run --synthetic` reproduces it without the lakehouse.

**Pricers.** Largest relative difference from the closed form across six validation cases (calls and puts, 30 days to a year, strikes 90 to 120, volatilities 15% to 35%, with a dividend yield):

| Measure | Central finite difference | Binomial, 2,000 steps | Monte Carlo, 400,000 paths |
|---|---|---|---|
| Price | | 2.1e-04 | 8.0e-04 |
| Delta | 9.3e-06 | 5.0e-05 | 1.7e-03 |
| Gamma | 7.6e-05 | 7.4e-04 | 1.9e-02 |
| Vega | 7.9e-08 | | 2.0e-03 |
| Theta | 2.2e-09 | 4.4e-04 | |
| Rho | 9.5e-11 | | |

Put-call parity holds to within 1e-14. The Monte Carlo gamma is a bump-and-revalue estimate on a kinked payoff and is the least precise number in the table. The tests also price the textbook case (call 4.76, put 0.81), an American put (6.09, against 5.57 European) and an American call without dividends, which the tree correctly never exercises early.

**Delta hedging.** A short at-the-money 63-day call (premium 5.47 on a spot of 100) sold at 25% implied volatility and hedged on simulated paths with the same realised volatility:

| Rebalance every | 1 day | 2 days | 5 days | 10 days | 21 days |
|---|---|---|---|---|---|
| Std. dev. of P&L, % of premium | 9.9% | 13.6% | 21.1% | 29.1% | 42.1% |

The error grows close to the square root of the interval (45.4% at 21 days if it did so exactly) while the mean stays within simulation noise of zero. With implied fixed at 25% and realised volatility from 15% to 35%, the mean hedged P&L matches ½ Γ S² (σ²implied − σ²realised) Δt summed along each path to within 0.02 per option: +1.99 at 15% realised, −2.01 at 35%. On every simulated path the daily P&L sums to the hedge account's terminal value to within 1e-12.

**The book.** Ten option positions and one stock position on the eight stocks of market-risk-lab's equity portfolio, defined by moneyness and days to expiry (`config/book.yml`). Net dollar delta +$1.11m, dollar gamma −$256k per 1% move, vega −$4.5k per volatility point and theta +$980 a calendar day: short gamma and short volatility, earning time decay in return. In the ladder, a fall of 20% in every underlying loses $501k on full revaluation; the delta-gamma approximation says $734k. A rise of 20% gains $10k on full revaluation and loses $292k on the approximation. Within ±2% the two agree to $166.

## Results on the lakehouse

`derivlab run` on the marketdata-lakehouse gold layer: 1,966 daily returns for the eight stocks, 2019-01-03 to 2026-10-28. market-risk-lab's validation flagged 49 outlier returns and kept them (most in March 2020, the rest on days such as AAPL's +15% on 2025-04-09); no return was dropped. Full output: [`outputs/report.md`](outputs/report.md).

**Options VaR backtest**, one day, 99%, 1,716 forecasts (17.2 exceptions expected):

| Method | Exceptions | Kupiec p | Kupiec at 5% | Loss / ES on exception days |
|---|---|---|---|---|
| Full revaluation HS | 23 | 0.18 | not rejected | 1.21 |
| Delta-gamma HS | 21 | 0.37 | not rejected | 1.28 |
| Delta HS | 31 | 0.003 | rejected | 1.21 |
| Delta-normal | 32 | 0.001 | rejected | 1.60 |

- **On this book, linearity is the error that matters.** Dropping convexity (delta HS against full revaluation, same empirical scenarios) adds 8 exceptions and fails Kupiec. The gap opens in 2022 to 2026 (2022 alone: 8 against 4). In 2020 all four methods miss a similar number of days (6 to 9), a problem they share: the 250 days before March 2020 were calm.
- **The normal assumption costs severity more than frequency.** It adds one exception to delta HS, but on exception days the loss averaged 1.60 times the expected shortfall it had forecast, against 1.21 for full revaluation.
- Delta-gamma HS recovers full revaluation's result (21 against 23). No method reached the Basel red zone: the most exceptions in any 250 consecutive forecasts was 9 (delta-normal) and 7 (full revaluation). At 97.5% no method is rejected.

**Worked hedge.** The short at-the-money 63-day call hedged daily along AAPL's last 63 returns, whose realised volatility (24.4%) was just under the 25% implied: +0.730 per option = gamma −2.696 + theta +3.956 + carry −0.476 + residual −0.053.

## Methodology

**Pricers.** Black-Scholes-Merton with a continuous dividend yield, vectorised over NumPy arrays, with closed-form delta, gamma, vega, theta and rho and an implied-volatility solver (Brent's method, bounded by the no-arbitrage limits). Cox-Ross-Rubinstein tree with European or American exercise; its delta and gamma come from the first two layers and its theta from the two-step middle node, so they are independent of the closed form. Monte Carlo under geometric Brownian motion with antithetic draws and the discounted terminal stock price as a control variate (known expectation S e^(−qT)); with both on, the control-variate coefficient is estimated on the antithetic pair averages, the units the standard error is computed on. Delta and vega are pathwise estimates; gamma is a central difference with common random numbers.

**Delta hedging.** Sell one option at the implied volatility, hold the Black-Scholes delta in stock, finance the difference at r, receive dividends at q and rebalance every k trading days. Each day's P&L is split into gamma (−½ Γ ΔS²), theta (−Θ Δt), carry (interest and dividends), mismatch ((hedge delta − option delta) × ΔS, non-zero between rebalances), transaction costs and a residual for the higher-order terms. The frequency study holds realised volatility equal to implied; the volatility study holds implied fixed and varies realised. The worked example hedges the same option along a real 63-day path, rescaled to start at 100.

**The book.** Every position is defined by moneyness and calendar days to expiry, so it is the same book on every date (constant moneyness, constant maturity) and its one-day P&L is a fixed function of that day's return vector: side × notional × (v(1 + r, m, T − h) − v(1, m, T)), with v the option value per unit of spot. Dollar gamma is the change in dollar delta for a 1% move; vega is per volatility point; theta per calendar day. The ladder applies the same instantaneous shock to every underlying and compares full revaluation with delta-gamma-vega.

**VaR.** One-day VaR and ES at 99% and 97.5% on a 250-day window, four ways:

| Method | P&L | Distribution |
|---|---|---|
| Full revaluation HS | every option repriced, one trading day of time decay | empirical |
| Delta-gamma HS | delta, gamma and one day of theta | empirical |
| Delta HS | delta and one day of theta (linear) | empirical |
| Delta-normal | dollar deltas and the rolling covariance | normal, zero mean, no theta |

Delta HS against full revaluation HS changes only the P&L, so the difference in exceptions measures what the book's convexity does to a linear model. Delta-normal against delta HS keeps the dollar deltas and replaces the empirical distribution with a zero-mean normal one, so that difference measures the normal assumption, including ignoring drift and theta. The historical-simulation methods call market-risk-lab's `rolling_hs` (order-statistic VaR, exact ES, no look-ahead) and delta-normal calls its `rolling_moments` and `rolling_parametric` with dollar deltas as weights. Realised P&L is the full revaluation on the day; an exception is a loss above that day's VaR. Kupiec's test and the Basel traffic light (99% only) are market-risk-lab's functions. Tests require the forecast for day t to be unchanged when the return on day t is perturbed, the three historical-simulation methods to coincide on a linear book, and delta-normal to equal market-risk-lab's parametric VaR there.

## Limitations

**Model**
- No option market data. Implied volatilities, the rate and dividend yields are fixed assumptions in `config/book.yml`; there is no smile, no term structure and no volatility dynamics. The VaR therefore captures spot risk only, and vega risk appears only in the ladder.
- The constant-moneyness, constant-maturity book is rolled every day. It isolates the effect of non-linearity, but it is not the P&L of a book that ages and drifts in moneyness.
- European options in the book; American exercise only in the tree.
- The hedging studies assume geometric Brownian motion with constant volatility, no jumps and, by default, no transaction costs (`cost_bps` turns them on).
- Delta-normal is zero-mean with no theta term, as in its textbook form.

**Backtest** (inherited from market-risk-lab)
- Kupiec tests frequency only; there is no independence test, so clustering is visible in the chart but not tested.
- Small exception counts and low test power; several methods and levels are tested without a multiple-testing adjustment.
- One horizon (one day) and one window (250 days).

**Data**
- The same lakehouse data as market-risk-lab, with the same caveats: free sources, adjusted closes that are restated after later corporate actions, and tickers chosen in 2026 from current index members.

**Engineering**
- The VaR backtest needs market-risk-lab installed and the lakehouse database. The tests cover the lakehouse path against a Postgres fixture with synthetic prices, not the live database.
- No scheduling, monitoring or results store; outputs are files.

## Project structure

```
config/book.yml           market assumptions, the book, scenarios, VaR, hedging and validation settings
src/derivlab/
  bs.py                   Black-Scholes-Merton price, Greeks, implied volatility
  tree.py                 Cox-Ross-Rubinstein tree, European and American, tree Greeks
  mc.py                   Monte Carlo with antithetics and a control variate, pathwise Greeks
  validation.py           the three pricers and finite differences against the closed form
  hedge.py                delta-hedging simulator, P&L attribution, frequency and volatility studies
  book.py                 positions, full revaluation, sensitivities, Taylor P&L, scenario ladder
  var.py                  four VaR methods on market-risk-lab's engine, backtest summary
  data.py                 lakehouse returns via market-risk-lab; synthetic returns for tests
  charts.py, report.py    five charts and report.md
  run.py, cli.py          the pipeline and the derivlab command
tests/                    unit tests, Postgres integration test, gold-schema fixture
outputs/                  generated by derivlab run
```

## Licence

MIT. See `LICENSE`.
