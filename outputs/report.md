# Derivatives risk lab: results

Data: marketdata-lakehouse gold layer via market-risk-lab; 1,966 daily returns, 2019-01-03 to 2026-10-28. Built 2026-11-01T19:10:52+00:00.

Price validation (market-risk-lab): outlier 49 (warnings: outlier returns are flagged and kept; returns next to a missing price or a calendar gap are dropped; each one is listed in `price_validation.csv`).

## Pricers and Greeks

Closed-form Greeks agree with central finite differences to within 7.6e-05 (largest relative difference). The binomial tree is within 0.02% of the closed form on price and 0.07% on gamma. Monte Carlo is within 0.08% on price and 1.94% on gamma; its gamma is a bump-and-revalue estimate on a kinked payoff, the noisiest number in the table.

| Measure | Finite difference | Binomial | Monte Carlo |
|---|---|---|---|
| price | n.a. | 2.06e-04 | 8.01e-04 |
| delta | 9.3e-06 | 5.01e-05 | 1.74e-03 |
| gamma | 7.6e-05 | 7.39e-04 | 1.94e-02 |
| vega | 7.9e-08 | n.a. | 1.99e-03 |
| theta | 2.2e-09 | 4.39e-04 | n.a. |
| rho | 9.5e-11 | n.a. | n.a. |

Largest relative difference from the closed form across the validation cases in `config/book.yml` (n.a.: that method does not produce the measure here).

## The book

Net dollar delta $1,105,619, dollar gamma -$256,383 per 1% move, vega -$4,539 per volatility point, theta $980 a calendar day: short gamma and short volatility overall.

| Underlying | Position | Value | Dollar delta | Dollar gamma (1%) | Vega (1 pt) | Theta (day) |
|---|---|---|---|---|---|---|
| AAPL | long 63d 100% call | 47,627 | 543,303 | 35,320 | 1,646 | -400 |
| AAPL | short 63d 90% put | -17,878 | 293,827 | -40,986 | -1,910 | 379 |
| JPM | long 91d 95% put | 72,487 | -894,043 | 86,521 | 5,177 | -631 |
| XOM | long 30d 100% call | 14,943 | 258,029 | 26,660 | 570 | -249 |
| XOM | long 30d 100% put | 14,697 | -240,576 | 26,660 | 570 | -241 |
| AMZN | short 63d 105% call | -74,028 | -812,684 | -56,586 | -3,223 | 925 |
| NEE | short 182d 90% put | -66,180 | 583,858 | -43,089 | -5,371 | 344 |
| CAT | long 91d 110% call | 36,958 | 436,835 | 35,449 | 2,563 | -432 |
| JNJ | short 30d 100% call | -39,512 | -1,028,920 | -163,166 | -2,280 | 667 |
| JNJ | short 30d 100% put | -38,037 | 965,990 | -163,166 | -2,280 | 618 |
| PG | long stock | 1,000,000 | 1,000,000 | 0 | 0 | 0 |

## Scenario ladder

The worst outcome in the grid is -$542,750, at a -20% move in every underlying with volatility +10 points. The delta-gamma-vega approximation is within $166 of full revaluation for moves of 2% or less and off by up to $344,317 at +20%, where the terms it drops matter.

| Spot shock | Full revaluation | Delta-gamma-vega | Approximation error |
|---|---|---|---|
| -20% | -500,755 | -733,891 | -233,136 |
| -10% | -210,958 | -238,754 | -27,796 |
| -5% | -85,670 | -87,329 | -1,659 |
| -2% | -27,248 | -27,240 | 8 |
| +0% | 0 | 0 | 0 |
| +2% | 17,150 | 16,985 | -166 |
| +5% | 27,228 | 23,233 | -3,995 |
| +10% | 23,075 | -17,630 | -40,705 |
| +20% | 9,802 | -291,643 | -301,446 |

Full revaluation by spot shock (rows) and volatility shock in points (columns):

| Spot | -10 | -5 | +0 | +5 | +10 |
|---|---|---|---|---|---|
| -20% | -456,410 | -477,966 | -500,755 | -522,534 | -542,750 |
| -10% | -170,014 | -191,952 | -210,958 | -229,969 | -250,335 |
| -5% | -57,925 | -68,231 | -85,670 | -107,443 | -131,608 |
| -2% | 9,972 | -5,812 | -27,248 | -51,338 | -76,905 |
| +0% | 40,063 | 21,523 | 0 | -23,595 | -48,595 |
| +2% | 46,602 | 34,753 | 17,150 | -3,755 | -26,795 |
| +5% | 35,643 | 35,437 | 27,228 | 13,046 | -5,093 |
| +10% | 20,863 | 22,991 | 23,075 | 18,717 | 9,549 |
| +20% | 5,865 | 8,798 | 9,802 | 9,324 | 7,287 |

## Delta hedging

A short at-the-money 63-day call (premium 5.47 on a spot of 100), sold at 25% implied volatility and hedged on paths with the same realised volatility, has a P&L standard deviation of 9.9% of the premium when hedged daily and 42.1% when hedged every 21 days, against 45.4% if the error grew exactly with the square root of the interval. The mean stays near zero: hedging less often adds noise, not expected loss.

| Rebalance every (days) | Hedges | Mean P&L | Std. dev. | Std. dev. / premium |
|---|---|---|---|---|
| 1 | 63 | 0.002 | 0.543 | 9.9% |
| 2 | 32 | 0.002 | 0.747 | 13.6% |
| 5 | 13 | 0.005 | 1.156 | 21.1% |
| 10 | 7 | -0.006 | 1.592 | 29.1% |
| 21 | 3 | -0.002 | 2.304 | 42.1% |

With the option sold at a fixed implied volatility and hedged daily, the mean P&L follows 1/2 x Gamma x S^2 x (implied^2 - realised^2) x dt summed along each path, to within 0.018 per option across these volatilities:

| Realised vol | Mean P&L | Expected from gamma | Std. dev. |
|---|---|---|---|
| 15% | 1.989 | 1.971 | 0.650 |
| 20% | 0.997 | 0.991 | 0.546 |
| 25% | -0.001 | 0.000 | 0.548 |
| 30% | -0.992 | -0.998 | 0.772 |
| 35% | -2.007 | -1.999 | 1.178 |

Worked example: the same short call, hedged daily along the last 63 daily returns of AAPL, whose realised volatility over the period was 24.4%. Total +0.730 per option = gamma -2.696 + theta +3.956 + carry -0.476 + residual -0.053 (higher-order terms). The breakdown by day is in `hedge_example_attribution.csv`.

## Options VaR backtest

At 99% over 1,716 forecast days (17.2 exceptions expected):

- Full revaluation HS: 23 exceptions (Kupiec not rejected).
- Delta-gamma HS: 21 (Kupiec not rejected); 2 fewer exceptions than full revaluation.
- Delta HS: 31 (Kupiec rejected). The same method on the linear P&L: 8 more exceptions than full revaluation, which is what ignoring the book's convexity costs.
- Delta-normal: 32 (Kupiec rejected). The same dollar deltas with a zero-mean normal distribution in place of the empirical one: 1 more exception than delta HS, which is what the normal assumption costs.

| Method | Level | Forecasts | Exceptions | Expected | Kupiec p | Kupiec at 5% | Red-zone share | Latest VaR | Loss / ES |
|---|---|---|---|---|---|---|---|---|---|
| Full revaluation HS | 99.0% | 1,716 | 23 | 17.2 | 0.1780 | not rejected | 0.0% | 81,809 | 1.21 |
| Full revaluation HS | 97.5% | 1,716 | 48 | 42.9 | 0.4389 | not rejected | n.a. | 53,217 | 1.17 |
| Delta-gamma HS | 99.0% | 1,716 | 21 | 17.2 | 0.3681 | not rejected | 0.0% | 81,957 | 1.28 |
| Delta-gamma HS | 97.5% | 1,716 | 48 | 42.9 | 0.4389 | not rejected | n.a. | 53,159 | 1.17 |
| Delta HS | 99.0% | 1,716 | 31 | 17.2 | 0.0026 | rejected | 0.0% | 59,090 | 1.21 |
| Delta HS | 97.5% | 1,716 | 54 | 42.9 | 0.0987 | not rejected | n.a. | 50,041 | 1.22 |
| Delta-normal | 99.0% | 1,716 | 32 | 17.2 | 0.0013 | rejected | 0.0% | 71,508 | 1.60 |
| Delta-normal | 97.5% | 1,716 | 44 | 42.9 | 0.8655 | not rejected | n.a. | 60,246 | 1.57 |

Red-zone share: share of days whose trailing 250 forecasts hold 10 or more exceptions, defined for 99% VaR only. Loss / ES: mean realised loss on exception days over the mean ES predicted for them.

Implied volatilities, rates and dividend yields are held fixed (the lakehouse has no option quotes), so the backtest measures the non-linearity in spot only; volatility risk is in the ladder, not in the VaR.
