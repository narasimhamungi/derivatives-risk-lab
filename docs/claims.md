# Claims ledger

Generated 2026-11-04 from the live run on marketdata-lakehouse gold layer (2026-10-30). Tags: Demonstrated / Inferred / Assumed / Unsupported.

| Claim | Tag | Evidence |
|---|---|---|
| Pricers agree with closed form (FD 7.6e-5, tree 7.4e-4 on gamma, MC 0.08% on price) | Demonstrated | `derivlab check` output, `price_validation.csv` |
| Hedging-error and realised-vs-implied results | Demonstrated (GBM simulation; not market evidence) | worked AAPL example, 63-day window |
| Book sensitivities and ladder | Demonstrated for the configured hypothetical book and assumed implied vols | report.md |
| Delta VaR fails on an options book (31 exceptions vs 23 full reval, Kupiec rejected) | Demonstrated on market data for this hypothetical book | backtest: 1,716 forecasts, 17.2 expected |
| Normal tail too thin in severity (loss/ES 1.60 vs 1.21) | Demonstrated, same scope | exception-day loss/ES figures |
| Vega risk in the VaR | Not modelled — implied vols are fixed; vega is in the ladder only | README must state this explicitly, not imply full Greeks coverage in VaR |
| Front-office trading or structuring experience | Not claimed — README already says so | — |

## Action
No overclaim found; README's existing caveat (capability demonstration, not front-office experience) and the vega-not-in-VaR point should both remain explicit. Re-run `derivlab run` once (open item in plan doc) so committed `outputs/` include `price_validation.csv` before this ledger is treated as final.
