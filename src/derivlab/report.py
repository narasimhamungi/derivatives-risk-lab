"""report.md from a run: pricer checks, the book, the ladder, hedging and the
options VaR backtest, each with the finding first."""
from __future__ import annotations

import math

import pandas as pd

LABELS = {"full_revaluation_hs": "Full revaluation HS", "delta_gamma_hs": "Delta-gamma HS",
          "delta_hs": "Delta HS", "delta_normal": "Delta-normal"}


def _t(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _na(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _m(x, d=0):
    return "n.a." if _na(x) else f"{x:,.{d}f}"


def _usd(x):
    return "n.a." if _na(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def _e(x, fmt):
    return "n.a." if _na(x) else format(x, fmt)


def _p(x):
    return "n.a." if _na(x) else ("<0.0001" if x < 1e-4 else f"{x:.4f}")


def _pricers(s, errors):
    g = s["greeks_max_relative_error"]
    fd = max(v["finite_difference"] for v in g.values() if not _na(v["finite_difference"]))
    L = ["## Pricers and Greeks", "",
         f"Closed-form Greeks agree with central finite differences to within {fd:.1e} (largest relative "
         f"difference). The binomial tree is within {g['price']['binomial']:.2%} of the closed form on price and "
         f"{g['gamma']['binomial']:.2%} on gamma. Monte Carlo is within {g['price']['monte_carlo']:.2%} on price "
         f"and {g['gamma']['monte_carlo']:.2%} on gamma; its gamma is a bump-and-revalue estimate on a kinked "
         "payoff, the noisiest number in the table.", ""]
    L += _t(["Measure", "Finite difference", "Binomial", "Monte Carlo"],
            [[r["measure"], _e(r["finite_difference"], ".1e"), _e(r["binomial"], ".2e"), _e(r["monte_carlo"], ".2e")]
             for _, r in errors.iterrows()])
    return L + ["", "Largest relative difference from the closed form across the validation cases in "
                    "`config/book.yml` (n.a.: that method does not produce the measure here)."]


def _book(s, sens, lad):
    b = s["book_totals"]
    L = ["## The book", "",
         f"Net dollar delta {_usd(b['dollar_delta'])}, dollar gamma {_usd(b['dollar_gamma_1pct'])} per 1% move, vega "
         f"{_usd(b['vega_per_point'])} per volatility point, theta {_usd(b['theta_per_day'])} a calendar day: "
         f"{'short' if b['dollar_gamma_1pct'] < 0 else 'long'} gamma and "
         f"{'short' if b['vega_per_point'] < 0 else 'long'} volatility overall.", ""]
    L += _t(["Underlying", "Position", "Value", "Dollar delta", "Dollar gamma (1%)", "Vega (1 pt)", "Theta (day)"],
            [[r.underlying, r.position, _m(r.value), _m(r.dollar_delta), _m(r.dollar_gamma_1pct),
              _m(r.vega_per_point), _m(r.theta_per_day)] for r in sens.itertuples(index=False)])
    base = lad[lad["vol_shock"] == 0].sort_values("spot_shock")
    worst = lad.loc[lad["full_revaluation"].idxmin()]
    err = lad.loc[lad["approximation_error"].abs().idxmax()]
    L += ["", "## Scenario ladder", "",
          f"The worst outcome in the grid is {_usd(worst['full_revaluation'])}, at a {worst['spot_shock']:+.0%} move in "
          f"every underlying with volatility {worst['vol_shock'] * 100:+.0f} points. The delta-gamma-vega "
          f"approximation is within {_usd(base.loc[base['spot_shock'].abs() <= 0.02, 'approximation_error'].abs().max())} "
          f"of full revaluation for moves of 2% or less and off by up to {_usd(abs(err['approximation_error']))} at "
          f"{err['spot_shock']:+.0%}, where the terms it drops matter.", ""]
    L += _t(["Spot shock", "Full revaluation", "Delta-gamma-vega", "Approximation error"],
            [[f"{r.spot_shock:+.0%}", _m(r.full_revaluation), _m(r.delta_gamma_vega), _m(r.approximation_error)]
             for r in base.itertuples(index=False)])
    piv = lad.pivot(index="spot_shock", columns="vol_shock", values="full_revaluation")
    L += ["", "Full revaluation by spot shock (rows) and volatility shock in points (columns):", ""]
    L += _t(["Spot"] + [f"{c * 100:+.0f}" for c in piv.columns],
            [[f"{i:+.0%}"] + [_m(v) for v in row] for i, row in piv.iterrows()])
    return L


def _every(k) -> str:
    return "daily" if int(k) == 1 else f"every {int(k)} days"


def _hedging(s):
    fq = pd.DataFrame(s["hedge_frequency"])
    e = s["hedge_example"]
    first, last = fq.iloc[0], fq.iloc[-1]
    rule = first["std_pct_premium"] * math.sqrt(last["rebalance_days"] / first["rebalance_days"])
    L = ["## Delta hedging", "",
         f"A short at-the-money {e['days']}-day {e['kind']} (premium {first['premium']:.2f} on a spot of 100), sold "
         f"at {e['implied_vol']:.0%} implied volatility and hedged on paths with the same realised volatility, has a "
         f"P&L standard deviation of {first['std_pct_premium']:.1%} of the premium when hedged "
         f"{_every(first['rebalance_days'])} and {last['std_pct_premium']:.1%} when hedged "
         f"{_every(last['rebalance_days'])}, against {rule:.1%} if the error grew exactly with the square root "
         "of the interval. The mean stays near zero: hedging less often adds noise, not expected loss.", ""]
    L += _t(["Rebalance every (days)", "Hedges", "Mean P&L", "Std. dev.", "Std. dev. / premium"],
            [[int(r.rebalance_days), int(r.hedges), f"{r.mean_pnl:.3f}", f"{r.std_pnl:.3f}", f"{r.std_pct_premium:.1%}"]
             for r in fq.itertuples(index=False)])
    vv = pd.DataFrame(s["hedge_realised_vol"])
    gap = (vv["mean_pnl"] - vv["expected_from_gamma"]).abs().max()
    L += ["", "With the option sold at a fixed implied volatility and hedged daily, the mean P&L follows "
              "1/2 x Gamma x S^2 x (implied^2 - realised^2) x dt summed along each path, to within "
              f"{gap:.3f} per option across these volatilities:", ""]
    L += _t(["Realised vol", "Mean P&L", "Expected from gamma", "Std. dev."],
            [[f"{r.realised_vol:.0%}", f"{r.mean_pnl:.3f}", f"{r.expected_from_gamma:.3f}", f"{r.std_pnl:.3f}"]
             for r in vv.itertuples(index=False)])
    L += ["", f"Worked example: the same short {e['kind']}, hedged daily along the last {e['days']} daily returns of "
              f"{e['underlying']}, whose realised volatility over the period was {e['realised_vol']:.1%}. Total "
              f"{e['total']:+.3f} per option = gamma {e['gamma']:+.3f} + theta {e['theta']:+.3f} + carry "
              f"{e['carry']:+.3f} + residual {e['residual']:+.3f} (higher-order terms). The breakdown by day is in "
              "`hedge_example_attribution.csv`."]
    return L


def _var(vs):
    top = vs[vs["alpha"] == vs["alpha"].max()].set_index("method")
    n, exp = int(top["forecasts"].iloc[0]), float(top["expected"].iloc[0])

    def x(m):
        return int(top.loc[m, "exceptions"])

    def k(m):
        return f"Kupiec {top.loc[m, 'kupiec_5pct']}"

    def diff(a, b, name):
        d = x(a) - x(b)
        noun = "exception" if abs(d) == 1 else "exceptions"
        return f"the same count as {name}" if d == 0 else f"{abs(d)} {'more' if d > 0 else 'fewer'} {noun} than {name}"

    conv, norm = x("delta_hs") - x("full_revaluation_hs"), x("delta_normal") - x("delta_hs")
    L = ["## Options VaR backtest", "",
         f"At {vs['alpha'].max():.0%} over {n:,} forecast days ({exp:.1f} exceptions expected):", "",
         f"- Full revaluation HS: {x('full_revaluation_hs')} exceptions ({k('full_revaluation_hs')}).",
         f"- Delta-gamma HS: {x('delta_gamma_hs')} ({k('delta_gamma_hs')}); "
         f"{diff('delta_gamma_hs', 'full_revaluation_hs', 'full revaluation')}.",
         f"- Delta HS: {x('delta_hs')} ({k('delta_hs')}). The same method on the linear P&L: "
         f"{diff('delta_hs', 'full_revaluation_hs', 'full revaluation')}"
         + (", which is what ignoring the book's convexity costs." if conv > 0 else "."),
         f"- Delta-normal: {x('delta_normal')} ({k('delta_normal')}). The same dollar deltas with a zero-mean "
         f"normal distribution in place of the empirical one: {diff('delta_normal', 'delta_hs', 'delta HS')}"
         + (", so here the normal model is the more conservative of the two." if norm < 0 else
            ", which is what the normal assumption costs." if norm > 0 else "."), ""]
    L += _t(["Method", "Level", "Forecasts", "Exceptions", "Expected", "Kupiec p", "Kupiec at 5%", "Red-zone share",
             "Latest VaR", "Loss / ES"],
            [[LABELS[r.method], f"{r.alpha:.1%}", f"{r.forecasts:,}", r.exceptions, f"{r.expected:.1f}",
              _p(r.kupiec_p), r.kupiec_5pct, _e(r.red_share, ".1%"), _m(r.latest_var), _e(r.loss_over_es, ".2f")]
             for r in vs.itertuples(index=False)])
    L += ["", "Red-zone share: share of days whose trailing 250 forecasts hold 10 or more exceptions, defined for 99% "
              "VaR only. Loss / ES: mean realised loss on exception days over the mean ES predicted for them.", "",
          "Implied volatilities, rates and dividend yields are held fixed (the lakehouse has no option quotes), so "
          "the backtest measures the non-linearity in spot only; volatility risk is in the ladder, not in the VaR."]
    return L


def markdown(s: dict, vs: pd.DataFrame, errors: pd.DataFrame, sens: pd.DataFrame, lad: pd.DataFrame) -> str:
    L = ["# Derivatives risk lab: results", ""]
    if s["synthetic"]:
        L += ["**SYNTHETIC DATA: invented returns and renamed underlyings, for testing the pipeline.**", ""]
    d = s["data"]
    L.append(f"Data: {d['source']}; {d['returns']:,} daily returns, {d['from']} to {d['to']}. Built {s['built_at']}.")
    if not s["synthetic"]:
        issues = d.get("issues") or {}
        L += ["", "Price validation (market-risk-lab): " + (
            ", ".join(f"{k} {v}" for k, v in sorted(issues.items())) + " (warnings: outlier returns are flagged and "
            "kept; returns next to a missing price or a calendar gap are dropped; each one is listed in "
            "`price_validation.csv`)" if issues else "no warnings") + "."]
    L.append("")
    for part in (_pricers(s, errors), _book(s, sens, lad), _hedging(s), _var(vs)):
        L += part + [""]
    return "\n".join(L).rstrip() + "\n"
