"""Value-at-Risk for the options book, built on market-risk-lab.

Because the book is the same on every date, its one-day P&L is a fixed
function of the day's return vector. Each method therefore reduces to a
P&L series that market-risk-lab's own engine can roll:

  full revaluation HS   P&L from repricing every option (market_risk_lab.risk.rolling_hs)
  delta-gamma HS        the second-order Taylor P&L      (rolling_hs)
  delta HS              the first-order (linear) P&L     (rolling_hs)
  delta-normal          dollar deltas and the rolling covariance (rolling_moments, rolling_parametric)

Delta HS against full revaluation HS isolates the book's non-linearity (same
method, different P&L); delta-normal against delta HS isolates the normal
assumption (same P&L, different method).

and the realised P&L is the full revaluation on the day itself. Exceptions,
Kupiec's test and the Basel traffic light use market_risk_lab.backtest, so
the conventions (order-statistic VaR, exact ES, no look-ahead) are the ones
already tested there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from market_risk_lab.backtest import kupiec_pof, rolling_exception_count, traffic_light
from market_risk_lab.risk import rolling_hs, rolling_moments, rolling_parametric

from derivlab import book

METHODS = {"full_revaluation_hs": "Full revaluation, historical simulation",
           "delta_gamma_hs": "Delta-gamma, historical simulation",
           "delta_hs": "Delta (linear), historical simulation", "delta_normal": "Delta-normal"}
BASEL_ALPHA = 0.99   # the traffic light is defined on 99% VaR only


def pnl_series(positions, mk, returns: pd.DataFrame, horizon_days: float = 1.0) -> pd.DataFrame:
    sens = book.sensitivities(positions, mk)
    return pd.DataFrame({
        "full_revaluation": book.pnl(positions, mk, returns, horizon_days),
        "delta_gamma": book.taylor_pnl(sens, returns, horizon_days, order=2),
        "delta": book.taylor_pnl(sens, returns, horizon_days, order=1),
    }, index=returns.index)


def rolling(positions, mk, returns: pd.DataFrame, window: int, alphas: list[float],
            horizon_days: float = 1.0) -> pd.DataFrame:
    """One row per forecast day, method and level: VaR, ES, realised P&L, exception."""
    if window >= len(returns):
        raise ValueError(f"window {window} needs more than {len(returns)} returns")
    p = pnl_series(positions, mk, returns, horizon_days)
    sens = book.sensitivities(positions, mk)
    dd = sens.groupby("underlying")["dollar_delta"].sum()
    unders = list(dd.index)
    out = {
        "full_revaluation_hs": rolling_hs(-p["full_revaluation"].to_numpy(), window, alphas),
        "delta_gamma_hs": rolling_hs(-p["delta_gamma"].to_numpy(), window, alphas),
        "delta_hs": rolling_hs(-p["delta"].to_numpy(), window, alphas),
    }
    mu, cov = rolling_moments(returns[unders].to_numpy(), window)
    out["delta_normal"] = rolling_parametric(mu, cov, dd.to_numpy(), alphas, zero_mean=True)
    realised = p["full_revaluation"].to_numpy()[window:]
    frames = []
    for method, by_alpha in out.items():
        for a, (var, es) in by_alpha.items():
            frames.append(pd.DataFrame({"date": returns.index[window:], "method": method, "alpha": a, "var": var,
                                        "es": es, "realised_pnl": realised, "exception": -realised > var}))
    return pd.concat(frames, ignore_index=True)


def summary(tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, a), g in tbl.groupby(["method", "alpha"], sort=False):
        n, x = len(g), int(g["exception"].sum())
        _, pval = kupiec_pof(x, n, 1 - a)
        if np.isclose(a, BASEL_ALPHA):
            zones = pd.Series(traffic_light(rolling_exception_count(g["exception"].to_numpy()))).dropna()
        else:
            zones = pd.Series(dtype=object)
        hit = g[g["exception"]]
        rows.append({"method": method, "alpha": a, "forecasts": n, "exceptions": x, "expected": n * (1 - a),
                     "rate": x / n, "kupiec_p": pval, "kupiec_5pct": "rejected" if pval < 0.05 else "not rejected",
                     "red_share": float((zones == "red").mean()) if len(zones) else np.nan,
                     "latest_var": float(g["var"].iloc[-1]), "latest_es": float(g["es"].iloc[-1]),
                     "mean_var": float(g["var"].mean()),
                     "loss_over_es": float(-hit["realised_pnl"].mean() / hit["es"].mean()) if x else np.nan})
    return pd.DataFrame(rows)
