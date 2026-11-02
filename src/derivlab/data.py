"""Daily returns for the book's underlyings.

`lakehouse`: market-risk-lab's own pipeline (the marketdata-lakehouse gold
layer, the same validation, simple returns), so the options book is measured
on exactly the data the equity VaR is. `synthetic`: invented returns with
volatility clustering and correlation, for tests and offline runs.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    pass


def lakehouse_returns(tickers: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Returns, a source description with issue counts, and every validation
    issue (written to price_validation.csv rather than logged line by line)."""
    from market_risk_lab.data import build_returns, pull_prices
    from market_risk_lab.store import connect

    conn = connect()
    try:
        prices, cal = pull_prices(conn, tickers, start, end)
    finally:
        conn.close()
    quiet = logging.getLogger("market_risk_lab.data")
    level = quiet.level
    quiet.setLevel(logging.ERROR)   # warnings go to the CSV and the report; failures still print
    try:
        returns, report = build_returns(prices, tickers, cal)
    finally:
        quiet.setLevel(level)
    return returns, {"source": "marketdata-lakehouse gold layer via market-risk-lab", "issues": report.counts()}, \
        list(report.issues)


def synthetic_returns(tickers: list[str], days: int = 1950, seed: int = 17) -> pd.DataFrame:
    """GARCH(1,1)-style volatility around a daily level drawn between 1.2% and
    2.2% (about 19% to 35% a year), one common factor; columns named as given."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    base = rng.uniform(0.012, 0.022, n)
    alpha, beta = 0.08, 0.90
    omega = 1 - alpha - beta          # unconditional variance multiplier of 1
    var = np.ones(n)
    out = np.empty((days, n))
    for t in range(days):
        common = rng.standard_normal()
        idio = rng.standard_normal(n)
        z = 0.55 * common + np.sqrt(1 - 0.55 ** 2) * idio
        shock = np.sqrt(var) * z
        out[t] = base * shock + 0.0003
        var = omega + alpha * shock ** 2 + beta * var
    idx = pd.bdate_range("2019-01-03", periods=days)
    return pd.DataFrame(out, index=idx, columns=tickers)


def prices_from_returns(returns: pd.Series, start: float = 100.0) -> pd.Series:
    return start * (1 + returns).cumprod()
