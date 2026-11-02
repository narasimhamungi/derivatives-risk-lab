"""An options book defined by moneyness and days to expiry, so it is the same
book on every date (constant moneyness, constant maturity). Positions are
sized by underlying notional in dollars, which makes the book's P&L for a
given vector of returns independent of the price level:

  P&L = side * notional * (v(1 + r, m, T - h) - v(1, m, T))

with v the option price per unit of spot and m = strike / spot. Stock
positions are linear: side * notional * r.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from derivlab import bs

YEAR_DAYS = 365.0
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class Position:
    underlying: str
    kind: str            # call, put or stock
    side: float          # +1 long, -1 short
    notional: float      # $ of underlying
    moneyness: float = 1.0
    days: int = 0        # calendar days to expiry

    @property
    def T(self) -> float:
        return self.days / YEAR_DAYS


@dataclass(frozen=True)
class Market:
    rate: float
    div_yield: dict[str, float]
    vols: dict[str, float]


def load(cfg: dict) -> tuple[list[Position], Market]:
    m = cfg["market"]
    pos = []
    for p in cfg["book"]:
        side = {"long": 1.0, "short": -1.0}[p["side"]]
        if p["kind"] == "stock":
            pos.append(Position(p["underlying"], "stock", side, float(p["notional"])))
        else:
            pos.append(Position(p["underlying"], p["kind"], side, float(p["notional"]), float(p["moneyness"]),
                                int(p["days"])))
    unders = {p.underlying for p in pos}
    missing = sorted(u for u in unders if u not in m["vols"])
    if missing:
        raise ValueError(f"no implied volatility for {missing}")
    divs = {u: float(m.get("dividend_yields", {}).get(u, m.get("dividend_yield", 0.0))) for u in unders}
    return pos, Market(float(m["rate"]), divs, {u: float(m["vols"][u]) for u in unders})


def rename(positions: list[Position], market: Market, mapping: dict[str, str]) -> tuple[list[Position], Market]:
    pos = [Position(mapping.get(p.underlying, p.underlying), p.kind, p.side, p.notional, p.moneyness, p.days)
           for p in positions]
    return pos, Market(market.rate, {mapping.get(k, k): v for k, v in market.div_yield.items()},
                       {mapping.get(k, k): v for k, v in market.vols.items()})


def unit_value(p: Position, mk: Market, spot, T, vol_shift=0.0):
    """Value per unit of starting spot (spot = 1 + return)."""
    if p.kind == "stock":
        return np.asarray(spot, dtype=float)
    vol = mk.vols[p.underlying] + vol_shift
    return bs.price(spot, p.moneyness, T, mk.rate, mk.div_yield[p.underlying], vol, p.kind)


def pnl(positions: list[Position], mk: Market, returns: pd.DataFrame | dict, horizon_days: float = 0.0,
        vol_shift: float = 0.0) -> np.ndarray:
    """Full-revaluation P&L for each row of returns (one column per underlying).
    horizon_days: trading days of time decay applied."""
    total = None
    h = horizon_days / TRADING_DAYS
    for p in positions:
        r = np.asarray(returns[p.underlying], dtype=float)
        if p.kind == "stock":
            v = p.side * p.notional * r
        else:
            v0 = unit_value(p, mk, 1.0, p.T)
            v1 = unit_value(p, mk, 1.0 + r, max(p.T - h, 0.0), vol_shift)
            v = p.side * p.notional * (v1 - v0)
        total = v if total is None else total + v
    return total


def sensitivities(positions: list[Position], mk: Market) -> pd.DataFrame:
    """Per position: dollar delta, dollar gamma (change in dollar delta for a
    1% move), vega per vol point, theta per calendar day, rho per 1%."""
    rows = []
    for p in positions:
        if p.kind == "stock":
            rows.append({"underlying": p.underlying, "position": f"{'long' if p.side > 0 else 'short'} stock",
                         "notional": p.notional, "value": p.side * p.notional, "dollar_delta": p.side * p.notional,
                         "dollar_gamma_1pct": 0.0, "vega_per_point": 0.0, "theta_per_day": 0.0, "rho_per_pct": 0.0})
            continue
        g = bs.greeks(1.0, p.moneyness, p.T, mk.rate, mk.div_yield[p.underlying], mk.vols[p.underlying], p.kind)
        val = unit_value(p, mk, 1.0, p.T)
        k = p.side * p.notional
        rows.append({"underlying": p.underlying,
                     "position": f"{'long' if p.side > 0 else 'short'} {p.days}d {p.moneyness:.0%} {p.kind}",
                     "notional": p.notional, "value": float(k * val), "dollar_delta": float(k * g["delta"]),
                     "dollar_gamma_1pct": float(k * g["gamma"] * 0.01), "vega_per_point": float(k * g["vega"] * 0.01),
                     "theta_per_day": float(k * g["theta"] / YEAR_DAYS), "rho_per_pct": float(k * g["rho"] * 0.01)})
    return pd.DataFrame(rows)


def taylor_pnl(sens: pd.DataFrame, returns: pd.DataFrame | dict, horizon_days: float = 0.0,
               vol_shift: float = 0.0, order: int = 2) -> np.ndarray:
    """Delta (order 1) or delta-gamma (order 2) approximation, plus vega for a
    vol shift and theta for the horizon."""
    by = sens.groupby("underlying")[["dollar_delta", "dollar_gamma_1pct", "vega_per_point", "theta_per_day"]].sum()
    total = 0.0
    for u, row in by.iterrows():
        r = np.asarray(returns[u], dtype=float)
        total = total + row["dollar_delta"] * r
        if order >= 2:
            # dollar gamma per 1% move = Gamma * S^2 * 0.01 per unit; 1/2 Gamma (S r)^2 = 1/2 * (g1 / 0.01) * r^2
            total = total + 0.5 * row["dollar_gamma_1pct"] / 0.01 * r ** 2
        total = total + row["vega_per_point"] * vol_shift * 100 + row["theta_per_day"] * horizon_days * \
            (YEAR_DAYS / TRADING_DAYS)
    return np.asarray(total, dtype=float)


def ladder(positions: list[Position], mk: Market, spot_shocks: list[float], vol_shocks: list[float]) -> pd.DataFrame:
    """Instantaneous parallel shocks to every underlying: full revaluation and
    the delta-gamma-vega approximation."""
    sens = sensitivities(positions, mk)
    unders = sorted({p.underlying for p in positions})
    rows = []
    for dv in vol_shocks:
        for ds in spot_shocks:
            shock = {u: np.array([ds]) for u in unders}
            full = float(pnl(positions, mk, shock, 0.0, dv)[0])
            approx = float(taylor_pnl(sens, shock, 0.0, dv, order=2)[0])
            rows.append({"spot_shock": ds, "vol_shock": dv, "full_revaluation": full, "delta_gamma_vega": approx,
                         "approximation_error": approx - full})
    return pd.DataFrame(rows)
