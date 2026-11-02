"""Delta hedging a short European option, with P&L attribution.

The hedger sells one option at the implied volatility, holds the Black-Scholes
delta in stock, finances the rest in cash at r, receives dividends at q and
rebalances every `rebalance` steps. Over each step the hedged P&L splits into

  gamma     -1/2 * Gamma * dS^2           (short gamma loses on any move)
  theta     -Theta * dt                   (the option's time decay accrues to the seller)
  carry     interest on cash + dividends on the stock held
  mismatch  (hedge delta - option delta) * dS, non-zero between rebalances
  costs     transaction costs on rebalancing
  residual  everything else (higher-order terms, discretisation)

and theta + carry + gamma ~ 1/2 * Gamma * S^2 * (sigma_implied^2 - sigma_realised^2) * dt, which is why a
short-option hedger earns when realised volatility comes in below implied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from derivlab import bs

TRADING_DAYS = 252


@dataclass(frozen=True)
class HedgeSpec:
    spot: float
    strike: float
    days: int                 # trading days to expiry
    rate: float
    div_yield: float
    implied_vol: float
    kind: str = "call"
    cost_bps: float = 0.0     # per unit of stock traded


def gbm_paths(spot: float, days: int, mu: float, vol: float, paths: int, seed: int) -> np.ndarray:
    """(paths, days + 1) daily prices under GBM."""
    dt = 1 / TRADING_DAYS
    z = np.random.default_rng(seed).standard_normal((paths, days))
    steps = np.exp((mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z)
    return spot * np.concatenate([np.ones((paths, 1)), np.cumprod(steps, axis=1)], axis=1)


def simulate(spec: HedgeSpec, prices: np.ndarray, rebalance: int = 1) -> dict:
    """prices: (paths, days + 1) or (days + 1,). Returns totals by path and the
    per-step attribution (paths x days arrays)."""
    S = np.atleast_2d(np.asarray(prices, dtype=float))
    n_paths, n = S.shape[0], S.shape[1] - 1
    if n != spec.days:
        raise ValueError(f"price paths have {n} steps; the option has {spec.days} days")
    dt = 1 / TRADING_DAYS
    r, q, vol, K = spec.rate, spec.div_yield, spec.implied_vol, spec.strike
    T = (spec.days - np.arange(n + 1)) * dt
    V = np.stack([bs.price(S[:, i], K, T[i], r, q, vol, spec.kind) for i in range(n + 1)], axis=1)
    g = [bs.greeks(S[:, i], K, T[i], r, q, vol, spec.kind) if T[i] > 0 else None for i in range(n)]
    opt_delta = np.stack([gi["delta"] for gi in g], axis=1)
    gamma = np.stack([gi["gamma"] for gi in g], axis=1)
    theta = np.stack([gi["theta"] for gi in g], axis=1)
    hedge = np.empty((n_paths, n))
    cost = np.zeros((n_paths, n))
    held = np.zeros(n_paths)
    for i in range(n):
        if i % rebalance == 0:
            new = opt_delta[:, i]
            cost[:, i] = spec.cost_bps / 1e4 * np.abs(new - held) * S[:, i]
            held = new
        hedge[:, i] = held
    cash = np.empty((n_paths, n + 1))
    cash[:, 0] = V[:, 0] - hedge[:, 0] * S[:, 0] - cost[:, 0]
    carry = np.empty((n_paths, n))
    for i in range(n):
        interest = cash[:, i] * (np.exp(r * dt) - 1)
        divs = hedge[:, i] * q * S[:, i] * dt
        carry[:, i] = interest + divs
        nxt = cash[:, i] + carry[:, i]
        if i + 1 < n:
            nxt = nxt - (hedge[:, i + 1] - hedge[:, i]) * S[:, i + 1] - cost[:, i + 1]
        cash[:, i + 1] = nxt
    dS = np.diff(S, axis=1)
    step = -(V[:, 1:] - V[:, :-1]) + hedge * dS + carry - np.concatenate([cost[:, 1:], np.zeros((n_paths, 1))],
                                                                          axis=1)
    gamma_pnl = -0.5 * gamma * dS ** 2
    theta_pnl = -theta * dt
    mismatch = (hedge - opt_delta) * dS
    costs = -np.concatenate([cost[:, 1:], np.zeros((n_paths, 1))], axis=1)
    residual = step - (gamma_pnl + theta_pnl + carry + mismatch + costs)
    total = step.sum(axis=1) - cost[:, 0]
    final = cash[:, n] + hedge[:, n - 1] * S[:, n] - V[:, n]
    return {"total": total, "final_check": final, "premium": V[:, 0], "step": step, "gamma": gamma_pnl,
            "theta": theta_pnl, "carry": carry, "mismatch": mismatch, "costs": costs, "residual": residual,
            "initial_cost": cost[:, 0], "gamma_dollar": gamma, "prices": S}


def expected_gamma_pnl(spec: HedgeSpec, prices: np.ndarray, realised_vol: float) -> np.ndarray:
    """Path-wise 1/2 * sum Gamma * S^2 * (implied^2 - realised^2) * dt."""
    S = np.atleast_2d(prices)
    dt = 1 / TRADING_DAYS
    T = (spec.days - np.arange(spec.days)) * dt
    gam = np.stack([bs.greeks(S[:, i], spec.strike, T[i], spec.rate, spec.div_yield, spec.implied_vol,
                              spec.kind)["gamma"] for i in range(spec.days)], axis=1)
    return 0.5 * (gam * S[:, :-1] ** 2).sum(axis=1) * (spec.implied_vol ** 2 - realised_vol ** 2) * dt


def frequency_study(spec: HedgeSpec, rebalances: list[int], paths: int, seed: int) -> pd.DataFrame:
    """Hedging error against rebalancing frequency, realised = implied volatility."""
    S = gbm_paths(spec.spot, spec.days, spec.rate, spec.implied_vol, paths, seed)
    rows = []
    for k in rebalances:
        res = simulate(spec, S, k)
        prem = float(res["premium"][0])
        rows.append({"rebalance_days": k, "hedges": int(np.ceil(spec.days / k)), "mean_pnl": float(res["total"].mean()),
                     "std_pnl": float(res["total"].std(ddof=1)), "std_pct_premium": float(res["total"].std(ddof=1)) / prem,
                     "premium": prem})
    return pd.DataFrame(rows)


def vol_study(spec: HedgeSpec, realised: list[float], paths: int, seed: int) -> pd.DataFrame:
    """Mean hedged P&L against realised volatility, daily rebalancing."""
    rows = []
    for i, rv in enumerate(realised):
        S = gbm_paths(spec.spot, spec.days, spec.rate, rv, paths, seed + i)
        res = simulate(spec, S, 1)
        exp = expected_gamma_pnl(spec, S, rv)
        rows.append({"realised_vol": rv, "implied_vol": spec.implied_vol, "mean_pnl": float(res["total"].mean()),
                     "std_pnl": float(res["total"].std(ddof=1)), "expected_from_gamma": float(exp.mean()),
                     "premium": float(res["premium"][0])})
    return pd.DataFrame(rows)


def attribution_table(res: dict, path: int = 0) -> pd.DataFrame:
    keys = ["step", "gamma", "theta", "carry", "mismatch", "costs", "residual"]
    df = pd.DataFrame({k: res[k][path] for k in keys})
    df.insert(0, "spot", res["prices"][path, 1:])
    df.index.name = "day"
    df.index = df.index + 1
    return df
