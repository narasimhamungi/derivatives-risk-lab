"""Black-Scholes-Merton for European options on a stock with a continuous
dividend yield: price, closed-form Greeks and implied volatility.

All functions are vectorised over NumPy arrays. Units: T in years, rates and
volatility as annual fractions. Vega and rho are per unit (1.00 = 100 vol
points or 100% rate); theta is per year. `per_day` / `per_point` helpers
give the desk conventions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vt
    return d1, d1 - vt


def price(S, K, T, r, q, sigma, kind="call"):
    """Option price; at expiry (T <= 0) the intrinsic value."""
    S, K, T = np.asarray(S, dtype=float), np.asarray(K, dtype=float), np.asarray(T, dtype=float)
    call = np.asarray(kind) == "call"
    live = T > 0
    Tl = np.where(live, T, 1.0)
    d1, d2 = _d1d2(S, K, Tl, r, q, np.where(live, sigma, 1.0))
    df_q, df_r = np.exp(-q * Tl), np.exp(-r * Tl)
    c = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    p = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    value = np.where(call, c, p)
    intrinsic = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(live, value, intrinsic)


def greeks(S, K, T, r, q, sigma, kind="call") -> dict[str, np.ndarray]:
    """Closed-form delta, gamma, vega, theta (per year) and rho."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    call = np.asarray(kind) == "call"
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    sq = np.sqrt(T)
    delta = np.where(call, df_q * norm.cdf(d1), -df_q * norm.cdf(-d1))
    gamma = df_q * pdf / (S * sigma * sq)
    vega = S * df_q * pdf * sq
    common = -S * df_q * pdf * sigma / (2 * sq)
    theta = np.where(call, common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1),
                     common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1))
    rho = np.where(call, K * T * df_r * norm.cdf(d2), -K * T * df_r * norm.cdf(-d2))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def per_day(theta, days_per_year: float = 365.0):
    return np.asarray(theta) / days_per_year


def per_point(vega):
    """Vega for a one-volatility-point (0.01) move."""
    return np.asarray(vega) * 0.01


def implied_vol(target: float, S: float, K: float, T: float, r: float, q: float, kind: str = "call",
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Volatility that reproduces `target`; raises if the price is outside no-arbitrage bounds."""
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    lower = max(S * df_q - K * df_r, 0.0) if kind == "call" else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if kind == "call" else K * df_r
    if not lower - 1e-12 <= target <= upper + 1e-12:
        raise ValueError(f"price {target} outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]")
    return float(brentq(lambda s: float(price(S, K, T, r, q, s, kind)) - target, lo, hi, xtol=1e-12, rtol=1e-12))
