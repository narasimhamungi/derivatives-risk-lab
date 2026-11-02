"""Monte Carlo for European options under geometric Brownian motion.

Antithetic draws and the discounted terminal stock price as a control
variate (its expectation is known exactly: S e^(-qT)). With both on, the
control-variate coefficient is estimated on the antithetic pair averages,
the same units the standard error is computed on. Pathwise estimators
for delta and vega; gamma by central differences with common random numbers.
Every estimate comes with its standard error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class McResult:
    price: float
    std_error: float
    delta: float
    vega: float
    gamma: float
    paths: int


def _terminal(S, T, r, q, sigma, z):
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)


def european(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str = "call",
             paths: int = 200_000, seed: int = 7, antithetic: bool = True, control_variate: bool = True) -> McResult:
    rng = np.random.default_rng(seed)
    n = paths // 2 if antithetic else paths
    z = rng.standard_normal(n)
    if antithetic:
        z = np.concatenate([z, -z])
    disc = np.exp(-r * T)
    st = _terminal(S, T, r, q, sigma, z)
    itm = st > K if kind == "call" else st < K
    pay = disc * (np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0))
    cv = disc * st
    if antithetic:   # the unit of observation is the average of each antithetic pair
        pay, cv = 0.5 * (pay[:n] + pay[n:]), 0.5 * (cv[:n] + cv[n:])
    if control_variate:   # coefficient estimated on the same units the standard error uses
        cov = np.cov(pay, cv)
        est = pay - cov[0, 1] / cov[1, 1] * (cv - S * np.exp(-q * T))
    else:
        est = pay
    se = est.std(ddof=1) / np.sqrt(len(est))
    sign = 1.0 if kind == "call" else -1.0
    delta = float(np.mean(disc * itm * sign * st / S))
    vega = float(np.mean(disc * itm * sign * st * (np.log(st / S) - (r - q + 0.5 * sigma ** 2) * T) / sigma))
    h = 0.01 * S
    up = _terminal(S + h, T, r, q, sigma, z)
    dn = _terminal(S - h, T, r, q, sigma, z)
    payoff = (lambda s: np.maximum(s - K, 0.0)) if kind == "call" else (lambda s: np.maximum(K - s, 0.0))
    gamma = float(disc * np.mean(payoff(up) - 2 * payoff(st) + payoff(dn)) / h ** 2)
    return McResult(float(est.mean()), float(se), delta, vega, gamma, len(z))
