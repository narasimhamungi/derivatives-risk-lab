"""Cox-Ross-Rubinstein binomial tree, European or American exercise.

Delta and gamma come from the first two layers of the tree and theta from
the two-step node, so the tree's Greeks are independent of the closed form
they are tested against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TreeResult:
    price: float
    delta: float
    gamma: float
    theta: float          # per year
    early_exercise: bool  # whether exercise was optimal at any node


def crr(S: float, K: float, T: float, r: float, q: float, sigma: float, steps: int = 500, kind: str = "call",
        american: bool = False) -> TreeResult:
    if steps < 3:
        raise ValueError("need at least three steps for the tree Greeks")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1); use more steps")
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    spot = S * u ** (steps - j) * d ** j
    payoff = np.maximum(spot - K, 0.0) if kind == "call" else np.maximum(K - spot, 0.0)
    values = payoff
    layers = {}
    exercised = False
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            s_n = S * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
            ex = np.maximum(s_n - K, 0.0) if kind == "call" else np.maximum(K - s_n, 0.0)
            exercised = exercised or bool(np.any(ex > values + 1e-12))
            values = np.maximum(values, ex)
        if n <= 2:
            layers[n] = values.copy()
    v0, v1, v2 = layers[0][0], layers[1], layers[2]
    su, sd = S * u, S * d
    delta = (v1[0] - v1[1]) / (su - sd)
    suu, sud, sdd = S * u * u, S, S * d * d
    gamma = ((v2[0] - v2[1]) / (suu - sud) - (v2[1] - v2[2]) / (sud - sdd)) / (0.5 * (suu - sdd))
    theta = (v2[1] - v0) / (2 * dt)
    return TreeResult(float(v0), float(delta), float(gamma), float(theta), exercised)
