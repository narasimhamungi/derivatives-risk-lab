"""Cross-checks between the three pricers and between closed-form and
numerical Greeks: the table the report shows and the tests assert on."""
from __future__ import annotations

import pandas as pd

from derivlab import bs, mc, tree


def finite_difference(S, K, T, r, q, sigma, kind, h_spot=1e-3, h_vol=1e-4, h_t=1e-5, h_r=1e-5) -> dict:
    """Central differences of the closed-form price (relative bump for spot)."""
    def p(**kw):
        args = dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
        args.update(kw)
        return float(bs.price(args["S"], K, args["T"], args["r"], q, args["sigma"], kind))
    dS = S * h_spot
    return {
        "delta": (p(S=S + dS) - p(S=S - dS)) / (2 * dS),
        "gamma": (p(S=S + dS) - 2 * p() + p(S=S - dS)) / dS ** 2,
        "vega": (p(sigma=sigma + h_vol) - p(sigma=sigma - h_vol)) / (2 * h_vol),
        "theta": -(p(T=T + h_t) - p(T=T - h_t)) / (2 * h_t),
        "rho": (p(r=r + h_r) - p(r=r - h_r)) / (2 * h_r),
    }


def greeks_table(cfg: dict) -> pd.DataFrame:
    g = cfg["greeks_check"]
    S, r, q = g["spot"], g["rate"], g["div_yield"]
    rows = []
    for K, days, vol, kind in g["cases"]:
        T = days / 365.0
        closed = {k: float(v) for k, v in bs.greeks(S, K, T, r, q, vol, kind).items()}
        fd = finite_difference(S, K, T, r, q, vol, kind)
        tr = tree.crr(S, K, T, r, q, vol, g["tree_steps"], kind)
        m = mc.european(S, K, T, r, q, vol, kind, g["mc_paths"], seed=7)
        price = float(bs.price(S, K, T, r, q, vol, kind))
        parity = float(bs.price(S, K, T, r, q, vol, "call") - bs.price(S, K, T, r, q, vol, "put")) - \
            (S * pow(2.718281828459045, -q * T) - K * pow(2.718281828459045, -r * T))
        base = {"strike": K, "days": days, "vol": vol, "kind": kind}
        rows.append({**base, "measure": "price", "closed_form": price, "finite_difference": None,
                     "binomial": tr.price, "monte_carlo": m.price, "mc_std_error": m.std_error,
                     "put_call_parity_gap": parity})
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            rows.append({**base, "measure": greek, "closed_form": closed[greek], "finite_difference": fd[greek],
                         "binomial": {"delta": tr.delta, "gamma": tr.gamma, "theta": tr.theta}.get(greek),
                         "monte_carlo": {"delta": m.delta, "gamma": m.gamma, "vega": m.vega}.get(greek),
                         "mc_std_error": None, "put_call_parity_gap": None})
    return pd.DataFrame(rows)


def error_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Largest relative difference from the closed form, by measure and method."""
    out = []
    for measure, g in table.groupby("measure", sort=False):
        row = {"measure": measure}
        for col in ("finite_difference", "binomial", "monte_carlo"):
            d = g.dropna(subset=[col])
            row[col] = float(((d[col] - d["closed_form"]).abs() / d["closed_form"].abs().clip(lower=1e-12)).max()) \
                if len(d) else None
        out.append(row)
    return pd.DataFrame(out)
