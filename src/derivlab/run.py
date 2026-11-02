"""The one command: validate the pricers, measure the book, run the hedging
studies and the options VaR backtest, write outputs/ and report.md."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from derivlab import book, charts, data, hedge, validation, var


def _r(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def run(cfg: dict, out_dir: Path, synthetic: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions, mk = book.load(cfg)
    unders = list(dict.fromkeys(p.underlying for p in positions))
    if synthetic:
        mapping = {u: f"SYN{i + 1}" for i, u in enumerate(unders)}
        positions, mk = book.rename(positions, mk, mapping)
        unders = [mapping[u] for u in unders]
        returns = data.synthetic_returns(unders)
        source, issues = {"source": "synthetic returns (invented)", "issues": {}}, []
        example = mapping.get(cfg["hedge"]["example_underlying"], unders[0])
    else:
        d = cfg["data"]
        returns, source, issues = data.lakehouse_returns(unders, d["start_date"], d.get("end_date"))
        example = cfg["hedge"]["example_underlying"]

    greeks = validation.greeks_table(cfg)
    errors = validation.error_summary(greeks)
    sens = book.sensitivities(positions, mk)
    lad = book.ladder(positions, mk, cfg["scenarios"]["spot_shocks"], cfg["scenarios"]["vol_shocks"])
    v = cfg["var"]
    tbl = var.rolling(positions, mk, returns, v["window"], v["alphas"], v["horizon_days"])
    vs = var.summary(tbl)

    h = cfg["hedge"]
    spec = hedge.HedgeSpec(h["spot"], h["strike"], h["days"], mk.rate, 0.0, h["implied_vol"], h["kind"],
                           h["cost_bps"])
    freq = hedge.frequency_study(spec, h["rebalance_days"], h["paths"], h["seed"])
    vols = hedge.vol_study(spec, h["realised_vols"], h["paths"], h["seed"])
    path_returns = returns[example].iloc[-h["days"]:]
    path = np.concatenate([[h["spot"]], h["spot"] * (1 + path_returns.to_numpy()).cumprod()])
    ex = hedge.simulate(spec, path, 1)
    ex_table = hedge.attribution_table(ex)
    realised = float(np.std(np.log(path[1:] / path[:-1]), ddof=1) * np.sqrt(hedge.TRADING_DAYS))

    greeks.to_csv(out / "greeks_validation.csv", index=False)
    sens.to_csv(out / "book_sensitivities.csv", index=False)
    lad.to_csv(out / "scenario_ladder.csv", index=False)
    tbl.to_csv(out / "var_backtest_daily.csv", index=False)
    vs.to_csv(out / "var_backtest_summary.csv", index=False)
    freq.to_csv(out / "hedge_frequency.csv", index=False)
    vols.to_csv(out / "hedge_realised_vol.csv", index=False)
    ex_table.to_csv(out / "hedge_example_attribution.csv")
    pd.DataFrame(issues, columns=["check", "severity", "action", "ticker", "date", "detail"]).to_csv(
        out / "price_validation.csv", index=False)
    files = [charts.ladder(lad, out / "scenario_ladder.png", synthetic),
             charts.hedge_frequency(freq, out / "hedge_frequency.png"),
             charts.hedge_vol(vols, out / "hedge_realised_vol.png"),
             charts.attribution(ex_table, out / "hedge_example_attribution.png", synthetic),
             charts.var_paths(tbl, max(v["alphas"]), out / "var_backtest.png", synthetic)]
    totals = sens[["value", "dollar_delta", "dollar_gamma_1pct", "vega_per_point", "theta_per_day", "rho_per_pct"]].sum()
    summary = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "synthetic": synthetic,
        "data": {**source, "returns": len(returns), "from": str(returns.index[0].date()),
                 "to": str(returns.index[-1].date()), "underlyings": unders},
        "greeks_max_relative_error": {r["measure"]: {k: _r(r[k]) for k in ("finite_difference", "binomial", "monte_carlo")}
                                      for _, r in errors.iterrows()},
        "book_totals": {k: float(x) for k, x in totals.items()},
        "ladder_extremes": {"worst": float(lad["full_revaluation"].min()),
                            "largest_approximation_error": float(lad["approximation_error"].abs().max())},
        "var": [{k: _r(x) for k, x in row.items()} for row in vs.to_dict(orient="records")],
        "hedge_frequency": freq.to_dict(orient="records"), "hedge_realised_vol": vols.to_dict(orient="records"),
        "hedge_example": {"underlying": example, "days": h["days"], "kind": h["kind"], "realised_vol": realised,
                          "implied_vol": h["implied_vol"],
                          "total": float(ex["total"][0]), "gamma": float(ex_table["gamma"].sum()),
                          "theta": float(ex_table["theta"].sum()), "carry": float(ex_table["carry"].sum()),
                          "residual": float(ex_table["residual"].sum()), "premium": float(ex["premium"][0])},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    from derivlab import report
    (out / "report.md").write_text(report.markdown(summary, vs, errors, sens, lad), encoding="utf-8")
    return {"summary": summary, "files": files + [out / "summary.json", out / "report.md"]}
