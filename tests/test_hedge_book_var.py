import numpy as np
import pandas as pd
import pytest
from market_risk_lab.risk import rolling_moments, rolling_parametric

from derivlab import book, data, hedge, var

SPEC = hedge.HedgeSpec(spot=100, strike=100, days=63, rate=0.04, div_yield=0.0, implied_vol=0.25, kind="call")


@pytest.fixture(scope="module")
def paths():
    return hedge.gbm_paths(100, 63, 0.04, 0.25, 4000, seed=1)


# --- delta hedging -------------------------------------------------------------------------------

def test_hedge_is_self_financing(paths):
    res = hedge.simulate(SPEC, paths, 1)
    # terminal wealth of cash + stock - option equals the sum of the step P&Ls
    assert np.allclose(res["total"], res["final_check"], atol=1e-10)


def test_attribution_adds_up_and_residual_is_small(paths):
    res = hedge.simulate(SPEC, paths, 1)
    parts = res["gamma"] + res["theta"] + res["carry"] + res["mismatch"] + res["costs"] + res["residual"]
    assert np.allclose(parts, res["step"])
    assert np.abs(res["mismatch"]).max() == 0.0                       # daily hedge holds the option delta
    assert np.abs(res["residual"].sum(1)).mean() < 0.05 * np.abs(res["gamma"].sum(1)).mean()


def test_hedging_error_grows_with_the_square_root_of_the_interval(paths):
    sd1 = hedge.simulate(SPEC, paths, 1)["total"].std()
    sd5 = hedge.simulate(SPEC, paths, 5)["total"].std()
    assert 0.8 * np.sqrt(5) < sd5 / sd1 < 1.2 * np.sqrt(5)


def test_mean_pnl_follows_implied_minus_realised_variance():
    df = hedge.vol_study(SPEC, [0.15, 0.25, 0.35], paths=4000, seed=3)
    se = df["std_pnl"] / np.sqrt(4000)
    assert (np.abs(df["mean_pnl"] - df["expected_from_gamma"]) < 4 * se + 0.02).all()
    assert df["mean_pnl"].iloc[0] > 0 > df["mean_pnl"].iloc[-1]     # short option: earns when realised < implied


def test_transaction_costs_lower_the_mean(paths):
    free = hedge.simulate(SPEC, paths, 1)["total"].mean()
    costly = hedge.simulate(hedge.HedgeSpec(100, 100, 63, 0.04, 0.0, 0.25, "call", cost_bps=10), paths, 1)
    assert costly["total"].mean() < free
    assert costly["costs"].sum(1).mean() < 0


def test_simulate_checks_path_length():
    with pytest.raises(ValueError):
        hedge.simulate(SPEC, np.full(10, 100.0), 1)


def test_attribution_table_shape(paths):
    t = hedge.attribution_table(hedge.simulate(SPEC, paths[:2], 1), path=1)
    assert len(t) == 63 and t.index[0] == 1 and {"spot", "gamma", "theta", "carry", "residual"} <= set(t.columns)


# --- the book ----------------------------------------------------------------------------------

@pytest.fixture
def the_book(cfg):
    return book.load(cfg)


def test_load_requires_a_volatility_for_every_underlying(cfg):
    bad = {**cfg, "market": {**cfg["market"], "vols": {"AAPL": 0.2}}}
    with pytest.raises(ValueError, match="no implied volatility"):
        book.load(bad)


def test_zero_move_zero_pnl_and_stock_is_linear(the_book):
    pos, mk = the_book
    unders = sorted({p.underlying for p in pos})
    flat = {u: np.zeros(1) for u in unders}
    assert book.pnl(pos, mk, flat, 0.0)[0] == pytest.approx(0.0, abs=1e-9)
    stock = [p for p in pos if p.kind == "stock"]
    r = {u: np.array([0.031]) for u in unders}
    assert book.pnl(stock, mk, r)[0] == pytest.approx(sum(p.side * p.notional for p in stock) * 0.031)


def test_dollar_delta_and_gamma_match_bumped_revaluation(the_book):
    pos, mk = the_book
    sens = book.sensitivities(pos, mk)
    h = 1e-4
    for i, p in enumerate(pos):
        if p.kind == "stock":
            continue
        up, dn = (book.pnl([p], mk, {p.underlying: np.array([x])})[0] for x in (h, -h))
        assert sens.loc[i, "dollar_delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-5)
        # dollar gamma per 1%: change in dollar delta for a 1% move = Gamma * S^2 * 0.01 per unit notional
        assert sens.loc[i, "dollar_gamma_1pct"] == pytest.approx((up + dn) / h ** 2 * 0.01, rel=1e-3)


def test_taylor_is_accurate_for_small_moves_and_degrades_for_large(the_book):
    pos, mk = the_book
    lad = book.ladder(pos, mk, [-0.2, -0.01, 0.0, 0.01, 0.2], [0.0])
    err = lad.set_index("spot_shock")["approximation_error"].abs()
    full = lad.set_index("spot_shock")["full_revaluation"].abs()
    assert err[0.0] == pytest.approx(0.0, abs=1e-9)
    assert err[0.01] < 0.01 * full[0.01] and err[-0.01] < 0.01 * full[-0.01]
    assert err[0.2] > 100 * err[0.01]


def test_vol_shock_moves_value_by_vega(the_book):
    pos, mk = the_book
    sens = book.sensitivities(pos, mk)
    flat = {p.underlying: np.zeros(1) for p in pos}
    one_point = book.pnl(pos, mk, flat, 0.0, vol_shift=0.01)[0]
    assert one_point == pytest.approx(sens["vega_per_point"].sum(), rel=1e-2)


def test_theta_matches_one_day_of_decay(the_book):
    pos, mk = the_book
    sens = book.sensitivities(pos, mk)
    flat = {p.underlying: np.zeros(1) for p in pos}
    one_day = book.pnl(pos, mk, flat, horizon_days=book.TRADING_DAYS / book.YEAR_DAYS)[0]   # one calendar day
    assert one_day == pytest.approx(sens["theta_per_day"].sum(), rel=1e-2)


def test_rename_keeps_the_pnl(the_book):
    pos, mk = the_book
    unders = sorted({p.underlying for p in pos})
    mapping = {u: f"X{i}" for i, u in enumerate(unders)}
    pos2, mk2 = book.rename(pos, mk, mapping)
    r = {u: np.array([0.02, -0.03]) for u in unders}
    r2 = {mapping[u]: v for u, v in r.items()}
    assert np.allclose(book.pnl(pos, mk, r, 1.0), book.pnl(pos2, mk2, r2, 1.0))


# --- options VaR -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def returns():
    return data.synthetic_returns(["AAA", "BBB", "CCC"], days=700, seed=4)


def _linear_book():
    pos = [book.Position("AAA", "stock", 1.0, 2e6), book.Position("BBB", "stock", -1.0, 1e6),
           book.Position("CCC", "stock", 1.0, 5e5)]
    mk = book.Market(0.04, {u: 0.0 for u in "AAA BBB CCC".split()}, {u: 0.2 for u in "AAA BBB CCC".split()})
    return pos, mk


def test_linear_book_methods_coincide(returns):
    pos, mk = _linear_book()
    tbl = var.rolling(pos, mk, returns, 250, [0.99], horizon_days=1.0)
    w = {m: g["var"].to_numpy() for m, g in tbl.groupby("method")}
    assert np.allclose(w["full_revaluation_hs"], w["delta_hs"])
    assert np.allclose(w["full_revaluation_hs"], w["delta_gamma_hs"])
    # delta-normal is market-risk-lab's parametric VaR with dollar weights
    mu, cov = rolling_moments(returns[["AAA", "BBB", "CCC"]].to_numpy(), 250)
    ref = rolling_parametric(mu, cov, np.array([2e6, -1e6, 5e5]), [0.99], zero_mean=True)[0.99][0]
    assert np.allclose(w["delta_normal"], ref)


def test_es_not_below_var_and_exceptions_consistent(the_book):
    pos, mk = the_book
    unders = sorted({p.underlying for p in pos})
    R = data.synthetic_returns(unders, days=600, seed=9)
    tbl = var.rolling(pos, mk, R, 250, [0.99, 0.975])
    assert (tbl["es"] >= tbl["var"] - 1e-9).all()
    assert (tbl["exception"] == (-tbl["realised_pnl"] > tbl["var"])).all()
    assert set(tbl["method"]) == set(var.METHODS)


def test_no_look_ahead(the_book):
    pos, mk = the_book
    unders = sorted({p.underlying for p in pos})
    R = data.synthetic_returns(unders, days=400, seed=2)
    base = var.rolling(pos, mk, R, 250, [0.99])
    day = R.index[300]
    R2 = R.copy()
    R2.loc[day] = R2.loc[day] * 5
    moved = var.rolling(pos, mk, R2, 250, [0.99])
    a = base[base["date"] == day].set_index("method")
    b = moved[moved["date"] == day].set_index("method")
    assert np.allclose(a["var"], b["var"]) and not np.allclose(a["realised_pnl"], b["realised_pnl"])


def test_summary(the_book):
    pos, mk = the_book
    unders = sorted({p.underlying for p in pos})
    R = data.synthetic_returns(unders, days=600, seed=9)
    s = var.summary(var.rolling(pos, mk, R, 250, [0.99, 0.975]))
    assert len(s) == 2 * len(var.METHODS)
    assert s["kupiec_p"].between(0, 1).all()
    assert s.loc[s["alpha"] == 0.975, "red_share"].isna().all()          # traffic light is a 99% concept
    assert s.loc[s["alpha"] == 0.99, "red_share"].notna().all()
    assert np.allclose(s["expected"], s["forecasts"] * (1 - s["alpha"]))


def test_window_must_leave_forecasts(the_book):
    pos, mk = the_book
    R = data.synthetic_returns(sorted({p.underlying for p in pos}), days=100, seed=1)
    with pytest.raises(ValueError):
        var.rolling(pos, mk, R, 250, [0.99])


def test_synthetic_returns_shape_and_level():
    R = data.synthetic_returns(["A", "B"], days=1500, seed=1)
    assert list(R.columns) == ["A", "B"] and len(R) == 1500 and isinstance(R.index, pd.DatetimeIndex)
    vol = R.std() * np.sqrt(252)
    assert ((vol > 0.12) & (vol < 0.45)).all()
