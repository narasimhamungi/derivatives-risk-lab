import numpy as np
import pytest

from derivlab import bs, mc, tree, validation

CASES = [  # S, K, T, r, q, sigma, kind
    (100, 90, 30 / 365, 0.04, 0.01, 0.20, "call"),
    (100, 100, 0.25, 0.04, 0.01, 0.25, "call"),
    (100, 110, 1.0, 0.04, 0.01, 0.30, "call"),
    (100, 90, 0.5, 0.04, 0.01, 0.35, "put"),
    (100, 120, 1.0, 0.04, 0.01, 0.25, "put"),
]


def test_textbook_values():
    # the standard textbook case (Hull, Options, Futures and Other Derivatives): S=42, K=40, r=10%,
    # sigma=20%, T=0.5 gives a call of 4.76 and a put of 0.81
    assert float(bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "call")) == pytest.approx(4.76, abs=0.005)
    assert float(bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "put")) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("S,K,T,r,q,sigma,kind", CASES)
def test_put_call_parity(S, K, T, r, q, sigma, kind):
    c = float(bs.price(S, K, T, r, q, sigma, "call"))
    p = float(bs.price(S, K, T, r, q, sigma, "put"))
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,q,sigma,kind", CASES)
def test_closed_form_greeks_match_finite_differences(S, K, T, r, q, sigma, kind):
    closed = bs.greeks(S, K, T, r, q, sigma, kind)
    fd = validation.finite_difference(S, K, T, r, q, sigma, kind)
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        assert float(closed[g]) == pytest.approx(fd[g], rel=1e-4, abs=1e-8), g


def test_intrinsic_value_at_expiry_and_vectorised():
    S = np.array([80.0, 100.0, 120.0])
    assert np.allclose(bs.price(S, 100, 0.0, 0.04, 0.0, 0.2, "call"), [0, 0, 20])
    assert np.allclose(bs.price(S, 100, 0.0, 0.04, 0.0, 0.2, "put"), [20, 0, 0])
    assert bs.price(S, 100, 0.5, 0.04, 0.0, 0.2, "call").shape == (3,)


def test_desk_unit_helpers():
    assert float(bs.per_day(365.0)) == pytest.approx(1.0)
    assert float(bs.per_point(25.0)) == pytest.approx(0.25)


@pytest.mark.parametrize("kind,sigma", [("call", 0.12), ("put", 0.45), ("call", 0.80)])
def test_implied_vol_round_trip(kind, sigma):
    p = float(bs.price(100, 105, 0.75, 0.03, 0.02, sigma, kind))
    assert bs.implied_vol(p, 100, 105, 0.75, 0.03, 0.02, kind) == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_rejects_arbitrage_prices():
    with pytest.raises(ValueError):
        bs.implied_vol(150.0, 100, 100, 1.0, 0.03, 0.0, "call")   # above the stock price
    with pytest.raises(ValueError):
        bs.implied_vol(0.5, 120, 100, 1.0, 0.03, 0.0, "call")     # below intrinsic value


@pytest.mark.parametrize("S,K,T,r,q,sigma,kind", CASES)
def test_binomial_converges_to_closed_form(S, K, T, r, q, sigma, kind):
    t = tree.crr(S, K, T, r, q, sigma, 2000, kind)
    g = bs.greeks(S, K, T, r, q, sigma, kind)
    assert t.price == pytest.approx(float(bs.price(S, K, T, r, q, sigma, kind)), rel=2e-3, abs=2e-3)
    assert t.delta == pytest.approx(float(g["delta"]), abs=2e-3)
    assert t.gamma == pytest.approx(float(g["gamma"]), rel=1e-2)
    assert t.theta == pytest.approx(float(g["theta"]), rel=1e-2)
    assert not t.early_exercise


def test_american_exercise():
    euro = tree.crr(100, 100, 1.0, 0.05, 0.0, 0.2, 1000, "put")
    amer = tree.crr(100, 100, 1.0, 0.05, 0.0, 0.2, 1000, "put", american=True)
    assert amer.price == pytest.approx(6.09, abs=0.01)          # standard reference value
    assert amer.price > euro.price and amer.early_exercise
    call = tree.crr(100, 100, 1.0, 0.05, 0.0, 0.2, 1000, "call", american=True)
    assert call.price == pytest.approx(tree.crr(100, 100, 1.0, 0.05, 0.0, 0.2, 1000, "call").price, abs=1e-12)
    assert not call.early_exercise                                 # never optimal without dividends


def test_tree_input_checks():
    with pytest.raises(ValueError):
        tree.crr(100, 100, 1.0, 0.05, 0.0, 0.2, 2)


@pytest.mark.parametrize("S,K,T,r,q,sigma,kind", CASES)
def test_monte_carlo_within_standard_errors(S, K, T, r, q, sigma, kind):
    m = mc.european(S, K, T, r, q, sigma, kind, 200_000, seed=11)
    closed = float(bs.price(S, K, T, r, q, sigma, kind))
    assert abs(m.price - closed) < 4 * m.std_error
    g = bs.greeks(S, K, T, r, q, sigma, kind)
    assert m.delta == pytest.approx(float(g["delta"]), rel=2e-2, abs=2e-3)
    assert m.vega == pytest.approx(float(g["vega"]), rel=2e-2)
    assert m.gamma == pytest.approx(float(g["gamma"]), rel=8e-2)


def test_variance_reduction_lowers_the_standard_error():
    def se(a, c):
        return mc.european(100, 100, 1.0, 0.04, 0.0, 0.25, "call", 100_000, seed=5, antithetic=a,
                           control_variate=c).std_error
    plain, anti, cv, both = se(False, False), se(True, False), se(False, True), se(True, True)
    assert anti < plain and cv < plain
    assert both < min(anti, cv) and both < 0.3 * plain


def test_validation_table(small_cfg):
    t = validation.greeks_table(small_cfg)
    assert t["put_call_parity_gap"].dropna().abs().max() < 1e-10
    e = validation.error_summary(t).set_index("measure")
    assert e.loc[["delta", "gamma", "vega", "theta", "rho"], "finite_difference"].max() < 1e-3
    assert e.loc["price", "binomial"] < 5e-3
    assert np.isnan(e.loc["rho", "binomial"]) and np.isnan(e.loc["price", "finite_difference"])
