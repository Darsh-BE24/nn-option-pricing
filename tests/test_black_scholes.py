"""Tests for black_scholes.py: textbook values, parity, bounds, IV, Greeks."""
import numpy as np
import pytest

from black_scholes import (
    ArbitrageBoundError,
    bs_greeks,
    bs_price,
    implied_volatility,
    implied_volatility_vec,
    no_arbitrage_bounds,
)

SEED = 12345
N = 2000


def random_params(n=N, seed=SEED):
    """Fresh, seeded draw per call so every test is order-independent."""
    rng = np.random.default_rng(seed)
    return dict(
        S=rng.uniform(50, 150, n),
        K=rng.uniform(50, 150, n),
        T=rng.uniform(0.05, 3.0, n),
        r=rng.uniform(0.0, 0.10, n),
        sigma=rng.uniform(0.05, 0.80, n),
        q=rng.uniform(0.0, 0.05, n),
    )


# ---------------------------------------------------------------- prices
def test_hull_textbook_values():
    # Hull, Options Futures & Other Derivatives, Example 15.6: C = 4.76, P = 0.81
    assert bs_price(42, 40, 0.5, 0.1, 0.2, "call") == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.1, 0.2, "put") == pytest.approx(0.8086, abs=1e-4)


def test_put_call_parity():
    p = random_params()
    c = bs_price(**p, option_type="call")
    pu = bs_price(**p, option_type="put")
    fwd = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
    np.testing.assert_allclose(c - pu, fwd, atol=1e-10)


def test_prices_within_no_arbitrage_bounds():
    p = random_params()
    for ot in ("call", "put"):
        price = bs_price(**p, option_type=ot)
        for i in range(0, N, 50):
            lo, hi = no_arbitrage_bounds(p["S"][i], p["K"][i], p["T"][i], p["r"][i], ot, p["q"][i])
            assert lo - 1e-12 <= price[i] <= hi + 1e-12


def test_vectorised_option_type_matches_scalar_loop():
    p = random_params(200)
    types = np.where(np.random.default_rng(7).random(200) < 0.5, "call", "put")
    vec = bs_price(**p, option_type=types)
    loop = [bs_price(p["S"][i], p["K"][i], p["T"][i], p["r"][i], p["sigma"][i], types[i], p["q"][i])
            for i in range(200)]
    np.testing.assert_allclose(vec, loop, rtol=0, atol=0)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, "straddle")


def test_T_zero_gives_intrinsic():
    S = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(bs_price(S, 100, 0.0, 0.05, 0.3, "call"), [0, 0, 20])
    np.testing.assert_allclose(bs_price(S, 100, 0.0, 0.05, 0.3, "put"), [20, 0, 0])


def test_sigma_zero_gives_discounted_forward_intrinsic():
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    assert bs_price(S, K, T, r, 0.0, "call") == pytest.approx(S - K * np.exp(-r * T))
    assert bs_price(S, K, T, r, 0.0, "put") == 0.0


def test_small_sigma_converges_to_sigma_zero_limit():
    S, K, T, r = 100.0, 103.0, 1.0, 0.05
    for ot in ("call", "put"):
        assert bs_price(S, K, T, r, 1e-7, ot) == pytest.approx(bs_price(S, K, T, r, 0.0, ot), abs=1e-9)


# ---------------------------------------------------------------- implied vol
def test_iv_user_example_raises_with_clear_message():
    # Original bug: price 10 < lower bound 200 - 150 e^{-0.2} = 77.19 -> no root.
    with pytest.raises(ArbitrageBoundError, match="outside no-arbitrage bounds"):
        implied_volatility(200, 150, 2, 0.1, 10, "call")


def test_iv_above_upper_bound_raises():
    with pytest.raises(ArbitrageBoundError):
        implied_volatility(100, 100, 1, 0.05, 100.5, "call")  # C > S
    with pytest.raises(ArbitrageBoundError):
        implied_volatility(100, 100, 1, 0.05, 96.0, "put")    # P > K e^{-rT} = 95.12


def test_iv_round_trip():
    p = random_params()
    for ot in ("call", "put"):
        prices = bs_price(**p, option_type=ot)
        vega = bs_greeks(**p, option_type=ot)["vega"]
        iv = implied_volatility_vec(p["S"], p["K"], p["T"], p["r"], prices, ot, p["q"])
        # price always reproduced
        repriced = bs_price(p["S"], p["K"], p["T"], p["r"], iv, ot, p["q"])
        np.testing.assert_allclose(repriced, prices, atol=1e-10)
        # sigma recovered wherever the price is actually sensitive to sigma.
        # ~7% of these random contracts are deep ITM/OTM with vega < 0.01: the
        # price is reproduced but sigma is not identifiable (errors up to ~0.35).
        # This is why real-market IV work uses OTM quotes / vega filters.
        ok = vega > 1e-2
        assert ok.mean() > 0.90
        np.testing.assert_allclose(iv[ok], p["sigma"][ok], atol=1e-8)


def test_iv_vec_nan_mode_is_explicit():
    out = implied_volatility_vec([200, 200], 150, 2, 0.1, [10, 90], "call", errors="nan")
    assert np.isnan(out[0]) and np.isfinite(out[1])
    with pytest.raises(ArbitrageBoundError):
        implied_volatility_vec([200, 200], 150, 2, 0.1, [10, 90], "call")


def test_iv_extreme_vol_recovered():
    price = bs_price(100, 100, 1.0, 0.0, 3.0, "call")
    assert implied_volatility(100, 100, 1.0, 0.0, price, "call") == pytest.approx(3.0, abs=1e-8)


# ---------------------------------------------------------------- Greeks
def _fd_greeks(p, ot):
    """Central finite differences.

    The spot bump is scaled by S*sigma*sqrt(T), the natural width of the
    terminal distribution: for short-dated/low-vol options the price curves
    sharply near the strike, and a fixed 0.1%-of-S bump gives O(h^2)
    truncation errors of ~1e-5 in delta.
    """
    def V(**kw):
        return bs_price(**{**p, **kw}, option_type=ot)

    S, T, r, sig = p["S"], p["T"], p["r"], p["sigma"]
    hS = 1e-3 * S * sig * np.sqrt(T)
    hT, hr, hs = 1e-5, 1e-5, 1e-5
    return {
        "delta": (V(S=S + hS) - V(S=S - hS)) / (2 * hS),
        "gamma": (V(S=S + hS) - 2 * V() + V(S=S - hS)) / hS**2,
        "vega": (V(sigma=sig + hs) - V(sigma=sig - hs)) / (2 * hs),
        "theta": -(V(T=T + hT) - V(T=T - hT)) / (2 * hT),
        "rho": (V(r=r + hr) - V(r=r - hr)) / (2 * hr),
    }


def test_greeks_match_finite_differences():
    p = random_params()
    for ot in ("call", "put"):
        an = bs_greeks(**p, option_type=ot)
        fd = _fd_greeks(p, ot)
        for g in an:
            np.testing.assert_allclose(an[g], fd[g], rtol=1e-5, atol=1e-6, err_msg=f"{ot} {g}")


def test_greek_identities():
    p = random_params()
    c, pu = bs_greeks(**p, option_type="call"), bs_greeks(**p, option_type="put")
    np.testing.assert_allclose(c["delta"] - pu["delta"], np.exp(-p["q"] * p["T"]), atol=1e-12)
    np.testing.assert_allclose(c["gamma"], pu["gamma"], rtol=1e-12)
    np.testing.assert_allclose(c["vega"], pu["vega"], rtol=1e-12)


def test_black_scholes_pde_satisfied():
    # Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0
    p = random_params()
    for ot in ("call", "put"):
        g = bs_greeks(**p, option_type=ot)
        V = bs_price(**p, option_type=ot)
        resid = (g["theta"] + 0.5 * p["sigma"] ** 2 * p["S"] ** 2 * g["gamma"]
                 + (p["r"] - p["q"]) * p["S"] * g["delta"] - p["r"] * V)
        np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_degenerate_greeks_are_finite():
    for T, sig in [(0.0, 0.2), (1.0, 0.0)]:
        g = bs_greeks(np.array([90.0, 110.0]), 100, T, 0.05, sig, "call")
        for v in g.values():
            assert np.all(np.isfinite(v))
        assert g["gamma"].tolist() == [0.0, 0.0]
