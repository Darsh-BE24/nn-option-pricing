"""Torch-free tests for the vectorised IV solver and the Stage-5 implied-vol machinery.
A fake 'network' (a plain function) stands in for the NN, including deliberately broken ones."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from black_scholes import ArbitrageBoundError, bs_greeks, bs_price, implied_volatility, implied_volatility_bisect
from generate_data import make_split
from smile import (PRESETS, fourier_price_fn, grid_inputs, iv_calls, iv_error_report, plot_smiles,
                   plot_surface, print_report)


def random_options(n, seed=0):
    rng = np.random.default_rng(seed)
    return dict(S=rng.uniform(50, 150, n), K=rng.uniform(50, 150, n), T=rng.uniform(0.1, 3.0, n),
                r=rng.uniform(0.0, 0.10, n), sigma=rng.uniform(0.08, 0.8, n), q=rng.uniform(0.0, 0.05, n))


# ------------------------------------------------------------ vectorised solver
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bisect_agrees_with_stage1_brent_solver(option_type):
    p = random_options(120)
    price = bs_price(**p, option_type=option_type)
    vega = bs_greeks(**p, option_type=option_type)["vega"]
    fast = implied_volatility_bisect(p["S"], p["K"], p["T"], p["r"], price, option_type, p["q"])
    for i in np.nonzero(vega > 0.05)[0][:60]:
        slow = implied_volatility(p["S"][i], p["K"][i], p["T"][i], p["r"][i], price[i], option_type, p["q"][i])
        assert fast[i] == pytest.approx(slow, abs=1e-9)
    ok = np.isfinite(fast)
    reprice = bs_price(p["S"][ok], p["K"][ok], p["T"][ok], p["r"][ok], fast[ok], option_type, p["q"][ok])
    np.testing.assert_allclose(reprice, price[ok], atol=1e-10)


def test_bisect_returns_nan_instead_of_raising_outside_bounds():
    # S = K = 1, T = 1, r = 0: valid call prices lie strictly between 0 and S = 1
    out = implied_volatility_bisect(1.0, 1.0, 1.0, 0.0, [-0.1, 0.0, 0.2, 0.5, 1.5])
    assert np.isnan(out[[0, 1, 4]]).all() and np.isfinite(out[[2, 3]]).all()
    with pytest.raises(ArbitrageBoundError):          # the scalar solver is loud about the same input
        implied_volatility(1.0, 1.0, 1.0, 0.0, 1.5)


def test_black_scholes_prices_give_a_flat_smile():
    X = grid_inputs(dict(v0=0.04, kappa=1.0, theta=0.04, xi=0.3, rho=-0.5), np.array([0.25, 1.0]), np.linspace(85, 120, 15))
    iv = iv_calls(X, bs_price(X[:, 0], 1.0, X[:, 1], X[:, 2], 0.2, "call"))
    np.testing.assert_allclose(iv, 0.2, atol=1e-9)


# ------------------------------------------------------------ Heston smile shape
def test_heston_produces_a_skew_and_a_smile_unlike_black_scholes():
    strikes = np.array([80.0, 100.0, 125.0])
    skew = PRESETS[next(iter(PRESETS))]
    X = grid_inputs(skew, np.array([0.5]), strikes)
    lo, atm, hi = 100 * iv_calls(X, fourier_price_fn(X), solver="brent")
    assert lo > atm > hi and (lo - hi) > 3.0                     # downward skew, several vol points
    X0 = grid_inputs({**skew, "rho": 0.0}, np.array([0.5]), strikes)
    lo0, atm0, hi0 = 100 * iv_calls(X0, fourier_price_fn(X0))
    assert 0.5 * (lo0 + hi0) > atm0 + 0.2                        # rho = 0: symmetric smile, wings above ATM
    assert abs(lo0 - hi0) < 1.0


# ------------------------------------------------------------ IV error report
@pytest.fixture(scope="module")
def val_set():
    d = make_split("heston", "val", 4000)
    return d["X"], d["y"]


def test_report_recovers_an_exact_one_vol_point_shift(val_set):
    X, y = val_set
    iv = iv_calls(X, y)
    shifted = bs_price(X[:, 0], 1.0, X[:, 1], X[:, 2], iv + 0.01, "call")      # NN exactly 1 vol point too high
    rep = iv_error_report(X, y, shifted, min_vega=0.05)
    o = rep["overall"]
    assert rep["frac_eligible"] > 0.9 and o["n"] == rep["n_eligible"]
    for key in ("mean_abs", "median_abs", "p99_abs", "max_abs", "mean_signed"):
        assert o[key] == pytest.approx(1.0, abs=1e-5), key
    assert rep["n_pred_price_outside_bounds"] == 0
    np.testing.assert_allclose(rep["bucket_mae"][rep["bucket_count"] > 0], 1.0, atol=1e-5)


def test_report_counts_unusable_nn_prices_and_respects_vega_filter(val_set):
    X, y = val_set
    bad = y.copy()
    iv = iv_calls(X, y)
    vega = bs_greeks(X[:, 0], 1.0, X[:, 1], X[:, 2], np.where(np.isfinite(iv), iv, 0.2))["vega"]
    good_rows = np.nonzero(vega > 0.2)[0][:5]
    bad[good_rows] = -0.01                                           # a network returning negative prices
    rep = iv_error_report(X, y, bad, min_vega=0.05)
    assert rep["n_pred_price_outside_bounds"] == 5
    assert rep["overall"]["n"] == rep["n_eligible"] - 5
    none_left = iv_error_report(X, y, y, min_vega=50.0)
    assert none_left["n_eligible"] == 0 and none_left["overall"]["n"] == 0
    print_report(none_left)                                          # must not crash when nothing is eligible
    print_report(rep)


# ------------------------------------------------------------ plots survive bad networks
def noisy_nn(X):
    rng = np.random.default_rng(0)
    return fourier_price_fn(X) + 3e-4 * rng.standard_normal(len(X))


def broken_nn(X):
    return np.full(len(X), 0.69)                                     # like an untrained softplus network


@pytest.mark.parametrize("nn", [None, noisy_nn, broken_nn], ids=["no_nn", "noisy", "broken"])
def test_plots_are_written_even_if_the_network_is_garbage(tmp_path, nn):
    s1 = plot_smiles(tmp_path / "smiles.png", nn)
    s2 = plot_surface(tmp_path / "surface.png", nn)
    assert (tmp_path / "smiles.png").stat().st_size > 10_000 and (tmp_path / "surface.png").stat().st_size > 10_000
    assert len(s1) == len(PRESETS) and s2["iv_max"] > s2["iv_min"]
