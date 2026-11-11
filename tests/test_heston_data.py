"""Torch-free tests for the Heston dataset generator and the finite-difference reference Greeks."""
import numpy as np
import pytest

from black_scholes import bs_greeks
from evaluate_heston import fd_heston_greeks, group_table
from generate_data import PARAM_RANGES, bounds, check_and_repair_labels, make_split
from heston import heston_price_quad


def test_heston_box_and_feature_order():
    r = PARAM_RANGES["heston"]
    assert list(r) == ["moneyness", "T", "r", "v0", "kappa", "theta", "xi", "rho"]
    assert r["rho"][1] <= 0 and r["xi"][0] > 0 and r["T"][0] > 0


def test_heston_split_is_inside_box_with_valid_independent_labels():
    tr, te = make_split("heston", "train", 1500), make_split("heston", "test", 1500)
    assert not set(map(tuple, tr["X"])) & set(map(tuple, te["X"]))
    for j, (lo, hi) in enumerate(PARAM_RANGES["heston"].values()):
        assert te["X"][:, j].min() >= lo and te["X"][:, j].max() <= hi
    lb, ub = bounds(te["X"])
    assert np.all(te["y"] >= lb - 1e-9) and np.all(te["y"] <= ub + 1e-9) and np.all(te["y"] >= 0)
    for i in range(0, 1500, 150):                                  # spot-check against the slow reference
        m, T, r, *hp = te["X"][i]
        assert te["y"][i] == pytest.approx(heston_price_quad(m, 1.0, T, r, *hp), abs=1e-9)
    assert te["label_quality"]["max_abs_diff_vs_adaptive_quad_on_subset"] < 1e-9


def test_label_checker_repairs_a_corrupted_label():
    te = make_split("heston", "val", 300)
    y_bad = te["y"].copy()
    y_bad[7] = -5e-5                                                # the kind of error a mis-priced option had
    y_fixed, info = check_and_repair_labels("heston", te["X"], y_bad)
    assert info["n_recomputed_outside_bounds"] == 1
    assert y_fixed[7] == pytest.approx(te["y"][7], abs=1e-9)


def test_fd_greeks_reproduce_black_scholes_in_the_zero_vol_of_vol_limit():
    # xi -> 0: variance is deterministic, so Heston = BS with total variance
    # sigma_eff^2 * T = integral of v(t) dt, and dV/dv0 = vega * d sigma_eff / d v0.
    S, K, T, r = np.array([85.0, 100.0, 118.0]), 100.0, 0.8, 0.03
    v0, kappa, theta = 0.06, 2.0, 0.04
    sig = np.sqrt(theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T))
    g = bs_greeks(S, K, T, r, sig, "call")
    fd = fd_heston_greeks(S, K, T, r, v0, kappa, theta, 1e-3, 0.0)
    np.testing.assert_allclose(fd["delta"], g["delta"], atol=2e-4)
    np.testing.assert_allclose(fd["gamma"], g["gamma"], rtol=2e-3, atol=1e-6)
    dsig_dv0 = (1 - np.exp(-kappa * T)) / (kappa * T) / (2 * sig)
    np.testing.assert_allclose(fd["dv0"], g["vega"] * dsig_dv0, rtol=2e-3)


def test_fd_greeks_are_insensitive_to_bump_size():
    p = (np.array([95.0, 105.0]), 100.0, 0.4, 0.03, 0.04, 1.5, 0.05, 0.5, -0.7)
    a, b = fd_heston_greeks(*p, rel=0.02), fd_heston_greeks(*p, rel=0.04)
    for g in ("delta", "gamma", "dv0", "theta"):
        np.testing.assert_allclose(a[g], b[g], rtol=5e-3, err_msg=g)


def test_group_table_counts():
    v = np.array([0.1, 0.3, 0.7, 1.5, 3.0, 9.0])
    rows = group_table(v, [0.0, 0.25, 0.5, 1.0, 2.0, np.inf], list("abcde"), np.ones(6))
    assert [r["n"] for r in rows] == [1, 1, 1, 1, 2]
