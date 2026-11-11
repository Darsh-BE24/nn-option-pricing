"""Torch-free tests for the evaluation helpers and the dataset generator."""
import numpy as np
import pytest

from black_scholes import bs_price
from evaluate import M_EDGES, T_EDGES, bucket_table, greek_metrics, price_metrics
from generate_data import PARAM_RANGES, make_split, sample_lhs


def test_price_metrics_known_values():
    y = np.array([0.0, 0.02, 0.10, 0.50])
    yh = y + np.array([1e-3, -1e-3, 2e-3, 0.0])
    pm = price_metrics(y, yh)
    assert pm["mae"] == pytest.approx(1e-3)
    assert pm["max_abs"] == pytest.approx(2e-3)
    assert pm["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4) / 4) * 1e-3)
    assert pm["frac_above_floor"] == 0.75                 # y=0 excluded from relative error
    assert pm["rel_max"] == pytest.approx(0.05)           # 1e-3 / 0.02


def test_bucket_table_counts_and_values():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.7, 1.3, 5000)
    T = rng.uniform(0.05, 2.0, 5000)
    err = np.where(m > 1.1, 1.0, 0.0)                     # only deep-ITM cells have error
    bt = bucket_table(m, T, err)
    assert bt["count"].sum() == 5000
    assert np.allclose(bt["mae"][-1], 1.0) and np.allclose(bt["mae"][:-1], 0.0)
    # edge values land in the first / last bucket, not outside
    bt2 = bucket_table(np.array([M_EDGES[0], M_EDGES[-1]]), np.array([T_EDGES[0], T_EDGES[-1]]), np.ones(2))
    assert bt2["count"][0, 0] == 1 and bt2["count"][-1, -1] == 1


def test_greek_metrics_zero_for_identical():
    g = {k: np.ones(3) for k in ["delta", "gamma", "vega", "theta", "rho"]}
    assert all(v["mae"] == 0 for v in greek_metrics(g, g).values())


def test_lhs_stays_in_box_and_is_stratified():
    rng_box = PARAM_RANGES["bs"]
    X = sample_lhs(rng_box, 1000, seed=0)
    for j, (lo, hi) in enumerate(rng_box.values()):
        assert X[:, j].min() >= lo and X[:, j].max() <= hi
        counts = np.histogram(X[:, j], bins=10, range=(lo, hi))[0]
        assert np.all(counts == 100)                      # exactly one point per 1/n stratum


def test_splits_are_independent_and_target_is_normalised_price():
    tr, te = make_split("bs", "train", 2000), make_split("bs", "test", 2000)
    assert not set(map(tuple, tr["X"])) & set(map(tuple, te["X"]))
    m, T, r, s = te["X"].T
    K = 250.0  # homogeneity: V(mK, K)/K == V(m, 1)
    np.testing.assert_allclose(bs_price(m * K, K, T, r, s) / K, te["y"], atol=1e-14)
