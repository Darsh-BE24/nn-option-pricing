import numpy as np
import pytest

from black_scholes import bs_greeks
from generate_data import PARAM_RANGES, sample_lhs
from train_utils import bs_vega_normalised, heston_effective_vol, intrinsic, vega_weights

FEATS = list(PARAM_RANGES["heston_cal"])


def test_normalised_vega_matches_black_scholes_module():
    m, T, r, s = 1.05, 0.3, 0.06, 0.2
    ref = bs_greeks(m, 1.0, T, r, s, "call")["vega"]
    assert bs_vega_normalised(m, T, r, s) == pytest.approx(float(ref), rel=1e-9)


def test_effective_vol_limits():
    # v0 == theta: variance is constant, so the effective vol is just sqrt(v0)
    assert heston_effective_vol(0.5, 0.04, 2.0, 0.04) == pytest.approx(0.2)
    # very fast mean reversion: effective vol -> sqrt(theta)
    assert heston_effective_vol(1.0, 0.09, 500.0, 0.04) == pytest.approx(0.2, rel=0.02)


def test_vega_weights_mean_one_and_favour_low_vega():
    X = sample_lhs(PARAM_RANGES["heston_cal"], 5000, seed=3)
    w = vega_weights(X, FEATS, floor=0.05)
    assert w.mean() == pytest.approx(1.0) and np.all(w > 0)
    vega = bs_vega_normalised(X[:, 0], X[:, 1], X[:, 2], heston_effective_vol(X[:, 1], X[:, 3], X[:, 4], X[:, 5]))
    assert np.corrcoef(vega, w)[0, 1] < -0.5          # lower vega -> larger weight
    assert w.max() / w.min() < 1000                   # the floor keeps the spread bounded


def test_vega_weights_black_scholes_layout():
    X = np.array([[1.0, 0.5, 0.05, 0.2], [1.0, 0.5, 0.05, 0.4]])
    w = vega_weights(X, ["moneyness", "T", "r", "sigma"])
    assert w.shape == (2,) and w.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        vega_weights(X, ["a", "b", "c", "d"])


def test_intrinsic_is_the_no_arbitrage_lower_bound():
    assert intrinsic(1.2, 0.5, 0.04) == pytest.approx(1.2 - np.exp(-0.02))
    assert intrinsic(0.8, 0.5, 0.04) == 0.0


def test_narrow_box_labels_respect_bounds():
    from generate_data import make_split
    d = make_split("heston_cal", "val", 300)
    assert d["label_quality"]["n_still_outside_bounds"] == 0
    assert d["X"][:, 3].max() <= 0.06 and d["X"][:, 6].max() <= 1.5
