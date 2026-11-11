import numpy as np
import pandas as pd
import pytest

import calibrate as cal
from nse_chain import load_chain

CSV = "data/nifty_chain.csv"


def test_load_chain_parses_commas_and_dashes():
    df = load_chain(CSV)
    assert df.strike.is_monotonic_increasing and df.strike.min() == 20950
    row = df[df.strike == 24000].iloc[0]
    assert row.c_bid == pytest.approx(598.70) and row.p_ask == pytest.approx(567.80)
    assert np.isnan(df[df.strike == 20950].iloc[0].c_ltp)          # '-' -> NaN


def test_forward_from_parity_is_near_24044():
    F = cal.forward_from_parity(load_chain(CSV))
    assert 24000 < F < 24090


def test_build_market_keeps_only_otm_two_sided_quotes():
    df = load_chain(CSV); F = cal.forward_from_parity(df)
    m = cal.build_market(df, F, 0.2247, 0.06)
    assert len(m) > 40
    assert ((m.typ == "put") == (m.strike < F)).all()
    assert (m.iv_bid <= m.iv + 1e-12).all() and (m.iv <= m.iv_ask + 1e-12).all()
    assert m.iv.between(0.05, 0.4).all()


def test_calibration_fits_synthetic_smile_and_respects_fixed():
    T, r = 0.2247, 0.06
    S = 24000 * np.exp(-r * T)
    K = np.arange(21500, 25001, 250.0)
    truth = dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    iv = cal.exact_iv(truth, S, K, T, r)
    mkt = pd.DataFrame(dict(strike=K, iv=iv, iv_bid=iv - 0.002, iv_ask=iv + 0.002, width=0.004))
    fits = cal.calibrate(cal.fourier_fn, mkt, S, T, r, n_starts=1, verbose=False, fixed={"kappa": 3.0})
    assert fits[0]["params"]["kappa"] == 3.0
    assert fits[0]["rmse_vol_pts"] < 0.1          # noise-free: the smile is reproduced
