import numpy as np
import pandas as pd
import pytest

import calibrate as cal
import calibrate_surface as cs


def test_parse_expiry_from_nse_file_name():
    assert cs.parse_expiry("data/option-chain-ED-NIFTY-25-Jan-2027.csv") == pd.Timestamp("2027-01-25")
    assert cs.parse_expiry("C:/x/option-chain-ED-NIFTY-3-Mar-2027 (1).csv") == pd.Timestamp("2027-03-03")
    with pytest.raises(ValueError):
        cs.parse_expiry("chain.csv")


def test_multi_expiry_fit_recovers_parameters():
    truth = dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    mkt = cs.synthetic_surface(0.06, 0.0, truth)
    fit = cal.calibrate(cal.fourier_fn, mkt, mkt.S.values, mkt["T"].values, 0.06, n_starts=2, verbose=False)[0]
    p = fit["params"]
    for n, tol in [("v0", 0.003), ("kappa", 0.4), ("theta", 0.006), ("xi", 0.05), ("rho", 0.05)]:
        assert abs(p[n] - truth[n]) < tol, (n, p[n], truth[n])


def test_per_expiry_report_covers_every_expiry():
    mkt = cs.synthetic_surface(0.06, 0.0, dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7))
    rep = cs.per_expiry(mkt, mkt.iv.values + 0.001)
    assert len(rep) == 4 and all(abs(d["rmse"] - 0.1) < 1e-9 for d in rep.values())


def test_sparse_chain_forward_uses_one_tight_strike():
    from nse_chain import load_chain
    df = load_chain("data/surface/option-chain-ED-NIFTY-30-Mar-2027.csv")   # strikes every 1500 points
    F = cal.forward_from_parity(df, r=0.06, T=146 / 365)
    assert 24400 < F < 24600


def test_strike_window_shrinks_for_short_expiries():
    from nse_chain import load_chain
    df = load_chain("data/surface/option-chain-ED-NIFTY-23-Nov-2026.csv")
    F = cal.forward_from_parity(df, r=0.06, T=0.05)
    m = cal.build_market(df, F, 0.05, 0.06)
    assert np.abs(np.log(m.strike / F)).max() <= 0.08 + 1e-9


def test_every_expiry_gets_equal_total_weight():
    mkt = cs.synthetic_surface(0.06, 0.0, dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7))
    mkt = mkt.iloc[[0, 1, 2, 3, 4, 5, 13, 14, 26, 27, 28, 39, 40, 41, 42]].reset_index(drop=True)
    w = cs.add_weights(mkt)
    tot = (w.wgt ** 2).groupby(w.expiry).sum()
    assert np.allclose(tot, tot.iloc[0])


def test_calibration_stays_inside_the_pricers_own_box():
    truth = dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    mkt = cs.synthetic_surface(0.06, 0.0, truth)

    def narrow(X):                                   # exact pricer that advertises a narrow training box
        return cal.fourier_fn(X)
    narrow.box = {"moneyness": (0.8, 1.25), "T": (0.03, 0.6), "r": (0.04, 0.08), "v0": (0.005, 0.015),
                  "kappa": (0.5, 6.0), "theta": (0.01, 0.06), "xi": (0.2, 1.5), "rho": (-0.8, -0.05)}
    fit = cal.calibrate(narrow, mkt, mkt.S.values, mkt["T"].values, 0.06, n_starts=1, verbose=False)[0]
    assert fit["params"]["v0"] <= 0.015               # truth 0.02 is outside the box, so the bound must bind
