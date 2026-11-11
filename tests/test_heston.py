"""Tests for heston.py: CF identities, BS limit, quadrature accuracy,
static no-arbitrage, Fourier vs Monte Carlo, external reference value."""
import numpy as np
import pytest
from scipy.stats import qmc

from black_scholes import bs_price, no_arbitrage_bounds
from heston import heston_cf, heston_mc, heston_price, heston_price_quad

# Parameter box that Stage 4 will sample from (Feller condition NOT imposed).
BOX = {"m": (0.7, 1.3), "T": (0.05, 2.0), "r": (0.0, 0.08), "v0": (0.005, 0.25),
       "kappa": (0.5, 5.0), "theta": (0.005, 0.25), "xi": (0.1, 1.0), "rho": (-0.95, 0.0)}
HPARAMS = ("v0", "kappa", "theta", "xi", "rho")


def sample_box(n, seed):
    lo, hi = np.array([b[0] for b in BOX.values()]), np.array([b[1] for b in BOX.values()])
    X = qmc.scale(qmc.LatinHypercube(d=len(BOX), seed=seed).random(n), lo, hi)
    return dict(zip(BOX, X.T))


def sigma_eff(v0, kappa, theta, T):
    """Root-mean expected variance over [0, T]: the BS vol Heston collapses to as xi -> 0."""
    return np.sqrt(theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T))


# ------------------------------------------------------------ characteristic fn
def test_cf_normalisation_and_martingale():
    p = sample_box(200, 1)
    args = [p[k] for k in ("T",) + HPARAMS]
    np.testing.assert_allclose(heston_cf(0.0, *args), 1.0, atol=1e-13)
    np.testing.assert_allclose(heston_cf(-1j, *args), 1.0, atol=1e-11)   # E[S_T] = forward


def test_cf_is_bounded_on_lewis_contour():
    # |phi(u - i/2)| = |E[e^{(iu + 1/2) X}]| <= E[e^{X/2}] <= 1 (Jensen) -> integrand is bounded
    p = sample_box(200, 2)
    u = np.linspace(0, 200, 401)[None, :]
    mag = np.abs(heston_cf(u - 0.5j, *[p[k][:, None] for k in ("T",) + HPARAMS]))
    assert mag.max() <= 1 + 1e-12


# ------------------------------------------------------------ Black-Scholes limit
@pytest.mark.parametrize("v0", [0.04, 0.09])
def test_bs_limit_rho_zero_is_second_order(v0):
    S, K, T, r, kappa, theta = 100.0, np.array([80.0, 100.0, 120.0]), 1.3, 0.03, 2.0, 0.04
    target = bs_price(S, K, T, r, sigma_eff(v0, kappa, theta, T))
    err = lambda xi: np.abs(heston_price(S, K, T, r, v0, kappa, theta, xi, 0.0) - target).max()
    assert err(1e-3) < 1e-5
    assert err(1e-2) / err(1e-3) == pytest.approx(100, rel=0.1)          # O(xi^2)


def test_bs_limit_with_correlation_is_first_order():
    # With rho != 0 the leading correction is proportional to rho * xi (skew), so O(xi).
    S, K, T, r, v0, kappa, theta = 100.0, np.array([80.0, 120.0]), 1.3, 0.03, 0.09, 2.0, 0.04
    target = bs_price(S, K, T, r, sigma_eff(v0, kappa, theta, T))
    err = lambda xi: np.abs(heston_price(S, K, T, r, v0, kappa, theta, xi, -0.5) - target).max()
    assert err(1e-4) < 5e-4
    assert err(1e-3) / err(1e-4) == pytest.approx(10, rel=0.05)


# ------------------------------------------------------------ quadrature accuracy
def test_fast_quadrature_matches_adaptive_reference():
    p = sample_box(300, 3)
    fast = heston_price(p["m"], 1.0, p["T"], p["r"], *[p[k] for k in HPARAMS])
    ref = np.array([heston_price_quad(p["m"][i], 1.0, p["T"][i], p["r"][i], *[p[k][i] for k in HPARAMS])
                    for i in range(300)])
    np.testing.assert_allclose(fast, ref, atol=1e-8)                      # prices are in units of K


def test_external_reference_value():
    # S=K=100, r=5%, T=5, v0=theta=0.09, kappa=2, xi=1, rho=-0.3: reference 34.9998
    # (value quoted by the open-source HestonFive project; matches independent MC below too)
    assert heston_price_quad(100, 100, 5, 0.05, 0.09, 2.0, 0.09, 1.0, -0.3) == pytest.approx(34.9998, abs=1e-4)
    assert heston_price(100, 100, 5, 0.05, 0.09, 2.0, 0.09, 1.0, -0.3) == pytest.approx(34.9998, abs=1e-4)


# ------------------------------------------------------------ no-arbitrage
def test_prices_respect_static_no_arbitrage():
    p = sample_box(60, 4)
    K = np.linspace(60, 140, 81)
    for i in range(60):
        c = heston_price(100.0, K, p["T"][i], p["r"][i], *[p[k][i] for k in HPARAMS])
        dc, d2c = np.diff(c), np.diff(c, 2)
        assert np.all(dc <= 1e-9), "call price must be non-increasing in strike"
        assert np.all(d2c >= -1e-9), "call price must be convex in strike (butterfly >= 0)"
        lo, hi = zip(*(no_arbitrage_bounds(100.0, k, p["T"][i], p["r"][i]) for k in K))
        assert np.all(c >= np.array(lo) - 1e-9) and np.all(c <= np.array(hi) + 1e-9)


def test_put_call_parity_holds():
    args = (100.0, np.array([80.0, 100.0, 120.0]), 0.7, 0.04, 0.05, 1.2, 0.07, 0.6, -0.6)
    c = heston_price(*args, option_type="call", q=0.02)
    p = heston_price(*args, option_type="put", q=0.02)
    np.testing.assert_allclose(c - p, 100 * np.exp(-0.02 * 0.7) - args[1] * np.exp(-0.04 * 0.7), atol=1e-10)


# ------------------------------------------------------------ Monte Carlo
MC_CASES = [
    # (S, T, r, q, v0, kappa, theta, xi, rho)
    (100.0, 1.0, 0.03, 0.0, 0.04, 1.5, 0.06, 0.5, -0.7),      # typical equity
    (100.0, 2.0, 0.01, 0.02, 0.04, 0.5, 0.04, 1.0, -0.9),     # Feller badly violated (2*k*th = 0.04 << xi^2 = 1)
    (100.0, 0.25, 0.05, 0.0, 0.09, 3.0, 0.04, 0.3, -0.3),     # short-dated, v0 above theta
]


@pytest.mark.parametrize("case", MC_CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fourier_matches_monte_carlo_qe(case, option_type):
    S, T, r, q, *hp = case
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    fourier = heston_price(S, K, T, r, *hp, option_type=option_type, q=q)
    mc, se = heston_mc(S, K, T, r, *hp, option_type=option_type, q=q, n_paths=200_000, seed=11)
    z = (mc - fourier) / se
    assert np.all(np.abs(z) < 4), f"MC disagrees with Fourier: z-scores {np.round(z, 2)}"


@pytest.mark.parametrize("case", MC_CASES)
def test_qe_scheme_is_martingale(case):
    S, T, r, q, *hp = case
    ST = heston_mc(S, 100.0, T, r, *hp, q=q, n_paths=200_000, seed=5, return_ST=True)
    disc = ST * np.exp(-(r - q) * T)
    half = len(disc) // 2
    pairs = 0.5 * (disc[:half] + disc[half:])
    assert abs(pairs.mean() - S) < 4 * pairs.std(ddof=1) / np.sqrt(half)


# ------------------------------------------------------------ pricer accuracy in the hard corner
# Found in Stage 4: with a fixed 256 Gauss-Legendre nodes, ~0.2% of the parameter box (low v0,
# high xi, strongly negative rho, short/medium T) was mis-priced by up to 4e-5 of the strike,
# including slightly negative call prices. The pricer now adds nodes where the integrand oscillates.
HARD_CASES = [  # (m, T, r, v0, kappa, theta, xi, rho), K = 1; 256 fixed nodes were wrong by 3e-6 .. 4e-5
    (1.007, 0.3467, 0.0713, 0.0111, 0.5473, 0.0265, 0.8202, -0.8926),
    (1.0786, 0.162, 0.0251, 0.0105, 1.974, 0.0052, 0.7409, -0.8195),
    (1.0118, 0.0526, 0.0788, 0.0119, 1.8788, 0.0806, 0.5923, -0.9447),
    (1.2567, 0.1697, 0.076, 0.0113, 3.4668, 0.0057, 0.9529, -0.598),
]


@pytest.mark.parametrize("case", HARD_CASES)
def test_pricer_accurate_on_known_hard_cases(case):
    m, T, r, *hp = case
    assert heston_price(m, 1.0, T, r, *hp) == pytest.approx(heston_price_quad(m, 1.0, T, r, *hp), abs=1e-9)


def test_pricer_accurate_in_hard_region_vs_high_node_reference():
    rng = np.random.default_rng(0)
    n = 1200
    p = dict(m=rng.uniform(0.7, 1.3, n), T=rng.uniform(0.05, 0.6, n), r=rng.uniform(0, 0.08, n),
             v0=rng.uniform(0.005, 0.03, n), kappa=rng.uniform(0.5, 5, n), theta=rng.uniform(0.005, 0.25, n),
             xi=rng.uniform(0.5, 1.0, n), rho=rng.uniform(-0.95, -0.5, n))
    args = (p["m"], 1.0, p["T"], p["r"], *[p[k] for k in HPARAMS])
    fast = heston_price(*args)
    reference = heston_price(*args, n_nodes=4096, adaptive=False)
    assert np.abs(fast - reference).max() < 1e-8


def test_no_label_violates_arbitrage_bounds_over_the_box():
    p = sample_box(20_000, 5)
    y = heston_price(p["m"], 1.0, p["T"], p["r"], *[p[k] for k in HPARAMS])
    lower = np.maximum(p["m"] - np.exp(-p["r"] * p["T"]), 0.0)
    assert np.all(y >= lower - 1e-9) and np.all(y <= p["m"] + 1e-9)
