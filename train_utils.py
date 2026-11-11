"""Torch-free helpers for training the calibration network (so they can be unit-tested anywhere).

vega_weights: the training loss is  w * (price_error)^2  with  w ~ 1 / vega^2.
    price_error / vega is (to first order) the IMPLIED-VOL error, so this loss makes a one-vol-point error
    count the same everywhere, instead of a one-basis-point PRICE error counting the same. Short-dated and
    far-out-of-the-money options have small vega, so they get more weight. A floor keeps the weight finite
    where vega -> 0 (there the vol is unidentifiable anyway, see Stage 1).
"""
from __future__ import annotations

import numpy as np

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def heston_effective_vol(T, v0, kappa, theta) -> np.ndarray:
    """sqrt(expected average variance over [0, T]) under Heston: a sensible Black-Scholes vol to evaluate vega at."""
    T, v0, kappa, theta = (np.asarray(a, dtype=float) for a in (T, v0, kappa, theta))
    total_var = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa
    return np.sqrt(np.maximum(total_var, 1e-12) / T)


def bs_vega_normalised(m, T, r, sigma) -> np.ndarray:
    """Black-Scholes vega of a call with K = 1 and S = m (per 1.00 of vol): m * phi(d1) * sqrt(T)."""
    m, T, r, sigma = (np.asarray(a, dtype=float) for a in (m, T, r, sigma))
    d1 = (np.log(m) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return m * np.exp(-0.5 * d1 ** 2) / _SQRT_2PI * np.sqrt(T)


def vega_weights(X: np.ndarray, features: list[str], floor: float = 0.05) -> np.ndarray:
    """Per-sample loss weights  1 / (vega^2 + floor^2), scaled to mean 1 (so learning rates stay comparable)."""
    X = np.asarray(X, dtype=float)
    col = {name: i for i, name in enumerate(features)}
    if "v0" in col:
        sigma = heston_effective_vol(X[:, col["T"]], X[:, col["v0"]], X[:, col["kappa"]], X[:, col["theta"]])
    elif "sigma" in col:
        sigma = X[:, col["sigma"]]
    else:
        raise ValueError(f"do not know how to get a volatility from features {features}")
    vega = bs_vega_normalised(X[:, col["moneyness"]], X[:, col["T"]], X[:, col["r"]], sigma)
    w = 1.0 / (vega ** 2 + floor ** 2)
    return w / w.mean()


def intrinsic(m, T, r) -> np.ndarray:
    """Normalised no-arbitrage lower bound of a call (K = 1, q = 0): max(m - e^{-rT}, 0)."""
    m, T, r = (np.asarray(a, dtype=float) for a in (m, T, r))
    return np.maximum(m - np.exp(-r * T), 0.0)
