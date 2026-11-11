"""Heston stochastic-volatility model: Fourier pricing + Monte Carlo.

Model (risk-neutral):
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2,      d<W1, W2> = rho dt
Parameters: v0 (initial variance), kappa (mean-reversion speed), theta (long-run
variance), xi (vol of vol), rho (spot-vol correlation).

Pricing methods
---------------
* ``heston_price``        : Lewis (2001) single-integral formula, vectorised fixed
                            Gauss-Legendre quadrature (fast; used for datasets).
* ``heston_price_quad``   : same formula with adaptive QUADPACK integration
                            (slow, one option at a time; the accuracy reference).
* ``heston_mc``           : Monte Carlo, Andersen (2008) QE scheme with
                            martingale correction, or full-truncation Euler.

Lewis formula (call):
    C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi * int_0^inf Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du
    with k = ln(S/K) + (r - q) T and phi the characteristic function of
    X_T = ln(S_T/S_0) - (r - q) T.
Characteristic function: Gatheral's "little trap" form (Albrecher et al. 2007),
which avoids the branch-cut discontinuity of the complex log in Heston's
original formula for long maturities.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.special import ndtr

from black_scholes import _is_call

__all__ = ["heston_cf", "heston_price", "heston_price_quad", "heston_mc", "PARAM_NAMES"]
PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")


# --------------------------------------------------------------------------
# characteristic function
# --------------------------------------------------------------------------
def heston_cf(u: ArrayLike, T, v0, kappa, theta, xi, rho) -> NDArray[np.complex128]:
    """E[exp(i u X_T)], X_T = ln(S_T/S_0) - (r-q)T. Broadcasts over all inputs.

    Checks: phi(0) = 1 (normalisation) and phi(-i) = E[e^X] = 1 (martingale).
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * v0)


def _call_to_type(call, S, K, T, r, q, is_call):
    """Put via put-call parity (exact for any model)."""
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    out = np.where(is_call, call, put)
    return out[()] if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Fourier pricing
# --------------------------------------------------------------------------
_SCAN = np.logspace(-1, 4, 80)


_RULE_CACHE: dict[int, tuple[NDArray, NDArray]] = {}
_PANEL = 1024


def _unit_rule(n: int) -> tuple[NDArray, NDArray]:
    """Quadrature nodes/weights on [0, 1] with n points.

    n <= 1024: one Gauss-Legendre rule. Larger n: n/1024 equal panels, each a
    1024-point Gauss-Legendre rule (computing a single 8192-point rule takes ~1 minute).
    """
    if n not in _RULE_CACHE:
        m = min(n, _PANEL)
        x, w = np.polynomial.legendre.leggauss(m)
        p = n // m
        t = ((np.arange(p)[:, None] + 0.5 * (x[None, :] + 1.0)) / p).reshape(-1)
        _RULE_CACHE[n] = (t, np.tile(0.5 * w / p, p))
    return _RULE_CACHE[n]


def _nodes_needed(U: NDArray, k: NDArray, n_min: int, factor: float, n_max: int) -> NDArray:
    """Gauss-Legendre nodes required per option (power of two, >= n_min).

    The integrand oscillates like exp(i u k) over [0, U], so the work needed grows
    with R = U * (|k| + 0.05), the approximate number of radians swept. Measured
    on 20,000 random parameter sets: 256 nodes fail (error > 1e-8) from R of about
    88 upward (worst error 3.8e-5), while ~3 nodes per unit of R suffice. We use
    ``factor`` = 6 per unit of R (a 2x margin), rounded up to a power of two.
    """
    R = U * (np.abs(k) + 0.05)
    need = np.maximum(n_min, factor * R)
    pow2 = 2 ** np.ceil(np.log2(need))
    return np.minimum(pow2, n_max).astype(int)


def heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type="call", q=0.0,
                 n_nodes: int = 256, tol: float = 1e-10, chunk: int = 4096,
                 adaptive: bool = True, node_factor: float = 6.0, max_nodes: int = 8192,
                 max_elems: int = 1_500_000):
    """Vectorised Heston price (Lewis formula, Gauss-Legendre quadrature).

    The infinite integral is truncated per option at U, found by scanning the
    integrand magnitude on a log grid until |f(u)| * u < tol (a tail estimate),
    then [0, U] is integrated with Gauss-Legendre nodes. ``n_nodes`` is the
    minimum node count; with ``adaptive=True`` (default) options whose integrand
    oscillates a lot (deep OTM, short-dated, large vol-of-vol: large U * |k|)
    get 2x, 4x, ... more nodes, in groups so the work stays vectorised.
    Without this, a fixed 256 nodes mis-prices about 0.2% of the Stage-4
    parameter box by more than 1e-8 and a few by up to 1e-4 of the strike
    (including slightly negative call prices). Validated against adaptive
    quadrature and a 4096-node reference in the tests.
    """
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                 for a in (S, K, T, r, v0, kappa, theta, xi, rho, q)))
    S, K, T, r, v0, kappa, theta, xi, rho, q = arrs
    is_call = np.broadcast_to(_is_call(option_type), S.shape)
    if np.any(T <= 0) or np.any(xi <= 0) or np.any(np.abs(rho) >= 1):
        raise ValueError("need T > 0, xi > 0, |rho| < 1 (use black_scholes for xi = 0)")
    flat = [a.reshape(-1) for a in (S, K, T, r, v0, kappa, theta, xi, rho, q)]
    call = np.empty(flat[0].size)

    def integrate(sel: NDArray, U: NDArray, n: int) -> NDArray:
        """Lewis integral for options ``sel`` (indices), truncation points ``U`` (aligned), n nodes."""
        t, w = _unit_rule(n)
        Sc, Kc, Tc, rc, v0c, kc, thc, xic, rhoc, qc = (a[sel, None] for a in flat)
        pars = (Tc, v0c, kc, thc, xic, rhoc)
        Uc = U[:, None]
        u = Uc * t
        k = np.log(Sc / Kc) + (rc - qc) * Tc
        f = (np.exp(1j * u * k) * heston_cf(u - 0.5j, *pars)).real / (u * u + 0.25)
        integral = (f * (Uc * w)).sum(axis=1)
        return (Sc[:, 0] * np.exp(-qc[:, 0] * Tc[:, 0])
                - np.sqrt(Sc[:, 0] * Kc[:, 0]) * np.exp(-(rc[:, 0] + qc[:, 0]) * Tc[:, 0] / 2)
                / np.pi * integral)

    for s in range(0, call.size, chunk):
        e = min(s + chunk, call.size)
        Tc, v0c, kc, thc, xic, rhoc = (flat[i][s:e, None] for i in (2, 4, 5, 6, 7, 8))
        # 1) per-option truncation point
        mag = np.abs(heston_cf(_SCAN - 0.5j, Tc, v0c, kc, thc, xic, rhoc)) / (_SCAN**2 + 0.25) * _SCAN
        last = np.where(mag > tol, np.arange(_SCAN.size), 0).max(axis=1)
        U = _SCAN[np.minimum(last + 1, _SCAN.size - 1)]
        # 2) nodes per option, then integrate group by group
        logm = np.log(flat[0][s:e] / flat[1][s:e]) + (flat[3][s:e] - flat[9][s:e]) * flat[2][s:e]
        nodes = (_nodes_needed(U, logm, n_nodes, node_factor, max_nodes) if adaptive
                 else np.full(e - s, n_nodes))
        for n in np.unique(nodes):
            idx = np.nonzero(nodes == n)[0]
            step = max(1, max_elems // int(n))
            for g in range(0, idx.size, step):
                sub = idx[g:g + step]
                rows = s + sub
                call[rows] = integrate(rows, U[sub], int(n))
    call = call.reshape(S.shape)
    return _call_to_type(call, S, K, T, r, q, is_call)


def heston_price_quad(S: float, K: float, T: float, r: float, v0: float, kappa: float,
                      theta: float, xi: float, rho: float, option_type: str = "call",
                      q: float = 0.0) -> float:
    """Scalar reference price: same Lewis formula, adaptive QUADPACK on [0, inf)."""
    k = np.log(S / K) + (r - q) * T

    def f(u: float) -> float:
        return (np.exp(1j * u * k) * heston_cf(u - 0.5j, T, v0, kappa, theta, xi, rho)).real / (u * u + 0.25)

    integral, _ = quad(f, 0.0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-12)
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-(r + q) * T / 2) / np.pi * integral
    return float(_call_to_type(np.asarray(call), S, K, T, r, q, _is_call(option_type)))


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def heston_mc(S: float, K: ArrayLike, T: float, r: float, v0: float, kappa: float,
              theta: float, xi: float, rho: float, option_type: str = "call", q: float = 0.0,
              n_paths: int = 200_000, steps_per_year: int = 100,
              scheme: Literal["qe", "euler"] = "qe", seed: int | None = 0,
              return_ST: bool = False):
    """Monte Carlo price(s) for one parameter set and a vector of strikes.

    * ``qe``   : Andersen (2008) quadratic-exponential variance step with
                 central (gamma1 = gamma2 = 1/2) discretisation of the integrated
                 variance and his martingale correction ("QE-M"), so the
                 discounted spot is an exact martingale in the discrete scheme.
    * ``euler``: full-truncation Euler (Lord, Koekkoek & van Dijk 2010): use
                 max(v, 0) in drift and diffusion; simple but biased at coarse dt.
    Antithetic variates (Z -> -Z) are used; standard errors are computed from
    the n_paths/2 independent antithetic pair averages.

    Returns (prices, standard_errors) as arrays shaped like K.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n_steps = max(10, int(np.ceil(T * steps_per_year)))
    dt = T / n_steps
    x = np.zeros(2 * half)                 # log(S_t / S_0)
    v = np.full(2 * half, float(v0))

    if scheme == "qe":
        ek = np.exp(-kappa * dt)
        K1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        K2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        K3 = K4 = 0.5 * dt * (1.0 - rho * rho)
        A = K2 + 0.5 * K4
        K0_plain = -rho * kappa * theta * dt / xi
    for _ in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        if scheme == "euler":
            vp = np.maximum(v, 0.0)
            sq = np.sqrt(vp * dt)
            x += (r - q - 0.5 * vp) * dt + sq * (rho * z1 + np.sqrt(1 - rho * rho) * z2)
            v = v + kappa * (theta - vp) * dt + xi * sq * z1
            continue
        # --- QE step for v ---
        m = theta + (v - theta) * ek
        s2 = v * xi * xi * ek / kappa * (1 - ek) + theta * xi * xi / (2 * kappa) * (1 - ek) ** 2
        psi = s2 / (m * m)
        quad_branch = psi <= 1.5
        v_new = np.empty_like(v)
        K0 = np.empty_like(v)
        # quadratic branch: v' = a (b + Z)^2
        pq = psi[quad_branch]
        b2 = 2 / pq - 1 + np.sqrt(2 / pq) * np.sqrt(2 / pq - 1)
        a = m[quad_branch] / (1 + b2)
        v_new[quad_branch] = a * (np.sqrt(b2) + z1[quad_branch]) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = A < 1 / (2 * a)
            mgf = np.exp(A * b2 * a / (1 - 2 * A * a)) / np.sqrt(1 - 2 * A * a)
            K0[quad_branch] = np.where(ok, -np.log(mgf) - (K1 + 0.5 * K3) * v[quad_branch], K0_plain)
        # exponential branch: v' = 0 w.p. p, else Exp(beta)
        eb = ~quad_branch
        pe = (psi[eb] - 1) / (psi[eb] + 1)
        beta = (1 - pe) / m[eb]
        uni = ndtr(z1[eb])                 # U = Phi(Z): keeps antithetic pairing
        with np.errstate(divide="ignore"):
            v_new[eb] = np.where(uni <= pe, 0.0, np.log((1 - pe) / np.maximum(1 - uni, 1e-300)) / beta)
            ok = A < beta
            mgf = pe + beta * (1 - pe) / (beta - A)
            K0[eb] = np.where(ok, -np.log(mgf) - (K1 + 0.5 * K3) * v[eb], K0_plain)
        # --- log-spot step (Z for spot independent of the variance draw) ---
        x += (r - q) * dt + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * v + K4 * v_new) * z2
        v = v_new

    ST = S * np.exp(x)
    if return_ST:
        return ST
    sign = 1.0 if _is_call(option_type) else -1.0
    disc = np.exp(-r * T)
    payoff = np.maximum(sign * (ST[:, None] - K[None, :]), 0.0) * disc   # (paths, strikes)
    pairs = 0.5 * (payoff[:half] + payoff[half:])
    return pairs.mean(axis=0), pairs.std(axis=0, ddof=1) / np.sqrt(half)


if __name__ == "__main__":
    import time
    p = dict(v0=0.04, kappa=1.5, theta=0.06, xi=0.5, rho=-0.7)
    S, T, r = 100.0, 1.0, 0.03
    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    t0 = time.perf_counter(); fast = heston_price(S, Ks, T, r, **p); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); ref = [heston_price_quad(S, k, T, r, **p) for k in Ks]; t_quad = time.perf_counter() - t0
    t0 = time.perf_counter(); mc, se = heston_mc(S, Ks, T, r, **p, n_paths=400_000); t_mc = time.perf_counter() - t0
    print(f"{'K':>5} {'quad':>10} {'GL-256':>10} {'MC (QE)':>10} {'SE':>7} {'(MC-quad)/SE':>12}")
    for i, k in enumerate(Ks):
        print(f"{k:5.0f} {ref[i]:10.5f} {fast[i]:10.5f} {mc[i]:10.5f} {se[i]:7.4f} {(mc[i] - ref[i]) / se[i]:12.2f}")
    print(f"time: quad {1e3 * t_quad / 5:.1f} ms/option, GL {1e3 * t_fast / 5:.2f} ms/option (tiny batch), MC {t_mc:.1f} s for all 5")
