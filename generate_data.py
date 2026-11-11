"""Generate (features -> normalised price) datasets for training NN pricers.

Normalisation (why it is valid): BS/Heston prices are homogeneous of degree 1
in (S, K):  V(lambda*S, lambda*K) = lambda*V(S, K).  So
    V(S, K, ...) / K = f(S/K, ...)
and one network input (moneyness m = S/K) replaces two (S, K). The network
learns a single dimensionless surface instead of every price level separately.

Train / val / test are drawn as INDEPENDENT Latin-hypercube samples with
different seeds and written to separate files, so the test set is never seen
during training, early stopping or tuning.

Label quality (Heston): labels come from the vectorised Fourier pricer. For each
split we record (in meta.json) the worst violation of the model-free
no-arbitrage bounds, and the maximum difference against an independent adaptive
integrator on a random subset. Any label outside the bounds by more than 1e-9
is recomputed with the adaptive integrator, so the network never trains on a
mis-priced option.

Usage:
    python generate_data.py --model bs                      # 1M / 100k / 100k
    python generate_data.py --model heston                  # same sizes, about 3-10 minutes
    python generate_data.py --model heston_cal              # narrow calibration box (Stage 6b)
    python generate_data.py --model heston --n-train 20000 --n-val 5000 --n-test 5000 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from black_scholes import bs_price
from heston import heston_price, heston_price_quad

DATA_DIR = Path("data")

# Sampling box for each model. Features are always in this order.
PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "bs": {
        "moneyness": (0.7, 1.3),   # S / K
        "T": (0.05, 2.0),          # years
        "r": (0.0, 0.08),
        "sigma": (0.05, 0.60),
    },
    "heston": {
        "moneyness": (0.7, 1.3),   # S / K
        "T": (0.05, 2.0),          # years
        "r": (0.0, 0.08),
        "v0": (0.005, 0.25),       # initial variance (vol 7% .. 50%)
        "kappa": (0.5, 5.0),       # mean-reversion speed
        "theta": (0.005, 0.25),    # long-run variance
        "xi": (0.1, 1.0),          # vol of vol (Feller condition NOT imposed)
        "rho": (-0.95, 0.0),       # spot-vol correlation (equity-style, <= 0)
    },
    # Narrower box built for CALIBRATING to index options (Stage 6): short/medium maturities only, the
    # moneyness window real strike ranges need (ln(S/K) within about +-0.22), and wider xi / lower Feller
    # corner where real fits land. Same capacity, far less ground to cover.
    "heston_cal": {
        "moneyness": (0.80, 1.25),
        "T": (0.03, 0.60),
        "r": (0.04, 0.08),
        "v0": (0.005, 0.06),
        "kappa": (0.5, 6.0),
        "theta": (0.01, 0.06),
        "xi": (0.2, 1.5),
        "rho": (-0.8, -0.05),
    },
}
SEEDS = {"train": 0, "val": 1, "test": 2}
BLOCK = 50_000
BOUND_TOL = 1e-9


def sample_lhs(ranges: dict[str, tuple[float, float]], n: int, seed: int) -> np.ndarray:
    """Latin hypercube sample of shape (n, d), scaled to `ranges`.

    LHS stratifies every 1-D marginal into n equal bins with exactly one point
    each, so it covers the box more evenly than i.i.d. uniform draws (no
    accidental gaps/clumps) at the same sample size.
    """
    lo = np.array([r[0] for r in ranges.values()])
    hi = np.array([r[1] for r in ranges.values()])
    u = qmc.LatinHypercube(d=len(ranges), seed=seed).random(n)
    return qmc.scale(u, lo, hi)


def price_bs(X: np.ndarray) -> np.ndarray:
    """Normalised call price C/K with K = 1 (so S = moneyness)."""
    m, T, r, sigma = X.T
    return np.asarray(bs_price(m, 1.0, T, r, sigma, "call"))


def price_heston(X: np.ndarray) -> np.ndarray:
    """Normalised Heston call price C/K with K = 1 (Fourier, adaptive Gauss-Legendre)."""
    m, T, r, v0, kappa, theta, xi, rho = X.T
    return np.asarray(heston_price(m, 1.0, T, r, v0, kappa, theta, xi, rho))


def price_heston_scalar(x: np.ndarray) -> float:
    m, T, r, v0, kappa, theta, xi, rho = x
    return heston_price_quad(m, 1.0, T, r, v0, kappa, theta, xi, rho)


PRICERS = {"bs": price_bs, "heston": price_heston, "heston_cal": price_heston}
SCALAR_REFERENCE = {"heston": price_heston_scalar, "heston_cal": price_heston_scalar}


def bounds(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model-free bounds on a normalised call (K = 1, q = 0): max(m - e^{-rT}, 0) <= C <= m."""
    m, T, r = X[:, 0], X[:, 1], X[:, 2]
    return np.maximum(m - np.exp(-r * T), 0.0), m


def price_with_progress(model: str, X: np.ndarray, label: str) -> np.ndarray:
    n = len(X)
    out = np.empty(n)
    t0 = time.perf_counter()
    for s in range(0, n, BLOCK):
        out[s:s + BLOCK] = PRICERS[model](X[s:s + BLOCK])
        if n >= 4 * BLOCK:
            done = min(s + BLOCK, n)
            el = time.perf_counter() - t0
            print(f"    {label}: priced {done:>9,d}/{n:,d}  ({el:5.0f}s elapsed, "
                  f"~{el / done * (n - done):4.0f}s left)", flush=True)
    return out


def check_and_repair_labels(model: str, X: np.ndarray, y: np.ndarray,
                            n_check: int = 400, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Verify labels against no-arbitrage bounds and an independent integrator."""
    lb, ub = bounds(X)
    bad = np.nonzero((y < lb - BOUND_TOL) | (y > ub + BOUND_TOL))[0]
    y = y.copy()
    if model in SCALAR_REFERENCE and len(bad):
        for i in bad:
            y[i] = SCALAR_REFERENCE[model](X[i])
    still_bad = int(np.sum((y < lb - BOUND_TOL) | (y > ub + BOUND_TOL)))
    n_neg = int(np.sum(y < 0))
    y = np.maximum(y, 0.0)                      # remove round-off (|y| < 1e-9) negatives
    info = {
        "n_recomputed_outside_bounds": int(len(bad)),
        "n_still_outside_bounds": still_bad,
        "n_tiny_negative_clipped": n_neg,
        "max_lower_bound_violation": float(max(0.0, np.max(lb - y))),
        "max_upper_bound_violation": float(max(0.0, np.max(y - ub))),
    }
    if model in SCALAR_REFERENCE:
        idx = np.random.default_rng(seed).choice(len(y), size=min(n_check, len(y)), replace=False)
        ref = np.array([SCALAR_REFERENCE[model](X[i]) for i in idx])
        info["max_abs_diff_vs_adaptive_quad_on_subset"] = float(np.max(np.abs(y[idx] - ref)))
        info["n_subset_checked"] = int(len(idx))
    if still_bad:
        raise RuntimeError(f"{still_bad} labels remain outside no-arbitrage bounds: {info}")
    return y, info


def make_split(model: str, split: str, n: int, seed_offset: int = 0, verbose: bool = False) -> dict:
    ranges = PARAM_RANGES[model]
    X = sample_lhs(ranges, n, SEEDS[split] + seed_offset)
    t0 = time.perf_counter()
    y = price_with_progress(model, X, split) if verbose else PRICERS[model](X)
    secs = time.perf_counter() - t0
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"non-finite prices in {model}/{split}")
    y, quality = check_and_repair_labels(model, X, y)
    return {"X": X, "y": y, "pricing_seconds": secs, "label_quality": quality}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRICERS), default="bs")
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--n-test", type=int, default=100_000)
    ap.add_argument("--tag", default="", help="suffix for output folder, e.g. 'smoke'")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)

    out = args.out / (args.model + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "features": list(PARAM_RANGES[args.model]),
        "ranges": PARAM_RANGES[args.model],
        "target": "call_price / K",
        "sampling": "Latin hypercube (scipy.stats.qmc), independent per split",
        "seeds": SEEDS,
        "splits": {},
    }
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        d = make_split(args.model, split, n, verbose=True)
        np.savez(out / f"{split}.npz", X=d["X"], y=d["y"])
        meta["splits"][split] = {"n": n, "pricing_seconds": d["pricing_seconds"],
                                 "us_per_option": 1e6 * d["pricing_seconds"] / n,
                                 "label_quality": d["label_quality"]}
        q = d["label_quality"]
        print(f"{split:5s}: {n:>9,d} samples, priced in {d['pricing_seconds']:.1f}s "
              f"({meta['splits'][split]['us_per_option']:.1f} us/option), "
              f"y in [{d['y'].min():.4g}, {d['y'].max():.4g}]")
        print(f"       label check: {q['n_recomputed_outside_bounds']} recomputed, "
              f"max bound violation {max(q['max_lower_bound_violation'], q['max_upper_bound_violation']):.1e}"
              + (f", max diff vs adaptive quad ({q['n_subset_checked']} options) "
                 f"{q['max_abs_diff_vs_adaptive_quad_on_subset']:.1e}"
                 if 'max_abs_diff_vs_adaptive_quad_on_subset' in q else ""))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
