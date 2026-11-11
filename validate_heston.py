"""Stage 3 validation report for heston.py (README tables + plots).

1. Andersen (2008) test cases: Fourier vs QE Monte Carlo.
2. Discretisation bias vs time step: QE vs full-truncation Euler (Andersen case I).
3. Convergence to Black-Scholes as vol-of-vol -> 0 (O(xi) with rho != 0, O(xi^2) with rho = 0).
4. Timing: adaptive quadrature vs vectorised Gauss-Legendre vs Monte Carlo.

Usage:  python validate_heston.py        (about 1-2 minutes)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from black_scholes import bs_price
from heston import heston_mc, heston_price, heston_price_quad

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
ANDERSEN = {   # (T, r, v0, kappa, theta, xi, rho), S = K = 100
    "I": (10.0, 0.00, 0.04, 0.5, 0.04, 1.0, -0.9),
    "II": (15.0, 0.00, 0.04, 0.3, 0.04, 0.9, -0.5),
    "III": (5.0, 0.05, 0.09, 1.0, 0.09, 1.0, -0.3),
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.25, linewidth=0.6)


def andersen_table(n_paths=200_000, steps_per_year=32):
    rows = []
    print(f"\n== Andersen (2008) cases, S = K = 100: Fourier vs QE Monte Carlo "
          f"({n_paths:,} paths, {steps_per_year} steps/yr) ==")
    print(f"{'case':>4} {'Feller 2k*th/xi^2':>18} {'Fourier':>10} {'MC':>10} {'SE':>7} {'z':>6}")
    for name, (T, r, v0, k, th, xi, rho) in ANDERSEN.items():
        f = heston_price_quad(100, 100, T, r, v0, k, th, xi, rho)
        mc, se = heston_mc(100, 100, T, r, v0, k, th, xi, rho, n_paths=n_paths,
                           steps_per_year=steps_per_year, seed=7)
        z = (mc[0] - f) / se[0]
        feller = 2 * k * th / xi**2
        print(f"{name:>4} {feller:18.2f} {f:10.4f} {mc[0]:10.4f} {se[0]:7.4f} {z:6.2f}")
        rows.append({"case": name, "feller_ratio": feller, "fourier": f, "mc": float(mc[0]),
                     "se": float(se[0]), "z": float(z)})
    return rows


def bias_vs_steps(path: Path, n_paths=200_000):
    T, r, v0, k, th, xi, rho = ANDERSEN["I"]
    ref = heston_price_quad(100, 100, T, r, v0, k, th, xi, rho)
    steps = [1, 2, 4, 8, 16, 32]
    out = {}
    for scheme in ("qe", "euler"):
        res = [heston_mc(100, 100, T, r, v0, k, th, xi, rho, n_paths=n_paths,
                         steps_per_year=s, scheme=scheme, seed=3) for s in steps]
        out[scheme] = {"err": [float(m[0] - ref) for m, _ in res], "se": [float(e[0]) for _, e in res]}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme, color, label in (("qe", BLUE, "QE with martingale correction"),
                                 ("euler", ORANGE, "Full-truncation Euler")):
        ax.errorbar(steps, out[scheme]["err"], yerr=2 * np.array(out[scheme]["se"]), color=color,
                    lw=2, marker="o", ms=5, capsize=3, label=label)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("time steps per year")
    ax.set_ylabel("MC price - Fourier price")
    ax.set_title("Discretisation bias, Andersen case I (T = 10y, ATM, error bars = 2 SE)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\n== Bias vs step size (case I, reference {ref:.4f}) ==")
    for i, s in enumerate(steps):
        print(f"{s:3d} steps/yr: QE {out['qe']['err'][i]:+.4f} (SE {out['qe']['se'][i]:.4f})   "
              f"Euler {out['euler']['err'][i]:+.4f} (SE {out['euler']['se'][i]:.4f})")
    return {"steps_per_year": steps, "reference": ref, **out}


def bs_limit(path: Path):
    S, T, r, v0, k, th = 100.0, 1.0, 0.03, 0.06, 2.0, 0.04
    K = np.array([80.0, 100.0, 120.0])
    s_eff = np.sqrt(th + (v0 - th) * (1 - np.exp(-k * T)) / (k * T))
    target = bs_price(S, K, T, r, s_eff)
    xis = np.logspace(-3.5, -0.5, 13)
    errs = {rho: [float(np.abs(heston_price(S, K, T, r, v0, k, th, x, rho) - target).max()) for x in xis]
            for rho in (-0.5, 0.0)}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(xis, errs[-0.5], color=ORANGE, lw=2, marker="o", ms=4, label="rho = -0.5")
    ax.loglog(xis, errs[0.0], color=BLUE, lw=2, marker="o", ms=4, label="rho = 0")
    ax.loglog(xis, errs[-0.5][0] * xis / xis[0], color=MUTED, ls="--", lw=1)
    ax.loglog(xis, errs[0.0][0] * (xis / xis[0]) ** 2, color=MUTED, ls=":", lw=1)
    ax.text(xis[6], errs[-0.5][0] * xis[6] / xis[0] * 3.0, "slope 1", color=MUTED, fontsize=9)
    ax.text(xis[6], errs[0.0][0] * (xis[6] / xis[0]) ** 2 * 0.25, "slope 2", color=MUTED, fontsize=9)
    ax.set_xlabel("vol of vol  xi")
    ax.set_ylabel("max |Heston - Black-Scholes|")
    ax.set_title("Heston converges to Black-Scholes as xi -> 0", loc="left", fontsize=10)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return {"xi": xis.tolist(), "max_err": {str(k_): v for k_, v in errs.items()}}


def timing():
    rng = np.random.default_rng(0)
    n = 20_000
    p = dict(S=rng.uniform(70, 130, n), K=100.0, T=rng.uniform(0.05, 2, n), r=rng.uniform(0, 0.08, n),
             v0=rng.uniform(0.005, 0.25, n), kappa=rng.uniform(0.5, 5, n), theta=rng.uniform(0.005, 0.25, n),
             xi=rng.uniform(0.1, 1, n), rho=rng.uniform(-0.95, 0, n))
    t0 = time.perf_counter(); heston_price(**p); t_gl = (time.perf_counter() - t0) / n
    m = 200
    t0 = time.perf_counter()
    for i in range(m):
        heston_price_quad(*(p[k] if np.isscalar(p[k]) else p[k][i] for k in p))
    t_quad = (time.perf_counter() - t0) / m
    t0 = time.perf_counter()
    heston_mc(100, 100, 1.0, 0.03, 0.04, 1.5, 0.06, 0.5, -0.7, n_paths=100_000, steps_per_year=100)
    t_mc = time.perf_counter() - t0
    out = {"gl256_us_per_option_batch20k": 1e6 * t_gl, "quad_ms_per_option": 1e3 * t_quad,
           "mc_s_per_option_100k_paths_T1": t_mc}
    print(f"\n== Timing (this machine) ==\nFourier, vectorised adaptive Gauss-Legendre (batch of {n:,}): "
          f"{out['gl256_us_per_option_batch20k']:.0f} us/option\nFourier, adaptive quad: "
          f"{out['quad_ms_per_option']:.2f} ms/option\nMonte Carlo QE (100k paths, 100 steps): "
          f"{t_mc:.2f} s/option-parameter-set")
    return out


def main():
    Path("plots").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    res = {"andersen": andersen_table(),
           "bias_vs_steps": bias_vs_steps(Path("plots/stage3_bias_vs_steps.png")),
           "bs_limit": bs_limit(Path("plots/stage3_bs_limit.png")),
           "timing": timing()}
    Path("results/stage3_heston_validation.json").write_text(json.dumps(res, indent=2))
    print("\nsaved results/stage3_heston_validation.json and plots/stage3_*.png")


if __name__ == "__main__":
    main()
