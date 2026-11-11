"""Stage 5: volatility smiles, and the NN surrogate's accuracy in implied-vol terms.

1. Smile plots: Heston prices (Fourier) -> implied vols with the Black-Scholes solver. Black-Scholes
   would give a flat line (one vol for every strike); Heston gives a skew/smile. The NN's prices are
   overlaid (dashed) to show they reproduce it.
2. Surface plot: implied vol over (strike, maturity), and the NN's error over the same grid.
3. Implied-vol accuracy on the held-out test set: IV(NN price) - IV(Fourier price), in vol points
   (1 vol point = 0.01 = one percentage point of volatility), for options whose IV is well identified
   (vega >= --min-vega), overall, by moneyness x maturity, by Feller ratio, and for a "market-like"
   region (0.9 <= S/K <= 1.1, T >= 0.1).

Why vol points rather than price error: IV error ~ price error / vega, so the same 1 bp price error
is a small vol error for a vega-rich near-the-money 1-year option and a large one for a cheap,
short-dated wing option. Options with tiny vega carry almost no volatility information (the price
barely moves with sigma), so they are excluded and the share excluded is reported.

Usage:
    python smile.py --model models/heston_mlp.pt --data data/heston
    python smile.py --no-nn                       # Fourier-only smile plots (no PyTorch needed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from black_scholes import bs_greeks, implied_volatility_bisect, implied_volatility_vec
from evaluate import (BLUE, INK, M_EDGES, M_LABELS, MUTED, ORANGE, T_EDGES, T_LABELS,
                      _style, bucket_table, format_table)
from evaluate_heston import FELLER_EDGES, FELLER_LABELS, feller_ratio, group_table
from heston import heston_price

PriceFn = Callable[[np.ndarray], np.ndarray]     # (n, 8) Heston features -> (n,) call price / K

R_RATE = 0.03
MATURITIES = (0.1, 0.25, 0.5, 1.0)
PRESETS = {
    "Equity-like skew (Feller holds: 2*kappa*theta/xi^2 = 1.25)":
        dict(v0=0.04, kappa=2.0, theta=0.05, xi=0.4, rho=-0.7),
    "Steep skew (Feller violated: ratio = 0.33)":
        dict(v0=0.03, kappa=1.5, theta=0.04, xi=0.6, rho=-0.8),
}
SHADES = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]    # one hue, light -> dark = short -> long maturity
DIVERGING = ["#eb6834", "#f0efec", "#2a78d6"]            # orange = NN too LOW (negative), grey = exact, blue = NN too HIGH (positive)


# --------------------------------------------------------------------------
# inputs and implied vols
# --------------------------------------------------------------------------
def grid_inputs(params: dict, T: np.ndarray, strike_pct: np.ndarray, r: float = R_RATE) -> np.ndarray:
    """Feature rows (S/K, T, r, v0, kappa, theta, xi, rho) for a strike x maturity grid (K = 1)."""
    TT, KK = np.meshgrid(np.asarray(T, float), np.asarray(strike_pct, float) / 100.0, indexing="ij")
    m = 1.0 / KK.ravel()                                    # S/K with S = 1
    n = m.size
    cols = [m, TT.ravel(), np.full(n, r)] + [np.full(n, params[k]) for k in ("v0", "kappa", "theta", "xi", "rho")]
    return np.column_stack(cols)


def fourier_price_fn(X: np.ndarray) -> np.ndarray:
    m, T, r, v0, kappa, theta, xi, rho = X.T
    return np.asarray(heston_price(m, 1.0, T, r, v0, kappa, theta, xi, rho))


def iv_calls(X: np.ndarray, price: np.ndarray, solver: str = "bisect") -> np.ndarray:
    """Implied vol of call prices (K = 1). 'brent' = the Stage-1 scalar solver, 'bisect' = vectorised."""
    m, T, r = X[:, 0], X[:, 1], X[:, 2]
    if solver == "brent":
        return implied_volatility_vec(m, 1.0, T, r, price, "call", errors="nan")
    return implied_volatility_bisect(m, 1.0, T, r, price, "call")


# --------------------------------------------------------------------------
# implied-vol accuracy on the test set (torch-free, unit-tested)
# --------------------------------------------------------------------------
def _stats(abs_err: np.ndarray, signed: np.ndarray) -> dict:
    if abs_err.size == 0:
        return {"n": 0}
    return {"n": int(abs_err.size), "mean_abs": float(abs_err.mean()), "median_abs": float(np.median(abs_err)),
            "p90_abs": float(np.quantile(abs_err, 0.90)), "p99_abs": float(np.quantile(abs_err, 0.99)),
            "max_abs": float(abs_err.max()), "mean_signed": float(signed.mean()),
            "frac_below_0.1": float(np.mean(abs_err < 0.1)), "frac_below_0.25": float(np.mean(abs_err < 0.25)),
            "frac_below_0.5": float(np.mean(abs_err < 0.5))}


def iv_error_report(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, min_vega: float = 0.05) -> dict:
    """IV(pred) - IV(true) in vol points, for options with vega >= min_vega (vega per unit strike)."""
    m, T, r = X[:, 0], X[:, 1], X[:, 2]
    iv_t, iv_p = iv_calls(X, y_true), iv_calls(X, y_pred)
    ok_t = np.isfinite(iv_t)
    vega = np.zeros(len(X))
    vega[ok_t] = bs_greeks(m[ok_t], 1.0, T[ok_t], r[ok_t], iv_t[ok_t])["vega"]
    eligible = ok_t & (vega >= min_vega)
    pred_invalid = eligible & ~np.isfinite(iv_p)             # NN price outside no-arbitrage bounds
    used = eligible & np.isfinite(iv_p)
    signed = 100.0 * (iv_p - iv_t)                           # vol points
    abs_err = np.abs(signed)

    market = used & (m >= 1 / 1.1) & (m <= 1 / 0.9) & (T >= 0.1)
    calib = used & (m >= 1 / 1.15) & (m <= 1 / 0.87) & (T >= 0.25)       # strikes 87%-115% of spot, T >= 0.25y
    bt = bucket_table(m[used], T[used], abs_err[used])
    kept = bucket_table(m[eligible], T[eligible], np.zeros(eligible.sum()))["count"]
    total = bucket_table(m, T, np.zeros(len(m)))["count"]
    fr = feller_ratio(X[:, 4], X[:, 5], X[:, 6])
    return {
        "min_vega": min_vega, "n_total": int(len(X)),
        "n_eligible": int(eligible.sum()), "frac_eligible": float(eligible.mean()),
        "n_pred_price_outside_bounds": int(pred_invalid.sum()),
        "overall": _stats(abs_err[used], signed[used]),
        "market_like": _stats(abs_err[market], signed[market]),
        "calibration_region": _stats(abs_err[calib], signed[calib]),
        "bucket_mae": bt["mae"], "bucket_max": bt["max"], "bucket_count": bt["count"],
        "bucket_kept_share": kept / np.maximum(total, 1),
        "by_feller": group_table(fr[used], FELLER_EDGES, FELLER_LABELS, abs_err[used]),
        "worst": [{"err_vol_points": float(signed[i]), "iv_true": float(iv_t[i]), "vega": float(vega[i]),
                   "features": X[i].tolist()} for i in np.nonzero(used)[0][np.argsort(-abs_err[used])[:5]]],
    }


def print_report(rep: dict) -> None:
    o, mk, cr = rep["overall"], rep["market_like"], rep["calibration_region"]
    print(f"\n== Implied-vol accuracy of the NN on the held-out test set (vol points; 1 = one percentage point) ==")
    print(f"Options with vega >= {rep['min_vega']}: {rep['n_eligible']:,} of {rep['n_total']:,} "
          f"({rep['frac_eligible']:.1%}); NN price outside no-arbitrage bounds for {rep['n_pred_price_outside_bounds']}")
    hdr = f"{'':>40s} {'n':>8s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'bias':>7s}"
    print(hdr)
    for name, s in (("all eligible options", o), ("market-like (0.9-1.1, T>=0.1)", mk),
                    ("calibration region (K 87-115%, T>=.25)", cr)):
        if s.get("n"):
            print(f"{name:>40s} {s['n']:8d} {s['mean_abs']:7.3f} {s['median_abs']:7.3f} {s['p90_abs']:7.3f} "
                  f"{s['p99_abs']:7.3f} {s['max_abs']:7.3f} {s['mean_signed']:+7.3f}")
    if o.get("n"):
        print(f"share of eligible options with |error| < 0.1 / 0.25 / 0.5 vol points: "
              f"{o['frac_below_0.1']:.1%} / {o['frac_below_0.25']:.1%} / {o['frac_below_0.5']:.1%}")
    print("\nMean |IV error| by bucket (vol points):\n" + format_table(rep["bucket_mae"], scale=1.0, fmt="{:8.3f}"))
    print("\nMax |IV error| by bucket (vol points):\n" + format_table(rep["bucket_max"], scale=1.0, fmt="{:8.3f}"))
    print("\nShare of each bucket kept by the vega filter:\n" + format_table(rep["bucket_kept_share"], scale=100.0, fmt="{:7.0f}%"))
    print("\nBy Feller ratio 2*kappa*theta/xi^2:")
    print(f"{'ratio':>20s} {'n':>8s} {'mean |err|':>11s} {'max |err|':>10s}")
    for g in rep["by_feller"]:
        print(f"{g['group']:>20s} {g['n']:8d} {g['mae']:11.3f} {g['max']:10.3f}")
    print("\n5 worst options (error in vol points):")
    for w in rep["worst"]:
        f = w["features"]
        print(f"  {w['err_vol_points']:+7.3f}  IV {w['iv_true']:.3f} vega {w['vega']:.3f}  S/K={f[0]:.3f} T={f[1]:.2f} "
              f"v0={f[3]:.3f} kappa={f[4]:.2f} theta={f[5]:.3f} xi={f[6]:.2f} rho={f[7]:.2f}")
    print("\nThe vega filter hides cheap short-dated wing options (see the smile plot): do not trust the NN there.")
    print("Rule of thumb (my judgement, not a standard): a mean error under ~0.1 vol point is negligible for\n"
          "calibration; a typical error above ~0.5 vol point would need a better network.")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def error_cmap():
    """Diverging map for (NN - Fourier) implied vol: negative -> orange, 0 -> grey, positive -> blue."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("nn_err", DIVERGING)


def plot_smiles(path: Path, nn_price_fn: PriceFn | None = None) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    strikes = np.linspace(80, 125, 46)
    fig, axes = plt.subplots(1, len(PRESETS), figsize=(12.5, 4.4), sharey=False)
    summary = {}
    for ax, (title, hp) in zip(np.atleast_1d(axes), PRESETS.items()):
        for T, color in zip(MATURITIES, SHADES):
            X = grid_inputs(hp, np.array([T]), strikes)
            iv = 100 * iv_calls(X, fourier_price_fn(X), solver="brent")        # the Stage-1 solver
            ax.plot(strikes, iv, color=color, lw=2.2, label=f"{T:g}y")
            if nn_price_fn is not None:
                iv_nn = 100 * iv_calls(X, nn_price_fn(X), solver="brent")
                ax.plot(strikes, iv_nn, color=color, lw=1.6, ls="--")
            if T == 1.0:
                atm = float(np.interp(100, strikes, iv))
                ax.axhline(atm, color=MUTED, lw=1, ls=":")
                ax.text(strikes[-1] + 0.5, atm + 0.15, "Black-Scholes:\none vol for\nevery strike", color=MUTED,
                        fontsize=8, va="bottom", ha="left")
                summary[title] = {"atm_iv_1y": atm, "iv_80pct": float(iv[0]), "iv_125pct": float(iv[-1])}
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("strike, % of spot")
        ax.set_ylabel("implied volatility, %")
        ax.set_xlim(strikes[0], strikes[-1] + 12)
        _style(ax)
        handles, labels = ax.get_legend_handles_labels()
        extra = [Line2D([], [], color=INK, lw=2.2), Line2D([], [], color=INK, lw=1.6, ls="--")] if nn_price_fn else []
        ax.legend(handles + extra, labels + (["Heston (Fourier)", "NN"] if nn_price_fn else []),
                  frameon=False, fontsize=8, title="maturity", title_fontsize=8)
    fig.suptitle("Heston produces a volatility skew; Black-Scholes cannot (r = 3%)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary


def plot_surface(path: Path, nn_price_fn: PriceFn | None = None, preset: str | None = None) -> dict:
    import matplotlib.pyplot as plt
    title, hp = (preset, PRESETS[preset]) if preset else next(iter(PRESETS.items()))
    strikes = np.linspace(80, 125, 46)
    mats = np.linspace(0.05, 2.0, 40)
    X = grid_inputs(hp, mats, strikes)
    iv = 100 * iv_calls(X, fourier_price_fn(X)).reshape(len(mats), len(strikes))
    ncol = 2 if nn_price_fn is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.4))
    axes = np.atleast_1d(axes)
    im = axes[0].pcolormesh(strikes, mats, np.ma.masked_invalid(iv), cmap="Blues", shading="auto")
    fig.colorbar(im, ax=axes[0], label="implied volatility, %")
    axes[0].set_title("Heston implied-vol surface (Fourier)", loc="left", fontsize=10)
    stats = {"preset": title, "iv_min": float(np.nanmin(iv)), "iv_max": float(np.nanmax(iv))}
    if nn_price_fn is not None:
        iv_nn = 100 * iv_calls(X, nn_price_fn(X)).reshape(iv.shape)
        err = iv_nn - iv                                     # vol points; + means NN too high
        lim = max(0.05, float(np.nanquantile(np.abs(err), 0.99)))
        im2 = axes[1].pcolormesh(strikes, mats, np.ma.masked_invalid(err), cmap=error_cmap(), vmin=-lim, vmax=lim, shading="auto")
        fig.colorbar(im2, ax=axes[1], label="NN minus Fourier, vol points (+ = NN too high)")
        axes[1].set_title("NN implied-vol error (blue: NN too high, orange: NN too low; white: no valid vol)",
                          loc="left", fontsize=9)
        stats.update({"grid_mean_abs_err": float(np.nanmean(np.abs(err))), "grid_max_abs_err": float(np.nanmax(np.abs(err)))})
    for ax in axes:
        ax.set_xlabel("strike, % of spot")
        ax.set_ylabel("maturity, years")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
def load_nn_price_fn(path: Path) -> PriceFn:
    """Wrap a trained checkpoint as a function (n, 8) features -> (n,) call price / K (float64, CPU)."""
    import torch
    from nn_pricer import load_model
    model, ckpt = load_model(path)
    from nn_pricer import HESTON_FEATURES
    cfg = ckpt["config"]

    def price(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        with torch.no_grad():
            return np.concatenate([model(torch.tensor(X[i:i + 50_000])).numpy()
                                   for i in range(0, len(X), 50_000)])
    price.box = dict(zip(HESTON_FEATURES, zip(cfg["lo"], cfg["hi"])))   # the network's training box
    return price


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/heston_mlp.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/heston"))
    ap.add_argument("--min-vega", type=float, default=0.05)
    ap.add_argument("--no-nn", action="store_true", help="Fourier-only plots; skip the NN comparison")
    args = ap.parse_args(argv)
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    nn_fn = None if args.no_nn else load_nn_price_fn(args.model)
    res: dict = {"smiles": plot_smiles(Path("plots/stage5_smiles.png"), nn_fn)}
    res["surface"] = plot_surface(Path("plots/stage5_surface.png"), nn_fn)
    print("Saved plots/stage5_smiles.png and plots/stage5_surface.png")
    for k, v in res["smiles"].items():
        print(f"  {k}: 1y ATM vol {v['atm_iv_1y']:.1f}%, vol at 80% strike {v['iv_80pct']:.1f}%, at 125% strike {v['iv_125pct']:.1f}%")

    if nn_fn is not None:
        test = np.load(args.data / "test.npz")
        rep = iv_error_report(test["X"], test["y"], nn_fn(test["X"]), args.min_vega)
        print_report(rep)
        res["iv_error"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rep.items()}
    Path("results/stage5_smile.json").write_text(json.dumps(res, indent=2))
    print("saved results/stage5_smile.json")
    return res


if __name__ == "__main__":
    main()
