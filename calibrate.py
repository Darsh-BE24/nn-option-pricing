"""Stage 6: calibrate Heston to a real NSE NIFTY option chain, using the NN surrogate as the pricer.

    python calibrate.py --csv data/nifty_chain.csv --date 2026-11-04 --expiry 2027-01-25 --r 0.06
    python calibrate.py --synthetic                      # parameter-recovery test on a fake chain

Method
  * Forward F from put-call parity (zero crossing of C_mid - P_mid), so no spot / dividend input is needed.
  * Price with S_eff = F*exp(-rT), q = 0: same terminal law as spot + dividends, so Heston is unchanged.
  * Use OTM quotes only (puts below F, calls above F): they are the liquid ones, and IV is the same either way.
  * Least squares on IV (vol points), weighted by 1/(bid-ask width in vol), multi-start, bounds = training box.
  * Afterwards re-price the fitted parameters with the EXACT Fourier pricer: that, not the NN's own
    number, is the honest fit quality.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import black_scholes as bs
from generate_data import PARAM_RANGES
from heston import heston_price
from nse_chain import load_chain

PriceFn = Callable[[np.ndarray], np.ndarray]
NAMES = ["v0", "kappa", "theta", "xi", "rho"]
BOX = PARAM_RANGES["heston"]


# ----------------------------------------------------------------------------- market side
def forward_from_parity(df: pd.DataFrame, max_spread: float = 30.0, r: float | None = None,
                        T: float | None = None) -> float:
    """F where C_mid - P_mid crosses zero (independent of the discount factor)."""
    d = df.dropna(subset=["c_bid", "c_ask", "p_bid", "p_ask"]).copy()
    d["y"] = (d.c_bid + d.c_ask) / 2 - (d.p_bid + d.p_ask) / 2
    d["sp"] = (d.c_ask - d.c_bid) + (d.p_ask - d.p_bid)
    d = d[np.abs(d.y) < 0.03 * (d.strike + d.y)].sort_values("strike")   # strikes within ~3% of the forward
    d = d.iloc[np.argsort(d.sp.values)[:8]].sort_values("strike")        # the 8 tightest two-sided quotes
    if len(d) < 3:
        if r is None or T is None or len(d) < 1:
            raise ValueError("not enough tight two-sided strikes near the forward to infer it")
        # sparse chain (e.g. 1500-point strike spacing): one tight strike + assumed discount factor
        k = d.iloc[np.argmin(d.sp.values)]
        return float(k.strike + k.y / np.exp(-r * T))
    # y = D(F-K): fit a line through the tight strikes nearest to zero, weights 1/spread^2
    near = d.iloc[np.argsort(np.abs(d.y.values))[:6]]
    A = np.column_stack([np.ones(len(near)), -near.strike.values])
    w = 1.0 / near.sp.values ** 2
    c0, D = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * near.y.values))
    return float(c0 / D) if 0.9 < D < 1.0 else float(np.average(near.strike + near.y / 0.985, weights=w))


def build_market(df: pd.DataFrame, F: float, T: float, r: float, max_rel_spread: float = 0.12,
                 max_abs_logm: float | None = None, min_iv_width: float = 0.001) -> pd.DataFrame:
    """OTM options with sane two-sided quotes -> IV from mid, with bid/ask IV as the uncertainty."""
    S = F * np.exp(-r * T)
    if max_abs_logm is None:                  # window of about +-2 standard deviations, 15% vol: 0.30*sqrt(T)
        max_abs_logm = float(np.clip(0.30 * np.sqrt(T), 0.08, 0.20))
    rows = []
    for _, x in df.iterrows():
        K = x.strike
        if abs(np.log(K / F)) > max_abs_logm:
            continue
        typ, b, a = ("put", x.p_bid, x.p_ask) if K < F else ("call", x.c_bid, x.c_ask)
        if not (np.isfinite(b) and np.isfinite(a)) or b <= 0 or a <= b:
            continue
        rel = (a - b) / ((a + b) / 2)
        if rel > max_rel_spread:
            continue
        iv = [float(bs.implied_volatility_bisect(S, K, T, r, p, typ)) for p in (b, (a + b) / 2, a)]
        if not np.all(np.isfinite(iv)):
            continue
        rows.append(dict(strike=K, typ=typ, bid=b, ask=a, rel_spread=rel,
                         iv_bid=iv[0], iv=iv[1], iv_ask=iv[2], width=max(iv[2] - iv[0], min_iv_width)))
    return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------------------- model side
def features(p: np.ndarray, S: float, K: np.ndarray, T: float, r: float) -> np.ndarray:
    v0, kappa, theta, xi, rho = p
    n = len(K)
    return np.column_stack([S / K, np.full(n, T), np.full(n, r), np.full(n, v0), np.full(n, kappa),
                            np.full(n, theta), np.full(n, xi), np.full(n, rho)])


def model_iv(price_fn: PriceFn, p: np.ndarray, S: float, K: np.ndarray, T: float, r: float) -> np.ndarray:
    """Model implied vol per strike (via call prices; OTM put IV == call IV at the same strike)."""
    X = features(p, S, K, T, r)
    c = np.asarray(price_fn(X), dtype=float) * K          # price / K -> price
    return bs.implied_volatility_bisect(S, K, T, r, c, "call")


def calibrate(price_fn: PriceFn, mkt: pd.DataFrame, S: float, T: float, r: float, n_starts: int = 8,
              seed: int = 0, verbose: bool = True, fixed: dict | None = None,
              x0: list | None = None, residual: str = "price", box: dict | None = None) -> list[dict]:
    """`fixed` pins some parameters (e.g. {'kappa': 2.0}); a single expiry cannot identify kappa."""
    fixed = fixed or {}
    free = np.array([n not in fixed for n in NAMES])
    fixed_vals = np.array([fixed.get(n, 0.0) for n in NAMES])
    K, iv = mkt.strike.values, mkt.iv.values
    w = mkt.wgt.values if "wgt" in mkt.columns else 1.0 / mkt.width.values   # 'wgt': equal weight per expiry
    w = w / w.mean()
    box = box or getattr(price_fn, "box", None) or BOX   # a network's own training box if it has one
    lo = np.array([box[n][0] for n in NAMES]); hi = np.array([box[n][1] for n in NAMES])
    pad = 0.02 * (hi - lo)
    lo, hi = lo + pad, hi - pad                            # stay strictly inside the training box

    # price-space residual: (C_model - C_mkt) / vega_mkt ~ IV error, with no IV inversion inside the loop
    c_mkt = np.asarray(bs.bs_price(S, K, T, r, iv, "call"), dtype=float)
    vega = np.asarray(bs.bs_greeks(S, K, T, r, iv, "call")["vega"], dtype=float)

    def full(q):
        p = fixed_vals.copy(); p[free] = q
        return p

    def resid(q):
        if residual == "price":
            m = np.asarray(price_fn(features(full(q), S, K, T, r)), dtype=float) * K
            return (m - c_mkt) / vega * w
        m = model_iv(price_fn, full(q), S, K, T, r)
        return np.where(np.isfinite(m), m - iv, 1.0) * w   # NaN (price outside bounds) -> big penalty

    rng = np.random.default_rng(seed)
    starts = [np.array([0.02, 2.0, 0.03, 0.5, -0.7])]      # a plausible equity-index start
    starts += [lo + rng.random(5) * (hi - lo) for _ in range(n_starts - 1)]
    if x0 is not None:                                     # explicit starting points (e.g. the NN solution)
        starts = [np.asarray(x, dtype=float) for x in x0]
    out = []
    for i, p0 in enumerate(starts):
        t0 = time.perf_counter()
        p0 = np.clip(p0, lo, hi)
        sol = least_squares(resid, p0[free], bounds=(lo[free], hi[free]), x_scale=(hi - lo)[free],
                            diff_step=1e-3, max_nfev=200)
        dt = time.perf_counter() - t0
        p = full(sol.x)
        m = model_iv(price_fn, p, S, K, T, r)
        rmse = float(np.sqrt(np.nanmean((m - iv) ** 2)) * 100)
        rec = dict(start=i, params=dict(zip(NAMES, map(float, p))), rmse_vol_pts=rmse, cost=float(sol.cost),
                   feller=float(2 * p[1] * p[2] / p[3] ** 2), nfev=int(sol.nfev), seconds=dt)
        out.append(rec)
        if verbose:
            print(f"  start {i}: rmse {rmse:.3f} vol pts  v0={p[0]:.4f} k={p[1]:.2f} th={p[2]:.4f} "
                  f"xi={p[3]:.3f} rho={p[4]:.2f}  Feller {rec['feller']:.2f}  ({dt:.1f}s, {sol.nfev} evals)")
    return sorted(out, key=lambda d: d["cost"])


def exact_iv(p: dict, S: float, K: np.ndarray, T: float, r: float) -> np.ndarray:
    c = heston_price(S / K, 1.0, T, r, p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"]) * K
    return bs.implied_volatility_bisect(S, K, T, r, np.asarray(c), "call")


def fourier_fn(X: np.ndarray) -> np.ndarray:
    m, T, r, v0, kappa, theta, xi, rho = X.T
    return np.asarray(heston_price(m, 1.0, T, r, v0, kappa, theta, xi, rho))


def load_nn(path: Path) -> PriceFn:
    from smile import load_nn_price_fn
    return load_nn_price_fn(path)


# ----------------------------------------------------------------------------- plots / report
def plot_fit(path: Path, mkt, iv_nn, iv_ex, F, title) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    k = np.log(mkt.strike / F)
    a.vlines(k, mkt.iv_bid * 100, mkt.iv_ask * 100, color="#9aa0a6", lw=1.2, label="market bid-ask")
    a.plot(k, mkt.iv * 100, "o", ms=4, color="#444", label="market mid")
    a.plot(k, iv_ex * 100, "-", lw=2, color="#1f77b4", label="Heston fit (exact Fourier)")
    a.plot(k, iv_nn * 100, "--", lw=1.6, color="#e8710a", label="Heston fit (NN pricer)")
    a.set_ylabel("implied vol (%)"); a.set_title(title, fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8)
    b.axhline(0, color="#888", lw=0.8)
    b.plot(k, (iv_ex - mkt.iv) * 100, "o-", ms=3, lw=1, color="#1f77b4")
    b.set_ylabel("fit - market\n(vol pts)"); b.set_xlabel("log-moneyness ln(K/F)")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/nifty_chain.csv")
    ap.add_argument("--date", default="2026-11-04"); ap.add_argument("--expiry", default="2027-01-25")
    ap.add_argument("--r", type=float, default=0.06, help="assumed risk-free rate (annual, cont.)")
    ap.add_argument("--model", default="models/heston_mlp.pt")
    ap.add_argument("--pricer", choices=["nn", "fourier"], default="nn")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true", help="recover known parameters from a fake chain")
    ap.add_argument("--noise", type=float, default=0.1, help="synthetic IV noise (vol pts)")
    ap.add_argument("--time-fourier", action="store_true", help="also run the same multi-start with the exact pricer")
    ap.add_argument("--polish", action="store_true", help="refine the NN solution with the exact pricer")
    ap.add_argument("--fix-kappa", type=float, default=None, help="pin kappa (single expiry cannot identify it)")
    ap.add_argument("--residual", choices=["price", "iv"], default="price",
                    help="price: (model-mkt price)/vega, fast; iv: invert every model price, slow")
    ap.add_argument("--tag", default="nifty")
    a = ap.parse_args(argv)

    price_fn = load_nn(Path(a.model)) if a.pricer == "nn" else fourier_fn
    T = (pd.Timestamp(a.expiry) - pd.Timestamp(a.date)).days / 365.0
    r = a.r
    if a.synthetic:
        truth = dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        F = 24000.0; S = F * np.exp(-r * T)
        K = np.arange(21000, 25001, 100.0)
        iv_true = exact_iv(truth, S, K, T, r)
        rng = np.random.default_rng(1)
        iv = iv_true + rng.normal(0, a.noise / 100, len(K))
        mkt = pd.DataFrame(dict(strike=K, iv=iv, iv_bid=iv - 0.002, iv_ask=iv + 0.002, width=0.004))
        print(f"synthetic chain: T={T:.3f}, {len(K)} strikes, noise {a.noise} vol pts, truth {truth}")
    else:
        df = load_chain(a.csv)
        F = forward_from_parity(df); S = F * np.exp(-r * T)
        mkt = build_market(df, F, T, r)
        truth = None
        print(f"T={T:.4f}y  forward={F:.1f}  r={r:.3f}  S_eff={S:.1f}  usable OTM quotes: {len(mkt)} "
              f"(K {mkt.strike.min():.0f}-{mkt.strike.max():.0f})")

    print(f"calibrating with {a.pricer} pricer, {a.starts} starts:")
    t0 = time.perf_counter()
    fixed = {'kappa': a.fix_kappa} if a.fix_kappa is not None else None
    fits = calibrate(price_fn, mkt, S, T, r, a.starts, fixed=fixed, residual=a.residual)
    total = time.perf_counter() - t0
    best = fits[0]; p = best["params"]; K = mkt.strike.values
    pv = np.array([p[n] for n in NAMES])
    iv_nn = model_iv(price_fn, pv, S, K, T, r)
    iv_ex = exact_iv(p, S, K, T, r)
    rm = lambda x: float(np.sqrt(np.mean(x ** 2)) * 100)
    res = dict(T=T, forward=F, r=r, n_quotes=int(len(mkt)), pricer=a.pricer, best=best,
               rmse_nn_vs_market=rm(iv_nn - mkt.iv.values), rmse_exact_vs_market=rm(iv_ex - mkt.iv.values),
               max_abs_exact_vs_market=float(np.max(np.abs(iv_ex - mkt.iv.values)) * 100),
               rmse_nn_vs_exact_at_fit=rm(iv_nn - iv_ex),
               frac_inside_bidask=float(np.mean((iv_ex >= mkt.iv_bid.values) & (iv_ex <= mkt.iv_ask.values))),
               calib_seconds_total=total, n_starts=a.starts, all_fits=fits)
    good = [f for f in fits if f["rmse_vol_pts"] < best["rmse_vol_pts"] + 0.05]
    res["n_starts_within_0.05pt"] = len(good)
    res["param_spread_among_good"] = {n: [min(f["params"][n] for f in good), max(f["params"][n] for f in good)] for n in NAMES}
    print(f"\nbest: {p}\n  Feller ratio {best['feller']:.2f}")
    print(f"  RMSE vs market  (NN pricer)    : {res['rmse_nn_vs_market']:.3f} vol pts")
    print(f"  RMSE vs market  (exact Fourier): {res['rmse_exact_vs_market']:.3f} vol pts   <- honest fit quality")
    print(f"  worst strike (exact)           : {res['max_abs_exact_vs_market']:.3f} vol pts")
    print(f"  NN vs exact at the fitted params: {res['rmse_nn_vs_exact_at_fit']:.3f} vol pts")
    if a.synthetic is False:
        print(f"  exact fit inside market bid-ask: {100 * res['frac_inside_bidask']:.0f}% of strikes")
    print(f"  {len(good)}/{a.starts} starts reach within 0.05 pt of the best; parameter ranges among them:")
    for n in NAMES:
        print(f"     {n:6s} {res['param_spread_among_good'][n][0]:8.4f} .. {res['param_spread_among_good'][n][1]:8.4f}")
    if truth:
        print("  truth  :", {n: round(truth[n], 4) for n in NAMES})
        print("  fitted :", {n: round(p[n], 4) for n in NAMES})
        res["truth"] = truth
    if a.polish:
        t1 = time.perf_counter()
        pol = calibrate(fourier_fn, mkt, S, T, r, 1, verbose=False, fixed=fixed, x0=[pv], residual=a.residual)[0]
        tp = time.perf_counter() - t1
        res["polish"] = pol; res["polish_seconds"] = tp
        res["total_nn_plus_polish_seconds"] = total + tp
        pp = pol["params"]
        res["polish_rmse_exact_vs_market"] = rm(exact_iv(pp, S, K, T, r) - mkt.iv.values)
        print(f"  POLISH (exact pricer from NN solution): rmse {pol['rmse_vol_pts']:.3f} vol pts in {tp:.1f}s "
              f"({pol['nfev']} evals); NN stage {total:.1f}s -> total {total + tp:.1f}s")
        print("    polished params:", {n: round(pp[n], 4) for n in NAMES}, f"Feller {pol['feller']:.2f}")
    if a.time_fourier:
        t1 = time.perf_counter()
        ex = calibrate(fourier_fn, mkt, S, T, r, a.starts, verbose=False, fixed=fixed, residual=a.residual)
        tf = time.perf_counter() - t1
        res["fourier_multistart_seconds"] = tf; res["fourier_best"] = ex[0]
        n_nn = sum(f["nfev"] for f in fits); n_ex = sum(f["nfev"] for f in ex)
        print(f"  EXACT pricer, same {a.starts} starts: {tf:.1f}s, best rmse {ex[0]['rmse_vol_pts']:.3f} vol pts, "
              f"{n_ex} evals ({1000 * tf / n_ex:.1f} ms/eval)")
        print(f"  NN pricer,    same {a.starts} starts: {total:.1f}s, best rmse(exact-checked) "
              f"{res['rmse_exact_vs_market']:.3f}, {n_nn} evals ({1000 * total / n_nn:.1f} ms/eval)")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/calibration_{a.tag}.json").write_text(json.dumps(res, indent=2))
    plot_fit(Path(f"plots/calibration_{a.tag}.png"), mkt, iv_nn, iv_ex, F,
             f"Heston fit to {'synthetic chain' if a.synthetic else 'NIFTY ' + a.expiry} (T={T:.2f}y), {a.pricer} pricer")
    return res


if __name__ == "__main__":
    main()
