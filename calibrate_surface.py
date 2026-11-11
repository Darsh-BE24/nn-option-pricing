"""Stage 6b: calibrate ONE Heston parameter set to several NIFTY expiries at once.

    python calibrate_surface.py --date 2026-11-05 --csv data/a.csv data/b.csv data/c.csv --polish --time-fourier
    python calibrate_surface.py --synthetic --pricer fourier          # parameter-recovery test

Expiry dates are read from the NSE file names (...-25-Jan-2027.csv). Each expiry gets its own forward
(put-call parity), so dividends / carry differ per expiry without changing Heston; r is common.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate as cal
from nse_chain import load_chain

NAMES = cal.NAMES


def parse_expiry(path: str) -> pd.Timestamp:
    m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", Path(path).name)
    if not m:
        raise ValueError(f"cannot find an expiry like 25-Jan-2027 in file name {path!r}")
    return pd.to_datetime(m.group(1), format="%d-%b-%Y")


def load_surface(csvs, date: str, r: float, min_T: float = 0.0, **kw) -> pd.DataFrame:
    parts = []
    for path in csvs:
        exp = parse_expiry(path)
        T = (exp - pd.Timestamp(date)).days / 365.0
        df = load_chain(path)
        try:
            F = cal.forward_from_parity(df, r=r, T=T)
        except ValueError as e:
            print(f"  {exp.date()}: skipped ({e})"); continue
        m = cal.build_market(df, F, T, r, **kw)
        m["T"], m["F"], m["S"], m["expiry"] = T, F, F * np.exp(-r * T), exp.strftime("%d-%b-%Y")
        if len(m) < 6:
            print(f"  {exp.date()}: skipped (only {len(m)} usable quotes)"); continue
        print(f"  {m.expiry.iloc[0]}: T={T:.3f}y  forward={F:.1f}  usable quotes={len(m)}")
        if T < min_T:
            print(f"  {exp.date()}: skipped (T={T:.3f} < --min-T {min_T})"); continue
        if not 0.05 <= T <= 2.0:
            print(f"    WARNING: T={T:.3f} is outside the network's training range [0.05, 2]")
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)
    return add_weights(out)


def add_weights(mkt: pd.DataFrame) -> pd.DataFrame:
    """1/width per quote, scaled so every expiry carries the same total weight (a 6-quote expiry counts as much as an 80-quote one)."""
    n = mkt.groupby("expiry")["iv"].transform("size")
    mkt = mkt.copy()
    mkt["wgt"] = (1.0 / mkt.width) / np.sqrt(n)
    return mkt


def synthetic_surface(r: float, noise: float, truth: dict) -> pd.DataFrame:
    """Fake chains with exactly-Heston vols (+ noise), on the same strike windows the real data uses
    (about +-2 sd: 0.30*sqrt(T), clipped to [0.08, 0.20] in log-moneyness)."""
    rng = np.random.default_rng(1)
    parts = []
    for T in (0.1, 0.25, 0.5, 1.0):
        S = 23700.0
        w = float(np.clip(0.30 * np.sqrt(T), 0.08, 0.20))
        K = S * np.exp(r * T) * np.exp(np.linspace(-w, w, 13))
        iv = cal.exact_iv(truth, S, K, T, r) + rng.normal(0, noise / 100, len(K))
        parts.append(pd.DataFrame(dict(strike=K, iv=iv, iv_bid=iv - 0.002, iv_ask=iv + 0.002, width=0.004,
                                       T=T, S=S, F=S * np.exp(r * T), expiry=f"T={T}")))
    return add_weights(pd.concat(parts, ignore_index=True))


def per_expiry(mkt, iv_model) -> dict:
    out = {}
    for e, g in mkt.groupby("expiry", sort=False):
        err = (iv_model[g.index] - g.iv.values) * 100
        out[e] = dict(T=float(g["T"].iloc[0]), n=int(len(g)), rmse=float(np.sqrt((err ** 2).mean())),
                      max_abs=float(np.abs(err).max()),
                      inside=float(np.mean((iv_model[g.index] >= g.iv_bid.values) & (iv_model[g.index] <= g.iv_ask.values))))
    return out


def plot_surface_fit(path: Path, mkt, iv_nn, iv_ex) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(mkt.groupby("expiry", sort=False))
    n = len(groups); cols = 2; rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(10, 3.6 * rows), squeeze=False)
    for ax, (e, g) in zip(axs.ravel(), groups):
        k = np.log(g.strike / g.F)
        ax.vlines(k, g.iv_bid * 100, g.iv_ask * 100, color="#9aa0a6", lw=1.2)
        ax.plot(k, g.iv * 100, "o", ms=3.5, color="#444", label="market mid")
        ax.plot(k, iv_ex[g.index] * 100, "-", lw=2, color="#1f77b4", label="fit, exact pricer")
        ax.plot(k, iv_nn[g.index] * 100, "--", lw=1.5, color="#e8710a", label="fit, NN pricer")
        ax.set_title(f"{e}  (T={g['T'].iloc[0]:.2f}y)", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
        ax.set_xlabel("ln(K/F)"); ax.set_ylabel("implied vol (%)")
    axs.ravel()[0].legend(frameon=False, fontsize=7)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=[])
    ap.add_argument("--date", default="2026-11-05", help="date the chains were downloaded (YYYY-MM-DD)")
    ap.add_argument("--r", type=float, default=0.06)
    ap.add_argument("--min-T", type=float, default=0.0, help="drop expiries shorter than this (years)")
    ap.add_argument("--model", default="models/heston_mlp.pt")
    ap.add_argument("--pricer", choices=["nn", "fourier"], default="nn")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--time-fourier", action="store_true")
    ap.add_argument("--tag", default="surface")
    a = ap.parse_args(argv)

    price_fn = cal.load_nn(Path(a.model)) if a.pricer == "nn" else cal.fourier_fn
    r = a.r
    truth = None
    if a.synthetic:
        truth = dict(v0=0.02, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        mkt = synthetic_surface(r, a.noise, truth)
        print(f"synthetic surface, truth {truth}")
    else:
        if not a.csv:
            ap.error("give --csv files (or --synthetic)")
        mkt = load_surface(a.csv, a.date, r, min_T=a.min_T)
    S, T, K = mkt.S.values, mkt["T"].values, mkt.strike.values
    box = getattr(price_fn, "box", None)
    if box:                                   # how many quotes fall outside what the network was trained on?
        m_ = S / K
        out = (m_ < box["moneyness"][0]) | (m_ > box["moneyness"][1]) | (T < box["T"][0]) | (T > box["T"][1])
        if out.any():
            print(f"  WARNING: {int(out.sum())}/{len(mkt)} quotes lie outside the network's training box "
                  f"(S/K {box['moneyness']}, T {box['T']}); the network extrapolates there")
    print(f"{len(mkt)} quotes over {mkt.expiry.nunique()} expiries; calibrating with {a.pricer} pricer, {a.starts} starts")

    t0 = time.perf_counter()
    fits = cal.calibrate(price_fn, mkt, S, T, r, a.starts)
    total = time.perf_counter() - t0
    best = fits[0]; p = best["params"]; pv = np.array([p[n] for n in NAMES])
    iv_nn = cal.model_iv(price_fn, pv, S, K, T, r)
    res = dict(n_quotes=int(len(mkt)), expiries=list(mkt.expiry.unique()), r=r, pricer=a.pricer, nn_best=best,
               nn_seconds=total, nn_exact_checked=None)
    iv_ex = cal.exact_iv(p, S, K, T, r)
    rm = lambda x: float(np.sqrt(np.mean(x ** 2)) * 100)
    res["nn_stage_rmse_own_prices"] = rm(iv_nn - mkt.iv.values)
    res["nn_stage_rmse_exact_checked"] = rm(iv_ex - mkt.iv.values)
    print(f"\nstage 1 ({a.pricer} pricer): params {({n: round(p[n], 4) for n in NAMES})}  Feller {best['feller']:.2f}")
    print(f"  RMSE own prices {res['nn_stage_rmse_own_prices']:.3f}, re-priced with the exact pricer "
          f"{res['nn_stage_rmse_exact_checked']:.3f} vol pts   ({total:.1f}s)")
    good = [f for f in fits if f["rmse_vol_pts"] < best["rmse_vol_pts"] + 0.05]
    print(f"  {len(good)}/{a.starts} starts within 0.05 pt of the best; parameter ranges among them:")
    for n in NAMES:
        lo_, hi_ = min(f["params"][n] for f in good), max(f["params"][n] for f in good)
        print(f"     {n:6s} {lo_:8.4f} .. {hi_:8.4f}")
    final_p, final_iv = p, iv_ex
    if a.polish or a.pricer == "fourier":
        t1 = time.perf_counter()
        pol = cal.calibrate(cal.fourier_fn, mkt, S, T, r, 1, verbose=False, x0=[pv])[0]
        tp = time.perf_counter() - t1
        final_p = pol["params"]
        final_iv = cal.exact_iv(final_p, S, K, T, r)
        res["polish"] = pol; res["polish_seconds"] = tp; res["total_seconds"] = total + tp
        print(f"\nPOLISH with the exact pricer: rmse {rm(final_iv - mkt.iv.values):.3f} vol pts in {tp:.1f}s "
              f"({pol['nfev']} evals); total {total + tp:.1f}s")
        print("  params:", {n: round(final_p[n], 4) for n in NAMES}, f"Feller {pol['feller']:.2f}")
    res["final_params"] = final_p
    res["final_rmse"] = rm(final_iv - mkt.iv.values)
    res["per_expiry"] = per_expiry(mkt, final_iv)
    print("\nFINAL fit (exact pricer) per expiry:")
    for e, d in res["per_expiry"].items():
        print(f"  {e:12s} T={d['T']:.2f}  n={d['n']:3d}  rmse {d['rmse']:.3f}  worst {d['max_abs']:.3f}  "
              f"inside bid-ask {100 * d['inside']:.0f}%")
    print(f"  ALL: rmse {res['final_rmse']:.3f} vol pts over {len(mkt)} quotes")
    if truth:
        print("  truth :", {n: round(truth[n], 4) for n in NAMES})
        res["truth"] = truth
    if a.time_fourier:
        t1 = time.perf_counter()
        ex = cal.calibrate(cal.fourier_fn, mkt, S, T, r, a.starts, verbose=False)
        tf = time.perf_counter() - t1
        res["fourier_multistart_seconds"] = tf; res["fourier_best"] = ex[0]
        print(f"\nEXACT pricer alone, same {a.starts} starts: {tf:.1f}s, best rmse {ex[0]['rmse_vol_pts']:.3f}")
        print("  params:", {n: round(ex[0]['params'][n], 4) for n in NAMES}, f"Feller {ex[0]['feller']:.2f}")
        if "total_seconds" in res:
            print(f"  NN + polish {res['total_seconds']:.1f}s vs exact only {tf:.1f}s -> {tf / res['total_seconds']:.1f}x")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/calibration_{a.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    plot_surface_fit(Path(f"plots/calibration_{a.tag}.png"), mkt, iv_nn, iv_ex if not (a.polish or a.pricer == 'fourier') else final_iv)
    return res


if __name__ == "__main__":
    main()
