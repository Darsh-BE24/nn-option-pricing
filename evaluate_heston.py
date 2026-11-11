"""Evaluate a trained NN Heston surrogate on the HELD-OUT test split (Stage 4).

Reports
  1. pricing error on V/K (MAE, RMSE, max; relative error for options worth >= 1% of K)
  2. error by moneyness x maturity, by Feller ratio 2*kappa*theta/xi^2, and the 5 worst options
  3. NN autograd sensitivities (delta, gamma, dV/dv0, theta) vs finite differences of the
     Fourier pricer (Heston has no closed-form Greeks), with a self-consistency check
     on the finite-difference reference itself
  4. speed: NN vs the vectorised Fourier pricer vs the slow adaptive-quadrature pricer,
     at several batch sizes, on THIS machine (speed-ups are only meaningful on one machine)
Writes results/heston_mlp.json and plots/heston_mlp_*.png.

Usage:
    python evaluate_heston.py --model models/heston_mlp.pt --data data/heston
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import (BLUE, INK, M_EDGES, M_LABELS, MUTED, ORANGE, T_EDGES, T_LABELS,
                      _style, best_time, bucket_table, format_table, plot_bucket_heatmap,
                      plot_history, price_metrics)
from heston import heston_price, heston_price_quad

GREEKS = ("delta", "gamma", "dv0", "theta")
FELLER_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
FELLER_LABELS = ["<0.25", "0.25-0.5", "0.5-1", "1-2", ">2 (Feller holds)"]


# --------------------------------------------------------------------------
# torch-free helpers (unit-tested)
# --------------------------------------------------------------------------
def fd_heston_greeks(S, K, T, r, v0, kappa, theta, xi, rho, price_fn=heston_price,
                     rel: float = 0.02) -> dict[str, np.ndarray]:
    """Central finite differences of a pricer: delta, gamma, dV/dv0, theta (= -dV/dT).

    Bump sizes scale with the natural width of each variable so they are well
    chosen across the whole box: spot by S*sqrt(v0*T) (the width of the terminal
    distribution), T and v0 relatively. rel = 0.02 keeps truncation error ~1e-4
    relative while keeping rounding noise (~1e-10 / h^2 for gamma) negligible.
    """
    S, K, T, r, v0, kappa, theta, xi, rho = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (S, K, T, r, v0, kappa, theta, xi, rho)))

    def V(S=S, T=T, v0=v0):
        return np.asarray(price_fn(S, K, T, r, v0, kappa, theta, xi, rho))

    hS, hT, hv = rel * S * np.sqrt(v0 * T), rel * T, rel * v0
    base = V()
    return {
        "price": base,
        "delta": (V(S=S + hS) - V(S=S - hS)) / (2 * hS),
        "gamma": (V(S=S + hS) - 2 * base + V(S=S - hS)) / hS**2,
        "dv0": (V(v0=v0 + hv) - V(v0=v0 - hv)) / (2 * hv),
        "theta": -(V(T=T + hT) - V(T=T - hT)) / (2 * hT),
    }


def greek_errors(true: dict, pred: dict) -> dict:
    out = {}
    for g in GREEKS:
        e = np.abs(pred[g] - true[g])
        scale = float(np.mean(np.abs(true[g])))
        out[g] = {"mae": float(e.mean()), "max_abs": float(e.max()), "mean_abs_true": scale,
                  "mae_rel_to_scale": float(e.mean() / scale)}
    return out


def group_table(values: np.ndarray, edges, labels, abs_err: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (values >= lo) & (values < hi)
        rows.append({"group": lab, "n": int(sel.sum()),
                     "mae": float(abs_err[sel].mean()) if sel.any() else float("nan"),
                     "max": float(abs_err[sel].max()) if sel.any() else float("nan")})
    return rows


def feller_ratio(kappa, theta, xi):
    return 2.0 * kappa * theta / xi**2


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_greek_slices(model, path: Path) -> None:
    import matplotlib.pyplot as plt
    from nn_pricer import nn_heston_price_and_greeks

    K, T, r = 100.0, 0.5, 0.03
    hp = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7)
    S = np.linspace(75, 125, 300)
    nn = nn_heston_price_and_greeks(model, S, K, T, r, **hp)
    fd = fd_heston_greeks(S, K, T, r, **hp)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, g, title in zip(axes, ("delta", "gamma", "dv0"), ("delta", "gamma", "dV/dv0 (variance sensitivity)")):
        ax.plot(S, fd[g], color=INK, lw=2, label="finite diff. of Fourier pricer")
        ax.plot(S, nn[g], color=ORANGE, lw=2, ls="--", label="NN (autograd)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("spot S  (K = 100)")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Heston NN sensitivities vs reference  (T=0.5, r=0.03, v0=0.04, kappa=1.5, "
                 "theta=0.05, xi=0.5, rho=-0.7)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speed(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, color, label in (("gl_us", INK, "Fourier (vectorised Gauss-Legendre)"),
                              ("nn_cpu_us", BLUE, "NN (CPU, float32)"),
                              ("nn_best_us", ORANGE, "NN (best device)")):
        pts = [(r["batch"], r[key]) for r in rows if r.get(key) is not None]
        if pts:
            ax.loglog(*zip(*pts), marker="o", ms=5, lw=2, color=color, label=label)
    ax.set_xlabel("batch size (options priced in one call)")
    ax.set_ylabel("microseconds per option")
    ax.set_title("Pricing cost per option vs batch size", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    import torch
    from nn_pricer import load_model, nn_heston_price_and_greeks, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/heston_mlp.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/heston"))
    ap.add_argument("--n-greeks", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)

    name = args.model.stem
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    model, ckpt = load_model(args.model)                     # float64, CPU
    test = np.load(args.data / "test.npz")
    X, y = test["X"], test["y"]
    m, T, r, v0, kappa, theta, xi, rho = X.T

    # 1. pricing error ---------------------------------------------------------
    with torch.no_grad():
        y_hat = np.concatenate([model(torch.tensor(X[i:i + 50_000])).numpy()
                                for i in range(0, len(X), 50_000)])
    err = np.abs(y_hat - y)
    pm = price_metrics(y, y_hat)
    print(f"\n== Pricing error on {pm['n']:,} held-out Heston test options (units: V/K) ==")
    print(f"MAE {pm['mae']:.3e}   RMSE {pm['rmse']:.3e}   max {pm['max_abs']:.3e}")
    print(f"  = MAE {pm['mae'] * 1e4:.2f} bp of strike, i.e. Rs {pm['mae'] * 100:.4f} on a Rs 100 strike")
    print(f"relative error (options worth >= {pm['rel_floor']:.0%} of K, {pm['frac_above_floor']:.0%} of test set): "
          f"median {pm['rel_median']:.2e}, p99 {pm['rel_p99']:.2e}, max {pm['rel_max']:.2e}")

    # 2. where does it struggle -------------------------------------------------
    bt = bucket_table(m, T, err)
    print("\nMAE by bucket (bp of strike):\n" + format_table(bt["mae"]))
    print("\nMax abs error by bucket (bp of strike):\n" + format_table(bt["max"]))
    fr = feller_ratio(kappa, theta, xi)
    ft = group_table(fr, FELLER_EDGES, FELLER_LABELS, err)
    print("\nBy Feller ratio 2*kappa*theta/xi^2 (ratio < 1: variance can hit zero):")
    print(f"{'ratio':>20s} {'n':>8s} {'MAE (bp)':>9s} {'max (bp)':>9s}")
    for g in ft:
        print(f"{g['group']:>20s} {g['n']:8d} {g['mae'] * 1e4:9.2f} {g['max'] * 1e4:9.2f}")
    print("\n5 worst test options:")
    for i in np.argsort(-err)[:5]:
        print(f"  err {err[i] * 1e4:6.1f} bp  price {y[i]:.4f}  S/K={m[i]:.3f} T={T[i]:.2f} v0={v0[i]:.3f} "
              f"kappa={kappa[i]:.2f} theta={theta[i]:.3f} xi={xi[i]:.2f} rho={rho[i]:.2f}")
    plot_bucket_heatmap(bt["mae"], Path("plots") / f"{name}_bucket_mae.png",
                        "Heston NN pricing error by moneyness and maturity (test set)")

    # 3. Greeks -----------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n_greeks, len(X)), replace=False)
    K = np.full(len(idx), 100.0)
    S = m[idx] * K
    hp = (K, T[idx], r[idx], v0[idx], kappa[idx], theta[idx], xi[idx], rho[idx])
    nn_g = nn_heston_price_and_greeks(model, S, *hp)
    fd_g = fd_heston_greeks(S, *hp)
    gm = greek_errors(fd_g, nn_g)
    # how trustworthy is the reference? compare two bump sizes on 500 options
    sub = slice(0, 500)
    hp_sub = tuple(a[sub] for a in hp)
    fd_a, fd_b = fd_heston_greeks(S[sub], *hp_sub, rel=0.02), fd_heston_greeks(S[sub], *hp_sub, rel=0.04)
    ref_self = {g: float(np.mean(np.abs(fd_a[g] - fd_b[g])) / np.mean(np.abs(fd_a[g]))) for g in GREEKS}
    print(f"\n== NN autograd sensitivities vs finite differences of the Fourier pricer, "
          f"{len(idx):,} test options, K = 100 ==")
    print(f"{'greek':>6s} {'MAE':>10s} {'max':>10s} {'mean|ref|':>11s} {'MAE/scale':>10s} {'ref self-check':>15s}")
    for g, v in gm.items():
        print(f"{g:>6s} {v['mae']:10.3e} {v['max_abs']:10.3e} {v['mean_abs_true']:11.3e} "
              f"{v['mae_rel_to_scale']:10.2%} {ref_self[g]:15.2%}")
    print("(ref self-check: mean difference between two finite-difference bump sizes, relative; "
          "NN errors below this level cannot be resolved.)")
    gamma_bt = bucket_table(m[idx], T[idx], np.abs(nn_g["gamma"] - fd_g["gamma"]))
    print("\nGamma MAE by bucket (x1e4):\n" + format_table(gamma_bt["mae"]))
    plot_greek_slices(model, Path("plots") / f"{name}_greek_slices.png")
    hist = args.model.with_name(f"{name}_history.json")
    if hist.exists():
        plot_history(hist, Path("plots") / f"{name}_training.png")

    # 4. speed ------------------------------------------------------------------
    dev = pick_device("auto")
    sizes = [1, 100, 5000, 100_000]
    gl_sizes = {1, 100, 5000}
    cpu32, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    best32 = cpu32 if dev.type == "cpu" else load_model(args.model, device=str(dev), dtype=torch.float32)[0]

    def nn_time(mdl, device, n):
        xb = torch.tensor(X[:n], dtype=torch.float32, device=device)

        def run():
            with torch.no_grad():
                out = mdl(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            return out
        return best_time(run, repeats=15 if n <= 100 else 7)

    rows = []
    for n in sizes:
        row = {"batch": n, "nn_cpu_us": 1e6 * nn_time(cpu32, torch.device("cpu"), n) / n}
        row["nn_best_us"] = 1e6 * nn_time(best32, dev, n) / n if dev.type != "cpu" else row["nn_cpu_us"]
        if n in gl_sizes:
            row["gl_us"] = 1e6 * best_time(lambda n=n: heston_price(m[:n], 1.0, T[:n], r[:n], v0[:n], kappa[:n],
                                                                      theta[:n], xi[:n], rho[:n]),
                                           repeats=7 if n <= 100 else 3) / n
            row["speedup_cpu_vs_gl"] = row["gl_us"] / row["nn_cpu_us"]
            row["speedup_best_vs_gl"] = row["gl_us"] / row["nn_best_us"]
        rows.append(row)
    nq = 200
    t_quad = best_time(lambda: [heston_price_quad(m[i], 1.0, T[i], r[i], v0[i], kappa[i], theta[i], xi[i], rho[i])
                                for i in range(nq)], repeats=2) / nq
    print(f"\n== Speed on this machine (device for 'best': {dev}) ==")
    print(f"{'batch':>8s} {'Fourier us/opt':>15s} {'NN cpu us/opt':>14s} {'NN best us/opt':>15s} "
          f"{'speed-up (cpu)':>15s} {'speed-up (best)':>16s}")
    for rw in rows:
        gl = f"{rw['gl_us']:15.1f}" if "gl_us" in rw else f"{'-':>15s}"
        s1 = f"{rw['speedup_cpu_vs_gl']:14.0f}x" if "gl_us" in rw else f"{'-':>15s}"
        s2 = f"{rw['speedup_best_vs_gl']:15.0f}x" if "gl_us" in rw else f"{'-':>16s}"
        print(f"{rw['batch']:8d} {gl} {rw['nn_cpu_us']:14.2f} {rw['nn_best_us']:15.2f} {s1} {s2}")
    print(f"adaptive-quadrature reference pricer: {1e6 * t_quad:.0f} us/option "
          f"(NN is {1e6 * t_quad / rows[2]['nn_cpu_us']:.0f}x faster at batch 5000 on CPU)")
    print("Headline comparison = vs the vectorised Fourier pricer, the fair baseline "
          "(the quadrature reference and Monte Carlo are slower still).")
    plot_speed(rows, Path("plots") / f"{name}_speed.png")

    results = {
        "model": str(args.model), "data": str(args.data),
        "n_train": int(len(np.load(args.data / "train.npz")["y"])), "n_test": pm["n"],
        "n_params": ckpt.get("n_params"), "best_epoch": ckpt.get("best_epoch"),
        "train_seconds": ckpt.get("train_seconds"), "pricing": pm,
        "bucket_mae": bt["mae"].tolist(), "bucket_max": bt["max"].tolist(),
        "bucket_count": bt["count"].tolist(), "feller_groups": ft,
        "worst5": [{"err": float(err[i]), "x": X[i].tolist()} for i in np.argsort(-err)[:5]],
        "greeks": gm, "greeks_reference_self_check": ref_self,
        "gamma_bucket_mae": gamma_bt["mae"].tolist(),
        "speed": {"device": str(dev), "rows": rows, "quad_us_per_option": 1e6 * t_quad},
    }
    out = Path("results") / f"{name}.json"
    out.write_text(json.dumps(results, indent=2, default=lambda o: None if o is None else float(o)))
    print(f"\nsaved {out} and plots/{name}_*.png")
    return results


if __name__ == "__main__":
    main()
