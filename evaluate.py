"""Evaluate a trained NN pricer on the HELD-OUT test split.

Reports (Black-Scholes model, Stage 2):
  1. pricing error on V/K: MAE, RMSE, max; relative error for options worth >= 1% of K
  2. error by moneyness x maturity bucket (where does the network struggle?)
  3. throughput: NN vs reference pricer on the same batch
  4. autograd Greeks vs analytical Black-Scholes Greeks
Writes results/<name>.json and plots/<name>_*.png.

Usage:
    python evaluate.py --model models/bs_mlp.pt --data data/bs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from black_scholes import bs_greeks, bs_price

M_EDGES = [0.7, 0.9, 0.97, 1.03, 1.1, 1.3]          # S/K
M_LABELS = ["deep OTM", "OTM", "ATM", "ITM", "deep ITM"]   # for a call
T_EDGES = [0.05, 0.25, 0.5, 1.0, 2.0]
T_LABELS = ["<3M", "3-6M", "6M-1Y", "1-2Y"]
GREEKS = ["delta", "gamma", "vega", "theta", "rho"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


# --------------------------------------------------------------------------
# pure-numpy metrics (unit-tested without torch)
# --------------------------------------------------------------------------
def price_metrics(y_true: np.ndarray, y_pred: np.ndarray, rel_floor: float = 0.01) -> dict:
    err = y_pred - y_true
    big = y_true >= rel_floor
    rel = np.abs(err[big]) / y_true[big]
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
        "rel_floor": rel_floor,
        "frac_above_floor": float(big.mean()),
        "rel_median": float(np.median(rel)),
        "rel_p99": float(np.quantile(rel, 0.99)),
        "rel_max": float(rel.max()),
    }


def bucket_table(m: np.ndarray, T: np.ndarray, abs_err: np.ndarray,
                 m_edges=M_EDGES, T_edges=T_EDGES) -> dict[str, np.ndarray]:
    """MAE / max / count per (moneyness, maturity) cell. Rows = moneyness."""
    mi = np.clip(np.digitize(m, m_edges) - 1, 0, len(m_edges) - 2)
    ti = np.clip(np.digitize(T, T_edges) - 1, 0, len(T_edges) - 2)
    shape = (len(m_edges) - 1, len(T_edges) - 1)
    mae, mx, cnt = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, int)
    for a in range(shape[0]):
        for b in range(shape[1]):
            sel = (mi == a) & (ti == b)
            cnt[a, b] = sel.sum()
            if cnt[a, b]:
                mae[a, b], mx[a, b] = abs_err[sel].mean(), abs_err[sel].max()
    return {"mae": mae, "max": mx, "count": cnt}


def greek_metrics(true: dict, pred: dict) -> dict:
    out = {}
    for g in GREEKS:
        e = np.abs(pred[g] - true[g])
        scale = float(np.mean(np.abs(true[g])))
        out[g] = {"mae": float(e.mean()), "max_abs": float(e.max()),
                  "mean_abs_true": scale, "mae_rel_to_scale": float(e.mean() / scale)}
    return out


def best_time(fn, repeats: int = 7) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def format_table(tab: np.ndarray, scale: float = 1e4, fmt: str = "{:8.2f}") -> str:
    head = " " * 10 + "".join(f"{t:>9s}" for t in T_LABELS)
    rows = [f"{M_LABELS[i]:>10s}" + "".join(" " + fmt.format(v * scale) for v in tab[i])
            for i in range(tab.shape[0])]
    return "\n".join([head] + rows)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_bucket_heatmap(tab: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(tab * 1e4, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(T_LABELS)), T_LABELS)
    ax.set_yticks(range(len(M_LABELS)), [f"{l}\n{a}-{b}" for l, a, b in zip(M_LABELS, M_EDGES[:-1], M_EDGES[1:])])
    vmax = np.nanmax(tab * 1e4)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab[i, j] * 1e4
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 * vmax else INK, fontsize=9)
    ax.set_xlabel("maturity"); ax.set_ylabel("moneyness S/K")
    ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, label="MAE (bp of strike)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_greek_slices(model, path: Path, T: float = 0.5, r: float = 0.03, sigma: float = 0.2) -> None:
    import matplotlib.pyplot as plt
    from nn_pricer import nn_call_price_and_greeks
    K = 100.0
    S = np.linspace(72, 128, 400)
    nn = nn_call_price_and_greeks(model, S, K, T, r, sigma)
    an = bs_greeks(S, K, T, r, sigma, "call")
    an["price"] = bs_price(S, K, T, r, sigma, "call")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, g in zip(axes, ["delta", "gamma", "vega"]):
        ax.plot(S, an[g], color=INK, lw=2, label="analytical BS")
        ax.plot(S, nn[g], color=ORANGE, lw=2, ls="--", label="NN (autograd)")
        ax.set_title(g, loc="left", fontsize=11); ax.set_xlabel("spot S  (K = 100)")
        _style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Autograd Greeks vs analytical  (T={T}, r={r}, sigma={sigma})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(history_path: Path, path: Path) -> None:
    import matplotlib.pyplot as plt
    h = json.loads(history_path.read_text())
    ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(ep, [x["train_rmse"] for x in h], color=BLUE, lw=2, label="train RMSE")
    ax.semilogy(ep, [x["val_rmse"] for x in h], color=ORANGE, lw=2, label="validation RMSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE of V/K (log scale)")
    ax.set_title("Training curve", loc="left", fontsize=11); ax.legend(frameon=False); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    import torch
    from nn_pricer import load_model, nn_call_price_and_greeks

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/bs_mlp.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/bs"))
    ap.add_argument("--n-greeks", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)

    name = args.model.stem
    Path("results").mkdir(exist_ok=True); Path("plots").mkdir(exist_ok=True)
    model, ckpt = load_model(args.model)              # float64, CPU
    test = np.load(args.data / "test.npz")
    X, y = test["X"], test["y"]
    m, T, r, sigma = X.T

    # 1. pricing error ---------------------------------------------------------
    with torch.no_grad():
        y_hat = model(torch.tensor(X)).numpy()
    pm = price_metrics(y, y_hat)
    print(f"\n== Pricing error on {pm['n']:,} held-out test options (units: V/K) ==")
    print(f"MAE {pm['mae']:.3e}   RMSE {pm['rmse']:.3e}   max {pm['max_abs']:.3e}")
    print(f"  = MAE {pm['mae'] * 1e4:.2f} bp of strike, i.e. Rs {pm['mae'] * 100:.4f} on a Rs 100 strike")
    print(f"relative error (options worth >= {pm['rel_floor']:.0%} of K, {pm['frac_above_floor']:.0%} of test set): "
          f"median {pm['rel_median']:.2e}, p99 {pm['rel_p99']:.2e}, max {pm['rel_max']:.2e}")

    # 2. buckets ----------------------------------------------------------------
    bt = bucket_table(m, T, np.abs(y_hat - y))
    print("\nMAE by bucket (bp of strike):\n" + format_table(bt["mae"]))
    print("\nMax abs error by bucket (bp of strike):\n" + format_table(bt["max"]))
    plot_bucket_heatmap(bt["mae"], Path("plots") / f"{name}_bucket_mae.png",
                        "NN pricing error by moneyness and maturity (test set)")

    # 3. speed (float32 on the fastest available device vs NumPy closed form) --
    from nn_pricer import pick_device
    dev = pick_device("auto")
    model32, _ = load_model(args.model, device=str(dev), dtype=torch.float32)
    X32 = torch.tensor(X, dtype=torch.float32, device=dev)

    def run_nn():
        with torch.no_grad():
            out = model32(X32)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        return out

    t_nn = best_time(run_nn)
    t_ref = best_time(lambda: bs_price(m, 1.0, T, r, sigma, "call"))
    speed = {"batch": len(X), "device": str(dev),
             "nn_us_per_option": 1e6 * t_nn / len(X), "ref_us_per_option": 1e6 * t_ref / len(X),
             "speedup_ref_over_nn": t_ref / t_nn}
    print(f"\n== Speed, batch of {len(X):,} ==\nNN ({dev}): {speed['nn_us_per_option']:.3f} us/option   "
          f"BS closed form (NumPy): {speed['ref_us_per_option']:.3f} us/option   "
          f"ratio ref/NN = {speed['speedup_ref_over_nn']:.2f}x")
    print("(Expected: no real speed-up for BS - the closed form is already cheap. "
          "The speed-up story is Heston, Stage 4.)")

    # 4. Greeks -----------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n_greeks, len(X)), replace=False)
    K = 100.0
    S = m[idx] * K
    nn_g = nn_call_price_and_greeks(model, S, K, T[idx], r[idx], sigma[idx])
    an_g = bs_greeks(S, K, T[idx], r[idx], sigma[idx], "call")
    gm = greek_metrics(an_g, nn_g)
    print(f"\n== Autograd Greeks vs analytical, {len(idx):,} test options, K = 100 ==")
    print(f"{'greek':>6s} {'MAE':>10s} {'max':>10s} {'mean|true|':>11s} {'MAE/scale':>10s}")
    for g, v in gm.items():
        print(f"{g:>6s} {v['mae']:10.3e} {v['max_abs']:10.3e} {v['mean_abs_true']:11.3e} {v['mae_rel_to_scale']:10.2%}")
    gamma_bt = bucket_table(m[idx], T[idx], np.abs(nn_g["gamma"] - an_g["gamma"]))
    print("\nGamma MAE by bucket (x1e4):\n" + format_table(gamma_bt["mae"]))
    plot_greek_slices(model, Path("plots") / f"{name}_greek_slices.png")
    hist = args.model.with_name(f"{name}_history.json")
    if hist.exists():
        plot_history(hist, Path("plots") / f"{name}_training.png")

    results = {
        "model": str(args.model), "data": str(args.data),
        "n_train": int(len(np.load(args.data / "train.npz")["y"])), "n_test": pm["n"],
        "n_params": ckpt.get("n_params"), "best_epoch": ckpt.get("best_epoch"),
        "train_seconds": ckpt.get("train_seconds"),
        "pricing": pm, "bucket_mae": bt["mae"].tolist(), "bucket_max": bt["max"].tolist(),
        "bucket_count": bt["count"].tolist(), "speed": speed, "greeks": gm,
        "gamma_bucket_mae": gamma_bt["mae"].tolist(),
    }
    out = Path("results") / f"{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out} and plots/{name}_*.png")
    return results


if __name__ == "__main__":
    main()
