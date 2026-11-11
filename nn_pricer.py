"""Feed-forward neural-network option pricer (PyTorch) + training loop.

Model:  features (moneyness, T, r, sigma, ...) --[fixed min-max scaling to [-1, 1]]-->
        MLP with smooth activations --> softplus --> normalised price V/K  (>= 0)

Design choices (interview talking points):
* Smooth activations (SiLU / softplus): Greeks are derivatives of the network.
  ReLU nets are piecewise linear, so their second derivative (gamma) is zero
  almost everywhere -- useless for hedging. SiLU/softplus are C-infinity.
* Input scaling uses the *known sampling box* (from meta.json), not statistics
  of the data, so nothing about val/test can leak into preprocessing; the
  scaler is stored inside the model as buffers, so a checkpoint is self-contained.
* Softplus output keeps prices non-negative (a no-arbitrage requirement).
* Early stopping + LR schedule look only at the validation split; the test
  split is touched exclusively by evaluate.py.

Usage:
    python nn_pricer.py --data data/bs                 # full training
    python nn_pricer.py --data data/bs_smoke --epochs 3 --tag smoke   # ~1 min check
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_utils import vega_weights

MODEL_DIR = Path("models")
ACTIVATIONS = {"silu": nn.SiLU, "softplus": nn.Softplus, "tanh": nn.Tanh}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class PricerMLP(nn.Module):
    def __init__(self, lo: list[float], hi: list[float], hidden: int = 256,
                 layers: int = 4, activation: str = "silu", time_value: bool = False):
        super().__init__()
        self.time_value = time_value
        self.config = {"lo": list(map(float, lo)), "hi": list(map(float, hi)),
                       "hidden": hidden, "layers": layers, "activation": activation,
                       "time_value": time_value}
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        act = ACTIVATIONS[activation]
        dims = [len(lo)] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), act()]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)
        if time_value:   # start the learned time value near zero (the true one is small), not at softplus(0)=0.69
            nn.init.constant_(self.net[-1].bias, -5.0)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d) raw features in the original units -> (N,) price / K.

        time_value=True: the network learns only the TIME VALUE (price minus the no-arbitrage lower bound
        max(m - e^{-rT}, 0)) and the bound is added back here. The sharp kink at the strike for short
        maturities lives in the bound, not in the weights, and the output can never fall below the bound
        (which was what made implied vols undefined for the old network). Features 0,1,2 must be (m, T, r).
        """
        out = F.softplus(self.net(self.scale(x))).squeeze(-1)
        if self.time_value:
            out = out + torch.clamp(x[:, 0] - torch.exp(-x[:, 2] * x[:, 1]), min=0.0)
        return out


def save_model(model: PricerMLP, path: Path, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, **extra}, path)


def load_model(path: str | Path, device: str = "cpu",
               dtype: torch.dtype = torch.float64) -> tuple[PricerMLP, dict]:
    """Load a checkpoint. Default float64 on CPU: best for autograd Greeks."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = PricerMLP(cfg["lo"], cfg["hi"], cfg["hidden"], cfg["layers"], cfg["activation"],
                      cfg.get("time_value", False))     # older checkpoints have no such key
    model.load_state_dict(ckpt["state_dict"])
    model.to(device=device, dtype=dtype).eval()
    return model, ckpt


# --------------------------------------------------------------------------
# pricing / Greeks in physical units (Black-Scholes feature layout)
# --------------------------------------------------------------------------
def nn_call_price_and_greeks(model: PricerMLP, S, K, T, r, sigma,
                             greeks: bool = True) -> dict[str, np.ndarray]:
    """Call price and autograd Greeks for a BS-feature model.

    V = K * f(S/K, T, r, sigma). We differentiate V w.r.t. the *physical*
    inputs, so the chain rule through S/K and the input scaling is handled by
    autograd automatically. Summing over the batch before calling grad is
    valid because sample i's price depends only on sample i's inputs, so
    d(sum_j V_j)/dS_i = dV_i/dS_i.
    """
    p = next(model.parameters())
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (S, K, T, r, sigma)))
    S_t, K_t, T_t, r_t, s_t = (torch.tensor(a.reshape(-1), dtype=p.dtype, device=p.device)
                               for a in arrays)
    if not greeks:
        with torch.no_grad():
            V = K_t * model(torch.stack([S_t / K_t, T_t, r_t, s_t], dim=1))
        return {"price": V.cpu().numpy()}

    for t in (S_t, T_t, r_t, s_t):
        t.requires_grad_(True)
    V = K_t * model(torch.stack([S_t / K_t, T_t, r_t, s_t], dim=1))
    dS, dT, dr, dsig = torch.autograd.grad(V.sum(), [S_t, T_t, r_t, s_t], create_graph=True)
    (d2S,) = torch.autograd.grad(dS.sum(), [S_t])
    out = {"price": V, "delta": dS, "gamma": d2S, "vega": dsig, "theta": -dT, "rho": dr}
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


HESTON_FEATURES = ("moneyness", "T", "r", "v0", "kappa", "theta", "xi", "rho")


def nn_heston_price_and_greeks(model: PricerMLP, S, K, T, r, v0, kappa, theta, xi, rho,
                               greeks: bool = True) -> dict[str, np.ndarray]:
    """Call price and autograd sensitivities for a Heston-feature model.

    V = K * f(S/K, T, r, v0, kappa, theta, xi, rho). Returned (all in physical units):
      price, delta = dV/dS, gamma = d2V/dS2, dv0 = dV/dv0 (sensitivity to initial
      variance, the Heston analogue of vega), theta = -dV/dT.
    There is no closed form for these, so evaluate.py compares them with finite
    differences of the Fourier pricer.
    """
    p = next(model.parameters())
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                   for a in (S, K, T, r, v0, kappa, theta, xi, rho)))
    S_t, K_t, T_t, r_t, v0_t, k_t, th_t, xi_t, rho_t = (
        torch.tensor(a.reshape(-1), dtype=p.dtype, device=p.device) for a in arrays)
    if not greeks:
        with torch.no_grad():
            V = K_t * model(torch.stack([S_t / K_t, T_t, r_t, v0_t, k_t, th_t, xi_t, rho_t], dim=1))
        return {"price": V.cpu().numpy()}

    for t in (S_t, T_t, v0_t):
        t.requires_grad_(True)
    V = K_t * model(torch.stack([S_t / K_t, T_t, r_t, v0_t, k_t, th_t, xi_t, rho_t], dim=1))
    dS, dT, dv0 = torch.autograd.grad(V.sum(), [S_t, T_t, v0_t], create_graph=True)
    (d2S,) = torch.autograd.grad(dS.sum(), [S_t])
    out = {"price": V, "delta": dS, "gamma": d2S, "dv0": dv0, "theta": -dT}
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_split(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch: int = 65536,
                   w: torch.Tensor | None = None) -> dict[str, float]:
    """mse/mae/max of the PRICE error; with weights w also 'wmse' (the training objective, for model selection)."""
    model.eval()
    err = torch.cat([model(X[i:i + batch]) - y[i:i + batch] for i in range(0, len(X), batch)])
    out = {"mse": err.pow(2).mean().item(), "mae": err.abs().mean().item(), "max": err.abs().max().item()}
    out["wmse"] = (w * err.pow(2)).mean().item() if w is not None else out["mse"]
    return out


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = pick_device(args.device)
    meta = json.loads((args.data / "meta.json").read_text())
    lo = [r[0] for r in meta["ranges"].values()]
    hi = [r[1] for r in meta["ranges"].values()]

    def load(split: str):
        d = np.load(args.data / f"{split}.npz")
        return (torch.tensor(d["X"], dtype=torch.float32, device=device),
                torch.tensor(d["y"], dtype=torch.float32, device=device))

    Xtr, ytr = load("train")
    Xva, yva = load("val")       # NOTE: test.npz is deliberately never loaded here
    n = len(Xtr)
    wtr = wva = None
    if args.vega_weight:         # loss weight 1/(vega^2 + floor^2): equalise IMPLIED-VOL error, not price error
        feats = meta["features"]
        wtr = torch.tensor(vega_weights(Xtr.cpu().numpy(), feats, args.vega_floor), dtype=torch.float32, device=device)
        wva = torch.tensor(vega_weights(Xva.cpu().numpy(), feats, args.vega_floor), dtype=torch.float32, device=device)
        print(f"vega weights: floor {args.vega_floor}, min {wtr.min().item():.3f} / max {wtr.max().item():.1f} "
              f"(mean 1)")

    model = PricerMLP(lo, hi, args.hidden, args.layers, args.activation, args.time_value).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6)
    gen = torch.Generator().manual_seed(args.seed)

    print(f"device={device}  train={n:,}  val={len(Xva):,}  params={n_params:,}  "
          f"arch={args.layers}x{args.hidden} {args.activation}")
    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, -1, 0
    history: list[dict] = []
    t_start = time.perf_counter()
    try:  # Ctrl+C stops training cleanly and still saves the best model so far
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            perm = torch.randperm(n, generator=gen).to(device)
            running = torch.zeros((), device=device)
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                err2 = (model(Xtr[idx]) - ytr[idx]).pow(2)
                loss = err2.mean() if wtr is None else (wtr[idx] * err2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.detach() * len(idx)
            train_mse = (running / n).item()
            val = evaluate_split(model, Xva, yva, w=wva)
            sel = val["wmse"]                      # model selection follows the training objective
            sched.step(sel)
            lr = opt.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_rmse": train_mse ** 0.5,
                            "val_rmse": val["mse"] ** 0.5, "val_wrmse": sel ** 0.5, "val_mae": val["mae"],
                            "val_max": val["max"], "lr": lr})
            improved = sel < best_val * (1 - 1e-4)
            if improved:
                best_val, best_epoch, bad_epochs = sel, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
            print(f"epoch {epoch:3d}  train_{'w' if wtr is not None else ''}rmse {train_mse ** 0.5:.3e}  "
                  f"val_rmse {val['mse'] ** 0.5:.3e}  "
                  + (f"val_wrmse {sel ** 0.5:.3e}  " if wtr is not None else "") +
                  f"val_mae {val['mae']:.3e}  val_max {val['max']:.3e}  lr {lr:.1e}  "
                  f"{time.perf_counter() - t0:.1f}s{'  *' if improved else ''}")
            if bad_epochs >= args.patience:
                print(f"early stopping: no val improvement for {args.patience} epochs")
                break
    except KeyboardInterrupt:
        print("\ninterrupted by user: saving the best model found so far")
    if best_state is None:
        raise SystemExit("stopped before the first epoch finished; nothing to save")

    model.load_state_dict(best_state)
    name = f"{meta['model']}{'_' + args.tag if args.tag else ''}_mlp"
    path = MODEL_DIR / f"{name}.pt"
    save_model(model, path, {
        "features": meta["features"], "target": meta["target"], "data": str(args.data),
        "best_epoch": best_epoch, "best_val_rmse": best_val ** 0.5,   # weighted if --vega-weight
        "n_params": n_params, "train_seconds": time.perf_counter() - t_start,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    })
    (MODEL_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    print(f"best epoch {best_epoch}, val_rmse {best_val ** 0.5:.3e}; saved {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/bs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--activation", choices=list(ACTIVATIONS), default="silu")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-patience", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience (epochs)")
    ap.add_argument("--time-value", action="store_true",
                    help="learn price minus the no-arbitrage lower bound (helps short maturities)")
    ap.add_argument("--vega-weight", action="store_true",
                    help="weight the loss by 1/(vega^2+floor^2) so implied-vol error is equalised")
    ap.add_argument("--vega-floor", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
