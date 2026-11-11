"""PyTorch tests (skipped automatically if torch is not installed)."""
import json

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # ImportError, or OSError for a broken DLL install (Windows WinError 126)
    pytest.skip(f"torch cannot be loaded: {exc}", allow_module_level=True)

import nn_pricer  # noqa: E402
from generate_data import PARAM_RANGES, make_split  # noqa: E402
from nn_pricer import PricerMLP, load_model, nn_call_price_and_greeks, parse_args, save_model  # noqa: E402

LO = [r[0] for r in PARAM_RANGES["bs"].values()]
HI = [r[1] for r in PARAM_RANGES["bs"].values()]


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PricerMLP(LO, HI, hidden=16, layers=2).double().eval()


def test_scaler_maps_box_to_unit_cube():
    m = tiny_model()
    x = torch.tensor([LO, HI], dtype=torch.float64)
    # Bounds are stored as float32 buffers (training runs in float32), so they are
    # only exact to ~1e-7. Training and evaluation use the same rounded bounds,
    # so this does not create a train/test mismatch.
    np.testing.assert_allclose(m.scale(x).numpy(), [[-1] * 4, [1] * 4], atol=1e-6)


def test_output_shape_and_non_negative():
    m = tiny_model()
    x = torch.tensor(make_split("bs", "val", 256)["X"])
    out = m(x)
    assert out.shape == (256,) and torch.all(out >= 0)


def test_autograd_greeks_match_finite_differences_of_same_network():
    """Checks the Greek *plumbing* (chain rule through S/K and scaling),
    independently of how well the network is trained."""
    m = tiny_model()
    rng = np.random.default_rng(1)
    K = 100.0
    S, T = rng.uniform(75, 125, 50), rng.uniform(0.1, 1.9, 50)
    r, s = rng.uniform(0.01, 0.07, 50), rng.uniform(0.1, 0.5, 50)
    g = nn_call_price_and_greeks(m, S, K, T, r, s)
    V = lambda **kw: nn_call_price_and_greeks(m, **{"S": S, "K": K, "T": T, "r": r, "sigma": s, **kw},
                                              greeks=False)["price"]
    h = 1e-4
    fd = {
        "delta": (V(S=S + h) - V(S=S - h)) / (2 * h),
        "gamma": (V(S=S + 1e-2) - 2 * V() + V(S=S - 1e-2)) / 1e-4,
        "vega": (V(sigma=s + h) - V(sigma=s - h)) / (2 * h),
        "theta": -(V(T=T + h) - V(T=T - h)) / (2 * h),
        "rho": (V(r=r + h) - V(r=r - h)) / (2 * h),
    }
    for k, v in fd.items():
        np.testing.assert_allclose(g[k], v, rtol=1e-5, atol=1e-8, err_msg=k)


def test_save_load_round_trip(tmp_path):
    m = tiny_model()
    p = tmp_path / "m.pt"
    save_model(m, p, {"n_params": 1})
    m2, ckpt = load_model(p)
    x = torch.tensor(make_split("bs", "val", 64)["X"])
    np.testing.assert_allclose(m(x).detach().numpy(), m2(x).detach().numpy(), rtol=1e-6)
    assert ckpt["n_params"] == 1


def test_short_training_run_learns_something(tmp_path, monkeypatch):
    data = tmp_path / "bs_tiny"
    data.mkdir()
    meta = {"model": "bs", "features": list(PARAM_RANGES["bs"]),
            "ranges": PARAM_RANGES["bs"], "target": "call_price / K"}
    for split, n in (("train", 4000), ("val", 1000)):
        d = make_split("bs", split, n)
        np.savez(data / f"{split}.npz", X=d["X"], y=d["y"])
    (data / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(nn_pricer, "MODEL_DIR", tmp_path / "models")
    args = parse_args(["--data", str(data), "--epochs", "15", "--hidden", "32", "--layers", "2",
                       "--batch", "256", "--device", "cpu", "--tag", "t"])
    path = nn_pricer.train(args)
    hist = json.loads((tmp_path / "models" / "bs_t_mlp_history.json").read_text())
    assert path.exists()
    assert hist[-1]["val_rmse"] < 0.5 * hist[0]["val_rmse"]
