"""PyTorch tests for the Heston surrogate (skipped automatically if torch cannot be loaded)."""
import json

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # ImportError, or OSError for a broken DLL install
    pytest.skip(f"torch cannot be loaded: {exc}", allow_module_level=True)

import nn_pricer  # noqa: E402
from generate_data import PARAM_RANGES, make_split  # noqa: E402
from nn_pricer import PricerMLP, nn_heston_price_and_greeks, parse_args  # noqa: E402

LO = [r[0] for r in PARAM_RANGES["heston"].values()]
HI = [r[1] for r in PARAM_RANGES["heston"].values()]


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PricerMLP(LO, HI, hidden=16, layers=2).double().eval()


def test_heston_autograd_sensitivities_match_finite_differences_of_same_network():
    """Checks the plumbing (chain rule through S/K, scaling, feature order), not the training."""
    m = tiny_model()
    rng = np.random.default_rng(1)
    n, K = 40, 100.0
    base = dict(S=rng.uniform(80, 120, n), K=K, T=rng.uniform(0.1, 1.9, n), r=rng.uniform(0.0, 0.07, n),
                v0=rng.uniform(0.01, 0.2, n), kappa=rng.uniform(0.6, 4.5, n), theta=rng.uniform(0.01, 0.2, n),
                xi=rng.uniform(0.15, 0.95, n), rho=rng.uniform(-0.9, -0.05, n))
    g = nn_heston_price_and_greeks(m, **base)
    V = lambda **kw: nn_heston_price_and_greeks(m, **{**base, **kw}, greeks=False)["price"]
    h = 1e-4
    fd = {
        "delta": (V(S=base["S"] + h) - V(S=base["S"] - h)) / (2 * h),
        "gamma": (V(S=base["S"] + 1e-2) - 2 * V() + V(S=base["S"] - 1e-2)) / 1e-4,
        "dv0": (V(v0=base["v0"] + h * 1e-2) - V(v0=base["v0"] - h * 1e-2)) / (2e-2 * h),
        "theta": -(V(T=base["T"] + h) - V(T=base["T"] - h)) / (2 * h),
    }
    np.testing.assert_allclose(g["price"], V(), rtol=1e-12)
    for k, v in fd.items():
        np.testing.assert_allclose(g[k], v, rtol=1e-4, atol=1e-7, err_msg=k)


def test_heston_short_training_run_learns_something(tmp_path, monkeypatch):
    data = tmp_path / "heston_tiny"
    data.mkdir()
    meta = {"model": "heston", "features": list(PARAM_RANGES["heston"]),
            "ranges": PARAM_RANGES["heston"], "target": "call_price / K"}
    for split, n in (("train", 4000), ("val", 1000)):
        d = make_split("heston", split, n)
        np.savez(data / f"{split}.npz", X=d["X"], y=d["y"])
    (data / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(nn_pricer, "MODEL_DIR", tmp_path / "models")
    args = parse_args(["--data", str(data), "--epochs", "20", "--hidden", "32", "--layers", "2",
                       "--batch", "256", "--device", "cpu", "--tag", "t"])
    path = nn_pricer.train(args)
    hist = json.loads((tmp_path / "models" / "heston_t_mlp_history.json").read_text())
    assert path.exists()
    assert hist[-1]["val_rmse"] < 0.5 * hist[0]["val_rmse"]
