"""PyTorch tests for the calibration-network options (time-value target, vega-weighted training)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch cannot be loaded: {exc}", allow_module_level=True)

from generate_data import PARAM_RANGES, make_split  # noqa: E402
from nn_pricer import PricerMLP, load_model, parse_args, save_model, train  # noqa: E402

LO = [r[0] for r in PARAM_RANGES["heston_cal"].values()]
HI = [r[1] for r in PARAM_RANGES["heston_cal"].values()]


def test_time_value_output_never_below_the_lower_bound():
    torch.manual_seed(0)
    net = PricerMLP(LO, HI, hidden=16, layers=2, time_value=True).double().eval()
    X = torch.tensor(np.random.default_rng(0).uniform(LO, HI, size=(2000, 8)))
    out = net(X)
    lb = torch.clamp(X[:, 0] - torch.exp(-X[:, 2] * X[:, 1]), min=0.0)
    assert torch.all(out >= lb) and torch.all(out - lb < 0.5)         # starts with a modest time value


def test_checkpoint_roundtrip_keeps_time_value_flag(tmp_path):
    net = PricerMLP(LO, HI, hidden=8, layers=1, time_value=True)
    save_model(net, tmp_path / "m.pt", {})
    loaded, _ = load_model(tmp_path / "m.pt")
    assert loaded.time_value is True
    old = PricerMLP(LO, HI, hidden=8, layers=1)
    assert old.time_value is False and old.config["time_value"] is False


def test_vega_weighted_time_value_training_runs_and_learns(tmp_path, monkeypatch):
    data = tmp_path / "heston_cal_tiny"
    data.mkdir()
    import json
    for split, n in (("train", 3000), ("val", 500)):
        d = make_split("heston_cal", split, n)
        np.savez(data / f"{split}.npz", X=d["X"], y=d["y"])
    (data / "meta.json").write_text(json.dumps({"model": "heston_cal", "features": list(PARAM_RANGES["heston_cal"]),
                                                "ranges": PARAM_RANGES["heston_cal"], "target": "call_price / K"}))
    import nn_pricer
    monkeypatch.setattr(nn_pricer, "MODEL_DIR", tmp_path / "models")
    args = parse_args(["--data", str(data), "--epochs", "8", "--hidden", "32", "--layers", "2", "--batch", "256",
                       "--time-value", "--vega-weight", "--device", "cpu", "--tag", "t"])
    path = train(args)
    model, ckpt = load_model(path)
    assert model.time_value and ckpt["args"]["vega_weight"] is True
    hist = json.loads((tmp_path / "models" / "heston_cal_t_mlp_history.json").read_text())
    assert hist[-1]["val_wrmse"] < hist[0]["val_wrmse"]
