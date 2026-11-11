"""PyTorch test for the Stage-5 wrapper around a trained checkpoint (skipped if torch cannot be loaded)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # ImportError, or OSError for a broken DLL install
    pytest.skip(f"torch cannot be loaded: {exc}", allow_module_level=True)

from generate_data import PARAM_RANGES, make_split  # noqa: E402
from nn_pricer import PricerMLP, save_model  # noqa: E402
from smile import iv_error_report, load_nn_price_fn, plot_smiles, plot_surface  # noqa: E402


def test_checkpoint_wrapper_matches_the_model_and_plots_survive_an_untrained_net(tmp_path):
    lo = [r[0] for r in PARAM_RANGES["heston"].values()]
    hi = [r[1] for r in PARAM_RANGES["heston"].values()]
    torch.manual_seed(0)
    model = PricerMLP(lo, hi, hidden=16, layers=2)
    path = tmp_path / "m.pt"
    save_model(model, path, {"n_params": 1})
    price = load_nn_price_fn(path)
    X = make_split("heston", "val", 300)["X"]
    out = price(X)
    assert out.shape == (300,) and np.all(np.isfinite(out)) and np.all(out >= 0)
    with torch.no_grad():
        direct = model.double()(torch.tensor(X)).numpy()
    np.testing.assert_allclose(out, direct, rtol=1e-5, atol=1e-6)
    # an untrained network prices far outside the no-arbitrage bounds: the code must report, not crash
    d = make_split("heston", "test", 500)
    rep = iv_error_report(d["X"], d["y"], price(d["X"]))
    assert rep["n_pred_price_outside_bounds"] + rep["overall"]["n"] == rep["n_eligible"]
    plot_smiles(tmp_path / "s.png", price)
    plot_surface(tmp_path / "u.png", price)
    assert (tmp_path / "s.png").exists() and (tmp_path / "u.png").exists()
