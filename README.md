# Neural Network Option Pricing Engine

*(Work in progress: motivation, method and results tables are filled in as stages complete.)*

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                  # all tests, including the PyTorch ones
```

## Stage 2: NN surrogate for Black-Scholes (sanity check)

**1. Smoke test (about 1 minute) to check everything runs end-to-end:**

```bash
python generate_data.py --model bs --n-train 20000 --n-val 5000 --n-test 5000 --tag smoke
python nn_pricer.py --data data/bs_smoke --epochs 3 --tag smoke
python evaluate.py --model models/bs_smoke_mlp.pt --data data/bs_smoke
```

The numbers will be poor after 3 epochs. The only goal is no errors.

**2. Full run:**

```bash
python generate_data.py --model bs                       # 1M train / 100k val / 100k test, ~2 s
python nn_pricer.py --data data/bs                        # 4x256 SiLU MLP, early stopping
python evaluate.py --model models/bs_mlp.pt --data data/bs
```

Training time depends on the machine: roughly 2-5 s per epoch on a GPU or Apple-silicon (MPS),
and 5-15 s per epoch on a laptop CPU. Early stopping usually ends it well before 150 epochs.
Add `--device cpu` if MPS/CUDA causes problems.

**Outputs:** `models/bs_mlp.pt` (weights) and `models/bs_mlp_history.json` (training log),
`results/bs_mlp.json` (all metrics), and `plots/bs_mlp_{bucket_mae,greek_slices,training}.png`.

## Stage 3: Heston model (Fourier + Monte Carlo), validated

```bash
pytest tests/test_heston.py      # 18 tests, about 30 s
python heston.py                 # quick demo: Fourier vs MC at 5 strikes
python validate_heston.py        # full validation report + plots, about 1 minute
```

**Fourier pricer.** Lewis (2001) single-integral formula with the Gatheral "little trap"
characteristic function, which avoids the complex-log branch-cut problem of Heston's original form.
`heston_price` is vectorised Gauss-Legendre quadrature with the truncation point chosen per option and the
number of nodes (256 up to 8192) chosen per option from how fast the integrand oscillates (see "Pricer bug found in Stage 4" below).
`heston_price_quad` uses adaptive QUADPACK and is the accuracy reference.
Over 400,000 random parameter sets the fast pricer never violates the no-arbitrage bounds by more than 2e-12 (units of K),
and the 300 hardest options match the adaptive reference to 4e-12.

**Monte Carlo.** Andersen's QE scheme with martingale correction, plus full-truncation Euler for
comparison. Both use antithetic variates.

**How we know it's right (all automated tests):**
- The characteristic function satisfies phi(0) = 1 and phi(-i) = 1. The second identity means the discounted spot is a martingale.
- Fourier and QE Monte Carlo agree within 4 standard errors for calls and puts at 5 strikes in 3 regimes, including one where the Feller condition is badly violated. Over 8 seeds the z-scores have mean 0.00 and standard deviation of about 1, so there is no detectable bias.
- Andersen (2008) cases I-III: Fourier 13.0847 / 16.6492 / 33.5968, all matched by MC within 1.3 SE.
- Black-Scholes limit: as xi -> 0, Heston converges to BS with the time-averaged variance. The error is O(xi) when rho != 0 (the skew term) and O(xi^2) when rho = 0. Both rates are verified (`plots/stage3_bs_limit.png`).
- No static arbitrage: prices are decreasing and convex in strike and within the model-free bounds. Put-call parity holds.
- External reference value 34.9998 is reproduced (34.99976).

**Discretisation bias** (`plots/stage3_bias_vs_steps.png`, Andersen case I, T = 10y):
at 1 step per year QE is off by +0.22 and Euler by +6.48, on a price of 13.08.
Euler needs over 32 steps per year to get within 0.3.

## Stage 4: NN surrogate for Heston

```bash
pip install -r requirements.txt            # nothing new
pytest                                     # 59 tests
python generate_data.py --model heston     # 1M / 100k / 100k labelled options; roughly 5-15 min, prints progress
python nn_pricer.py --data data/heston --layers 5 --hidden 256 --batch 2048 --epochs 150
python evaluate_heston.py --model models/heston_mlp.pt --data data/heston
```

Smoke test first (about 3 minutes): add `--n-train 20000 --n-val 5000 --n-test 5000 --tag smoke` to the generator,
then `--data data/heston_smoke --epochs 3 --tag smoke` to the trainer and `--model models/heston_smoke_mlp.pt
--data data/heston_smoke` to the evaluator.

**Inputs (8):** S/K, T, r, v0, kappa, theta, xi, rho. **Target:** call price / K. Parameter box: S/K in [0.7, 1.3],
T in [0.05, 2], r in [0, 0.08], v0 and theta in [0.005, 0.25], kappa in [0.5, 5], xi in [0.1, 1], rho in [-0.95, 0].
The Feller condition is deliberately not imposed (real calibrations often violate it). The surrogate is only valid
inside this box. Train / validation / test are independent Latin-hypercube samples, in separate files.

**Label quality is checked automatically.** For each split the generator records the worst no-arbitrage-bound
violation and the maximum difference from the independent adaptive integrator on a random subset (in `meta.json`);
any label outside the bounds by more than 1e-9 is recomputed with the slow integrator.

**Pricer bug found in Stage 4.** The first version of `heston_price` used a fixed 256 quadrature nodes. When generating
the dataset I found 72 of 200,000 labels were negative (worst -8e-5 of the strike), larger than the network's own
Black-Scholes error. Cause: for low initial variance, high vol-of-vol, strongly negative correlation and
short-to-medium maturities (both in and out of the money) the integrand oscillates over a long range and 256 nodes
cannot resolve it. On a fresh 20,000-option sample, 37 options (0.19%) were wrong by more than 1e-8 and the worst by
3.8e-5. Stage 3's validation sampled only 300 options and missed it. Fix: nodes per option are chosen from the
oscillation measure R = U * (|log-moneyness + drift| + 0.05) (6 nodes per unit of R, rounded up to a power of two,
built from 1024-node panels above that). Afterwards: no error above 1e-8 on the 20,000-option sample (max 3.7e-9
against a 4096-node reference), no bound violation above 2e-12 over 400,000 options, and the 300 highest-R options
match the adaptive integrator to 4e-12. Speed is unchanged. Regression tests are in `tests/test_heston.py`.

**Greeks.** Heston has no closed form, so the network's autograd delta, gamma, dV/dv0 and theta are compared with
central finite differences of the Fourier pricer. `evaluate_heston.py` also reports how much that reference itself
changes when the bump size is doubled, so errors below that level are not over-interpreted.

**Speed.** Reported at batch sizes 1, 100, 5000 and 100,000 against the vectorised Fourier pricer (the fair
baseline), plus the slow adaptive integrator for context, all on the same machine.

## Stage 5: volatility smiles and implied-vol accuracy

```bash
pytest                                            # 70+ tests
python smile.py --model models/heston_mlp.pt --data data/heston
python smile.py --no-nn                           # Fourier-only plots, no PyTorch needed
```

**Smile.** Heston prices are converted to implied volatilities with the Black-Scholes solver. Black-Scholes would
give one vol for every strike; Heston gives a downward skew (with rho < 0), steeper when the Feller condition is
violated (`plots/stage5_smiles.png`). With the first preset (v0=0.04, kappa=2, theta=0.05, xi=0.4, rho=-0.7) the
1-year vol is 24.4% at an 80% strike, 20.6% at-the-money and 16.9% at a 125% strike.

**Fast IV solver.** `implied_volatility_bisect` (vectorised bisection, about 22 us/option) matches the scalar Brent
solver to 1e-9 on well-identified options and returns NaN, never an exception, for prices outside the no-arbitrage
bounds. The smile plots use the original Brent solver; the 100,000-option test-set statistics use the fast one.

**Why implied-vol error is the metric that matters.** IV error is approximately price error divided by vega. A fixed
price error is a tiny vol error for a near-the-money 1-year option and a large one for a cheap short-dated wing
option, so absolute price error flatters the network. `smile.py` reports IV(NN) - IV(Fourier) in vol points
(1 vol point = one percentage point of volatility) for options with vega >= 0.05 per unit strike (95% of the test
set), overall, by moneyness x maturity, by Feller ratio and for a market-like region (0.9 <= S/K <= 1.1,
T >= 0.1). Options with tiny vega are excluded because their price carries almost no volatility information; the
share excluded per bucket is printed. Any NN price outside the no-arbitrage bounds is counted separately.

**Known weakness (visible in `plots/stage5_smiles.png`).** The vega filter hides cheap short-dated wing options, and
that is exactly where the network is unreliable: at T = 0.1y it is off by about 6 vol points at a 125% strike and
returns prices below the no-arbitrage bound for strikes under about 87%. The network matches the Fourier smile for
T >= 0.5y and in the middle strikes at 0.25y. The report therefore also gives a "calibration region" (strikes
87-115% of spot, T >= 0.25y), which is where Stage 6 will use the network.

## Project layout

| File | Purpose |
|---|---|
| `black_scholes.py` | Vectorised BSM pricer, analytical Greeks, IV solver with no-arbitrage checks |
| `generate_data.py` | Latin-hypercube datasets; features (S/K, T, r, sigma), target V/K |
| `nn_pricer.py` | PyTorch MLP (smooth activations), training loop, autograd Greeks |
| `evaluate.py` | Test-set errors, error buckets, speed, NN Greeks vs analytical |
| `heston.py` | Heston characteristic function, Lewis Fourier pricer (fast + reference), QE / Euler Monte Carlo |
| `validate_heston.py` | Stage 3 validation report and plots |
| `smile.py` | Stage 5: smile and surface plots, implied-vol error of the NN on the test set |
| `evaluate_heston.py` | Stage 4: Heston NN test errors, buckets by moneyness/maturity/Feller ratio, Greeks vs finite differences, speed |
| `tests/` | pytest suite |

## Stage 6 - calibrating Heston to a real NIFTY option chain

`nse_chain.py` parses the CSV from NSE's option-chain page; `calibrate.py` fits Heston to it with the NN as the pricer.

```
python calibrate.py --csv data/nifty_chain.csv --date 2026-11-04 --expiry 2027-01-25 --r 0.06 --time-fourier
python calibrate.py --synthetic --fix-kappa 2.0          # sanity test on a fake chain
```

How it works: forward from put-call parity; price with `S_eff = F e^{-rT}`; OTM quotes only; least squares on
implied vol weighted by 1/(bid-ask width); 8 random starts inside the training box; then the fitted parameters are
re-priced with the exact Fourier pricer, and that error is the one reported.

Known limit (found on a synthetic test): one expiry cannot identify the parameters. The smile is reproduced to noise
level, but `v0`, `theta` and `kappa` trade off against each other and hit the box edge. Report the fit, not the parameters.


## Stage 6b - multi-expiry calibration, and a network built for calibration

`calibrate_surface.py` fits ONE Heston parameter set to several NIFTY expiries at once (expiry read from the NSE file name).
Measured on 6 chains of 2026-11-04 (exact-pricer re-check): 0.31 vol pts RMSE (4 expiries, T >= 0.09) and 0.38 (6 expiries).
The general network cannot calibrate alone (0.72 vol pts once re-priced exactly); as a warm start plus an exact polish it is
4.1-4.5x faster than the exact pricer alone with an identical final fit. The network's own fit error is never quoted.

Calibration-focused retrain (narrow box, time-value target, vega-weighted loss):
```
python generate_data.py --model heston_cal
python nn_pricer.py --data data/heston_cal --time-value --vega-weight --layers 5 --hidden 256 --batch 2048 --epochs 150
python calibrate_surface.py --date 2026-11-04 --csv (Get-ChildItem data\surface\*.csv).FullName --model models\heston_cal_mlp.pt --polish --time-fourier --tag surface_cal
```
