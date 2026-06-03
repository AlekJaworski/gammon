# gamrs

Generalised Additive Models in Rust — clean-room reimplementation built on
six composable trait layers (`Basis`, `BasisTransform`, `Loss`/`Link`/`VarianceFn`,
`InnerSolver`, `ScoreDerivatives`, `OuterSolver`). Designed for parity with
R's `mgcv`.

**Status: beta (v0.7).** Faster than `mgcv_rust` 0.23 at every tested
fixture and scale (see [Performance](#performance)), and `mgcv` R-parity
on µ across all ten families. Multi-smooth additive (`y ~ s(x0) + s(x1)`),
n-margin tensor products (`te(x0, x1, …)` / `ti(…)`) and thin-plate splines
(`s(x0, x1, bs="tp")`) all ship. NegBin and Tweedie fit multi-smooth, with
Tweedie offering both profile-p (`tw()`) and fixed-p (`Tweedie(p)`).

## Install

```bash
pip install gamrs            # base wheel
pip install gamrs[quantile]  # + scipy for SHASH-calibrated quantile fits
```

## Quickstart

```python
from gamrs import Gam, CrTerm, TeTerm

# Single 1-D smooth, Gaussian
g = Gam(family="gaussian").fit(X, y)
mu  = g.predict(X)
mu, lo, hi = g.predict_ci(X, level=0.95)

# Multi-smooth additive
g = Gam(terms=[CrTerm("x0", k=10), CrTerm("x1", k=15)]).fit(df, df["y"])

# Tensor product
g = Gam(terms=[TeTerm(cols=("x0", "x1"), k=(5, 5))]).fit(df, df["y"])

# Large-n GLM — switch to the bam()-style fREML optimiser
g = Gam(family="poisson", method="fREML").fit(X_big, y_big)
```

Full walkthrough: [`docs/quickstart.md`](docs/quickstart.md).
Optimiser & large-n notes: [`docs/perf.md`](docs/perf.md).

## Families

All ten families land 1-D parity against `mgcv`:

| Family          | Link     | Inner solver | Outer optimiser     | Parity (µ rel-err) |
| --------------- | -------- | ------------ | ------------------- | ------------------ |
| Gaussian        | identity | one-Cholesky | 1-D Newton          | ~3e-6              |
| Bernoulli       | logit    | PIRLS        | 1-D Newton          | ~1e-3              |
| Poisson         | log      | PIRLS        | 1-D Newton          | ~8e-5              |
| QuasiPoisson    | log      | PIRLS        | 1-D Newton (prof φ) | ~2e-4              |
| QuasiBinomial   | logit    | PIRLS        | 1-D Newton (prof φ) | ~7e-5              |
| Gamma           | log      | PIRLS        | 1-D Newton (prof φ) | ~2e-2              |
| InverseGaussian | log      | PIRLS        | 1-D Newton (prof φ) | ~3e-4              |
| NegBin          | log      | PIRLS        | 2-D joint Newton    | ~9e-7              |
| Tweedie         | log      | PIRLS        | 3-D joint Newton    | ~5e-3              |
| TDist (`scat`)  | identity | PIRLS        | 3-D joint Newton    | ~2e-2              |
| Ocat            | logit    | gam.fit5     | joint β + threshold | smoke              |
| Quantile (ELF)  | identity | Armijo BT    | 1-D Newton          | smoke              |

Multi-smooth (`s(x0) + s(x1) + …`) ships with `mgcv` R parity tests for
Gaussian / Bernoulli / Poisson / QuasiPoisson / QuasiBinomial / Gamma /
InvGauss / NegBin / Tweedie. scat / TDist multi-smooth fits run and
converge; reference parity tests are pending. Ocat and Quantile/ELF
remain single-smooth-only.

## Smooths

- **Single 1-D** — `s(x0)` via `CrTerm` (cubic regression spline default).
- **Additive multi** — `s(x0) + s(x1) + s(x2)`.
- **Tensor product** — `te(x0, x1, …)` via `TeTerm`, `ti(…)` via `TiTerm`,
  any n-margin.
- **Thin-plate** — `s(x0, x1, bs="tp")` via `TpsTerm`.
- **Random effects** — `s(g, bs="re")` via `ReTerm`.
- **Parametric (linear)** — unsmoothed raw column via `ParametricTerm` or
  `predictor_basis_map={"x": "parametric"}` (alias `"linear"`). Use for
  0/1 indicators, counts, or anything you want unpenalised. mgcv R's
  "pterms" block.

## Performance

`gamrs` vs `mgcv_rust` 0.23.2, single smooth (k=20), best-of-3 wall time.
Numbers >1× mean gamrs is faster. See `scripts/bench_large_n.py`.

| family    | n=10K | n=100K | n=1M |
| --------- | ----: | -----: | ---: |
| Gaussian  | 0.98× | 1.46×  | 1.62×|
| Poisson   | 1.50× | 1.20×  | 1.27×|
| Bernoulli | 1.06× | 1.42×  | 1.49×|

For GLM families at large n, set `method="fREML"` (mgcv R's `bam()`
equivalent — Wood & Fasiolo 2017 Fellner-Schall multiplicative updates
with single-step IRLS per outer iteration). The defaults are already
sensible at small/medium n; the [perf guide](docs/perf.md) covers when
to switch.

## Rust API

```rust
use gamrs::{TermSpec, MarginKind, DesignStrategy};
use ndarray::Array2;

let x: Array2<f64> = /* (n, n_input_dims) */;
let y = /* Array1<f64> */;

let fit = gamrs::fit(gamrs::family::gaussian_identity(), x.view(), y.view(), None, 10)?;

let fit = gamrs::fit_with_design(
    gamrs::family::gaussian_identity(),
    DesignStrategy::Additive { terms: vec![
        TermSpec::Cr { col: 0, k: 10 },
        TermSpec::Cr { col: 1, k: 15 },
    ]},
    x.view(), y.view(), None,
)?;

let fit = gamrs::fit_with_design(
    gamrs::family::gaussian_identity(),
    DesignStrategy::Additive { terms: vec![
        TermSpec::Tensor { col_a: 0, col_b: 1, k_a: 5, k_b: 5,
                           bs_a: MarginKind::Cr, bs_b: MarginKind::Cr },
    ]},
    x.view(), y.view(), None,
)?;

let mu = fit.predict(x.view())?;
```

## Python API

PyO3 bindings + numpy. sklearn-like surface: `fit` / `predict` /
`predict_ci` / `predict_diff` / `vcov_` / `coef_` / `lambda_` /
`edf_` / `fit_stats_`, plus serialize / deserialize and `GamPredictor`
for inference-only deployment.

## Architecture

Trait layering (`src/traits.rs`):

```
Layer 1   Basis              ←  CrBasis, RandomEffectsBasis, TensorProductBasis<A, B>
Layer 1.5 BasisTransform     ←  SumToZero, StableReparam
Layer 2   Loss/Link/Variance ←  10 families (see table above)
Layer 3   InnerSolver        ←  GaussianClosedFormInner, PirlsInner, GamFit5Inner, ArmijoInner
Layer 4   ScoreDerivatives   ←  EnvelopeScore, ShapeAwareEnvelopeScore
Layer 5   OuterSolver        ←  NewtonWithHalving, FellnerSchall
Layer 6   FittedGam          ←  predict, predict_ci, predict_diff, vcov, serialize
```

Outer optimisers (Newton, Fellner-Schall) and per-family tolerances are
selected through `Loss::outer_tuning()` and `Loss::allows_no_refresh()`,
so adding a family is a `Loss` impl, not a fork of the optimiser.

## Versioning

Beta (`0.7.x`). The API is stabilising; minor bumps may carry breaking
changes until the remaining shape-aware families (scat/Ocat/ELF) gain
multi-smooth support and the 1.0 surface is locked.

## License

MIT.
