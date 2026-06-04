# gamrs

Generalised Additive Models in Rust — clean-room reimplementation built on
six composable trait layers (`Basis`, `BasisTransform`, `Loss`/`Link`/`VarianceFn`,
`InnerSolver`, `ScoreDerivatives`, `OuterSolver`). Designed for parity with
R's `mgcv`.

**Status: beta (v0.10).** Faster than `mgcv_rust` 0.23 at every tested
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
converge; reference parity tests are pending. Quantile/ELF is
single-smooth-only.

Multi-smooth Ocat fits run, produce well-defined `predict_proba`, and
reach 99%+ classification accuracy on synthetic fixtures.

v0.10 ports the full mgcv R outer-Newton stabilisation stack (smart
θ-init from category frequencies, diagonal Hessian preconditioning,
Gill-Murray-Wright eigen-fix, subset Newton, rank-deficient KKT
convergence check). After the ports, **single-smooth ocat converges
cleanly on every tested seed**. Multi-smooth still hits a flat
coordinated-shift ridge on the most pathological synthetic fixtures
(both θ axes against the bound) where `converged_=False` lingers
despite correct predictions — same regime mgcv R itself bails out
of with a "did not converge after 200 iterations" warning. See
`~/ObsidianVault/Projects/gamrs/gamrs - mgcv outer-Newton
stabilisation techniques (port catalogue) 2026-06-03.md` for the
full port story.

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

`gamrs` vs `mgcv_rust` 0.23.2, single-smooth, best-of-20 wall time
after long warm-up. Numbers >1× mean `gamrs` is faster. Measured on a
single 12th-gen Intel core. See `scripts/bench_matters.py`.

Single-smooth at gamrs-default `k=10`, n=2K:

| family    | speedup vs mgcv_rust |
| --------- | -------------------: |
| Gaussian  | 1.6×                 |
| Poisson   | 1.9×                 |
| Bernoulli | 2.1×                 |
| Tweedie   | 2.6×                 |
| NegBin    | 1.0×                 |
| scat      | 0.77×                |
| ocat      | 14× *(†)*            |

*(†)* ocat speedup is fixture-dependent; mgcv_rust's ocat path is
slow on multi-category outputs by construction.

scat closed substantially in v0.11.1 — from v0.10.0's 0.07× to 0.77× of
mgcv_rust. Two structural wins drove it: (a) analytic gradient + Level-2
analytic Hessian + observed-W PIRLS + warm-start (v0.11.0), and (b)
broadcast-expression conversion + batched-h_diag matmul across the IFT
chain (v0.11.1). The remaining ~25 % gap is per-pair work in the
Hessian assembly that's already broadcast-friendly — closing it would
need either algorithmic shortcuts or fundamentally different BLAS
dispatch at small p.

For GLM families at large n, set `method="fREML"` (mgcv R's `bam()`
equivalent — Wood & Fasiolo 2017 Fellner-Schall multiplicative updates
with single-step IRLS per outer iteration). The defaults are sensible
at small/medium n; the [perf guide](docs/perf.md) covers when to switch.

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

Beta (`0.10.x`). The API is stabilising; minor bumps may carry breaking
changes until the remaining shape-aware families (scat/Ocat/ELF) gain
multi-smooth support and the 1.0 surface is locked.

## License

MIT.
