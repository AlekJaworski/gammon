# gammon

Generalised Additive Models in Rust — a clean-room reimplementation built on
six composable trait layers (`Basis`, `BasisTransform`, `Loss`/`Link`/`VarianceFn`,
`InnerSolver`, `ScoreDerivatives`, `OuterSolver`). Designed for parity with
R's `mgcv` and the sibling `mgcv_rust` crate.

**Status: alpha.** Multi-smooth additive (`y ~ s(x0) + s(x1)`) and tensor
products (`te(x0, x1)`) both ship in this release — the latter leap-frogs
`mgcv_rust`. Shape-aware families (scat, NegBin, Tweedie, Ocat, ELF) are
still single-smooth-only; production users with multi-smooth shape-aware
models should stay on `mgcv-rust` for now.

## What's in this alpha

### Families (all 1-D parity, 10 families)

| Family          | Link     | Inner solver | Outer Newton        | Parity (μ rel-err) |
| --------------- | -------- | ------------ | ------------------- | ------------------ |
| Gaussian        | identity | one-Cholesky | 1-D Newton          | ~3e-6              |
| Bernoulli       | logit    | PIRLS        | 1-D Newton          | ~1e-3              |
| Poisson         | log      | PIRLS        | 1-D Newton          | ~8e-5              |
| QuasiPoisson    | log      | PIRLS        | 1-D Newton (prof φ) | ~2e-4              |
| QuasiBinomial   | logit    | PIRLS        | 1-D Newton (prof φ) | ~7e-5              |
| Gamma           | log      | PIRLS        | 1-D Newton (prof φ) | ~2e-2              |
| InverseGaussian | log      | PIRLS        | 1-D Newton (prof φ) | ~3e-4              |
| NegBin          | log      | PIRLS        | 2-D joint Newton    | ~9e-7              |
| Tweedie         | log      | PIRLS        | 3-D joint Newton    | ~2e-1              |
| TDist (scat)    | identity | PIRLS        | 3-D joint Newton    | ~2e-2              |
| Ocat            | logit    | gam.fit5     | joint β + threshold | smoke              |
| Quantile (ELF)  | identity | Armijo BT    | 1-D Newton          | smoke              |

### Bases

- `Cr` — cubic regression splines (default for 1-D smooths)
- `Re` — random effects (`bs="re"`)
- `Tensor<A, B>` — anisotropic tensor product (`te(x0, x1)`)

### Smooth strategies

- **Single 1-D smooth** — `y ~ s(x0)`
- **Additive multi-smooth** — `y ~ s(x0) + s(x1) + s(x2)` (parity with mgcv_rust on Gaussian/Bernoulli/Poisson/Gamma/InvGauss/QuasiPoisson/QuasiBinomial)
- **Tensor product** — `y ~ te(x0, x1)` (leap-frogs mgcv_rust; v0.x doesn't have this)

### Python API

PyO3 bindings + numpy. sklearn-like with `vcov`, `predict_ci`, `predict_diff`,
serialize/deserialize, GamPredictor for inference-only deployment.

## Not in this alpha (follow-ups)

- **Multi-smooth shape-aware families** — scat/TDist, NegBin, Tweedie, Ocat,
  ELF/quantile gated at single-smooth via runtime error. Lifting θ packing
  from `[ρ, shape…]` to `[ρ_0, …, ρ_{T-1}, shape…]` is tracked.
- **`s(x0, x1)` TPRS** — isotropic thin-plate; separate basis kind from `te()`.
- **3+ margin tensor products** `te(x0, x1, x2)` — mechanical generalization.
- **`ti(x0, x1)`** centred tensor interaction — needs main-effect orthogonalisation.

## Use (Rust)

```rust
use gammon::{TermSpec, MarginKind, DesignStrategy};
use ndarray::Array2;

let x: Array2<f64> = /* (n, n_input_dims) */;
let y = /* Array1<f64> */;

// Single 1-D smooth
let fit = gammon::fit(gammon::family::gaussian_identity(), x.view(), y.view(), None, 10)?;

// Multi-smooth additive
let fit = gammon::fit_with_design(
    gammon::family::gaussian_identity(),
    DesignStrategy::Additive { terms: vec![
        TermSpec::Cr { col: 0, k: 10 },
        TermSpec::Cr { col: 1, k: 15 },
    ]},
    x.view(), y.view(), None,
)?;

// Tensor product
let fit = gammon::fit_with_design(
    gammon::family::gaussian_identity(),
    DesignStrategy::Additive { terms: vec![
        TermSpec::Tensor { col_a: 0, col_b: 1, k_a: 5, k_b: 5, bs_a: MarginKind::Cr, bs_b: MarginKind::Cr },
    ]},
    x.view(), y.view(), None,
)?;

let mu = fit.predict(x.view())?;
```

## Use (Python)

```python
from gammon import Gam, CrTerm, ReTerm, TeTerm

# Additive multi-smooth
g = Gam(terms=[CrTerm("x0", k=10), CrTerm("x1", k=15)])
g.fit(df, "y")
mu = g.predict(df)

# Tensor product
g = Gam(terms=[TeTerm(cols=("x0", "x1"), k=(5, 5))])
g.fit(df, "y")
```

## Architecture

See `architecture-assumptions.md` in the repo root and the v2 plan note in
`~/ObsidianVault/Projects/mgcv_rust/plans/mgcv_rust - v2 Architecture Plan 2026-05-22.md`.

The trait layering (`src/traits.rs`):

```
Layer 1   Basis              ←  CrBasis, RandomEffectsBasis, TensorProductBasis<A, B>
Layer 1.5 BasisTransform     ←  SumToZero, StableReparam
Layer 2   Loss/Link/Variance ←  10 families (see table above)
Layer 3   InnerSolver        ←  GaussianClosedFormInner, PirlsInner, GamFit5Inner, ArmijoInner
Layer 4   ScoreDerivatives   ←  EnvelopeScore, ShapeAwareEnvelopeScore
Layer 5   OuterSolver        ←  NewtonWithHalving (ρ-dim generic)
Layer 6   FittedGam          ←  predict, predict_ci, predict_diff, vcov, serialize
```

## Versioning

`0.1.0-alpha.x` indicates pre-stable. Breaking changes are expected on every
minor bump until shape-aware multi-smooth lands.

## License

MIT.
