# Performance guide

When `gamrs` is fast, when it isn't, and what to do about it.

## TL;DR

- **`n ≤ 50_000`** — defaults are fine. `gamrs` matches or beats
  `mgcv_rust` on every family.
- **`n > 50_000` and family in {Gaussian, Poisson, Bernoulli, Gamma}**
  — pass `method="fREML"`. Shaves 30-60% off wall time.
- **NegBin / Tweedie / TDist** — defaults are fine, fREML doesn't apply
  (these use bespoke joint-θ outer solvers).
- **Shape-aware quantile (ELF)** — only the σ search is in Python; the
  inner GAM is the same fast Rust core. Speed depends on `K_folds` and
  the Brent tolerance (`xatol`).

## Choosing the outer optimiser

| `method=`       | Algorithm                                  | Inner step                         | When to use                        |
| --------------- | ------------------------------------------ | ---------------------------------- | ---------------------------------- |
| `"REML"` (def)  | Damped Newton on REML score                | Full PIRLS to convergence per iter | Default; small/medium n            |
| `"fREML"`       | Fellner-Schall multiplicative λ updates    | Single IRLS step per iter          | Large n; Gaussian-ish GLM families |

`"fREML"` is the algorithm R's `bam()` uses. Wood & Fasiolo (2017) showed
the multiplicative update is equivalent to a fixed-point iteration with
much lower per-iter cost than damped Newton, at the price of slightly
slower convergence near optimum and a Hessian that's an approximation
rather than the exact REML Hessian.

```python
g_default = Gam(family="poisson").fit(X, y)                 # REML, damped Newton
g_fast    = Gam(family="poisson", method="fREML").fit(X, y) # bam() equivalent
```

The two converge to within ~1e-4 on ρ̂ and µ̂ in practice — the predictive
performance is indistinguishable. `predict_ci` widths are very close but
not bit-identical (the fREML Hessian is an approximation).

## Reading `fit_stats_`

Every fitted `Gam` carries a `fit_stats_` dict with counters from the
solver, useful when investigating a slow or flaky fit:

```python
g.fit_stats_
# {
#   'outer_iterations':           7,    # outer loop iters (Newton or FS)
#   'line_search_trials':        12,    # step-halving attempts (Newton only)
#   'no_refresh_attempts':         9,   # IFT no-refresh shortcut tries
#   'no_refresh_hits':             7,   # successful no-refresh shortcuts
#   'no_refresh_hit_rate':       0.78,  # cheap-path hit ratio
#   'inner_pirls_calls':         15,    # PIRLS invocations
#   'inner_pirls_iterations_total': 47, # total PIRLS iters across all calls
#   'pirls_iters_per_call':       3.1,  # avg PIRLS depth
# }
```

What to look for:

- **`outer_iterations` ≥ 50** — outer solver is struggling. Probably a
  family / data mismatch (e.g. binary y with `family="gaussian"`),
  numerical issues, or `k` too large for the data. Check `g.converged_`.
- **`pirls_iters_per_call > 10`** — inner solver is grinding. Often
  indicates an ill-conditioned design (correlated predictors, near-zero
  smoothing parameter).
- **`no_refresh_hit_rate < 0.3`** — the IFT shortcut is firing but
  failing the safety check; the line search is doing real work. This is
  diagnostic, not actionable — the solver is correctly falling back.
- **`line_search_trials` ≫ `outer_iterations`** — the Newton step is
  often rejected. The damping is doing its job; not a bug, but if you're
  in a perf-sensitive setting consider `method="fREML"`.

`fit_stats_` is a zero-cost capture in release builds (it lives in `Cell`s
on the stack and the optimiser short-circuits the writes when no consumer
holds a reference).

## Reproducing the benches

The numbers in the [README perf table](../README.md#performance) come
from `scripts/bench_large_n.py`:

```bash
.venv/bin/python scripts/bench_large_n.py
```

Each cell is best-of-3 wall time. Synthetic data; single smooth (k=20);
fixed seed. `mgcv_rust` 0.23.2 is the reference.

`scripts/bench_vs_mgcv_rust.py` runs a broader matrix (all families,
multi-smooth, tensor product).

`scripts/compare_iters.py` reports outer-iteration counts side-by-side
between `gamrs` and `mgcv_rust` — useful for diagnosing why one solver
is taking more iterations than the other on a given dataset.

## Architectural levers (advanced)

Adding a new family means implementing the `Loss` trait. The trait
exposes three perf knobs:

```rust
fn allows_no_refresh(&self) -> bool        { false }  // IFT shortcut on/off
fn outer_tuning(&self) -> OuterTuning      { default }  // grad / REML tols
fn pirls_dev_rel_tol(&self) -> f64         { 1e-8 }   // PIRLS convergence
```

`allows_no_refresh = true` enables Wood (2011) §4.2's no-refresh shortcut
(propagate β through the IFT instead of re-running PIRLS) for trial
steps. This is responsible for most of the GLM-family speedup. Safe for
families with canonical-link Fisher = observed-info equivalence; risky
for shape-aware families where the score depends on more than just β.

`outer_tuning` lets a family relax the outer tolerances when its
parameter scale doesn't need 1e-9 precision (NegBin θ in particular).

`pirls_dev_rel_tol` controls inner PIRLS convergence. Looser → faster
fits, but at the cost of µ accuracy.

See `src/family/*.rs` for current per-family pickings.

## What about `discrete=True`?

`gamrs` accepts the kwarg for `mgcv_rust` source compat, but it's a
no-op — the dense-PIRLS path is already faster than `mgcv_rust`'s
discrete-binning path at the scales we've tested up to n=1M. We may add
it back if data shows up where binning helps; the bench harness
(`scripts/bench_discretizable.py`) tracks this.
