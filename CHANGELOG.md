# Changelog

All notable changes to **gamrs** are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); this project is in
beta (`0.x`), so minor bumps may carry breaking changes until the 1.0 surface
is locked. Versions correspond to the published PyPI wheels.

## [Unreleased]

## [0.11.8] — 2026-06-10

### Changed
- **Wheels now build with `panic = "unwind"`** (was `abort`). PyO3 wraps every
  binding in `catch_unwind`, so a Rust panic on a degenerate input now
  surfaces as a catchable Python `PanicException` instead of `SIGABRT`-ing the
  host interpreter. Correct failure mode for a library; negligible cost for a
  numerical crate.

### Fixed
- Penalty pseudo-determinant eigendecomposition now returns a typed
  `GamrsError::Linalg` on failure instead of `expect()`-panicking
  (`src/design/mod.rs::rank_and_log_pseudo_det`). Degenerate / rank-deficient
  designs surface as a catchable error rather than a host-interpreter abort.

### Added
- **Multi-smooth `scat` (scaled-t) mgcv reference parity.** 2-D and 3-D
  additive fixtures (`tests/parity_additive_scat.rs`,
  `tests/python/test_parity_multismooth.py::test_additive_scat_parity`,
  `scripts/r/gen_scat_multismooth_fixtures.R`): µ rel-err ~9e-3 (2-D) /
  ~1.7e-2 (3-D), σ̂² matching mgcv to ~0.1 %. Closes the README's
  "multi-smooth scat reference parity pending" gap.
- **First mgcv `ocat` multi-smooth parity** (`parity_ocat.rs` was smoke-only):
  `test_additive_ocat_parity` against `ocat(R=4)` on a well-posed noisy-latent
  DGP where both gamrs and mgcv converge cleanly — `predict_proba` agrees to
  ~1.8e-3 mean abs / ~98 % class agreement
  (`scripts/r/gen_ocat_multismooth_fixtures.R`).
- Synthetic, committable regression test for the v0.11.2 `scat`
  robustness-direction fix
  (`tests/parity_scat.rs::scat_downweights_high_outliers_synthetic`) — locks
  the directional contract (scat pulls *below* Gaussian under positive
  outliers) without depending on the proprietary housing fixture.
- Degenerate-input smoke test
  (`test_degenerate_input_never_aborts_the_interpreter`): NaN / inf /
  zero-variance fits raise catchable Python exceptions, never `SIGABRT`.
- `profile`-feature timers on the NegBin/Ocat profile-θ path
  (`rho_only_total`, `fit_inner_pirls`, `frozen_beta_probe`,
  `no_refresh_probe`, `hess_ift_rho`, `fit_inner_build`), matching the
  joint-Newton path's instrumentation. `cargo bench --bench bench_nb
  --features profile` now emits a per-phase breakdown; `bench_nb` gained
  realistic-size synthetic 2-D NegBin cases (n=2K, n=5K).

### Docs
- `docs/scat_parity_bug.md` rewritten as a self-contained **RESOLVED** note
  (was a stale "open" handoff referencing gitignored proprietary data).
- README: multi-smooth families line now lists `scat`; ocat paragraph documents
  the near-separable regime where mgcv itself diverges (θ≈181) or aborts while
  gamrs's θ∈(−3,3) bound stays stable.
- This `CHANGELOG.md` added.

### Internal
- `data/` (real housing sales) and `uv.lock` added to `.gitignore`.

## [0.11.7] — release the GIL during Python fits
- Fits release the GIL (`py.detach`) for the entire solve, so independent
  `Gam.fit` / `fit_quantile` calls run truly concurrently on a
  `ThreadPoolExecutor`. Set `OPENBLAS_NUM_THREADS=1` when fanning fits across a
  thread pool to avoid BLAS oversubscription.

## [0.11.6] — scat/link robustness + coverage
- scat: start `µ` at the response (mgcv `mustart <- y`); standardize the
  response in the fit core for raw-scale robustness (Rust API too).
- Links: floor `Log`/`Inverse` `d2`/`d3` denominators (no ±inf at µ≈0).
- Added additive + tensor multi-smooth extrapolation behaviour tests and
  non-identity-link extrapolation parity vs mgcv.

## [0.11.5] — wheel portability fix
- Pin `OPENBLAS_TARGET` + add an SDE AVX2 guard to fix the 0.11.4 `SIGILL` on
  AVX2-only hosts.

## [0.11.4] — scat raw-scale Rust API
- scat raw-scale standardization relocated into the fit core (Rust API now
  fits large-`y` scat correctly, not just the Python wrapper).
- CI: Node-24 / portability guard; wheel version derived from `Cargo.toml`
  (`pyproject` `dynamic`).

## [0.11.2] — scat mgcv parity (robustness-direction fix)
- **Fixed the `scat` wrong-direction robustness bug**: the IRLS outlier
  fallback paired the expected weight with the Fisher response, pulling
  outliers *toward* `y` (Gaussian-like) instead of down-weighting them. Now
  matches mgcv R to ~0.05 % on the fitted level. `min.df` 2 → 3 to match
  mgcv `scat(min.df = 3)`. See `docs/scat_parity_bug.md`.

## [0.11.1] — scat perf
- scat `0.52× → 0.77×` (vs mgcv_rust) via broadcast-expression conversions and
  batched `h_diag` matmul.

## [0.11.0] — analytic Hessian for scat + warm-start PIRLS
- Analytic Level-2 Hessian + observed-W PIRLS + warm-start for scat
  (`0.07× → ~0.52×`).

## [0.10.0] — full mgcv R outer-Newton stabilisation stack
- Ported smart θ-init from category frequencies, diagonal Hessian
  preconditioning, Gill-Murray-Wright eigen-fix, subset Newton, and the
  rank-deficient KKT convergence check. Single-smooth ocat now converges
  cleanly on every tested seed.

## [0.9.0] — predict_proba + parametric terms
- `predict_proba` for ocat; parametric (unpenalised, linear) terms baseline.

## [0.6.0] — fREML
- Single-step IRLS `fREML` (`bam()`-equivalent) — beats mgcv_rust at every
  tested scale for GLM families at large `n`.

## [0.4.0] — beta
- First beta. Multi-smooth additive, tensor products, and the ten-family
  parity battery.

[Unreleased]: https://github.com/AlekJaworski/gamrs/compare/v0.11.8...HEAD
[0.11.8]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.8
[0.11.7]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.7
[0.11.6]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.6
[0.11.5]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.5
[0.11.4]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.4
[0.11.2]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.2
[0.11.1]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.1
[0.11.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.0
[0.10.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.10.0
[0.9.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.9.0
[0.6.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.6.0
[0.4.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.4.0
