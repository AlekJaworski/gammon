# Changelog

All notable changes to **gamrs** are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); this project is in
beta (`0.x`), so minor bumps may carry breaking changes until the 1.0 surface
is locked. Versions correspond to the published PyPI wheels.

## [Unreleased]

## [0.12.2] — 2026-08-20

### Fixed
- **An all-parametric design is now fitted instead of refused (gaussian).** A
  predictor with only 2–3 distinct values is demoted to a parametric term, so a
  design made entirely of low-cardinality columns had no smooths at all —
  and `fit` raised *"All terms are parametric; a gamrs fit needs at least one
  smooth or random-effect term"*. That made gamrs unusable as a drop-in for
  mgcv anywhere a caller fits one submodel per feature: mgcv fits such a design
  without complaint, and any binary column (`pool`, a flag) hits it.

  With no smoothing parameters there is nothing for the outer Newton to search
  over, so `src/fit/gaussian.rs` skips it and takes the single unpenalised inner
  fit, and `src/inner/closed_form.rs` assembles a zero penalty of the design's
  own width (`combined_s` cannot infer that width from an empty `s_list`). The
  result is exactly weighted least squares: coefficients and predictions match
  `numpy.linalg.lstsq` to ~3e-7, `edf_total` equals the coefficient count,
  `scale_` is the residual variance on `n − p`, and `get_lambdas()` is empty.
  `n_iters` is 0 and `converged` is True because the answer is closed-form
  rather than iterated.

  **Gaussian only.** Every other family reaches the penalised PIRLS solvers,
  which still assume a non-empty `s_list`, so they keep an explicit refusal
  (now naming the family) rather than reaching a native panic.

## [0.12.1] — 2026-06-11

### Fixed
- **`fit_shash` robustness on near-Gaussian (and skewed / heavy-tailed) data.**
  v0.12.0 errored `SingularSystem: −Hess (Hp) is not SPD` on data with little
  skew/kurtosis (ε≈φ≈0) — common, and exactly where shash should reduce to a
  Gaussian. Two causes, both fixed in the outer REML (`src/gamlss/shash_reml.rs`):
  (1) `reml_eval`/`reml_grad` now perturb the penalised Hessian to SPD (mgcv's
  "ensure negative-definiteness" step) instead of erroring when the weakly-
  identified ε/φ directions carry ~0 curvature; (2) the smoothing-parameter
  search now clamps `ρ` to a sane range and rejects non-finite criteria — at
  absurd `ρ` the penalty `exp(ρ)·S0` overflowed the Hessian into a *finite
  garbage* `laml` the line search wrongly accepted, driving the fit to
  over-smooth μ all the way to linear (EDF→Mp). `fit_shash` now converges and
  recovers mgcv's fit on Gaussian/skewed/heavy-tailed data (e.g. Gaussian:
  μ̂ vs truth corr 0.999, EDF 12.1 vs mgcv 12.05, log-sp 4.90 vs 4.97).
  Regression guard: `fit_shash_converges_on_{gaussian,skewed,heavy_tailed}_data`
  unit tests (the earlier mgcv parity test only exercised shash-distributed
  data, so it never caught this).

### Docs
- README: added the `fit_shash` (sinh-arcsinh GAMLSS) section.

## [0.12.0] — 2026-06-11

### Added
- **GAMLSS: sinh-arcsinh (`shash`) — the first NON-orthogonal multi-linear-
  predictor family.** `gamrs.fit_shash(X, y, mu_terms=…, tau_terms=…,
  eps_terms=…, phi_terms=…)` fits all four sinh-arcsinh parameters — location
  μ(x), log-scale τ(x) (`logeb` link, σ = exp τ ≥ b), skewness ε(x), and
  log-kurtosis φ(x) — as smooths, **jointly**, to the shash likelihood. Unlike
  `gaulss`, shash's Fisher information is *not* block-diagonal, so it cannot
  alternate single-predictor fits: β for all four predictors is solved together
  by a **dense penalised block-Newton** inner solve (Levenberg-perturbed for
  ascent + step-halving), under **outer REML/LAML** smoothing-parameter
  selection driven by an **analytic gradient** (third derivatives `l3` →
  `d log|Hp|/dρ` via the implicit `dβ̂/dρ`). One fit yields every quantile via
  `ShashGamFit.predict_quantile` (the R `.shashQf` inverse-CDF); also
  `predict_eta` and `predict_params` → (μ, σ, ε, δ). Native Rust driver
  (`src/fit/shash.rs`, exported as `gamrs::fit_shash`); engine in
  `src/gamlss/{shash,shash_init,shash_inner,shash_reml}.rs`.

  Built component-by-component, each confronted with mgcv (1.9.4) **and** finite
  differences: density/grad/Hessian + the `logeb` link chain (vs mgcv `l0` and
  FD, ~1e-8); initialisation (bit-exact vs `pen.reg`); the joint inner Newton
  (recovers mgcv's intercept-only shash MLE to <1e-5, cross-checked against an
  independent BFGS optimum); `l3` and the analytic REML gradient (vs FD, ~3e-8);
  and the **outer REML fed mgcv's own designs recovers its smoothing parameters
  to ~1e-5 and EDF exactly**. End-to-end (gamrs building its OWN CR designs vs a
  2-smooth mgcv `shash` fit with `bs="cr"`): fitted η, smoothing parameters,
  total EDF and quantiles all match to **~1e-6**. (NB the reference smooths MUST
  request `bs="cr"`: mgcv's default `s(x, k)` is a *thin-plate* spline — a
  different basis — and comparing gamrs's `Cr` to it is apples-to-oranges; that
  mismatch, not any deficiency, accounted for an early ~1e-2 discrepancy seen
  during development.) Distinct from the two-stage
  `fit_quantile_lss(shape="shash")` (a per-residual MLE with a single global
  shape) — `fit_shash` is the genuine joint GAMLSS.
  Tests: `src/fit/shash.rs::fit_shash_matches_mgcv_two_smooth`,
  `tests/python/test_shash_gam.py`, `src/gamlss/*` unit suites (FD + mgcv).

### Known limitations
- `fit_shash` v1 supports at most one smooth term per predictor (one penalty
  per block); multi-smooth-per-predictor is a follow-up.
- shash, like mgcv's, needs an identifiable scale/shape — near-deterministic
  data can leave the penalised Hessian non-SPD, surfaced as a clear
  `SingularSystem` error rather than a silently bad fit.

## [0.11.9] — 2026-06-10

### Added
- **GAMLSS: Gaussian location-scale (`gaulss`) — the first multi-linear-
  predictor family.** `gamrs.fit_gaulss(X, y, mu_terms=…, sigma_terms=…)` fits
  `y ~ N(μ(x), σ(x)²)` with smooth μ(x) AND σ(x), jointly. Because the Gaussian
  location-scale Fisher information is block-diagonal (μ ⟂ log σ), the joint
  MLE is computed by **orthogonal alternating Fisher scoring** — an alternation
  of two single-predictor penalised weighted-Gaussian REML fits (μ reweighted
  by 1/σ²(x); log σ via the scale IRLS) — reusing the existing single-predictor
  fit stack rather than a dense block-Newton. One fit yields every quantile
  (`GaulssFit.predict_quantile`), monotone in τ so bands never cross. Native
  Rust driver (`src/fit/gaulss.rs`, exported as `gamrs::fit_gaulss`). Confronted
  with mgcv `gaulss`: recovers the same μ̂/σ̂ (RMSE ~3e-4 / ~1e-3) and matches
  OOS pinball to ~0.05%, ~70× faster than R `gaulss` at n=800
  (`tests/python/test_parity_multismooth.py::test_gaulss_joint_parity`,
  `src/fit/gaulss.rs` unit test). This is the seam for the broader GAMLSS class
  (`shash`/`gevlss`); non-orthogonal families will extend the alternation.
- **Distributional location-scale quantiles — `fit_quantile_lss`.** A new
  quantile path that models the whole conditional distribution (location μ(x)
  + scale σ(x), two Gaussian GAMs) and derives every quantile as
  `q_τ(x) = μ(x) + σ(x)·z_τ` from a single fit — the mgcv `gaulss`/`shash`
  view. Quantiles never cross (z_τ ↑ in τ, σ > 0) and τ is a predict-time
  argument (`QuantileLSSFit.predict_quantile(X, tau)`), so one fit serves all
  τ. `shape="gaussian"` (default, `z_τ = Φ⁻¹(τ)`, no scipy needed) or
  `shape="shash"` (fits skew/kurtosis on the standardised residuals via the
  existing SHASH helper; needed for skewed data). Pure Python over existing
  Gaussian fits — no Rust change. Confronted with mgcv `gaulss`: 2-D
  heteroskedastic OOS pinball within ~1% at τ ∈ {0.1, 0.5, 0.9}, zero
  crossings (`scripts/r/gen_quantile_lss_fixture.R`,
  `tests/python/test_parity_multismooth.py::test_additive_quantile_lss_parity`);
  the SHASH shape corrects the Gaussian shape's tail mis-coverage on skewed
  data (`test_quantile_shash.py::test_lss_shash_fixes_skewed_tails`).

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
- **Wheel portability: AVX-512 leak in the Linux build.** The first 0.11.8
  build tripped the SDE Haswell portability guard on an `vbroadcastsd zmm0`
  (AVX-512) instruction — the guard correctly blocked the PyPI publish (no
  broken wheel shipped). Root cause: the manylinux Rust toolchain was not
  emitting baseline x86-64 codegen. Pinned `RUSTFLAGS=-C target-cpu=x86-64`
  in the Linux release job so our Rust is deterministically baseline (and the
  sccache key can't reuse a native-arch object); the rebuilt wheel disassembles
  to zero `zmm` instructions and passes the guard. Also gated `publish-pypi` to
  release events so a candidate can be `workflow_dispatch`-tested through
  build + guard without publishing.

### Added
- **Multi-smooth (additive) Quantile/ELF.** `fit_quantile` gained a `terms=`
  argument (`CrTerm` / `TeTerm` / …) so quantile fits run on additive designs
  (`y ~ s(x0) + s(x1) + …`), not just a single smooth. The SHASH σ pilot, the
  K-fold CV σ search, and the final fit all use the additive design; σ stays a
  single family-level scale. First multi-smooth quantile parity test against
  qgam (the mgcv-family ground truth): 2-D additive OOS pinball within ±0.6%
  of qgam at τ ∈ {0.1, 0.5, 0.9}
  (`scripts/r/gen_quantile_multismooth_fixture.R`,
  `tests/python/test_parity_multismooth.py::test_additive_quantile_oos_parity`,
  `tests/quantile_smoke.rs::quantile_multismooth_additive_monotone_and_converged`).
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

[Unreleased]: https://github.com/AlekJaworski/gamrs/compare/v0.11.9...HEAD
[0.11.9]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.9
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
