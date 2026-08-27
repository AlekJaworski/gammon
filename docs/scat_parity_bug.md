# `scat` family parity — wrong-direction robustness bug (RESOLVED v0.11.2)

**Status: RESOLVED** — shipped in `release: 0.11.2 — scat mgcv parity
(robustness-direction fix)`. Pinned by a committable, synthetic regression
test: `tests/parity_scat.rs::scat_downweights_high_outliers_synthetic`.

This note is self-contained; the original real-data reproduction used a
proprietary housing fixture that is **not** in this repo (gitignored under
`data/`).

## The bug

`scat` (scaled-t) is meant to be outlier-robust: large-residual points are
down-weighted, so the fitted mean is pulled *toward the bulk* — below an
ordinary Gaussian/OLS fit in the presence of high-side outliers. gamrs's
`scat` did the opposite: it inflated the fitted mean *above* even the
Gaussian fit, breaking the README's "mgcv R-parity on µ across all ten
families" claim. On a real housing time-curve it produced +13.5%/yr
recent-year appreciation where mgcv's `scat` (and the empirical truth) gave
~+2.8%, biasing the curve the wrong way. (`scat` and `t-dist` are aliases in
gamrs — identical output — so this was not a name mix-up.)

## Root cause — inverted IRLS fallback

`scat`'s PIRLS down-weights outliers via the *observed* curvature
`W = ½·D''_η`, which goes **negative** for `|r| > √(νσ²)`.
`TDist::irls_observed_pair` replaced those outlier rows with the *expected*
(Fisher) curvature **and** the Fisher response `z = η + r`. That pairing
gives working response `W·(z − η) = w_exp·r`, so the IRLS fixed point became
`λSβ = X'·w_exp·r` — which pulls outliers *toward* y (Gaussian-like) instead
of down-weighting them. Net effect: the fit landed *above* Gaussian.

Verified against Simon Wood's mgcv R source (`R/gam.fit4.r:368-416`,
`R/efam.r:1248` `scat`), not against mgcv_rust (whose `fit_pirls_tdist` uses
an EM weight that is *not* what real mgcv does).

## The fix (`src/family/tdist.rs`)

- **`irls_observed_pair`**: outlier rows now use the expected weight
  `w_exp = (ν+1)/((ν+3)σ²)` (a factor-2 correction vs the old
  `(ν+1)/(2(ν+3)σ²)`, matching mgcv `EDmu2/2`) paired with the
  Fisher-consistent response `z = η − ½·D'_η/w_exp`, so `W·(z − η) = −½·D'_η`
  for every row and the fixed point is the true penalised-deviance
  stationary point `λSβ = X'·(−½·D'_η)` — exactly mgcv.
- **`min.df` 2 → 3** (`TDist::MIN_DF`): mgcv `scat(min.df = 3)`
  (`efam.r:1248`). The `ν − min.df` factor permeates every ν-derivative, so
  this shifts the ν-optimum, not just the floor.
- **Internal standardization**: scat's observed weights are `W ~ 1/σ²`, so
  raw responses (`var(y) ~ 1e11`) underflow `X'WX`. The fit core
  standardizes the response and rescales the identity-link fit back
  (`python.rs::rescale_scat_fit`); scat is location-scale equivariant, so
  this only fixes conditioning. Exercised raw-scale via the Rust API by
  `tests/parity_scat.rs::scat_raw_scale_via_rust_api`.
- The matching `Loss::ift_trace_weight_derivs` hook keeps the analytic REML
  shape-gradient consistent
  (`tests/score_tests.rs::tdist_analytic_shape_grad_matches_fd`).

## Result

On the original CR fixture, gamrs matches mgcv R (`gam(family=scat())`, same
per-feature CR basis) to **0.05 % on the fitted level**, with `ν = 3.0` in
both (mgcv itself floors at `min.df = 3` on that data). The directional
contract — scat pulls *below* Gaussian under positive outliers — is now
locked in CI by the synthetic regression test, so it cannot silently regress
again.

## Remaining caveats

- gamrs reports `converged = False` at the `ν` boundary on some fixtures —
  cosmetic; the fitted `(ν, σ², curve)` match mgcv.
- Multi-smooth scat reference parity tests against mgcv are still pending
  (single-smooth + the directional/raw-scale guards are in place).
- **A separate, still-open scat disagreement was measured on 2026-08-26**, on the
  single-term refits the TrueFootage adjuster publishes from: curves up to $1,349
  apart on one term where the same two fitters agree to $1.60 on the joint
  Gaussian model. Not this bug (the direction fix is in and its regression test
  passes) and not the 0.13.0 rank fix (A/B: max |Δ| = 0.0). See
  [`scat_adjuster_parity_2026-08.md`](./scat_adjuster_parity_2026-08.md).
