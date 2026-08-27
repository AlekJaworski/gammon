# `scat` vs mgcv on the adjuster refits — open, measured 2026-08-26

**Status: OPEN, cause characterised 2026-08-27 — a gamrs defect.** Not a reproduction of
[`scat_parity_bug.md`](./scat_parity_bug.md) (resolved in 0.11.2, direction-of-robustness);
that fix is in and its regression test still passes. This is a *magnitude* disagreement on
single-smooth `scat` refits, and it lands squarely on that note's last caveat:

> Multi-smooth scat reference parity tests against mgcv are still pending
> (single-smooth + the directional/raw-scale guards are in place).

## Why it matters

Every dollar figure TrueFootage's adjustment product publishes is a two-point secant of a
**single-term `scat` GAM fitted on partial residuals** — not of the joint Gaussian model. The
call site is `api_custom_clusters.adjustments_per_feature`, which passes
`sp_res.residual_model._model` into `get_slope`, overriding the joint model. So `scat` is not a
side family here; it is the only family the product's published numbers come from.

## The measurement

One order, 620 rows, 11 predictors, REML on both sides, gamrs 0.13.1 against mgcv 1.9.4 under
R 4.6.0. Same `(X, y)` in one process. Per feature, the fitted curve on a 200-point grid.

**Joint Gaussian model — the two agree to within rounding:**

| | value |
|---|---|
| curve RMSE across all terms | $0.03 – $4.20 |
| response range | $358,256 |
| every term's edf | agrees within 0.0017 (total 19.2195 vs 19.2163) |
| in-sample RMSE | 40,919.30 vs 40,919.36 |
| full-range secants | within 0.0044% |

**Single-term `scat` refits on the same data — 10× to 380× worse:**

| feature | joint curve RMSE | scat curve RMSE | scat max abs diff | scat RMSE / range | published secant Δ |
|---|---|---|---|---|---|
| `gla` | $0.3 | $20.4 | $58.0 | 0.019% | −0.06% |
| `lot_sqft` | $0.4 | $30.2 | $47.2 | 0.083% | −0.12% |
| `bedrooms` | $0.1 | $12.1 | $15.6 | 0.149% | −0.09% |
| `bathrooms` | $0.1 | $13.7 | $32.8 | 0.059% | +0.02% |
| `garage_spaces` | $0.5 | $62.7 | $79.1 | 0.377% | +0.16% |
| `stories` | $0.0 | $10.9 | $17.0 | 0.025% | −0.03% |
| `pool` | $0.2 | $26.6 | $40.1 | 0.058% | +0.07% |
| **`condition`** | $1.6 | **$611.3** | **$1,348.9** | **0.656%** | **−1.29%** |
| `quality` | $4.2 | $146.1 | $333.3 | 0.277% | +0.47% |

The secant is not amplifying a negligible difference — the `scat` curves themselves disagree, by
up to $1,349 on `condition`. The secant merely reports it. (Confirmed the secant is faithful:
recomputing it from the committed `scat` curves reproduces the published `price_per_unit` to
**4.6e-09** relative.)

The two worst terms are the two with the most distinct values against their `k`: `condition` 19
distinct against k=12, `quality` 16 against k=12. That is where a harder likelihood surface would
bite first. Read on 2026-08-26 as equally consistent with two correct implementations landing in
different optima and with a defect; **the 2026-08-27 measurements below separate the two — it is a
defect.** The distinguishing feature turned out not to be `n_unique` against `k` but the shape of
the λ ridge: on both worst terms the REML score has no interior optimum in λ at all.

## Ruled out

- **The 0.13.0 `scat` fix.** `TDist::score_rank_adjustment() == -1` (commit `1d1a8e9`) landed
  between the two candidate capture builds, so it was the obvious suspect. A/B at pinned 0.13.0
  and 0.13.1, same pickled call object so frame and spans are bit-identical:
  **max |Δ| = 0.0 on every feature, both arms.** It is not this.
- **The `k` cap.** Both captures pass `k_cap_offset=0`, `min_k=3`, and the cap can only bind where
  `n_unique ≤ k`. `condition` (19 distinct, k=12) and `quality` (16, k=12) cannot be capped, and
  they are the two worst terms.
- **The estimator word.** `gamrs/fREML` comes back bit-identical to `gamrs/REML` (gamrs has no
  `bam.fit`; fREML resolves to its own REML optimum), and the arm is correctly labelled REML.
- **Prediction grid / interpolation.** Both fitters go through the same 1000-point + 2-anchor
  predictor table.
- **A capture artefact.** Slopes, `scat` curves and joint curves now come from one process, one
  day, one `(X, y)`, with a `versions` block asked of the running toolchain.

## What would settle it — asked 2026-08-26, all three answered below

1. **The `scat` refits' own diagnostics, per arm** — estimated `ν`, `σ²`, edf, and convergence
   flag per feature. Only the *joint* model's `sp`/`edf` were captured. If gamrs's `scat` is
   landing on a different `ν`, that is the mechanism and it is a small capture to add. Start
   here; it is the cheapest and the most likely to be decisive.
2. **Multi-start on the same `scat` problem, in mgcv.** If mgcv itself lands in different places
   from different starting values, this is a flat or multimodal basin and neither implementation
   is wrong. Note gamrs has **no optimiser warm start** where mgcv does — a plausible route to a
   different optimum on exactly this kind of surface.
3. **Whether the `k` cap and the parametric auto-promotion apply inside the per-feature `scat`
   refits.** On the joint fit gamrs emits structural warnings mgcv does not — `bathrooms` capped
   6→3, `garage_spaces` 6→5, `bedrooms` auto-promoted to a parametric unpenalised term. Harmless
   there (the joint curves still agree to a dollar), but nobody has checked what happens per
   feature under `scat`.

## The cause — measured 2026-08-27

**gamrs's analytic REML gradient with respect to `ρ = log λ` is wrong for `scat`, and on a shallow
λ ridge it comes back with the wrong sign.** The outer Newton then steps the wrong way, step-halving
finds no decrease, and `outer.rs` returns the point it is standing on. That leaves λ too small, edf
above mgcv's, and the fitted curve hundreds of dollars off — with `ν` and `σ²` in agreement
throughout, because those are the two axes that were tested.

The measurement, on the synthetic fixture, at the exact θ the outer Newton stopped at
(`gamrs::fit`, traced out of `src/outer.rs`; the score value agrees with
`evaluate_reml_at_scat` to 4e-8, so it is the same score function):

| | ∂score/∂ρ | ∂score/∂log σ² | ∂score/∂log(ν−3) |
|---|---|---|---|
| gamrs analytic | **+0.027467** | +0.0853 | −0.0113 |
| central FD of gamrs's own score | **−0.035481** | −0.3449 | −0.0113 |

The FD is stable across `h` = 1e-2, 1e-3, 1e-4. Along the Newton path the error is additive and
roughly flat in absolute terms while the true gradient shrinks, so it is harmless early and decisive
late:

| iter | ρ | analytic ∂/∂ρ | FD ∂/∂ρ |
|---|---|---|---|
| 0 | 0.63 | −4.9066 | −4.9055 |
| 2 | 6.70 | −1.5046 | −1.5047 |
| 3 | 11.28 | −0.0835 | −0.1060 |
| 4 | 12.96 | **+0.0993** | **−0.0174** |
| 6 | 12.30 | **+0.0275** | **−0.0355** |

The Hessian is finite-differenced *on this same analytic gradient*
(`outer.rs`: "`value_grad_hess()` runs PIRLS + analytic-grad + FD-on-grad Hessian"), so a wrong
`g[0]` poisons `H[0][0]` too — the trace shows `H_ρρ = −0.0576`, negative curvature at a point where
the surface is monotone.

**Why nothing caught it.** `tests/score_tests.rs::tdist_analytic_shape_grad_matches_fd` checks
`i in 1..3` — the two shape axes only — and says of the third: *"g[0] is the λ-envelope gradient
(verified separately)"*. For `scat` it was not. `tests/score_tests.rs::tdist_analytic_rho_grad_matches_fd`
now probes it (`#[ignore]`d, it fails); the ρ axis fails that same test's own 1e-3 bar even at ρ = 0.

### Where it bites, and where it does not

A ~0.06 gradient error is nothing against a strongly curved surface and everything against a flat
one. That is exactly the split observed:

| feature | λ ridge | gamrs edf | mgcv edf | curve gap |
|---|---|---|---|---|
| `sale_date` | sharp interior optimum | 9.1628 | 9.0921 | $98 |
| `gla` | interior optimum | 3.6908 | 3.6740 | $43 |
| `lot_sqft` | interior optimum | 3.1793 | 3.1897 | $171 |
| `bathrooms` | interior optimum | 2.6113 | 2.6162 | $38 |
| `garage_spaces` | shallow | 2.1540 | 2.2132 | $683 |
| **`condition`** | **no interior optimum** | **2.2709** | **2.0031** | **$1,549** |
| **`quality`** | **no interior optimum** | **2.8746** | **2.0005** | **$661** |

On `condition` and `quality` the signal is near-linear in a coarse ordinal predictor, so REML has no
interior optimum in λ at all: the score descends monotonically toward the λ→∞ null-space limit,
where the CR smooth collapses to a straight line (edf → 2). mgcv walks that ridge to `sp` ≈ 9.3e6 /
1.0e9. gamrs stops at λ ≈ 9.0e4 / 2.8e5.

### Ruled out on the way

- **A different objective.** Fed the *same* `(λ, ν, σ²)`, gamrs's deviance reproduces mgcv's to
  1e-8 – 1e-6 on every feature. The CR basis, the identifiability constraint, the penalty
  normalisation and the inner PIRLS are the same; gamrs's reported λ is in mgcv's `sp` units.
  Every difference is in where the outer optimiser stops.
- **A different ν.** gamrs's ν is within 0.03–0.28 of mgcv's on every feature, and *closer to
  mgcv's own multi-start landing* than mgcv's default-start fit is (`condition`: gamrs 6.4646,
  mgcv multi-start 6.4796, mgcv default 6.4419). ν is not the mechanism. `MIN_DF = 3` and the
  `ν − min.df` reparameterisation still match at these values.
- **A flat basin with two legitimate answers.** mgcv lands on an identical ν and edf to six decimal
  places from all 11 starting `(ν, σ)` pairs on every feature that fits — zero spread. And
  *gamrs's own criterion* is better at mgcv's point than at gamrs's: −0.031 on `condition`,
  −0.559 on `quality`. It is not two optima; it is one, and gamrs is short of it.
- **The REML-change convergence escape** (`outer.rs:283-296`, which returns `converged: true` on
  `|ΔREML|/|REML| < reml_tol` with no gradient check — a real divergence from mgcv's `newton()`,
  which clears `converged` whenever any `|grad| > score.scale·conv.tol`). A/B with `reml_tol = 0`:
  `condition` and `quality` come back **bit-identical**. Not this.
- **The `k` cap and parametric auto-promotion inside the per-feature refits.** They do fire —
  `bathrooms` 6→3, `garage_spaces` 6→5, and `bedrooms`/`pool`/`stories` (2 distinct values each)
  are auto-promoted to unpenalised parametric terms where mgcv keeps a 2-column smooth. But
  `condition` (k=12) and `quality` (k=12) are untouched, and they are the two worst terms.
- **The input.** The mgcv and gamrs arms fit `scat` on partial residuals from their own joint
  models, which differ by up to $8.60 on a ~$300k range. Refitting both fitters on *both* arms'
  residuals moves the curve gap by less than $2. The disagreement is the fitter, not the input.

### Cost, in REML units

The reason this is invisible to a convergence test and loud in dollars: on `condition`, gamrs's own
score falls by **0.030** between its landing point and the λ→∞ limit, and the curve moves **$1,549**.
mgcv's criterion agrees the surface is that flat — its score at gamrs's λ and at its own differ by
0.022. A criterion flat to a hundredth of a REML unit spans a thousand-plus dollars of published
adjustment.

## Code paths

| what | where |
|---|---|
| the call site that makes `scat` the published family | `tf_adjustments/engine/api_custom_clusters.py:119` — `adjustments_per_feature` passes `sp_res.residual_model._model` into `get_slope`, overriding the joint model |
| the single-term `scat` fit itself | `tf_adjustments/engine/service_location_time_adjustments.py:230-241` — `FeatureAdjuster.two_step_adjustments` builds a one-predictor `Gam(family="t-dist")` and fits it on `sp.residuals_x[non_nan] / sp.residuals_y[non_nan]` |
| gamrs fitting entry point | `src/fit/family_impls.rs` — `FamilyFitWithSolver<IdentityLink, TVariance, S> for TDist::fit_from_prep_canonical`, which standardizes the response (`scat_response_scale`), builds `θ₀ = [ρ_init, log σ̃², log(ν−3)]` and calls `fit_shape_aware` |
| the outer optimiser | `src/outer.rs:266-599` — damped subset Newton; gradient test at `:270-282`, REML-change escape at `:283-296`, step-halving at `:411-437`, the `!accepted` fallback and its four convergence clauses at `:459-599` |
| the wrong gradient | `src/score/shape_aware/gradient.rs` — the assembled REML gradient; `g[0]` (the ρ axis) is the component that fails |
| mgcv's side | `mgcv` R `gam.fit5` / `newton()` in `R/mgcv.r`; `scat` in `R/efam.r:1248` |

## Verdict

**A gamrs defect, not a difference of objective.** The two fitters share the basis, the penalty, the
inner solve and the criterion — matched-parameter deviances agree to 1e-8. gamrs's own REML score is
lower at mgcv's answer than at gamrs's on both disputed terms. And the proximate cause is
identified: `g[0]` of `scat`'s assembled REML gradient disagrees with a finite difference of gamrs's
own score, by enough to flip its sign once the λ ridge goes shallow — the one axis of that gradient
nothing tested.

Not yet established: *why* `g[0]` is wrong. The error grows with ρ while the true gradient decays,
which points at a term in the λ-envelope derivative that fails to cancel in the over-penalised
limit, but that has not been traced to a line. Fixing it is a separate change and the parity claim
should stay withdrawn until it lands: **on the joint Gaussian model gamrs and mgcv are the same fit;
on the `scat` refits the published adjustments come from, they are not.**

## Reproducing

The fixture is proprietary and not in this repo (same constraint as `scat_parity_bug.md`). The
capture harness and the committed measurements live in `/home/alex/gitlab/2025/gamrs_evidence`:

- `tools/build_baselines.py` — builds the frame from a recorded `/v2` request and runs the
  engine's vendored `adjustments_for_cc_all` once per arm. Contains the 0.13.0/0.13.1 A/B.
- `data/scat_curves_<order>.json` — the per-feature `scat` residual model, 200 points on the
  engine's own span, in raw feature units, with a `secant` field. **This is the object to compare
  against mgcv.**
- `data/curves_<order>.json` — the joint Gaussian partial effects, for contrast.
- `data/adjuster_slopes_<order>.json` — published slopes per arm, plus the resolved
  `slope_spans`, the per-arm warnings, and `gamrs_build_ab`.

Its inputs (the recorded request, the property-api population, `r_fitting`, the vendored
`tf_adjustments`) live under `~/trufutaz/tf9963-gamrs-parity/`; paths are constants at the top of
`build_baselines.py`.

## What can and cannot be claimed meanwhile

- **Can:** on the joint Gaussian model gamrs and mgcv are the same fit, on one clean run.
- **Can:** held-out accuracy is a tie — three K-fold splits, +0.001% to +0.004% for gamrs, which
  is nothing on a ~$42,900 RMSE.
- **Cannot:** that the published adjustments agree to the same standard. They come from `scat`,
  and on `scat` the gap is up to 1.29% with the cause uncharacterised.
