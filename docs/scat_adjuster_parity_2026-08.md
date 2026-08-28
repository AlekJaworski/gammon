# `scat` vs mgcv on the adjuster refits — measured 2026-08-26, four defects fixed 2026-08-27

**Status: FIXED 2026-08-27 — a gamrs defect, root-caused to one term.** Not a reproduction of
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

## The root cause, to a line — 2026-08-27

`src/score/shape_aware/gradient.rs::compute_rho_envelope_gradient` carried a term for the
derivative of a ridge that is not in the matrix the score differentiates.

The shape-aware score reads `log|A|` and `tr(A⁻¹S_j)` off the factor the inner solver hands back.
Two inner solvers feed it:

- `OcatInner` (`src/inner/gam_fit5.rs:246-256`) builds its final-pass factor as
  `A = X'WX + Σλ_jS_j + c·max|diag(A_post_pen)|·I` with `c = 1e-5·(1 + √n_pen)`. That ridge depends
  on λ, so the ρ-gradient genuinely owes a `∂ridge/∂ρ_j · tr(A⁻¹)/2` term.
- `PirlsInner` — what scat, NegBin and Tweedie use — hands back the **unridged** factor. Its ridge
  is `1e-12·max_diag` and is applied to a *copy* used only for the β̂ solve
  (`src/inner/linalg.rs:353-364`, which factors `a` twice on purpose and returns the unridged one).

`compute_rho_envelope_gradient` is shared by both and hard-coded ocat's `c`. For every PIRLS-path
family it therefore added `½·c·λ_j·S_j[i*,i*]·tr(A⁻¹)` — a term proportional to λ — to a gradient
whose true value decays like 1/λ. Measured on `score_tests.rs::tdist_analytic_rho_grad_matches_fd`,
the error was exactly proportional to λ:

| ρ | analytic − FD, before | after |
|---|---|---|
| 0 | 1.13e-2 | 1.44e-4 |
| 4 | 5.65e-1 | 1.40e-6 |
| 8 | 3.08e1 | 6.93e-6 |
| 12 | 1.68e3 | 1.19e-5 |
| 16 | 9.18e4 | analytic decayed to −2.5e-8 |

Each +4 in ρ multiplied it by e⁴. That is the "λ-envelope derivative term that fails to cancel in
the over-penalised limit" the first pass guessed at — it does not fail to cancel, it should never
have been there.

**The fix** (`ShapeInnerBuilder::score_ridge_scale`, `src/score/shape_aware/builder.rs`): the
builder declares the ridge coefficient actually baked into the factor it returns — `0.0` by
default, ocat's `1e-5·(1 + √n_pen)` for `OcatInnerBuilder` — and the gradient differentiates that.

A second, much smaller correction went in beside it: the β-chain term in `log|H|` now prefers
`Loss::ift_trace_weight_derivs`' `dw_dmu` over `½·Dmu3` where the family supplies it. For scat the
two differ on the outlier rows, where the working weight is the μ-independent *expected* curvature
and so contributes nothing to `∂W/∂μ`.

### What it moved

| | before | after | mgcv |
|---|---|---|---|
| `parity_scat_flat_ridge` edf | 2.1316 | 2.0015 | 2.0102 |
| `parity_scat_flat_ridge` curve gap | $291.2 | $22.3 | — |
| `3d_scat_identity_n800` μ rel-err | 2.2e-3 | 5.7e-4 | — |
| `2d_scat_identity_n600` μ rel-err | 5.7e-4 | 1.08e-3 | — |
| Tweedie ρ-axis rel-err (first probe) | 2.78e-2 | 2.42e-3 | — |

The 2-D scat fixture is the one that moved the wrong way, and it is not a regression in the fit:
gamrs's ρ̂ went `[3.809, 10.012]` → `[3.799, 10.077]` against mgcv's log `sp` of `[3.736, 9.898]`,
and its total edf `9.05` → `9.03` against mgcv's `8.9798`. Its bound was retuned to 1.5e-3 and the
3-D one tightened 4× to 1e-3.

**Not fully closed.** A residual of about 1e-3 RELATIVE remains in scat's ρ-gradient at moderate ρ
(analytic +1.528723e-1 against FD +1.530165e-1 at ρ = 0). One candidate: the IFT step uses
`∂β/∂ρ_j = −λ_j·A⁻¹·S_j·β` with `A = X'WX + λS`, but on scat's outlier rows the working weight is
the *expected* curvature rather than `½·D''_η`, so `A` is not the Hessian of the penalised deviance
there. Untested.

## Verdict

**A gamrs defect, not a difference of objective.** The two fitters share the basis, the penalty, the
inner solve and the criterion — matched-parameter deviances agree to 1e-8. gamrs's own REML score is
lower at mgcv's answer than at gamrs's on both disputed terms. And the proximate cause is
identified: `g[0]` of `scat`'s assembled REML gradient disagrees with a finite difference of gamrs's
own score, by enough to flip its sign once the λ ridge goes shallow — the one axis of that gradient
nothing tested.

Traced and fixed — see "The root cause, to a line" above. The parity claim on the `scat` refits can
be restated once it is re-measured on the adjuster capture; the synthetic evidence is that the
flat-ridge regime now lands on mgcv (edf 2.0015 vs 2.0102, $22 rather than $291), and the real
`condition` / `quality` terms have NOT yet been re-measured.

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

---

## 2026-08-27, later — FOUR defects, not one, and the first fix was not the cause

The `score_ridge_scale` fix earlier in the day was real but did not move the adjuster numbers
(`garage_spaces` 577.1 → 577.1 dollars RMSE). Chasing that led to three more, each traced against
mgcv's own source rather than guessed.

**1. The IFT bracket was the wrong matrix.** `compute_rho_envelope_gradient` derived `∂β/∂ρ` by
solving against `fit.a_factor`. `irls_observed_pair` substitutes the positive *expected* curvature
wherever observed `½·D_μμ ≤ 0`, which preserves β̂ (the working response keeps
`W(z−η) = −½·D_μ` whatever the weight) but means `A` is not the Hessian of the penalised deviance on
those rows — and the implicit-function bracket must be that Hessian. PIRLS stationarity at the
converged fit is 1.4e-8, so a central FD of β̂ is a sound oracle:

| ρ | `∂β/∂ρ` vs FD, against `A` | against observed `H` |
|---|---|---|
| 0 | 2.576e-2 | **1.988e-8** |
| 4 | 2.926e-2 | 3.227e-5 (tracks stationarity 3.3e-6) |

**2. The criterion itself differed from mgcv's.** mgcv `gam.fit4.r:367` sets `w <- dd$Deta2 * .5` —
observed curvature, negatives kept (`gdi.c`'s `pls_fit1` takes them via `sqrt(|w|)` with sign
tracking) — and retries Fisher only if `X'WX + E'E` comes back indefinite. gamrs substituted
instead, and the score reads `log|H|` off that factor, so **gamrs was optimising a different
function of (λ, ν, σ²) than mgcv on any data with outlier rows.** Proof on the real order: gamrs
with observed `log|H|` evaluated at mgcv's own ρ gives **861.391445** against mgcv's recorded
**861.3914448621**; `condition` **650.132633** against **650.1326331246**. The argmin moves onto
mgcv's point too — `garage_spaces` `|ρ*−ρ_mgcv|` **1.4500 → 0.0250**. The observed penalised Hessian
is positive definite at every probe despite the negative rows, so plain Cholesky suffices.

**3. The outer loop's convergence test was inverted.** mgcv `fast-REML.r:1587-1603` sets
`converged <- TRUE`, clears it if any `|grad|` exceeds tolerance, then clears it *again* if the REML
value is still moving — re-enabling every axis, because it "can't progress" otherwise. It never
concludes convergence *from* a small score change. gamrs returned `converged: true` when
`|ΔREML| < reml_tol`, so it stopped wherever the steps went quiet, which on a flat λ ridge is
nowhere near the argmin (fixture: argmin ρ ≈ 21.92, stopping at ρ ≈ 16.77).

**2 and 3 are interlocking, and neither alone works** — `garage_spaces` curve RMSE against mgcv:

| | stop-early (old) | mgcv rule (new) |
|---|---|---|
| `log\|A\|` (shipped) | 577.1 | **806.7 — worse** |
| `log\|H_obs\|` | 182.7 | **20.8** |

Optimising harder only helps once the function is right. That is independent evidence for 2.

**4. edf was reported from a mismatched pair.** `compute_edf` forms `tr(A⁻¹·X'WX)` and was handed
`working_weights` while `A⁻¹` came from `a_factor` — different vectors once the factor is observed.
`GaussianInnerFit` now carries `a_weights`, the weights its factor was actually built from. Reporting
only; the fit never changed, which is why the curves did not move with it.

### Where it stands on the real order

- published secant worst `|Δ|` **1.288% → 0.391%**. `condition` −1.288% → **−0.001%**, `quality`
  0.470% → 0.053%, `garage_spaces` 0.156% → −0.048%, `lot_sqft` −0.116% → −0.006%. Nine of ten
  terms inside 0.09%.
- worst curve RMSE **600.0 → 91.3**; `garage_spaces` **577.1 → 20.8**.
- edf within 0.055 of mgcv on every term, most within 0.006.

### Still open

- **`sale_date`** is the lone secant outlier at 0.391% (was 0.495% — barely moved by any of this)
  and the worst curve at 91.3 RMSE. `k = 25`, edf 9.2: a far richer basis than the 2-edf terms, so
  probably a different mechanism.
- **Outer-loop start-sensitivity.** `tf9963_garage_spaces_scat_lands_on_mgcv_optimum` breaches its
  1e-3 bound (1.064e-3) with the new criterion, and it is the *start* that decides, not the
  criterion: `init_sigma2` is standardized as `init_sigma2/s²`, so the test's
  `tdist_identity(5.0, y_var*0.1)` (σ²_std ≈ 0.0999) lands at 9.456e-4 while `Gam`'s default
  (σ²_std ≈ 5.5e-10) lands at **1.095e-4**. The sensible-looking start does worse. This is the
  observation in "Ruled out" above — mgcv lands identically from 11 starts, gamrs does not — now
  the live defect. **Not to be resolved by moving the bound.**

### Two measurement traps, recorded so nobody re-walks them

- The scat score carries PIRLS-convergence noise of order 1e-2 in ρ (`dev_rel_tol` is 1e-9 against a
  score of ~7000). An FD profile of the score near a landing point is not trustworthy below that.
- ν is `MIN_DF + exp(θ)` with `MIN_DF = 3`, so a probe written as `ln(5)` means **ν = 6, not 5**.
  Hardcoding the wrong ν in a diagnostic makes a correct fix look only partly effective.

---

## 2026-08-28 — SIX defects, the criterion proven, and the switch deleted

Two adversarial critic passes over the 2026-08-27 work found two more defects and
two faults in the work itself. Final state.

### The criterion is mgcv's — proven, not inferred

`parity_scat_tf9963::scat_criterion_matches_mgcv_on_its_own_sp_ladder` evaluates
gamrs's score on mgcv's own `sp_ladder` — a pure λ-slice at fixed (ν, σ), since
the generator pins `theta` across every rung — and reproduces mgcv's REML at all
seven rungs to **3e-6 absolute** on values of ~7325 (4e-10 relative). The
working-weight `log|A|` misses the same ladder by 2.8e-2 on the steep side and
4.9e-3 on the shallow side. The test FAILS on the old criterion, which is what
forced the flip. **`GAMRS_OBSERVED_LOG_DET_H` is gone**; one seam
(`family_observed_score_weights`) decides what `fit.a_factor` is, and every
consumer reads that factor.

### The two further defects

**5. `max_half` — the actual root cause of the residual early stop.** The line
search carried an adaptive cap ported from mgcv_rust whose `stalled ⇒ 1 halving`
branch fired when `|ΔREML|/|v| < 1e-4`, i.e. *on a flat λ ridge, where more
halving is needed*. mgcv uses `maxHalf = 30` unconditionally (`gam.fit3.r:1230`,
`mgcv.r:2212`). Removing it took the low-σ² start from 9.133e-4 above the optimum
to −2.481e-8, and `tf9963` from failing at 1.064e-3 to 1.335e-5. It also
invalidated an earlier `step_min` sweep: `max_half = 1` exits the loop before
`step_min` can bind, so that null result meant nothing.

**6. edf and vcov come from the FISHER pair.** mgcv keeps two factorisations:
`gdi.c:2260-2290` re-QRs `[sqrt(wf)X; E]` *after* the observed-weight work,
because edf and `rV` must use the expected curvature. `wf = pmax(0, EDeta2*.5)`
(`gam.fit4.r:563`) and for scat `efam.r:1327-1329` makes `EDmu2` the constant
`2(ν+1)/((ν+3)σ²)` — one scalar on every row. Verified in R with mgcv's own
design and penalty:

| | vs mgcv |
|---|---|
| Fisher `tr[(c·X'X+λS)⁻¹c·X'X]` | edf to **4.5e-14** |
| observed pair | edf off by 7.2e-3 |
| Fisher `A_f⁻¹` | `Vp` to **7.8e-15** |
| observed `A_obs⁻¹` | `Vp` off by 2.2e-2 |

And σ² must NOT be applied on top: `c` already contains `1/σ²`, which is why mgcv
reports `scale = 1` for scat. gamrs was double-counting it — reported vcov sat
exactly `(1 − σ²) = 31.4%` from mgcv's. Every standard error, CI and
`predict(se=TRUE)` on a scat fit was affected.

### Where the real order stands

Common (standardized) problem — gamrs vs mgcv 1.9.4 run directly on each term:

| term | curve rel | $ max |
|---|---|---|
| `sale_date` | 3.667e-07 | 0.2 |
| `condition` | 3.381e-05 | 17.0 |
| `garage_spaces` | 3.476e-05 | 20.6 |
| worst (`quality`) | 3.809e-05 | 23.4 |

Seven of ten terms at 1e-6 or better. Published secant vs mgcv-raw: worst |Δ|
**1.288% → 0.391%**; `condition` **−1.288% → +0.040%**, `garage_spaces`
+0.156% → −0.026%, `quality` 0.470% → 0.053%. Nine of ten inside 0.09%.

**`sale_date` is not an open defect.** It is the most exact term on a shared
problem AND the worst against mgcv-raw, because it has the largest
raw-vs-standardized sensitivity of the ten: mgcv's OWN ν moves 6.4020 → 6.6465
(3.8%). Each term's raw residual tracks that sensitivity — mgcv shifts 0.63% /
0.79% / 3.8% on condition / garage_spaces / sale_date and the raw RMSEs come out
15.4 / 34.5 / 91.3 in the same order. What remains is the standardization
convention, a product decision.

### Faults in the 2026-08-27 work, for the record

- The flagship IFT test was **vacuous**: it asserted on a local rebuild of the
  formula, so reverting the production fix left it green. Proven by experiment.
  Now asserts on the assembled gradient and fails at 9.480e-4 without the fix.
- The tolerance citations were mis-sourced from `fast.REML.fit` (the Gaussian
  fREML path) instead of `gam.fit3.r`'s `newton`, which scat actually uses
  (`conv.tol = 1e-6` ⇒ gradient test `5e-6·score.scale`, axis filter
  `1e-7·score.scale`). The `dim_tol` change made on that basis was reverted.
- `grad_tol` is `sqrt(eps)`, deliberately ~336× tighter than mgcv's rule. That is
  an empirical choice, NOT parity — do not cite mgcv for it.

### Known-loose bounds — tighten before trusting

The fixture bounds were calibrated against the old criterion and now have
10-28000× headroom: `2d_scat` measures 5.502e-6 against 1.5e-3, `parity_scat`
measures 1.1e-7 and 3.6e-8 against 1e-3, `tf9963`'s edf bracket is `(2.2..2.6)`
where the gap is 0.006. They would pass through a large regression.

`parity_scat_flat_ridge` now takes 36 outer iterations (was 22) — the cost of
removing the halving cap.
