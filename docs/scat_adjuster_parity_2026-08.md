# `scat` vs mgcv on the adjuster refits — open, measured 2026-08-26

**Status: OPEN, cause uncharacterised.** Not a reproduction of
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
bite first — consistent with two correct implementations landing in different optima, and equally
consistent with a defect. **The data below cannot separate those two readings.**

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

## What would settle it — none of this is captured yet

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
