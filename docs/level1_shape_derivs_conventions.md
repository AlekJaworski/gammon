# `Level1ShapeDerivs::dmu3` — convention reconciliation

This note reconciles two candidate definitions for the `Dmu3` slot of the
`Level1ShapeDerivs` payload (`src/traits.rs:55`) consumed by the shape-aware
envelope score's Tk·KK' β-chain term (`src/score/shape_aware/gradient.rs:147`)
and the IFT analytic θ-gradient (`gradient.rs:340`).

## What the consumer actually computes

Both consumers compute `tr(H⁻¹ · ∂(X'WX)/∂s)` for some scalar `s` (either a
ρ_j or a shape param θ_k), via

```
tr(H⁻¹ · ∂(X'WX)/∂s) = Σ_i h_diag_i · (∂W_i/∂s)
                     = Σ_i h_diag_i · (∂W_i/∂η_i) · ∂η_i/∂s        (★)
```

where `h_diag_i = (X H⁻¹ X')_ii`. The code at `gradient.rs:147` and
`gradient.rs:340` writes this as

```rust
// gradient.rs:147 (Tk·KK')
s += 0.5 * level1.dmu3[i] * eta1_j[i] * h_diag[i];

// gradient.rs:340 (IFT)
let s_ki = 0.5 * (level1.dmu2th[[i, k]] + level1.dmu3[i] * x_db_i);
trace_term += s_ki * h_diag[i];
```

i.e. it assumes `½ · level1.dmu3[i] = ∂W_i/∂η_i`. So the consumer's expected
contract is **`Dmu3 := 2 · ∂W/∂η`**, evaluated at the converged η.

## What `Dmu3 = ∂³D/∂μ³` actually gives (ocat)

Ocat uses its own inner solver (`OcatInner`, `src/inner/gam_fit5.rs:118`) with
**Newton observed-info working weight** and **identity link**:

```
W_i = ½ · Dmu2_i           (identity link ⇒ μ ≡ η)
∂W_i/∂η = ½ · ∂Dmu2_i/∂μ = ½ · Dmu3_i           where  Dmu3 := ∂³D/∂μ³
⇒ 2 · ∂W_i/∂η = Dmu3_i.
```

So for ocat the two conventions **coincide**, and `ocat_dd_level1` filling
`Dmu3 = ∂³D/∂μ³` is correct. tdist/scat does the same thing (`tdist.rs:182`
uses `∂³D/∂μ³`, identity link, and matching Newton weight in its inner —
the parity tests on scat confirmed this works).

## Where the conventions diverge

Standard `PirlsInner` (used by NegBin and Tweedie) uses the **Fisher-scoring
working weight**:

```
W_i = 1 / [V(μ_i) · g'(μ_i)²]              (src/inner/pirls.rs:147)
```

This is generally NOT equal to `½ · Dmu2`, even modulo a constant — and
critically its η-derivative is NOT `½ · Dmu3`. The link factor `g'(μ)²` and
the variance function `V(μ)` both contribute extra chain terms.

### NegBin under standard PIRLS

Log link: `g'(μ) = 1/μ`, `∂μ/∂η = μ`. Variance `V(μ) = μ + μ²/θ`.

```
W   = μ² / (μ + μ²/θ) = θμ² / (μ + θ)
∂W/∂μ = θμ(μ + 2θ) / (μ + θ)²
∂W/∂η = μ · ∂W/∂μ = θμ²(μ + 2θ) / (μ + θ)²
⇒ 2·∂W/∂η = 2θμ²(μ + 2θ) / (μ + θ)²            ← convention CONSUMER expects
```

For comparison, the stash's `negbin_dd_level1` fills

```
Dmu3 = ∂³D/∂μ³ = -4y/μ³ + 4(θ + y)/(μ + θ)³    ← what the stash STORES
```

These two quantities **differ** in general:

- `2·∂W/∂η` is purely μ- and θ-dependent (no `y`).
- `∂³D/∂μ³` carries `y`-dependence.
- At converged β the working residual `y − μ` has zero weighted sum, but
  pointwise `y_i − μ_i ≠ 0`, so the two arrays disagree row-by-row and
  the Tk·KK' contribution at `gradient.rs:147` will be systematically off.

### Tweedie under standard PIRLS

Log link, `V(μ) = μ^p`, so

```
W   = μ² / μ^p = μ^{2−p}
∂W/∂μ = (2−p) · μ^{1−p}
∂W/∂η = μ · (2−p) · μ^{1−p} = (2−p) · μ^{2−p} = (2−p) · W
⇒ 2·∂W/∂η = 2(2−p) · μ^{2−p}                    ← convention CONSUMER expects
```

vs.

```
Dmu3 = ∂³D/∂μ³ = 2 · [p(p−1) · μ − p(p+1) · y] / μ^{p+2}        (from d2_loss_dmu)
```

Again — different functions, different signs, different `y` dependence.

## Verification of the stash's `∂³D/∂μ³` arithmetic

Independently of whether `∂³D/∂μ³` is the *right* quantity, the stash's
algebra is internally consistent — useful if a future refactor switches
the consumer to genuinely consume `∂³D/∂μ³`.

NegBin deviance (mgcv, y > 0):

```
D(y, μ; θ) = 2 [y log(y/μ) − (y + θ) log((y + θ)/(μ + θ))].
```

Differentiate in μ:

```
∂D/∂μ   = −2y/μ + 2(θ + y)/(μ + θ)          ← matches stash's claim ✓
∂²D/∂μ² =  2y/μ² − 2(θ + y)/(μ + θ)²
∂³D/∂μ³ = −4y/μ³ + 4(θ + y)/(μ + θ)³        ← stash's Dmu3 ✓
```

Equivalence of `∂D/∂μ` with existing `d_loss_dmu = 2θ(μ−y)/[μ(μ+θ)]`:

```
−2y/μ + 2(θ+y)/(μ+θ) = [−2y(μ+θ) + 2(θ+y)μ] / [μ(μ+θ)]
                     = [−2yμ − 2yθ + 2θμ + 2yμ] / [μ(μ+θ)]
                     = 2θ(μ − y) / [μ(μ+θ)]                ✓
```

So the **arithmetic for Dmu3 in the stash is correct**; it's just labelling
the wrong slot.

θ-derivatives (with `α = log θ`, so `∂/∂α = θ · ∂/∂θ`):

| Quantity     | Derivation                                                                                   | Stash formula                                      | Match |
|--------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------|-------|
| `Dth`        | `θ · ∂D/∂θ`, `∂D/∂θ = −2 log((y+θ)/(μ+θ)) − 2(μ−y)/(μ+θ)`                                     | `θ · [−2 log((y+θ)/(μ+θ)) − 2(μ−y)/(μ+θ)]`         | ✓     |
| `Dmuth`      | `θ · ∂(∂D/∂μ)/∂θ = θ · 2(μ−y)/(μ+θ)²`                                                          | `2θ(μ−y)/(μ+θ)²`                                   | ✓     |
| `Dmu2th`     | `θ · ∂(∂²D/∂μ²)/∂θ = θ · (−2)·[(μ+θ)−2(θ+y)]/(μ+θ)³ = 2θ(2y+θ−μ)/(μ+θ)³`                       | `2θ(2y + θ − μ)/(μ+θ)³`                            | ✓     |

For y = 0 the deviance reduces to `D = 2θ log((μ+θ)/θ)`. The stash branches
the `log_ratio` calculation in `Dth` to avoid `log(0)`; the limit
substitution back into the general formula gives `Dth = 2θ log((μ+θ)/θ) −
2θμ/(μ+θ)`, matching the stash's y=0 branch. (`Dmu3`, `Dmuth`, `Dmu2th` are
all polynomial in y so the y=0 limit is just substitution — no extra
branching needed.)

**Conclusion on stash arithmetic**: Dth/Dmuth/Dmu2th are correct as
`∂(∂ᵏD/∂μᵏ)/∂(log θ)`. Dmu3 is correct as `∂³D/∂μ³`. But the consumer
expects `2·∂W/∂η` in the Dmu3 slot.

## Which convention does `Level1ShapeDerivs` actually expect?

**The consumer expects `2·∂W/∂η`**, where W is the per-row working weight
that the GaussianInnerFit reports back in `fit.working_weights` and uses to
build `X'WX`. This is what (★) demands.

The doc-comment at `traits.rs:50` says `dmu3: ∂³D/∂μ³`. That doc-comment is
**accidentally correct only for ocat/tdist** because their inner solvers use
`W = ½·Dmu2` and identity link, so `2·∂W/∂η = Dmu3`. The doc is silent on
the general (link, weighting) case — and the existing IFT consumer
mathematically wires `Dmu3` into the `(∂W/∂η)·η₁` slot, not the
`∂(∂²D/∂μ²)/∂μ` slot.

This is the root of the "convention mismatch" called out in the 0.3.1
checkpoint: an earlier port attempt filled `Dmu3 = 2·∂W/∂η` (math-correct
for the consumer); the stash's NegBin port fills `Dmu3 = ∂³D/∂μ³`
(doc-correct, math-wrong under log link / Fisher weights).

## Consequences for NegBin and Tweedie

Plugging `∂³D/∂μ³` into a slot the math wants to be `2·∂W/∂η` produces:

1. **Wrong sign / magnitude on the Tk·KK' β-chain term in the ρ-gradient**
   (`gradient.rs:147`). Concretely: the term has `y`-dependence that doesn't
   belong; at large `y_i`, `Dmu3 ≈ 4y/(μ+θ)³` whereas `2·∂W/∂η` is
   `y`-free and ≈ `2θμ²(μ+2θ)/(μ+θ)²`. For the typical NegBin signal
   `y/μ ~ O(1)`, the magnitudes are within an order of magnitude but the
   sign of the `y/μ³` piece flips contributions for rows with small `μ_i`
   and large `y_i`.
2. **Wrong analytic θ-gradient** (`gradient.rs:340`). The
   `Dmu3 · x_db_i` term is supposed to capture how β shifts the working
   weights through the link; with the wrong Dmu3, the shift estimate is
   off and the IFT gradient won't match the FD-on-score reference.

If the analytic θ-grad already disagrees with the FD reference at the
2-D NB fixture (which the diagnostic harness can check), it's almost
certainly the Dmu3 slot. Symptoms: the Dth and `tr(H⁻¹·∂H/∂θ)` decomposed
pieces of `g[n_terms+k]` agree component-wise except for the Dmu3-bearing
contribution.

## Recommendation

Two paths forward; both are deferrable past the 0.4.x line.

1. **Re-label the slot.** Rename `dmu3 → w_eta_deriv` in the trait, document
   its definition as `2·∂W/∂η`, and fix ocat (identity link, W = ½·Dmu2 ⇒
   Dmu3 happens to satisfy the new contract) and tdist (same property).
   Then NegBin / Tweedie fill it with the analytical `2·∂W/∂η` derived
   above. This is the lowest-friction option but renames a load-bearing
   field in a published 0.4.x API surface.
2. **Two-slot split.** Keep `dmu3 = ∂³D/∂μ³` (useful for any future
   consumer that wants the pure deviance third deriv) AND add a separate
   `w_eta_deriv` slot; the consumer at `gradient.rs:147` switches to
   `w_eta_deriv`. Ocat/tdist fill both with the same value (their
   coincidence), NegBin/Tweedie fill them independently.

Either way, the NegBin port should not land as-is. The stash's analytical
work on `Dth / Dmuth / Dmu2th` is salvageable verbatim. The `Dmu3` formula
needs to be replaced with `2·θμ²(μ + 2θ) / (μ + θ)²` (NegBin under log
link + Fisher weights), and the equivalent for Tweedie.

## Cross-reference

- Trait definition: `src/traits.rs:44-60` and `:160-175`.
- Consumer (Tk·KK'): `src/score/shape_aware/gradient.rs:108-154`.
- Consumer (IFT analytic θ-grad): `src/score/shape_aware/gradient.rs:269-346`.
- Working ocat impl: `src/family/ocat.rs:226-247` and `ocat_dd_level1` at
  `:303-416`.
- Working tdist impl: `src/family/tdist.rs:133-214`.
- Standard PIRLS Fisher weight: `src/inner/pirls.rs:144-148`.
- gam.fit5 Newton weight (ocat): `src/inner/gam_fit5.rs:117-123`.
