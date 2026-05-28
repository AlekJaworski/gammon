//! Ordered categorical (`ocat`) with identity link (Phase 10 / v0.2 port).

use crate::traits::{Loss, VarianceFn};

use super::link::IdentityLink;
use super::Family;

/// Ordered categorical likelihood with `R ≥ 3` categories. Latent-variable
/// cumulative-logit model:
///
/// ```text
///   P(Y ≤ k | x) = F(α_{k+1} − μ),   F(z) = 1 / (1 + exp(−z))
/// ```
///
/// with thresholds `α_1 = −∞, α_2 = −1, α_{2+k} = −1 + Σ_{j ≤ k} exp(θ_j)`
/// for `k ∈ 0..R−2`, `α_{R+1} = +∞`. The free shape parameters are the
/// `R − 2` log-gap quantities `θ_j`; the log transform enforces monotone
/// ordering automatically (mgcv `efam.r:2618-2945`, ported in
/// `src/ocat.rs` of v0.x).
///
/// **Stateful loss** — `thresholds` (the log-gap θ slice) and `n_cats`
/// (`R`) are family-shape parameters. The score body and the dedicated
/// `inner::OcatInner` both read them via `Loss::n_shape_params /
/// set_shape_params / get_shape_params`. Saturated log-lik is zero
/// (mgcv `efam.r:2918` — categorical likelihood has no σ² / no `ls` term).
#[derive(Clone)]
pub struct OcatLoss {
    /// Log-gap thresholds, length `n_cats - 2`. Maps to `α_j` via
    /// `crate::special::ocat_alpha` (v0.x port).
    pub thresholds: ndarray::Array1<f64>,
    /// Number of categories `R ≥ 3`.
    pub n_cats: usize,
}

impl OcatLoss {
    /// Construct an `OcatLoss` at given thresholds. `thresholds.len()` MUST
    /// equal `n_cats - 2`.
    pub fn new(thresholds: ndarray::Array1<f64>, n_cats: usize) -> Self {
        assert!(n_cats >= 3, "Ocat requires R ≥ 3, got {n_cats}");
        assert_eq!(
            thresholds.len(),
            n_cats - 2,
            "Ocat thresholds length must equal n_cats - 2"
        );
        Self { thresholds, n_cats }
    }

    /// Boundary-aware logistic-CDF difference `F(b) − F(a)`, cancellation
    /// resistant. Mirrors v0.x `ocat::fdiff_boundary`. `pub(crate)` so
    /// `OcatInner` in `inner.rs` can share the implementation without
    /// duplicating the cancellation-safe branches.
    pub(crate) fn fdiff_boundary(a: f64, b: f64) -> f64 {
        // ±∞ short-circuits.
        if a.is_infinite() && a.is_sign_negative() {
            if b.is_infinite() && b.is_sign_positive() {
                1.0
            } else {
                1.0 / (1.0 + (-b).exp())
            }
        } else if b.is_infinite() && b.is_sign_positive() {
            1.0 / (1.0 + a.exp())
        } else {
            // Cancellation-safe `F(b) − F(a)` per v0.x ocat::fdiff.
            let ha = if a > 0.0 { -1.0 } else { 1.0 };
            let hb = if b > 0.0 { -1.0 } else { 1.0 };
            let ea = (a * ha).exp();
            let eb = (b * hb).exp();
            if b < 0.0 {
                eb / (1.0 + eb) - ea / (1.0 + ea)
            } else if a > 0.0 {
                (ea - eb) / ((ea + 1.0) * (eb + 1.0))
            } else {
                (1.0 - ea * eb) / ((eb + 1.0) * (ea + 1.0))
            }
        }
    }

    /// Compute the `(R+1)`-vector `α` from the log-gap thresholds.
    /// Mirrors v0.x `ocat::ocat_alpha`.
    pub fn alpha(&self) -> Vec<f64> {
        let r = self.n_cats;
        let mut alpha = vec![0.0_f64; r + 1];
        alpha[0] = f64::NEG_INFINITY;
        alpha[1] = -1.0;
        let mut acc = -1.0_f64;
        for k in 0..(r - 2) {
            acc += self.thresholds[k].exp();
            alpha[2 + k] = acc;
        }
        alpha[r] = f64::INFINITY;
        alpha
    }
}

/// Identity-style variance for ocat: constant 1. The actual working
/// weights come from `OcatInner` directly via `ocat_dd::Dmu2` — not
/// through `V(μ) · g'(μ)²`. This trivial `V(μ)=1` impl exists so a
/// `Family<OcatLoss, IdentityLink, OcatVariance>` can be aggregated for
/// constructor uniformity; the standard `PirlsInner` is NOT used for
/// ocat (see `inner::OcatInner`).
#[derive(Clone)]
pub struct OcatVariance;

impl Loss for OcatLoss {
    /// `μ_i = (α_{y-1} + α_y) / 2` (interior) with finite endpoints
    /// `−2, α_R + 1` for the boundary categories. Mirrors mgcv
    /// `efam.r:2947` initialisation expression and v0.x's `fit_pirls_ocat`
    /// start. Not strictly required (the outer fit entry point computes
    /// its own η₀), but provided so anything reading the trait gets a
    /// sane default.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let alpha = self.alpha();
        let r = self.n_cats;
        let lo_inf = -2.0_f64;
        let hi_inf = alpha[r - 1] + 1.0;
        y.iter()
            .map(|&yi| {
                let yi_c = (yi.round() as i64).clamp(1, r as i64) as usize;
                let lo = if yi_c == 1 { lo_inf } else { alpha[yi_c - 1] };
                let hi = if yi_c == r { hi_inf } else { alpha[yi_c] };
                0.5 * (lo + hi)
            })
            .collect()
    }

    /// Per-observation deviance `D_i = −2 · log F_i` where
    /// `F_i = F(α_{y_i} − μ_i) − F(α_{y_i − 1} − μ_i)`. Floors `F_i` at
    /// `f64::MIN_POSITIVE` to avoid `log(0)`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let r = self.n_cats;
        let alpha = self.alpha();
        let yi = (y.round() as i64).clamp(1, r as i64) as usize;
        let al0 = alpha[yi - 1] - mu;
        let al1 = alpha[yi] - mu;
        let f = Self::fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);
        -2.0 * f.ln()
    }

    /// Saturated log-lik is identically 0 for ocat (mgcv `efam.r:2918`).
    fn saturated_log_lik(&self, _y: f64, _scale: f64) -> f64 {
        0.0
    }

    /// `∂D/∂μ = −2 · (a1 − a0) / F` where `a_k = F'(α_k − μ)`. Internally
    /// uses the cancellation-safe `abcd` polynomial branch from v0.x
    /// `ocat::abcd`. Defined inline (no separate cache) — used by the
    /// generic PIRLS-validity check (`Loss::deviance_per_obs.is_finite()`)
    /// and by `OcatInner` for working-response assembly fallback. The
    /// hot path inside `OcatInner` calls `ocat_dd` directly to share the
    /// `α`-alloc across all `n` rows.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = self.n_cats;
        let alpha = self.alpha();
        let yi = (y.round() as i64).clamp(1, r as i64) as usize;
        let al0 = alpha[yi - 1] - mu;
        let al1 = alpha[yi] - mu;
        let f = Self::fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);
        let a0 = Self::abcd_a(al0);
        let a1 = Self::abcd_a(al1);
        -2.0 * (a1 - a0) / f
    }

    /// `∂²D/∂μ² = 2 · (a² / F − b) / F` with `a = a1 − a0`,
    /// `b = b1 − b0`. Used for working-weights fallback. Hot path uses
    /// `ocat_dd` per the `d_loss_dmu` note.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = self.n_cats;
        let alpha = self.alpha();
        let yi = (y.round() as i64).clamp(1, r as i64) as usize;
        let al0 = alpha[yi - 1] - mu;
        let al1 = alpha[yi] - mu;
        let f = Self::fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);
        let a0 = Self::abcd_a(al0);
        let a1 = Self::abcd_a(al1);
        let b0 = Self::abcd_b(al0);
        let b1 = Self::abcd_b(al1);
        let a = a1 - a0;
        let b = b1 - b0;
        2.0 * (a * a / f - b) / f
    }

    /// Dispersion fixed at 1 — ocat has no σ² (categorical likelihood).
    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    /// `R − 2` log-gap thresholds. The outer Newton joint-optimises these
    /// with `log λ`; same shape-aware machinery as scat/Tweedie/NegBin.
    fn n_shape_params(&self) -> usize {
        self.n_cats - 2
    }

    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(
            params.len(),
            self.n_cats - 2,
            "Ocat expects n_cats - 2 shape params"
        );
        self.thresholds = ndarray::Array1::from_vec(params.to_vec());
    }

    fn get_shape_params(&self) -> Vec<f64> {
        self.thresholds.to_vec()
    }

    /// Match v0.x's mgcv-style rank heuristic for the `Σ rank·log λ`
    /// score-formula term: report rank `(positive_eigvals − 1)` for the
    /// centered CR(k) penalty mgcv treats as having a 2-dim null space.
    /// gamrs's `rank_and_log_pseudo_det` returns the mathematical
    /// `positive_eigvals = k − 2`; this −1 brings the score formula
    /// in line with v0.x at `Σ rank·log λ` and closes the
    /// 5.95% → 0.06% multi-smooth ocat parity gap (2026-05-28 diagnostic).
    fn score_rank_adjustment(&self) -> i32 {
        -1
    }

    /// Match v0.x's `fit_pirls_ocat` β-tolerance (`lib.rs:968` passes
    /// `1e-8` to the PIRLS call). gamrs's general `PirlsOpts::default`
    /// uses `1e-9` which stops one decimal later than v0.x — the residual
    /// β difference between the two engines at the same θ was driven
    /// largely by this gap (Layer-3 parity diagnostic 2026-05-28).
    fn pirls_dev_rel_tol(&self) -> f64 {
        1.0e-8
    }

    /// Provide Level-1 derivatives (`Dmu3, Dth, Dmuth, Dmu2th`) to the
    /// shape-aware score's analytic θ-gradient assembly. Ports v0.x's
    /// `ocat_dd` at `OcatDerivLevel::Level1`. The score uses these to
    /// compute `∂REML/∂θ_k` via IFT — closes the 5.9% multi-smooth ocat
    /// parity gap (2026-05-27 parity report).
    fn level1_shape_derivatives(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<crate::traits::Level1ShapeDerivs> {
        let thresholds_vec: Vec<f64> = self.thresholds.iter().copied().collect();
        let (dmu3, dth, dmuth, dmu2th) =
            ocat_dd_level1(y, eta, &thresholds_vec, self.n_cats, prior_w);
        Some(crate::traits::Level1ShapeDerivs {
            dmu3,
            dth,
            dmuth,
            dmu2th,
        })
    }
}

impl OcatLoss {
    /// `aj = -ex / (ex+1)²` where `ex = exp(x · sign-flip)` per v0.x
    /// `ocat::abcd`. Cancellation-resistant. `pub(crate)` for `OcatInner`.
    #[inline]
    pub(crate) fn abcd_a(x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let h = if x > 0.0 { -1.0 } else { 1.0 };
        let ex = (x * h).exp();
        let ex1 = ex + 1.0;
        -ex / (ex1 * ex1)
    }

    /// `bj = h · (ex - ex²) / (ex+1)³` per v0.x `ocat::abcd`. `pub(crate)`
    /// for `OcatInner`.
    #[inline]
    pub(crate) fn abcd_b(x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let h = if x > 0.0 { -1.0 } else { 1.0 };
        let ex = (x * h).exp();
        let ex1 = ex + 1.0;
        let ex2 = ex * ex;
        h * (ex - ex2) / (ex1 * ex1 * ex1)
    }

    /// `cj = (-ex³ + 4·ex² − ex) / (ex+1)⁴` per v0.x `ocat::abcd` level 1.
    /// Third derivative of F, used by the analytic θ gradient (Dmu3).
    #[inline]
    pub(crate) fn abcd_c(x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let h = if x > 0.0 { -1.0 } else { 1.0 };
        let ex = (x * h).exp();
        let ex1 = ex + 1.0;
        let ex2 = ex * ex;
        let ex3 = ex2 * ex;
        let ex1_pow4 = ex1 * ex1 * ex1 * ex1;
        (-ex3 + 4.0 * ex2 - ex) / ex1_pow4
    }
}

impl VarianceFn for OcatVariance {
    /// Constant 1.0. Working weights are NOT built from `V(μ) · g'(μ)²`
    /// for ocat — see `OcatInner` for the family-specific `0.5·Dmu2` path.
    fn variance(&self, _mu: f64) -> f64 {
        1.0
    }
    // No shape-param sync needed — variance is μ- and threshold-free.
}

/// Per-row Level-1 ocat derivatives — port of v0.x `ocat::ocat_dd` at
/// `OcatDerivLevel::Level1`. Returns `(Dmu3, Dth, Dmuth, Dmu2th)` evaluated
/// at the converged η at the current θ. Required by the analytic θ_ocat
/// gradient (`reml_grad_ocat_theta_block_analytic` port; closes the 5.9%
/// multi-smooth ocat parity gap, 2026-05-27).
///
/// Convention: derivatives are computed WITHOUT prior-weights — caller
/// multiplies in `wt[i]` at the per-θ assembly step (mgcv efam.r:2814-2832).
///
/// Returns `Dth, Dmuth, Dmu2th` as `(n × n_θ)` matrices (where `n_θ = r-2`)
/// and `Dmu3` as a length-n vector.
pub fn ocat_dd_level1(
    y: ndarray::ArrayView1<f64>,
    eta: ndarray::ArrayView1<f64>,
    thresholds: &[f64],
    n_cats: usize,
    prior_w: Option<ndarray::ArrayView1<f64>>,
) -> (
    ndarray::Array1<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    use ndarray::{Array1, Array2};
    let n = y.len();
    let r = n_cats;
    let n_theta = r.saturating_sub(2);
    let alpha = {
        // Inline alpha computation — same shape as OcatLoss::alpha. α has
        // length r+1: [α_0=-∞, α_1=-1, α_{k+1} = α_k + exp(θ_{k-1})]; the
        // boundary handling above mirrors `OcatLoss::alpha`.
        let mut a = vec![0.0_f64; r + 1];
        a[0] = f64::NEG_INFINITY;
        a[1] = -1.0;
        let mut acc: f64 = -1.0;
        for k in 0..n_theta {
            acc += thresholds[k].exp();
            a[k + 2] = acc;
        }
        a[r] = f64::INFINITY;
        a
    };

    let mut dmu3 = Array1::<f64>::zeros(n);
    let mut dth = Array2::<f64>::zeros((n, n_theta));
    let mut dmuth = Array2::<f64>::zeros((n, n_theta));
    let mut dmu2th = Array2::<f64>::zeros((n, n_theta));

    // Per-row workspace: a/b/c/d/f and a0/a1/b0/b1/c0/c1 buffers from the
    // Level-1 assembly. mgcv efam.r:2796-2811.
    for i in 0..n {
        let yi = (y[i].round() as i64).clamp(1, r as i64) as usize;
        let mu_i = eta[i];
        let wt_i = prior_w.map(|w| w[i]).unwrap_or(1.0);
        let al0 = alpha[yi - 1] - mu_i;
        let al1 = alpha[yi] - mu_i;
        let f = OcatLoss::fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);

        let a0 = OcatLoss::abcd_a(al0);
        let a1 = OcatLoss::abcd_a(al1);
        let b0 = OcatLoss::abcd_b(al0);
        let b1 = OcatLoss::abcd_b(al1);
        let c0 = OcatLoss::abcd_c(al0);
        let c1 = OcatLoss::abcd_c(al1);
        let a = a1 - a0;
        let b = b1 - b0;
        let c = c1 - c0;
        let a2 = a * a;
        let a3 = a2 * a;
        let f2 = f * f;

        // Dmu3 includes wt (mgcv convention for length-n derivatives).
        dmu3[i] = 2.0 * wt_i * (-c - 2.0 * a3 / f2 + 3.0 * a * b / f) / f;

        // Per-row scalar building blocks (mgcv efam.r:2803-2811; written
        // WITHOUT wt — wt enters at the per-θ assembly below).
        let dmua0 = 2.0 * (a0 * a / f - b0) / f;
        let dmua1 = -2.0 * (a1 * a / f - b1) / f;
        let dmu2a0 = -2.0 * (c0 + (a0 * (2.0 * a2 / f - b) - 2.0 * b0 * a) / f) / f;
        let dmu2a1 = 2.0 * (c1 + (2.0 * (a1 * a2 / f - b1 * a) - a1 * b) / f) / f;
        let da0 = -2.0 * a0 / f;
        let da1 = 2.0 * a1 / f;

        // Assemble Dth/Dmuth/Dmu2th rows by which α boundary the y_i lands
        // on. mgcv efam.r:2814-2832 — k is 1-based in mgcv; here k is
        // 0-based, so the brackets become:
        //   yi == k+2  → upper boundary  (carries θ_k)
        //   yi == r    → lower boundary  (only meaningful for k = r-3)
        //   otherwise  → both (yi > k+2 && yi < r)
        for k in 0..n_theta {
            let etk = thresholds[k].exp();
            let k_plus_2 = k + 2;
            let bracket_upper = yi == k_plus_2;
            let bracket_lower = yi == r;
            let bracket_middle = yi > k_plus_2 && yi < r && r > k_plus_2;

            if bracket_upper {
                dth[[i, k]] = wt_i * da1 * etk;
                dmuth[[i, k]] = wt_i * dmua1 * etk;
                dmu2th[[i, k]] = wt_i * dmu2a1 * etk;
            } else if bracket_middle {
                dth[[i, k]] = wt_i * (da1 + da0) * etk;
                dmuth[[i, k]] = wt_i * (dmua1 + dmua0) * etk;
                dmu2th[[i, k]] = wt_i * (dmu2a1 + dmu2a0) * etk;
            } else if bracket_lower {
                dth[[i, k]] = wt_i * da0 * etk;
                dmuth[[i, k]] = wt_i * dmua0 * etk;
                dmu2th[[i, k]] = wt_i * dmu2a0 * etk;
            }
        }
    }

    (dmu3, dth, dmuth, dmu2th)
}

/// Phase 10 convenience constructor — Ocat + identity link at given log-gap
/// thresholds. Pair with `inner::OcatInner` (NOT `PirlsInner`) for the
/// joint β + threshold fit. `thresholds.len()` MUST equal `n_cats - 2`.
pub fn ocat_identity(
    thresholds: ndarray::Array1<f64>,
    n_cats: usize,
) -> Family<OcatLoss, IdentityLink, OcatVariance> {
    Family::new(
        OcatLoss::new(thresholds, n_cats),
        IdentityLink,
        OcatVariance,
    )
}

/// Initial log-gap threshold heuristic from category counts. Mirrors mgcv
/// `efam.r:2927-2945` (`preinitialize`). Returns θ of length `R − 2`.
///
/// Strategy: empirical cumulative proportions → latent-eta via logit →
/// diffs (clamped positive) → log. Falls back to `[-1, -1, …]` on edge
/// cases. Same algorithm as v0.x `ocat::ocat_init_theta`.
pub fn ocat_init_theta(y: ndarray::ArrayView1<f64>, n_cats: usize) -> ndarray::Array1<f64> {
    if n_cats < 3 {
        return ndarray::Array1::zeros(0);
    }
    let r = n_cats;
    let n_theta = r - 2;
    let n = y.len();
    if n == 0 {
        return ndarray::Array1::from_elem(n_theta, -1.0);
    }
    let mut counts = vec![1_usize; r]; // Laplace +1 like v0.x
    for &yi in y.iter() {
        let yi_round = yi.round() as i64;
        if yi_round >= 1 && (yi_round as usize) <= r {
            counts[yi_round as usize - 1] += 1;
        }
    }
    let total: usize = counts.iter().sum();
    let mut cum = vec![0.0_f64; r];
    let mut acc = 0.0_f64;
    for k in 0..r {
        acc += counts[k] as f64 / total as f64;
        cum[k] = acc;
    }
    let p1 = cum[0];
    let eta = if p1 <= 0.0 || p1 >= 1.0 {
        5.0
    } else {
        -1.0 - (p1 / (1.0 - p1)).ln()
    };
    let mut theta_alpha = vec![-1.0_f64; r - 1];
    for i in 1..(r - 1) {
        let pi = cum[i].clamp(1e-9, 1.0 - 1e-9);
        theta_alpha[i] = (pi / (1.0 - pi)).ln() + eta;
    }
    let mut diffs = ndarray::Array1::<f64>::zeros(r - 2);
    for i in 0..(r - 2) {
        diffs[i] = (theta_alpha[i + 1] - theta_alpha[i]).max(0.01).ln();
    }
    diffs
}
