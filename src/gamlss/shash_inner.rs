//! `shash` GAMLSS joint inner solver (TDD phase 4) — the penalised
//! coefficient estimate β̂ given fixed per-block penalties.
//!
//! Unlike the orthogonal Gaussian location-scale family (`gaulss`, whose
//! block-diagonal Fisher information lets it alternate single-predictor REML
//! fits), shash is **non-orthogonal**: the 4×4 per-observation Hessian
//! ([`ShashDensity::l2_eta`]) has nonzero cross-parameter terms, so β for all
//! four linear predictors must be solved **jointly** by a dense penalised
//! Newton step. This module builds that machinery component-by-component:
//!
//!   - phase 4a (here): the penalised log-likelihood, its full-vector gradient,
//!     and the dense `(Σpᵦ)×(Σpᵦ)` block Hessian assembled from the per-obs
//!     η-space derivatives — each validated against finite differences.
//!   - phase 4b: the damped Newton iteration (step-halving + Hessian
//!     perturbation to stay in the ascent cone), converging to β̂; confronted
//!     with mgcv's shash MLE on intercept-only data and FD-checked on smooths.
//!
//! Each block `b` carries a design `Xᵦ` (n×pᵦ) and an **already-λ-combined**
//! penalty `Sᵦ` (pᵦ×pᵦ, i.e. `Σⱼ λⱼ Sⱼ`); the outer REML (phase 5) supplies
//! the combined `Sᵦ`. The penalised objective is
//! `L(β) = Σᵢ ℓ₀(yᵢ; linkinv(ηᵢ)) − ½ Σᵦ βᵦᵀ Sᵦ βᵦ`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::shash::{ShashDensity, L2_INDEX};
use crate::error::{GamrsError, Result};
use crate::inner::{CholeskySolver, LinearSolver};

/// The four per-predictor designs and (λ-combined) penalties for a shash fit.
#[derive(Clone, Copy)]
pub struct ShashBlocks<'a> {
    /// Per-block design matrices `Xᵦ`, each `n×pᵦ` (same `n`, distinct `pᵦ`).
    pub x: [ArrayView2<'a, f64>; 4],
    /// Per-block λ-combined penalty matrices `Sᵦ`, each `pᵦ×pᵦ`.
    pub s: [ArrayView2<'a, f64>; 4],
    /// `logeb` bound `b` for the τ link (mgcv default 1e-2).
    pub b: f64,
}

impl<'a> ShashBlocks<'a> {
    /// Number of observations (rows of any block design).
    pub fn n(&self) -> usize {
        self.x[0].nrows()
    }
    /// Per-block coefficient counts `[p₁,p₂,p₃,p₄]`.
    pub fn p(&self) -> [usize; 4] {
        [
            self.x[0].ncols(),
            self.x[1].ncols(),
            self.x[2].ncols(),
            self.x[3].ncols(),
        ]
    }
    /// Total coefficient count `Σ pᵦ`.
    pub fn total_p(&self) -> usize {
        self.p().iter().sum()
    }
    /// Offset of block `b` within the flat coefficient vector.
    pub fn offset(&self, b: usize) -> usize {
        self.p()[..b].iter().sum()
    }

    /// The `n×4` matrix of linear predictors `ηᵢ = (X₁β₁, X₂β₂, X₃β₃, X₄β₄)ᵢ`
    /// for a flat coefficient vector `beta` (block `b` occupies
    /// `[offset(b)..offset(b)+pᵦ]`).
    pub fn eta(&self, beta: ArrayView1<f64>) -> Array2<f64> {
        let n = self.n();
        let mut eta = Array2::<f64>::zeros((n, 4));
        for b in 0..4 {
            let off = self.offset(b);
            let pb = self.p()[b];
            let beta_b = beta.slice(ndarray::s![off..off + pb]);
            let eta_b = self.x[b].dot(&beta_b); // n
            eta.column_mut(b).assign(&eta_b);
        }
        eta
    }
}

/// Penalised log-likelihood `L(β) = Σᵢ ℓ₀ − ½ Σᵦ βᵦᵀ Sᵦ βᵦ`.
pub fn penalized_loglik(
    density: &ShashDensity,
    blocks: &ShashBlocks,
    beta: ArrayView1<f64>,
    y: ArrayView1<f64>,
) -> f64 {
    let eta = blocks.eta(beta);
    let n = blocks.n();
    let mut ll = 0.0;
    for i in 0..n {
        let [mu, tau, eps, phi] = ShashDensity::linkinv(
            [eta[[i, 0]], eta[[i, 1]], eta[[i, 2]], eta[[i, 3]]],
            blocks.b,
        );
        ll += density.l0(y[i], mu, tau, eps, phi);
    }
    // − ½ Σᵦ βᵦᵀ Sᵦ βᵦ
    let mut pen = 0.0;
    for b in 0..4 {
        let off = blocks.offset(b);
        let pb = blocks.p()[b];
        let beta_b = beta.slice(ndarray::s![off..off + pb]);
        let sb = blocks.s[b];
        let sbeta = sb.dot(&beta_b);
        pen += beta_b.dot(&sbeta);
    }
    ll - 0.5 * pen
}

/// Penalised gradient `∇L` (length `Σpᵦ`) and dense block Hessian `∇²L`
/// (`Σpᵦ × Σpᵦ`, symmetric, negative-definite near the maximum).
///
/// Per block `b`: `gᵦ = Xᵦᵀ G[:,b] − Sᵦ βᵦ` where `G[i,:] = l1_eta(ηᵢ)`.
/// Per block pair `(b,c)`: `H_{bc} = Xᵦᵀ diag(Hessη[:,b,c]) X_c − δ_{bc} Sᵦ`
/// where `Hessη[i]` is the 4×4 unpacked from [`ShashDensity::l2_eta`].
pub fn penalized_grad_hess(
    density: &ShashDensity,
    blocks: &ShashBlocks,
    beta: ArrayView1<f64>,
    y: ArrayView1<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let n = blocks.n();
    let tp = blocks.total_p();
    let p = blocks.p();
    let off: [usize; 4] = [
        blocks.offset(0),
        blocks.offset(1),
        blocks.offset(2),
        blocks.offset(3),
    ];
    let eta = blocks.eta(beta);

    let mut grad = Array1::<f64>::zeros(tp);
    let mut hess = Array2::<f64>::zeros((tp, tp));

    for i in 0..n {
        let eta_i = [eta[[i, 0]], eta[[i, 1]], eta[[i, 2]], eta[[i, 3]]];
        let g_i = density.l1_eta(y[i], eta_i, blocks.b); // 4
        let h_packed = density.l2_eta(y[i], eta_i, blocks.b); // 10
                                                              // Unpack to dense 4×4.
        let mut h_i = [[0.0_f64; 4]; 4];
        for (idx, &(a, c)) in L2_INDEX.iter().enumerate() {
            h_i[a][c] = h_packed[idx];
            h_i[c][a] = h_packed[idx];
        }
        // Gradient: gᵦ += Xᵦ[i,:] · g_i[b].
        for b in 0..4 {
            let xb_row = blocks.x[b].row(i);
            for r in 0..p[b] {
                grad[off[b] + r] += xb_row[r] * g_i[b];
            }
        }
        // Hessian: H_{bc} += Xᵦ[i,:]ᵀ X_c[i,:] · h_i[b][c].
        for b in 0..4 {
            let xb_row = blocks.x[b].row(i);
            for c in 0..4 {
                let w = h_i[b][c];
                if w == 0.0 {
                    continue;
                }
                let xc_row = blocks.x[c].row(i);
                for r in 0..p[b] {
                    let xbr_w = xb_row[r] * w;
                    let hrow = off[b] + r;
                    for s in 0..p[c] {
                        hess[[hrow, off[c] + s]] += xbr_w * xc_row[s];
                    }
                }
            }
        }
    }

    // Subtract the penalty: gᵦ −= Sᵦ βᵦ ; H_{bb} −= Sᵦ.
    for b in 0..4 {
        let sb = blocks.s[b];
        let beta_b = beta.slice(ndarray::s![off[b]..off[b] + p[b]]);
        let sbeta = sb.dot(&beta_b);
        for r in 0..p[b] {
            grad[off[b] + r] -= sbeta[r];
            for s in 0..p[b] {
                hess[[off[b] + r, off[b] + s]] -= sb[[r, s]];
            }
        }
    }

    (grad, hess)
}

#[inline]
fn inf_norm(v: ArrayView1<f64>) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Solve `(−H + τI) Δ = g` for the Newton ascent direction. `−H` is SPD near
/// the maximum (since `H = ∇²L` is negative-definite there); when it is not —
/// far from the optimum or because shash is non-orthogonal — the diagonal is
/// inflated by a growing `τ` (a Levenberg-style perturbation, mgcv's
/// "ensure negative definiteness" step) until the Cholesky succeeds, blending
/// the Newton step toward gradient ascent.
fn solve_ascent(hess: &Array2<f64>, grad: ArrayView1<f64>, tp: usize) -> Result<Array1<f64>> {
    let neg_h = hess.mapv(|v| -v);
    let diag_max = (0..tp).map(|i| neg_h[[i, i]].abs()).fold(1.0_f64, f64::max);
    let mut tau = 0.0_f64;
    for _ in 0..32 {
        let mut a = neg_h.clone();
        if tau > 0.0 {
            for i in 0..tp {
                a[[i, i]] += tau;
            }
        }
        if let Ok(fact) = CholeskySolver::factorize(a) {
            return Ok(CholeskySolver::solve(&fact, grad));
        }
        tau = if tau == 0.0 {
            1e-8 * diag_max
        } else {
            tau * 10.0
        };
    }
    Err(GamrsError::SingularSystem(
        "shash inner Newton: Hessian could not be stabilised to SPD".into(),
    ))
}

/// Options for the shash joint inner Newton solve.
#[derive(Clone, Copy, Debug)]
pub struct ShashInnerOpts {
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// Converge when `‖∇L‖∞ < grad_tol`.
    pub grad_tol: f64,
    /// Also converge when an accepted step has `‖Δ‖∞ < step_tol`.
    pub step_tol: f64,
    /// Maximum step-halvings per iteration (sufficient-increase line search).
    pub max_halvings: usize,
}

impl Default for ShashInnerOpts {
    fn default() -> Self {
        Self {
            max_iter: 100,
            grad_tol: 1e-8,
            step_tol: 1e-11,
            max_halvings: 40,
        }
    }
}

/// Result of the joint inner solve: the penalised MLE `β̂` and diagnostics.
#[derive(Clone, Debug)]
pub struct ShashInnerFit {
    /// Penalised coefficient estimate (flat, block `b` at `offset(b)`).
    pub beta: Array1<f64>,
    /// Penalised log-likelihood at `β̂`.
    pub loglik: f64,
    /// `‖∇L‖∞` at `β̂`.
    pub grad_norm: f64,
    /// Newton iterations taken.
    pub n_iter: usize,
    /// Whether a convergence criterion was met (vs hitting `max_iter`).
    pub converged: bool,
}

/// Joint penalised Newton solve for the shash coefficients given fixed
/// per-block penalties: maximise `L(β)` (see [`penalized_loglik`]) by damped
/// Newton steps with a Hessian perturbation ([`solve_ascent`]) and a
/// sufficient-increase line search (step-halving).
pub fn fit_inner(
    density: &ShashDensity,
    blocks: &ShashBlocks,
    beta0: ArrayView1<f64>,
    y: ArrayView1<f64>,
    opts: ShashInnerOpts,
) -> Result<ShashInnerFit> {
    let tp = blocks.total_p();
    let mut beta = beta0.to_owned();
    let mut ll = penalized_loglik(density, blocks, beta.view(), y);
    let mut grad_norm = f64::INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    while n_iter < opts.max_iter {
        n_iter += 1;
        let (grad, hess) = penalized_grad_hess(density, blocks, beta.view(), y);
        grad_norm = inf_norm(grad.view());
        if grad_norm < opts.grad_tol {
            converged = true;
            break;
        }
        let delta = solve_ascent(&hess, grad.view(), tp)?;

        // Sufficient-increase line search (any strict increase accepted).
        let mut t = 1.0_f64;
        let mut accepted = false;
        for _ in 0..=opts.max_halvings {
            let trial = &beta + &(&delta * t);
            let ll_t = penalized_loglik(density, blocks, trial.view(), y);
            if ll_t > ll {
                let step = inf_norm((&delta * t).view());
                beta = trial;
                ll = ll_t;
                accepted = true;
                if step < opts.step_tol {
                    converged = true;
                }
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            // No increase even at the smallest step: a stationary point within
            // numerical resolution (grad already small relative to scale).
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }

    Ok(ShashInnerFit {
        beta,
        loglik: ll,
        grad_norm,
        n_iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array1, Array2};

    fn frac(v: f64) -> f64 {
        v - v.floor()
    }
    fn pnormal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// A small synthetic shash problem: distinct block widths, a real (SPD)
    /// penalty on the μ and τ blocks, intercept-only ε/φ. Returns (blocks-owned
    /// designs/penalties, y, beta) and the b bound; the caller builds views.
    struct Prob {
        x: [Array2<f64>; 4],
        s: [Array2<f64>; 4],
        y: Array1<f64>,
        beta: Array1<f64>,
        b: f64,
    }

    fn make_problem() -> Prob {
        let n = 60usize;
        // Block widths: μ=3, τ=2, ε=1, φ=1.
        let pw = [3usize, 2, 1, 1];
        let mut x: [Array2<f64>; 4] = [
            Array2::zeros((n, pw[0])),
            Array2::zeros((n, pw[1])),
            Array2::zeros((n, pw[2])),
            Array2::zeros((n, pw[3])),
        ];
        for i in 0..n {
            let c1 = frac((i as f64 + 0.5) * 0.6180339887498949);
            let c2 = frac((i as f64 + 0.5) * 0.3819660112501051 + 0.137);
            // μ design: intercept + 2 smooth-ish covariates.
            x[0][[i, 0]] = 1.0;
            x[0][[i, 1]] = c1;
            x[0][[i, 2]] = c2 - 0.5;
            // τ design: intercept + 1 covariate.
            x[1][[i, 0]] = 1.0;
            x[1][[i, 1]] = c1 - 0.5;
            // ε, φ: intercept only.
            x[2][[i, 0]] = 1.0;
            x[3][[i, 0]] = 1.0;
        }
        // Penalties: ridge-like SPD on the non-intercept directions of μ, τ;
        // zero for ε/φ (1×1 zero). (Penalising the intercept too is fine here —
        // we only need a well-defined quadratic for the FD checks.)
        let s: [Array2<f64>; 4] = [
            arr2(&[[0.3, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.5]]),
            arr2(&[[0.2, 0.0], [0.0, 0.4]]),
            arr2(&[[0.0]]),
            arr2(&[[0.0]]),
        ];
        // A response with location/scale signal so derivatives are nontrivial.
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let c1 = x[0][[i, 1]];
            let mu = 0.4 + 1.1 * c1 + 0.6 * x[0][[i, 2]];
            let sg = (0.1 + 0.3 * x[1][[i, 1]]).exp() * 0.4;
            let u1 = frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31);
            let u2 = frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59);
            y[i] = mu + sg * pnormal(u1, u2);
        }
        // A non-trivial β to evaluate derivatives at (not the optimum).
        let beta = Array1::from(vec![0.3, 0.9, 0.5, -0.6, 0.25, 0.1, -0.05]);
        Prob {
            x,
            s,
            y,
            beta,
            b: 1e-2,
        }
    }

    fn blocks_of(p: &Prob) -> ShashBlocks<'_> {
        ShashBlocks {
            x: [p.x[0].view(), p.x[1].view(), p.x[2].view(), p.x[3].view()],
            s: [p.s[0].view(), p.s[1].view(), p.s[2].view(), p.s[3].view()],
            b: p.b,
        }
    }

    #[test]
    fn block_offsets_and_sizes() {
        let p = make_problem();
        let blk = blocks_of(&p);
        assert_eq!(blk.p(), [3, 2, 1, 1]);
        assert_eq!(blk.total_p(), 7);
        assert_eq!(
            [blk.offset(0), blk.offset(1), blk.offset(2), blk.offset(3)],
            [0, 3, 5, 6]
        );
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let prob = make_problem();
        let blk = blocks_of(&prob);
        let d = ShashDensity::default();
        let (grad, _h) = penalized_grad_hess(&d, &blk, prob.beta.view(), prob.y.view());
        let tp = blk.total_p();
        let h = 1e-6;
        for k in 0..tp {
            let mut bp = prob.beta.clone();
            let mut bm = prob.beta.clone();
            bp[k] += h;
            bm[k] -= h;
            let fd = (penalized_loglik(&d, &blk, bp.view(), prob.y.view())
                - penalized_loglik(&d, &blk, bm.view(), prob.y.view()))
                / (2.0 * h);
            assert!(
                (grad[k] - fd).abs() < 1e-5,
                "∂L/∂β[{k}]: analytic {} vs FD {}",
                grad[k],
                fd
            );
        }
    }

    #[test]
    fn hessian_matches_finite_difference_and_is_symmetric() {
        let prob = make_problem();
        let blk = blocks_of(&prob);
        let d = ShashDensity::default();
        let (_g, hess) = penalized_grad_hess(&d, &blk, prob.beta.view(), prob.y.view());
        let tp = blk.total_p();
        let h = 1e-6;
        // Symmetry.
        for r in 0..tp {
            for c in 0..tp {
                assert!(
                    (hess[[r, c]] - hess[[c, r]]).abs() < 1e-9,
                    "Hessian asymmetry at ({r},{c}): {} vs {}",
                    hess[[r, c]],
                    hess[[c, r]]
                );
            }
        }
        // H[:,k] = ∂(grad)/∂β_k via central FD of the analytic gradient.
        for k in 0..tp {
            let mut bp = prob.beta.clone();
            let mut bm = prob.beta.clone();
            bp[k] += h;
            bm[k] -= h;
            let (gp, _) = penalized_grad_hess(&d, &blk, bp.view(), prob.y.view());
            let (gm, _) = penalized_grad_hess(&d, &blk, bm.view(), prob.y.view());
            for j in 0..tp {
                let fd = (gp[j] - gm[j]) / (2.0 * h);
                assert!(
                    (hess[[j, k]] - fd).abs() < 1e-4,
                    "∂²L/∂β[{j}]∂β[{k}]: analytic {} vs FD {}",
                    hess[[j, k]],
                    fd
                );
            }
        }
    }

    // --- phase 4b: damped Newton convergence -------------------------------

    /// mgcv's intercept-only shash MLE (cross-checked against an independent
    /// BFGS optimum) — see `scripts/r/gen_shash_inner_mle_fixture.R`.
    #[derive(serde::Deserialize)]
    struct MleFixture {
        n: usize,
        y: Vec<f64>,
        b: f64,
        coef_mu: f64,
        coef_tau: f64,
        coef_eps: f64,
        coef_phi: f64,
        loglik: f64,
    }

    fn load_mle_fixture() -> MleFixture {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/shash_inner_mle_mgcv.json");
        let txt = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
        serde_json::from_str(&txt).expect("malformed shash_inner MLE fixture")
    }

    #[test]
    fn fit_inner_recovers_mgcv_intercept_mle() {
        let fx = load_mle_fixture();
        let n = fx.n;
        // Intercept-only designs (ones), zero penalty -> pure MLE.
        let ones = Array2::<f64>::ones((n, 1));
        let zero = Array2::<f64>::zeros((1, 1));
        let blk = ShashBlocks {
            x: [ones.view(), ones.view(), ones.view(), ones.view()],
            s: [zero.view(), zero.view(), zero.view(), zero.view()],
            b: fx.b,
        };
        let y = Array1::from(fx.y.clone());
        let d = ShashDensity::default();
        let beta0 = Array1::<f64>::zeros(4);
        let fit = fit_inner(
            &d,
            &blk,
            beta0.view(),
            y.view(),
            ShashInnerOpts {
                grad_tol: 1e-9,
                ..Default::default()
            },
        )
        .expect("inner fit");

        assert!(fit.converged, "did not converge (iters={})", fit.n_iter);
        assert!(
            fit.grad_norm < 1e-7,
            "gradient not driven to ~0: {}",
            fit.grad_norm
        );
        let mgcv = [fx.coef_mu, fx.coef_tau, fx.coef_eps, fx.coef_phi];
        for (k, &m) in mgcv.iter().enumerate() {
            assert!(
                (fit.beta[k] - m).abs() < 1e-5,
                "coef[{k}] = {} vs mgcv {m}",
                fit.beta[k]
            );
        }
        // Σℓ₀ at β̂ matches mgcv's logLik (no penalty here).
        assert!(
            (fit.loglik - fx.loglik).abs() < 1e-4,
            "loglik {} vs mgcv {}",
            fit.loglik,
            fx.loglik
        );
    }

    #[test]
    fn fit_inner_converges_on_penalized_smooth() {
        // Real penalty + multi-column μ/τ designs; start from the phase-3 init,
        // the actual pipeline. The Newton must drive ‖∇L‖→0 (penalised MLE).
        let prob = make_problem();
        let blk = blocks_of(&prob);
        let d = ShashDensity::default();
        let init = crate::gamlss::shash_init::shash_init(
            blk.x[0],
            blk.x[1],
            blk.x[2],
            blk.x[3],
            prob.y.view(),
            0.0,
        )
        .expect("init");
        let mut beta0 = Array1::<f64>::zeros(blk.total_p());
        beta0.slice_mut(ndarray::s![0..3]).assign(&init.beta_mu);
        beta0.slice_mut(ndarray::s![3..5]).assign(&init.beta_tau);
        // ε/φ blocks start at 0 (already zero).

        let fit = fit_inner(
            &d,
            &blk,
            beta0.view(),
            prob.y.view(),
            ShashInnerOpts::default(),
        )
        .expect("inner fit");
        assert!(fit.converged, "did not converge (iters={})", fit.n_iter);
        assert!(
            fit.grad_norm < 1e-6,
            "penalised gradient not ~0 at solution: {}",
            fit.grad_norm
        );

        // The fit must improve on (or equal) the starting objective.
        let ll0 = penalized_loglik(&d, &blk, beta0.view(), prob.y.view());
        assert!(
            fit.loglik >= ll0 - 1e-9,
            "objective decreased: {} < {}",
            fit.loglik,
            ll0
        );
    }

    #[test]
    fn fit_inner_recovers_from_a_far_start() {
        // Damping/perturbation robustness: a deliberately bad start must still
        // reach the same intercept MLE as the zero start.
        let fx = load_mle_fixture();
        let n = fx.n;
        let ones = Array2::<f64>::ones((n, 1));
        let zero = Array2::<f64>::zeros((1, 1));
        let blk = ShashBlocks {
            x: [ones.view(), ones.view(), ones.view(), ones.view()],
            s: [zero.view(), zero.view(), zero.view(), zero.view()],
            b: fx.b,
        };
        let y = Array1::from(fx.y.clone());
        let d = ShashDensity::default();
        let beta0 = Array1::from(vec![-3.0, 2.0, 1.5, 0.8]); // far from MLE
        let fit = fit_inner(
            &d,
            &blk,
            beta0.view(),
            y.view(),
            ShashInnerOpts {
                grad_tol: 1e-9,
                ..Default::default()
            },
        )
        .expect("inner fit");
        assert!(fit.converged, "did not converge from far start");
        let mgcv = [fx.coef_mu, fx.coef_tau, fx.coef_eps, fx.coef_phi];
        for (k, &m) in mgcv.iter().enumerate() {
            assert!(
                (fit.beta[k] - m).abs() < 1e-4,
                "coef[{k}] = {} vs mgcv {m} (far start)",
                fit.beta[k]
            );
        }
    }

    #[test]
    fn penalty_term_matches_quadratic() {
        // With β fixed and a known SPD penalty, the penalty contribution to L
        // is exactly −½ Σᵦ βᵦᵀ Sᵦ βᵦ; check the gradient picks up −Sᵦ βᵦ by
        // comparing the penalised gradient to the unpenalised one (S=0).
        let prob = make_problem();
        let blk = blocks_of(&prob);
        let d = ShashDensity::default();
        let (g_pen, _) = penalized_grad_hess(&d, &blk, prob.beta.view(), prob.y.view());

        let zero: [Array2<f64>; 4] = [
            Array2::zeros((3, 3)),
            Array2::zeros((2, 2)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 1)),
        ];
        let blk0 = ShashBlocks {
            x: blk.x,
            s: [
                zero[0].view(),
                zero[1].view(),
                zero[2].view(),
                zero[3].view(),
            ],
            b: prob.b,
        };
        let (g_unp, _) = penalized_grad_hess(&d, &blk0, prob.beta.view(), prob.y.view());

        // g_pen − g_unp should equal −Sᵦ βᵦ per block.
        for b in 0..4 {
            let off = blk.offset(b);
            let pb = blk.p()[b];
            let beta_b = prob.beta.slice(ndarray::s![off..off + pb]);
            let sbeta = prob.s[b].dot(&beta_b);
            for r in 0..pb {
                let diff = g_pen[off + r] - g_unp[off + r];
                assert!(
                    (diff + sbeta[r]).abs() < 1e-10,
                    "penalty grad block {b}[{r}]: {diff} vs −{}",
                    sbeta[r]
                );
            }
        }
    }
}
