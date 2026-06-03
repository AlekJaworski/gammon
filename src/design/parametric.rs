//! Parametric (linear, unsmoothed) DesignStrategy + Predictor.
//!
//! A "parametric" term enters the linear predictor as a single raw column
//! with no spline expansion and **no smoothing penalty** — the coefficient
//! retains its full degree of freedom. This is mgcv R's "pterms" block;
//! mgcv_rust calls it `bs="parametric"` (alias `"linear"`).
//!
//! Architectural fit in gamrs:
//! - Contributes ONE column to the assembled design (the raw covariate).
//! - Contributes ZERO entries to the penalty `s_list` — there's no
//!   smoothing parameter to optimise. The unpenalised coefficient lives
//!   in the design and bumps `Mp` (null-space dim) by 1.
//! - No sum-to-zero centring. Parametric columns are deliberately raw —
//!   centring would absorb part of the effect into the intercept and
//!   destroy the "slope of x_param" interpretation (critical for 0/1
//!   indicators where the coefficient should mean "x=1 vs x=0").
//!
//! `Predictor::Parametric` carries the column index it reads from new x
//! and reproduces the same raw column at predict-time.

use ndarray::{Array2, ArrayView2};

use crate::error::Result;

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct ParametricPredictor {
    /// Column index this term reads from `x`. For the single-term path
    /// (not currently exposed) the predictor reads `x.column(0)`; the
    /// additive path passes a single-column slice so this stays at 0.
    pub col: usize,
}

impl ParametricPredictor {
    /// Sub-design called by `AdditivePredictor`. Receives a `(n, 1)` view
    /// of the parametric column and returns `[1 | x_param]` so the
    /// caller can strip the leading intercept like every other sub-design.
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x_new.nrows();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = 1.0;
            out[[i, 1]] = x_new[[i, 0]];
        }
        Ok(out)
    }

    /// `∂design/∂x` is `[0 | 1]` — the intercept is constant, the
    /// parametric column has unit slope w.r.t. its own covariate.
    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, _axis: usize) -> Result<Array2<f64>> {
        let n = x_new.nrows();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 1]] = 1.0;
        }
        Ok(out)
    }
}
