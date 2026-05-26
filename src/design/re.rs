//! Random-effect DesignStrategy + Predictor (`bs="re"` in mgcv).
//!
//! One-hot encoding over training levels + identity penalty. The intercept
//! column is retained (matches mgcv: `s(g, bs="re")` always rides with an
//! unpenalised constant). The single-smooth path produces
//! `s_list.len() == 1`; multi-smooth callers compose Re terms via
//! [`crate::design::Additive`].

use ndarray::{Array2, ArrayView2};

use crate::basis::RandomEffectsBasis;
use crate::error::{GammonError, Result};
use crate::traits::Basis;

use super::{
    prepend_intercept, rank_and_log_pseudo_det, DesignStrategy, PreparedDesign, Predictor,
};

#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RePredictor {
    /// Unique sorted levels from the training data. Unseen levels at
    /// predict-time map to a zero row (matches mgcv `bs="re"`).
    pub levels: Vec<f64>,
}

impl RePredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let basis = RandomEffectsBasis {
            levels: self.levels.clone(),
        };
        let raw = basis.evaluate(x_new);
        Ok(prepend_intercept(raw.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, _axis: usize) -> Result<Array2<f64>> {
        // Random effects are a step function over a categorical predictor —
        // derivative w.r.t. the input is identically zero (matches v0.x).
        let n = x_new.nrows();
        let p = 1 + self.levels.len();
        Ok(Array2::<f64>::zeros((n, p)))
    }

    /// Build the raw one-hot design + sorted levels for column
    /// `x_col_idx`. Shared with `Additive` so a single column slice flows
    /// through both code paths.
    pub(crate) fn fit_raw(
        x: ArrayView2<f64>,
        x_col_idx: usize,
    ) -> Result<RawReFit> {
        let x_col = x.column(x_col_idx);
        let basis = RandomEffectsBasis::from_data(x_col);
        let levels = basis.levels.clone();
        if levels.is_empty() {
            return Err(GammonError::InvalidParameter(
                "Re basis: x is empty".into(),
            ));
        }
        let x_view = x.slice(ndarray::s![.., x_col_idx..x_col_idx + 1]);
        let raw = basis.evaluate(x_view);
        Ok(RawReFit { levels, raw })
    }
}

/// Internal helper carrying the raw one-hot design + sorted levels.
pub(crate) struct RawReFit {
    pub levels: Vec<f64>,
    pub raw: Array2<f64>,
}

/// Random-effect basis (mgcv `bs="re"`). One-hot encoding over training
/// levels + identity penalty. The intercept column is retained (matches
/// mgcv: `s(g, bs="re")` always rides with an unpenalised constant) and
/// the identity penalty leaves the intercept row/col at zero.
pub struct Re;

impl DesignStrategy for Re {
    fn prepare(&self, x: ArrayView2<f64>) -> Result<PreparedDesign> {
        let fit = RePredictor::fit_raw(x, 0)?;
        let n_levels = fit.levels.len();
        let x_design = prepend_intercept(fit.raw.view());
        let p = x_design.ncols();

        let mut s_block = Array2::<f64>::zeros((p, p));
        for j in 0..n_levels {
            s_block[[1 + j, 1 + j]] = 1.0;
        }

        let (rank_s, log_pseudo_det_s) = rank_and_log_pseudo_det(s_block.view());
        let mp = p - rank_s;

        Ok(PreparedDesign {
            x_design,
            s_list: vec![s_block],
            rank_s_list: vec![rank_s],
            log_pseudo_det_s_list: vec![log_pseudo_det_s],
            mp,
            predictor: Predictor::Re(RePredictor { levels: fit.levels }),
        })
    }
}
