//! Thin-plate regression spline (TPRS) — 2-D (or higher) isotropic
//! smooth, mgcv's `bs="tp"`.
//!
//! Low-rank radial-basis implementation following Duchon thin-plate
//! construction (Wood 2003) with knots placed at a uniform-stride
//! subsample of data points. Produces a usable D-D isotropic smooth
//! with a single smoothing parameter.
//!
//! Construction (smoothness order m=2):
//!
//! 1. Pick `k` knot points (uniform stride over the data rows).
//! 2. Radial-basis matrix `E[i, j] = η(||x_i − xk_j||)` with
//!    dimension-dependent kernel `η` (Wood §5.5.1).
//! 3. Polynomial null-space `T[i, :] = [x_i.0, x_i.1, …]` (linear part).
//! 4. Design `[E | T_lin]` of width `k + D`.
//! 5. Penalty `S = block_diag(E_kk, 0_{D×D})`.

use ndarray::{Array2, ArrayView2};

use crate::error::{GamrsError, Result};

use super::{prepend_intercept, prepend_zero_column};

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct TpsPredictor {
    pub cols: Vec<usize>,
    pub knots: Array2<f64>,
}

impl TpsPredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw = self.tps_design(x_new, None);
        Ok(prepend_intercept(raw.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw_d1 = self.tps_design(x_new, Some(axis));
        Ok(prepend_zero_column(raw_d1.view()))
    }

    fn check_cols(&self, x_new: ArrayView2<f64>) -> Result<()> {
        for &c in &self.cols {
            if c >= x_new.ncols() {
                return Err(GamrsError::InvalidParameter(format!(
                    "TpsPredictor: term reads column {c} but x has only {} columns",
                    x_new.ncols()
                )));
            }
        }
        Ok(())
    }

    fn tps_design(&self, x_new: ArrayView2<f64>, deriv_axis: Option<usize>) -> Array2<f64> {
        let d = self.cols.len();
        let k = self.knots.nrows();
        let n = x_new.nrows();
        let mut out = Array2::<f64>::zeros((n, k + d));
        let deriv_margin: Option<usize> =
            deriv_axis.and_then(|ax| self.cols.iter().position(|&c| c == ax));
        if deriv_axis.is_some() && deriv_margin.is_none() {
            return out;
        }
        for i in 0..n {
            for j in 0..k {
                let mut r2 = 0.0_f64;
                let mut diff_axis = 0.0_f64;
                for m in 0..d {
                    let dxm = x_new[[i, self.cols[m]]] - self.knots[[j, m]];
                    r2 += dxm * dxm;
                    if Some(m) == deriv_margin {
                        diff_axis = dxm;
                    }
                }
                let r = r2.sqrt();
                if deriv_margin.is_some() {
                    out[[i, j]] = tps_eta_deriv(r, d, diff_axis);
                } else {
                    out[[i, j]] = tps_eta(r, d);
                }
            }
            for m in 0..d {
                let val = x_new[[i, self.cols[m]]];
                if let Some(m_idx) = deriv_margin {
                    out[[i, k + m]] = if m == m_idx { 1.0 } else { 0.0 };
                } else {
                    out[[i, k + m]] = val;
                }
            }
        }
        out
    }
}

pub(super) struct TpsTermFit {
    pub(super) design: Array2<f64>,
    pub(super) s_smooth: Array2<f64>,
    pub(super) predictor: TpsPredictor,
}

impl TpsTermFit {
    pub(super) fn build(x: ArrayView2<f64>, cols: &[usize], k: usize) -> Result<Self> {
        if cols.len() < 2 {
            return Err(GamrsError::InvalidParameter(format!(
                "tps term needs at least 2 input dims (got {})",
                cols.len()
            )));
        }
        for i in 0..cols.len() {
            for j in (i + 1)..cols.len() {
                if cols[i] == cols[j] {
                    return Err(GamrsError::InvalidParameter(format!(
                        "tps term must have distinct cols (cols[{i}] == cols[{j}] == {})",
                        cols[i]
                    )));
                }
            }
        }
        for &c in cols {
            if c >= x.ncols() {
                return Err(GamrsError::InvalidParameter(format!(
                    "tps term reads column {c} but x has only {} columns",
                    x.ncols()
                )));
            }
        }
        let n = x.nrows();
        let d = cols.len();
        if k < d + 1 {
            return Err(GamrsError::InvalidParameter(format!(
                "tps term needs k > d+1 (got k={k}, d={d})"
            )));
        }
        if k > n {
            return Err(GamrsError::InvalidParameter(format!(
                "tps term k={k} cannot exceed n={n}"
            )));
        }
        let mut knots = Array2::<f64>::zeros((k, d));
        if n == k {
            for j in 0..k {
                for m in 0..d {
                    knots[[j, m]] = x[[j, cols[m]]];
                }
            }
        } else {
            for j in 0..k {
                let frac = (j as f64) / ((k - 1) as f64);
                let idx = (frac * ((n - 1) as f64)).round() as usize;
                let idx = idx.min(n - 1);
                for m in 0..d {
                    knots[[j, m]] = x[[idx, cols[m]]];
                }
            }
        }
        let predictor = TpsPredictor {
            cols: cols.to_vec(),
            knots: knots.clone(),
        };
        let raw_design = predictor.tps_design(x, None);
        let mut e_kk = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                let mut r2 = 0.0_f64;
                for m in 0..d {
                    let dxm = knots[[r, m]] - knots[[c, m]];
                    r2 += dxm * dxm;
                }
                e_kk[[r, c]] = tps_eta(r2.sqrt(), d);
            }
        }
        for r in 0..k {
            for c in (r + 1)..k {
                let avg = 0.5 * (e_kk[[r, c]] + e_kk[[c, r]]);
                e_kk[[r, c]] = avg;
                e_kk[[c, r]] = avg;
            }
        }
        let p_term = k + d;
        let mut s_smooth = Array2::<f64>::zeros((p_term, p_term));
        for r in 0..k {
            for c in 0..k {
                s_smooth[[r, c]] = e_kk[[r, c]];
            }
        }
        let s_max = leading_eig_abs(s_smooth.view())?;
        if s_max > 0.0 {
            s_smooth /= s_max;
        }
        Ok(Self {
            design: raw_design,
            s_smooth,
            predictor,
        })
    }
}

fn tps_eta(r: f64, d: usize) -> f64 {
    if r < 1e-12 {
        return 0.0;
    }
    match d {
        1 => r.powi(3) / 12.0,
        2 => r.powi(2) * r.ln() / (8.0 * std::f64::consts::PI),
        3 => -r / 8.0,
        _ => {
            let power = 2 * 2 - d as i32;
            if power > 0 {
                if power % 2 == 0 { r.powi(power) * r.ln() } else { r.powi(power) }
            } else {
                r.ln()
            }
        }
    }
}

fn tps_eta_deriv(r: f64, d: usize, diff: f64) -> f64 {
    if r < 1e-12 {
        return 0.0;
    }
    let eta_prime = match d {
        1 => r.powi(2) / 4.0,
        2 => (2.0 * r * r.ln() + r) / (8.0 * std::f64::consts::PI),
        3 => -1.0 / 8.0,
        _ => {
            let power = 2 * 2 - d as i32;
            if power > 0 {
                if power % 2 == 0 {
                    (power as f64) * r.powi(power - 1) * r.ln() + r.powi(power - 1)
                } else {
                    (power as f64) * r.powi(power - 1)
                }
            } else {
                1.0 / r
            }
        }
    };
    eta_prime * diff / r
}

fn leading_eig_abs(s: ArrayView2<f64>) -> Result<f64> {
    use ndarray_linalg::Eigh;
    let s_owned = s.to_owned();
    let (eigs, _) = s_owned
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(format!("eigh failed in tps rescale: {e}")))?;
    Ok(eigs.iter().map(|e| e.abs()).fold(0.0_f64, f64::max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tps_design_shape() {
        let n = 100;
        let d = 2;
        let k = 12;
        let mut x = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            x[[i, 0]] = (i as f64) / (n as f64);
            x[[i, 1]] = ((i as f64) * 0.31).sin();
        }
        let fit = TpsTermFit::build(x.view(), &[0, 1], k).unwrap();
        assert_eq!(fit.design.shape(), &[n, k + d]);
        assert_eq!(fit.s_smooth.shape(), &[k + d, k + d]);
        for r in k..(k + d) {
            for c in k..(k + d) {
                assert_eq!(fit.s_smooth[[r, c]], 0.0);
            }
        }
    }
}
