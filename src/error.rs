use thiserror::Error;

#[derive(Debug, Error)]
pub enum GamrsError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("singular system: {0}")]
    SingularSystem(String),
    #[error("solver did not converge after {iters} iterations (last grad norm = {grad_norm:.3e})")]
    NotConverged { iters: usize, grad_norm: f64 },
    #[error("linalg failure: {0}")]
    Linalg(String),
}

pub type Result<T> = std::result::Result<T, GamrsError>;
