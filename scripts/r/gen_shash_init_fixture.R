#!/usr/bin/env Rscript
# Ground-truth fixture for the shash GAMLSS initialiser (TDD phase 3).
#
# Reproduces mgcv's `shash` `initialize` expression (gamlss.r:3974-4024) in the
# ZERO-PENALTY limit, where mgcv's `pen.reg` reduces exactly to OLS
# (`qr.coef`). That limit is the well-defined oracle we match bit-for-bit:
#   beta_mu  = pen.reg(X_mu , 0, y)                       # location, identity link
#   lres     = log|y - X_mu %*% beta_mu|                  # log abs residuals
#   beta_tau = pen.reg(X_tau, 0, lres)                    # log-scale, via logeb (init ignores b)
#   beta_eps = 0 ; beta_phi = 0                           # identity links, target linkfun(0)=0
#
# mgcv's adaptive EDF-targeting penalty weight (the `while` loops in pen.reg)
# is a heuristic regulariser for the warm start only — Newton refines it and
# Phase-6 parity does not depend on its exact value — so we do NOT replicate it.
# We confront mgcv where it is exact (OLS) and property-test the penalised path.
#
# Designs are moderately conditioned (rnorm, n >> p) so the gamrs normal-
# equations + Cholesky OLS matches qr.coef to ~1e-10.

suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

set.seed(20260611)

n  <- 50
p_mu <- 4; p_tau <- 3; p_eps <- 2; p_phi <- 2

# Block designs (intercept column + smooth-like covariates), distinct widths so
# the test also exercises per-block column routing.
mkX <- function(p) {
  X <- matrix(rnorm(n * p), n, p)
  X[, 1] <- 1                      # intercept-like first column
  X
}
X_mu  <- mkX(p_mu)
X_tau <- mkX(p_tau)
X_eps <- mkX(p_eps)
X_phi <- mkX(p_phi)

# Response: heteroskedastic-ish so log|resid| has real signal for the tau block.
mu_true  <- X_mu  %*% c(0.5, 1.2, -0.7, 0.3)
sig_true <- exp(X_tau %*% c(-0.2, 0.4, 0.1))
y <- as.numeric(mu_true + sig_true * rnorm(n))

# Zero penalty roots (square root of total penalty), one per block.
E0 <- function(p) matrix(0, p, p)

# --- mgcv initialize, zero-penalty limit ---------------------------------
beta_mu <- mgcv:::pen.reg(X_mu, E0(p_mu), y)
lres    <- log(abs(y - as.numeric(X_mu %*% beta_mu)))   # identity mu link
beta_tau <- mgcv:::pen.reg(X_tau, E0(p_tau), lres)
beta_eps <- rep(0, p_eps)   # identity link, regress on linkfun(0)=0  -> 0
beta_phi <- rep(0, p_phi)

stopifnot(all(is.finite(beta_mu)), all(is.finite(beta_tau)))

out <- list(
  n = n,
  p = list(mu = p_mu, tau = p_tau, eps = p_eps, phi = p_phi),
  # row-major flattening (C order) for direct ndarray::Array2::from_shape_vec
  X_mu  = as.numeric(t(X_mu)),
  X_tau = as.numeric(t(X_tau)),
  X_eps = as.numeric(t(X_eps)),
  X_phi = as.numeric(t(X_phi)),
  y = y,
  lres = lres,
  beta_mu  = as.numeric(beta_mu),
  beta_tau = as.numeric(beta_tau),
  beta_eps = beta_eps,
  beta_phi = beta_phi
)

outfile <- "tests/fixtures/shash_init_mgcv.json"
write_json(out, outfile, digits = 16, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", outfile, "\n")
cat("beta_mu :", beta_mu, "\n")
cat("beta_tau:", beta_tau, "\n")
