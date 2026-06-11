#!/usr/bin/env Rscript
# Ground-truth fixture for the shash outer REML / smoothing-parameter selection
# (TDD phase 5a — the mgcv confrontation the unit tests lacked).
#
# Fits a genuine 2-smooth shash GAMLSS:
#     y ~ shash(  mu  = beta0 + s(x0),
#                 tau = beta0 + s(x1),     # log-scale (logeb link)
#                 eps = const,             # skewness  (intercept only)
#                 phi = const )            # log-kurtosis (intercept only)
# and exports mgcv's OWN per-predictor model-matrix blocks Xb and penalty
# blocks S0b (unscaled), plus the selected smoothing parameters, total EDF,
# coefficients and fitted linear predictors.
#
# The gamrs test feeds these EXACT Xb/S0b into `fit_reml` and confronts:
#   - selected log-smoothing-params  rho_hat ~ log(mgcv sp),
#   - total effective degrees of freedom,
#   - fitted linear predictors eta_b = Xb %*% beta_hat_b  (basis-invariant).
# Using mgcv's own design columns means gamrs fits in mgcv's basis, so beta
# and eta are directly comparable; mgcv's internal stability reparam leaves the
# REML criterion (hence argmax + fit) invariant.

suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

set.seed(20260611)

n <- 400
x0 <- runif(n)
x1 <- runif(n)

# Smooth location + smooth log-scale; mild constant skew/kurtosis, drawn from
# the sinh-arcsinh law (mgcv parameterisation: W=sinh(del*asinh(z)-eps)~N(0,1)).
mu_t  <- 0.6 + 1.5 * sin(2 * pi * x0)
sig_t <- exp(-0.4 + 0.7 * sin(1.5 * pi * x1))   # nonlinear in x1 -> finite sp
eps_t <- 0.25
del_t <- 1.1
W <- rnorm(n)
z <- sinh((asinh(W) + eps_t) / del_t)
y <- mu_t + sig_t * del_t * z

fit <- gam(list(y ~ s(x0, k = 10, bs = "cr"), ~ s(x1, k = 10, bs = "cr"), ~ 1, ~ 1),
           family = shash(), method = "REML")

# Full linear-predictor model matrix with the per-predictor column index list.
lpmat <- predict(fit, type = "lpmatrix")
lpi <- attr(lpmat, "lpi")
stopifnot(length(lpi) == 4)

cf <- coef(fit)
eta <- predict(fit, type = "link")        # n x 4 fitted linear predictors
sp <- as.numeric(fit$sp)                   # one per smooth, formula order
edf_total <- sum(fit$edf)

# Per-block design + penalty (block-local). Blocks 1,2 carry one smooth each;
# 3,4 are intercept-only (no penalty).
block <- function(k) {
  cols <- lpi[[k]]
  Xb <- lpmat[, cols, drop = FALSE]
  pk <- length(cols)
  S0 <- matrix(0, pk, pk)
  rank <- 0L
  penalised <- FALSE
  # find a smooth whose global columns fall inside this block
  for (sm in fit$smooth) {
    gcols <- sm$first.para:sm$last.para
    if (all(gcols %in% cols)) {
      loc <- match(gcols, cols)
      Smooth <- sm$S[[1]]
      S0[loc, loc] <- Smooth
      rank <- as.integer(sm$rank)
      penalised <- TRUE
    }
  }
  list(X = Xb, p = pk, S0 = S0, rank = rank, penalised = penalised,
       coef = as.numeric(cf[cols]))
}

blocks <- lapply(1:4, block)

flat_rm <- function(M) as.numeric(t(M))   # row-major flatten for ndarray

out <- list(
  n = n,
  b = 1e-2,
  sp = sp,                       # mgcv selected smoothing params (formula order)
  log_sp = log(sp),
  edf_total = edf_total,
  p = sapply(blocks, function(b) b$p),
  penalised = sapply(blocks, function(b) b$penalised),
  rank = sapply(blocks, function(b) b$rank),
  X1 = flat_rm(blocks[[1]]$X), X2 = flat_rm(blocks[[2]]$X),
  X3 = flat_rm(blocks[[3]]$X), X4 = flat_rm(blocks[[4]]$X),
  S1 = flat_rm(blocks[[1]]$S0), S2 = flat_rm(blocks[[2]]$S0),
  coef1 = blocks[[1]]$coef, coef2 = blocks[[2]]$coef,
  coef3 = blocks[[3]]$coef, coef4 = blocks[[4]]$coef,
  eta = flat_rm(eta),            # n x 4 row-major
  y = y
)

outfile <- "tests/fixtures/shash_reml_mgcv.json"
write_json(out, outfile, digits = 16, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", outfile, "\n")
cat("sp:", sp, "  log(sp):", log(sp), "\n")
cat("edf_total:", edf_total, "\n")
cat("block p:", sapply(blocks, function(b) b$p),
    " rank:", sapply(blocks, function(b) b$rank),
    " penalised:", sapply(blocks, function(b) b$penalised), "\n")
