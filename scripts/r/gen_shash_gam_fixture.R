#!/usr/bin/env Rscript
# Ground-truth fixture for the END-TO-END shash GAMLSS driver (`fit_shash`,
# TDD phase 6a). Unlike `gen_shash_reml_fixture.R` — which exports mgcv's OWN
# per-block design + penalty so the gamrs REML core fits in mgcv's basis — this
# fixture exports only the RAW covariates, the response, and mgcv's fitted
# summaries. gamrs's `fit_shash` must build its OWN CR designs from `x0`/`x1`
# and STILL recover mgcv's fitted linear predictors, total EDF and quantiles.
# That is the real end-to-end parity proof: it exercises gamrs's design
# construction (CR knot placement + sum-to-zero centring + intercept), not just
# its REML criterion.
#
# Model (identical generative law to the reml fixture so the two are siblings):
#     y ~ shash(  mu  = beta0 + s(x0, k=10),
#                 tau = beta0 + s(x1, k=10),   # log-scale (logeb link, b=1e-2)
#                 eps = const,                 # skewness  (intercept only)
#                 phi = const )                # log-kurtosis (intercept only)
#
# Exports:
#   x0, x1   raw covariates (length n)            -> gamrs builds Cr{col,k=10}
#   y        response (length n)
#   eta      n x 4 fitted linear predictors (mu, tau, eps, phi), row-major
#   edf_total  sum(fit$edf)
#   b          logeb bound (1e-2)
#   q10/q50/q90  mgcv-derived per-obs quantiles at p in {0.1,0.5,0.9}, computed
#                from the fitted params with the SAME shash quantile formula the
#                gamrs driver uses (mirrors R qgam's .shashQf):
#                  q(p) = mu + del*sig*sinh( (asinh(qnorm(p)) + eps) / del )
#                with mu=eta[,1], sig=exp(tau)=exp(log(exp(eta[,2])+b)),
#                     eps=eta[,3], del=exp(eta[,4]).

suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

set.seed(20260611)

n <- 400
x0 <- runif(n)
x1 <- runif(n)

mu_t  <- 0.6 + 1.5 * sin(2 * pi * x0)
sig_t <- exp(-0.4 + 0.7 * sin(1.5 * pi * x1))
eps_t <- 0.25
del_t <- 1.1
W <- rnorm(n)
z <- sinh((asinh(W) + eps_t) / del_t)
y <- mu_t + sig_t * del_t * z

# bs="cr": cubic regression spline — MUST match gamrs's `Cr` term. (mgcv's
# default `s(x, k)` is a THIN-PLATE spline `tprs`, a different basis; comparing
# gamrs Cr to an mgcv thin-plate fit is apples-to-oranges and was the sole
# source of the earlier ~1.2e-2 end-to-end gap.)
fit <- gam(list(y ~ s(x0, k = 10, bs = "cr"), ~ s(x1, k = 10, bs = "cr"), ~ 1, ~ 1),
           family = shash(), method = "REML")

b <- 1e-2
eta <- predict(fit, type = "link")        # n x 4 fitted linear predictors
edf_total <- sum(fit$edf)

# Per-obs fitted params on the response scale (mgcv shash links).
mu_hat  <- eta[, 1]
tau_hat <- log(exp(eta[, 2]) + b)          # logeb link inverse
sig_hat <- exp(tau_hat)
eps_hat <- eta[, 3]
del_hat <- exp(eta[, 4])

# shash quantile function (mirrors R qgam .shashQf; same form as gamrs driver).
shash_qf <- function(p) {
  zp <- qnorm(p)
  mu_hat + del_hat * sig_hat * sinh((asinh(zp) + eps_hat) / del_hat)
}
q10 <- shash_qf(0.10)
q50 <- shash_qf(0.50)
q90 <- shash_qf(0.90)

flat_rm <- function(M) as.numeric(t(M))   # row-major flatten for ndarray

out <- list(
  n = n,
  b = b,
  edf_total = edf_total,
  x0 = as.numeric(x0),
  x1 = as.numeric(x1),
  y = as.numeric(y),
  eta = flat_rm(eta),            # n x 4 row-major
  q10 = as.numeric(q10),
  q50 = as.numeric(q50),
  q90 = as.numeric(q90)
)

outfile <- "tests/fixtures/shash_gam_mgcv.json"
write_json(out, outfile, digits = 16, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", outfile, "\n")
cat("edf_total:", edf_total, "\n")
cat("eta col means:", colMeans(eta), "\n")
cat("q50 range:", range(q50), "\n")
