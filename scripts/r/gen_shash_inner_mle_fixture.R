#!/usr/bin/env Rscript
# Ground-truth fixture for the shash joint inner solver (TDD phase 4b).
#
# Confronts the dense penalised Newton with mgcv's shash MLE in the cleanest
# case: an INTERCEPT-ONLY model (no smooths ⇒ no penalty ⇒ pure maximum
# likelihood). mgcv's `gam(list(y~1, ~1, ~1, ~1), family=shash())` therefore
# returns argmax Σ ℓ₀ over the 4 scalar linear-predictor coefficients
# (μ, η₂ for the logeb scale link, ε, φ). The gamrs `fit_inner` with
# intercept-only designs and zero penalty maximises the same Σ ℓ₀ (its per-obs
# phiPen matches mgcv's), so it must recover the same 4 coefficients.
#
# Data are drawn from a known sinh-arcsinh law (mild skew + kurtosis) so the
# MLE has non-trivial ε and φ — but the *oracle* is mgcv's fitted coefficients,
# not the planted truth.

suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

set.seed(20260611)

n <- 600
# Planted sinh-arcsinh params (mgcv's parameterisation: z=(y-mu)/(sig*del),
# W = sinh(del*asinh(z) - eps) ~ N(0,1)).
mu_t  <- 1.0
sig_t <- 0.7
eps_t <- 0.3       # skewness
del_t <- 1.1       # kurtosis (delta>1 -> lighter tails)

W <- rnorm(n)
z <- sinh((asinh(W) + eps_t) / del_t)
y <- mu_t + sig_t * del_t * z

# Intercept-only shash GAMLSS fit -> 4 scalar coefficients.
# (`fit$converged` is left unset for this family path; we instead verify the
# coefficients are a genuine MLE via an independent BFGS optimum below.)
fit <- gam(list(y ~ 1, ~ 1, ~ 1, ~ 1), family = shash(), method = "REML")
cf <- as.numeric(coef(fit))   # order: mu int, tau(eta2) int, eps int, phi int
stopifnot(length(cf) == 4, all(is.finite(cf)))

# Independent cross-check: BFGS on the shash negative log-likelihood (coef
# space, eta2 -> tau via logeb) must land on the same point.
b <- 1e-2; phiPen <- 1e-3
nll <- function(p) {
  mu <- p[1]; tau <- log(exp(p[2]) + b); eps <- p[3]; phi <- p[4]
  sig <- exp(tau); del <- exp(phi); zz <- (y - mu) / (sig * del)
  dT <- del * asinh(zz) - eps
  l0 <- -tau - 0.5 * log(2 * pi) + log(cosh(dT)) - 0.5 * log(zz^2 + 1) -
    0.5 * sinh(dT)^2 - phiPen * phi^2
  -sum(l0)
}
o <- optim(c(0, 0, 0, 0), nll, method = "BFGS",
           control = list(reltol = 1e-12, maxit = 500))
stopifnot(max(abs(cf - o$par)) < 1e-4)   # gam coef == independent MLE

# Total log-likelihood at the MLE (sum of per-obs l0), for a sanity anchor.
ll <- as.numeric(logLik(fit))

out <- list(
  n = n,
  y = y,
  b = 1e-2,                 # logeb bound (shash default)
  phiPen = 1e-3,
  coef_mu  = cf[1],
  coef_tau = cf[2],         # eta2 (pre-logeb)
  coef_eps = cf[3],
  coef_phi = cf[4],
  loglik = ll
)

outfile <- "tests/fixtures/shash_inner_mle_mgcv.json"
write_json(out, outfile, digits = 16, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", outfile, "\n")
cat("coef (mu, eta2, eps, phi):", cf, "\n")
cat("logLik:", ll, "\n")
