suppressMessages({library(mgcv); library(jsonlite)})
fam <- shash(); B <- 1e-2; PHIPEN <- 1e-3   # logeb bound + kurtosis penalty (mgcv defaults)

# mgcv l0 at a PARAM point (mu, tau=logsig, eps, phi=logdel): set eta via inverse links.
#   eta1=mu (identity); eta2 = log(exp(tau)-b)  [inverse logeb]; eta3=eps; eta4=phi.
l0_at <- function(y, mu, tau, eps, phi) {
  eta <- cbind(mu, log(exp(tau) - B), eps, phi)
  X <- diag(1, nrow=length(y), ncol=4); attr(X,"lpi") <- list(1L,2L,3L,4L)
  fam$ll(y=y, X=X, coef=rep(0,4), wt=rep(1,length(y)), family=fam, deriv=0, eta=eta)$l0
}

# Varied test points: skew +/-, scales, y above/below mu, kurtosis +/-.
pts <- data.frame(
  y   = c( 1.5, 2.3, -0.5,  0.7,  3.0, -1.2,  0.4,  2.1),
  mu  = c( 0.0, 1.0,  0.5, -0.3,  2.0, -1.0,  0.0,  1.5),
  tau = c(-0.5, 0.0, -1.0,  0.2,  0.3, -0.7,  0.5, -0.2),
  eps = c( 0.0, 0.2, -0.1,  0.0,  0.4, -0.3,  0.15, -0.25),
  phi = c( 0.0, 0.0,  0.3, -0.2,  0.1,  0.2, -0.15,  0.05)
)
h <- 1e-4
nm <- c("mu","tau","eps","phi")
out <- list()
for (i in seq_len(nrow(pts))) {
  p <- as.list(pts[i,]); y <- p$y
  base <- function(d) l0_at(y, p$mu+d["mu"], p$tau+d["tau"], p$eps+d["eps"], p$phi+d["phi"])
  z <- setNames(rep(0,4), nm)
  l0 <- base(z)
  # FD gradient (central) wrt each param
  l1 <- sapply(nm, function(k){ dp<-z; dp[k]<-h; dm<-z; dm[k]<--h; (base(dp)-base(dm))/(2*h) })
  # FD Hessian: diagonal + cross (lower-tri order mm,mt,me,mp,tt,te,tp,ee,ep,pp)
  H <- matrix(0,4,4)
  for (a in 1:4) for (b2 in a:4) {
    if (a==b2){ dp<-z; dp[a]<-h; dm<-z; dm[a]<--h; H[a,a]<-(base(dp)-2*l0+base(dm))/(h*h) }
    else { pp<-z;pp[a]<-h;pp[b2]<-h; pm<-z;pm[a]<-h;pm[b2]<--h; mp<-z;mp[a]<--h;mp[b2]<-h; mm<-z;mm[a]<--h;mm[b2]<--h
           H[a,b2]<-(base(pp)-base(pm)-base(mp)+base(mm))/(4*h*h); H[b2,a]<-H[a,b2] }
  }
  l2 <- c(H[1,1],H[1,2],H[1,3],H[1,4],H[2,2],H[2,3],H[2,4],H[3,3],H[3,4],H[4,4])
  out[[i]] <- list(y=y, mu=p$mu, tau=p$tau, eps=p$eps, phi=p$phi,
                   l0=l0, l1=as.numeric(l1), l2=as.numeric(l2))
}
fx <- list(schema_version=1L, name="shash_derivs_mgcv",
           description="mgcv shash per-obs l0 + finite-difference l1/l2 (param space mu,tau,eps,phi)",
           metadata=list(engine="mgcv", mgcv_version=as.character(packageVersion("mgcv")),
                         b_logeb=B, phiPen=PHIPEN, fd_h=h,
                         l2_order="mm,mt,me,mp,tt,te,tp,ee,ep,pp"),
           points=out)
cat(toJSON(fx, auto_unbox=TRUE, digits=14), file="tests/fixtures/shash_derivs_mgcv.json")
cat("wrote shash_derivs_mgcv.json with", length(out), "points\n")
cat("sample l0:", out[[1]]$l0, " l1:", round(out[[1]]$l1,4), "\n")
