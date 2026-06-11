suppressMessages({library(mgcv); library(jsonlite)})
fam <- shash(); B <- 1e-2; PHIPEN <- 1e-3   # logeb bound + kurtosis penalty (mgcv defaults)

# mgcv l0 at an ETA point (eta1,eta2,eta3,eta4): the family applies its own
# links internally (mu identity; tau via logeb tau=log(exp(eta2)+b); eps/phi
# identity) and adds phiPen. So we feed RAW eta here — no inverse-link round trip.
l0_at_eta <- function(y, e1, e2, e3, e4) {
  eta <- cbind(e1, e2, e3, e4)
  X <- diag(1, nrow=length(y), ncol=4); attr(X,"lpi") <- list(1L,2L,3L,4L)
  fam$ll(y=y, X=X, coef=rep(0,4), wt=rep(1,length(y)), family=fam, deriv=0, eta=eta)$l0
}

# Varied eta test points: skew +/-, scales (via eta2), y above/below mu (=eta1),
# kurtosis +/- (via eta4).
pts <- data.frame(
  y    = c( 1.5, 2.3, -0.5,  0.7,  3.0, -1.2,  0.4,  2.1),
  eta1 = c( 0.0, 1.0,  0.5, -0.3,  2.0, -1.0,  0.0,  1.5),
  eta2 = c(-0.5, 0.0, -1.0,  0.2,  0.3, -0.7,  0.5, -0.2),
  eta3 = c( 0.0, 0.2, -0.1,  0.0,  0.4, -0.3,  0.15, -0.25),
  eta4 = c( 0.0, 0.0,  0.3, -0.2,  0.1,  0.2, -0.15,  0.05)
)
h <- 1e-4
nm <- c("eta1","eta2","eta3","eta4")
out <- list()
for (i in seq_len(nrow(pts))) {
  p <- as.list(pts[i,]); y <- p$y
  base <- function(d) l0_at_eta(y, p$eta1+d["eta1"], p$eta2+d["eta2"],
                                   p$eta3+d["eta3"], p$eta4+d["eta4"])
  z <- setNames(rep(0,4), nm)
  l0 <- base(z)
  # FD gradient (central) wrt each eta
  l1 <- sapply(nm, function(k){ dp<-z; dp[k]<-h; dm<-z; dm[k]<--h; (base(dp)-base(dm))/(2*h) })
  # FD Hessian: diagonal + cross (lower-tri order mm,mt,me,mp,tt,te,tp,ee,ep,pp)
  H <- matrix(0,4,4)
  for (a in 1:4) for (b2 in a:4) {
    if (a==b2){ dp<-z; dp[a]<-h; dm<-z; dm[a]<--h; H[a,a]<-(base(dp)-2*l0+base(dm))/(h*h) }
    else { pp<-z;pp[a]<-h;pp[b2]<-h; pm<-z;pm[a]<-h;pm[b2]<--h; mp<-z;mp[a]<--h;mp[b2]<-h; mm<-z;mm[a]<--h;mm[b2]<--h
           H[a,b2]<-(base(pp)-base(pm)-base(mp)+base(mm))/(4*h*h); H[b2,a]<-H[a,b2] }
  }
  l2 <- c(H[1,1],H[1,2],H[1,3],H[1,4],H[2,2],H[2,3],H[2,4],H[3,3],H[3,4],H[4,4])
  out[[i]] <- list(y=y, eta1=p$eta1, eta2=p$eta2, eta3=p$eta3, eta4=p$eta4,
                   l0=l0, l1=as.numeric(l1), l2=as.numeric(l2))
}
fx <- list(schema_version=1L, name="shash_derivs_eta_mgcv",
           description="mgcv shash per-obs l0 + finite-difference l1/l2 (eta space; logeb link on tau)",
           metadata=list(engine="mgcv", mgcv_version=as.character(packageVersion("mgcv")),
                         b_logeb=B, phiPen=PHIPEN, fd_h=h,
                         links="mu=identity,tau=logeb(b),eps=identity,phi=identity",
                         l2_order="mm,mt,me,mp,tt,te,tp,ee,ep,pp"),
           points=out)
cat(toJSON(fx, auto_unbox=TRUE, digits=14), file="tests/fixtures/shash_derivs_eta_mgcv.json")
cat("wrote shash_derivs_eta_mgcv.json with", length(out), "points\n")
cat("sample l0:", out[[1]]$l0, " l1:", round(out[[1]]$l1,4), "\n")
