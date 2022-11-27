devtools::load_all()

n <- 100
d <- 7
L <- 5
d0 <- 3           # row-sparsity
L0 <- rep(2, d0)  # per-row-sparsity

#B <- 10*generate.doubly.sparse(L, d, d0, L0)
B <- matrix(c(
  6.481, 0, 0, 0, 0, 0, 0,
  3.5761, 0, 0, 0, 0, 0, 6.4018,
  0, 0, 0, 0, 0, 0, -3.0419,
  0, 7.9408, 0, 0, 0, 0, 0,
  0, 5.395, 0, 0, 0, 0, 0
), nrow = 7, ncol = 5)
X <- matrix(rnorm(n*d), nrow=n, ncol=d)

p <- exp(X%*%B)/rowSums(exp(X%*%B))
#Y <- t(apply(p, MARGIN = 1, function(row)(rmultinom(1, 1, row))))
Y <- apply(p, MARGIN = 1, function(row)(which.max(rmultinom(1, 1, row))))
Y.test <- apply(p, MARGIN = 1, function(row)(which.max(rmultinom(1, 1, row))))

Y.gaussian <- X%*%B + matrix(rnorm(n*L), nrow=n, ncol=L)


A<- 1/5*sqrt(n)
lambda <- A*sapply(1:d, function(i)(sqrt(log(d*exp(1)/i)/n)))
alpha <- A*sapply(1:L, function(j)(sqrt(log(L*exp(1)/j)/n)))

library(Matrix)
library(tictoc)
tic("mfista1")
model <- SparseGroupSLOPE(
  X=X, y=diag(L)[Y,], eta=0.9, tol = 1e-8, monotone=T, accelerated=T, lambda = lambda, alpha = alpha, #B
  minStepSize = 1e-24, maxIter=1000, intercept=F, BInit = 1, standardize=T, saveData=T
  )
r <- toc()
r
print(c("Iters:", model$iterations))
print(c("Per iters:", (r$toc - r$tic) / model$iterations))
print(B)
print(model$coefficients)
print(model$coefficients - B)
