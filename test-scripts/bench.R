#devtools::load_all()
library(sparse.group.slope)

generate.doubly.sparse <- function(L, d, d0, L0){
  row.reorder <- sample(d)
  B <- rbind(
    t(sapply(1:d0, FUN = function(idx)(sample(c(rnorm(L0[idx]), rep(0, L-L0[idx])))))),
    matrix(0, nrow=d-d0, ncol=L)
  )
  return(B[row.reorder,])
}

n <- 100
d <- 1000
L <- 200
d0 <- 3           # row-sparsity
L0 <- rep(2, d0)  # per-row-sparsity

B <- 10*generate.doubly.sparse(L, d, d0, L0)
X <- matrix(rnorm(n*d), nrow=n, ncol=d)
X[abs(X) < 0.5] = 0

print(mean(X==0))

p <- exp(X%*%B)/rowSums(exp(X%*%B))
#Y <- t(apply(p, MARGIN = 1, function(row)(rmultinom(1, 1, row))))
Y <- apply(p, MARGIN = 1, function(row)(which.max(rmultinom(1, 1, row))))
Y.test <- apply(p, MARGIN = 1, function(row)(which.max(rmultinom(1, 1, row))))

Y.gaussian <- X%*%B + matrix(rnorm(n*L), nrow=n, ncol=L) 


A<- 1/5
lambda <- A*sapply(1:d, function(i)(sqrt(log(d*exp(1)/i)/n)))
alpha <- A*sapply(1:L, function(j)(sqrt(log(L*exp(1)/j)/n)))

library(Matrix)
Z <- as(X, "dgCMatrix")
library(tictoc)
tic("mfista1")
model <- SparseGroupSLOPE(
  X = X, y = diag(L)[Y, ], eta = 0.9, tol = 8e-4, monotone = TRUE,
  accelerated = TRUE, lambda = lambda, alpha = alpha, BInit = 1, #B
  maxIter = 50
  )
r <- toc()
r
print(c("Iters:", model$iterations))
print(c("Per iters:", (r$toc - r$tic) / model$iterations))

if (FALSE) {
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


A<- 1/5
lambda <- A*sapply(1:d, function(i)(sqrt(log(d*exp(1)/i)/n)))
alpha <- A*sapply(1:L, function(j)(sqrt(log(L*exp(1)/j)/n)))

library(Matrix)
library(tictoc)
tic("mfista1")
model <- SparseGroupSLOPE(
  X=X, y=diag(L)[Y,], eta=0.9, tol = 1e-10, monotone=T, accelerated=T, lambda = lambda, alpha = alpha, BInit = 1, #B
  maxIter=1000, intercept=F
  )
r <- toc()
r
print(c("Iters:", model$iterations))
print(c("Per iters:", (r$toc - r$tic) / model$iterations))
print(B)
print(model$coefficients)
print(model$coefficients - B)
}
