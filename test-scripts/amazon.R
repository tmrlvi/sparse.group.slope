library(dplyr)
library(stringr)
library(Matrix)
library(sparse.group.slope)

df <- farff::readARFF("./data/amazon_reviews/cleanedAmazon.arff")

X <- df[-ncol(df)]
X <- sweep(X, 2, sqrt(colSums(X^2)), "/")
X <- as(as.matrix(X), "dgCMatrix")
y <- df[,ncol(df)]


cl <- parallel::makeCluster(5)
doParallel::registerDoParallel(cl)
# Do 4-fold cross validation on a lambda sequence of length 100.
# The sequence is decreasing from the data derived lambda.max to 0.2*lambda.max

fit.cv <- msgl::cv(X, y, fold = 10, lambda = 1e-4, d=100, use_parallel = T, standardize = T, )

parallel::stopCluster(cl)
# Print information about models
# and cross validation errors (estimated expected generalization error)
print(fit.cv)

library(sparse.group.slope)
n <- nrow(X)
d <- ncol(X)
L <- nlevels(y)

idx <- sample(1:n, n)
folds <- 10
fold.size <- ceiling(n/folds)

As <- seq(0.01, 0.001, -0.001)
cvs <- rep(0, length(As))
all_results <- list()
Binits <- as.list(rep(0, folds))

for (j in 1:length(As)) {
  A <- As[j]

  lambda <- A*sapply(1:d, function(i)(sqrt(log(d*exp(1)/i)/n)))
  alpha <- A*sapply(1:L, function(j)(sqrt(log(L*exp(1)/j)/n)))

  fold <- function(i){
    #test.group <- fit.cv$cv.indices[[i]]
    test.group <- idx[(i*fold.size+1):((i+1)*fold.size+1)]
    model <- sparse.group.slope::SparseGroupSLOPE(
      X=X[-test.group,], y=diag(L)[y[-test.group],], eta=0.9, tol = 1e-5, 
      monotone=F, accelerated=T, lambda = lambda, alpha = alpha, BInit = Binits[[i]], #B
      maxIter=1000000, standardize=FALSE
      )
    list(
      B = model$coefficient,
      A = A,
      non_zeros = sum(model$coefficients != 0),
      non_zero_groups = sum(apply(model$coefficients != 0, 1, any)),
      iters = model$iterations,
      error = mean(apply(predict.sgs(model, X[test.group,]), 1, which.max) != as.numeric(y[test.group]))
    )
  }

  res <- lapply(1:folds, fold)
  all_results <- append(all_results, res)
  Binits <- map(res, pluck, "B")
  iters <- unlist(map(res, pluck, "iters"))
  cvs[j] <- mean(unlist(map(res, pluck, "error")))
  print(c(A, mean(iters), cvs[j]))
}
plot(As, cvs)
