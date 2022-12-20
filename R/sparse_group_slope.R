predict.sgs <- function(model, X=NULL){
  if (is.null(X)){
    X <- model$data$X
  } else {
    if (model$standardize) {
      X <- sweep(X, 2, model$col.scale, "*")*nrow(X)
    }
  }
  return(sweep(X %*% model$coefficients, 2, model$intercepts, "+"))
}

fit.cv <- function(X, y, grid.alpha, grid.lambda, cv.indices) {
    library(purrr)
    library(sparse.group.slope)
    cvs <- matrix(-1, length(grid.alpha), length(grid.lambda))
    n <- nrow(X)
    d <- ncol(X)
    L <- nlevels(y)

    Binits <- as.vector(rep(0, length(cv.indices)), "list")
    for (i in seq_along(grid.alpha)) {
        alpha <- grid.alpha[i]
        for (j in seq_along(grid.lambda)) {
            A <- grid.lambda[j]
            lambda <- alpha * A * sapply(
                1:d, function(i) (sqrt(log(d * exp(1) / i) / n))
            )
            kappa <- (1 - alpha) * A * sapply(
                1:L, function(j) (sqrt(log(L * exp(1) / j) / n ))
            )
            fold <- function(i){
                test.group <- cv.indices[[i]]
                model <- sparse.group.slope::SparseGroupSLOPE(
                    X = X[-test.group, ], y = diag(L)[y[-test.group], ],
                    eta = 0.5, tol = 1e-4, monotone = F, accelerated = T,
                    lambda = lambda, alpha = kappa, BInit = Binits[[i]],
                    maxIter = 100000, standardize = FALSE, minStepSize = 1e-5,
                    stepSize = 0.001,
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

            res <- lapply(1:length(cv.indices), fold)
            cvs[i, j] <- mean(unlist(map(res, pluck, "error")))
        }
        return(cvs)
    }
}

compare <- function(
    X, y, grid.alpha, grid.lambda, folds, n.boostrap = 100, test.freq = 0.75,
    indices = NULL, train.indices = NULL, model.cv = stop("missing model_cv"),
    model.fit_evaluate = stop("missing model.fit_evaluate"), ...
) {
    n <- nrow(X)
    d <- ncol(X)
    L <- ncol(y)

    n.train <- floor(n * test.freq)
    if (is.null(indices)) {
        idx <- sample(1:n.train, n.train)
        indices <- as.list(split(idx, sort(seq_along(idx) %% folds)))
        names(indices) <- NULL
    }
    if (is.null(train.indices)) {
        train.indices <- lapply(1:n.boostrap, function(i)(sample(1:n, size = n.train)))
    }

    results <- list()
    for (B in 1:n.boostrap) {
        train.idx <- train.indices[[B]]
        Xtrain <- X[train.idx, ]
        ytrain <- y[train.idx]
        Xtest <- X[-train.idx, ]
        ytest <- y[-train.idx]


        cvs <- model.cv(
            Xtrain, ytrain, grid.alpha = grid.alpha, grid.lambda = grid.lambda,
            cv.indices = indices, ...
        )
        loc <- which(cvs == min(cvs), arr.ind = TRUE)[1,]
        alpha <- grid.alpha[loc["row"]]
        lambda <- grid.lambda[loc["col"]]

        results[[B]] <- model.fit_evaluate(
            Xtrain, ytrain, Xtest, ytest, alpha = alpha, lambda = lambda, ...
        )
    }
    return(list(
        results=results,
        indices=indices,
        train.idx=train.idx
    ))
}