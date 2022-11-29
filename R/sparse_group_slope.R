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